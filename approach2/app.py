import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

# Enforce offline behavior for Hugging Face-backed components after initial setup.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import faiss
import numpy as np
import streamlit as st
from llama_cpp import Llama
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer


# --------------------------
# Config
# --------------------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_FILE_NAME = "phi-3-mini-4k-instruct-q4_K_M.gguf"

HIGH_THRESHOLD = 0.70
MEDIUM_THRESHOLD = 0.50
FUZZY_THRESHOLD = 80
TOP_K = 3

UNKNOWN_MESSAGE = (
    "I'm sorry, I couldn't find an answer in the official NUST FAQ. "
    "Please try rephrasing or ask another admissions-related question."
)

llm = None


# --------------------------
# Data and Index Loading
# --------------------------
def _resolve_faq_path() -> Path:
    """Try common FAQ locations; require local file only."""
    candidates = [
        Path("faq.json"),
        Path("data/faq.json"),
        Path("data/nust_faq.json"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "FAQ JSON file not found. Expected one of: faq.json, data/faq.json, data/nust_faq.json"
    )


@st.cache_resource(show_spinner=False)
def load_faq_data() -> Tuple[List[Dict[str, str]], List[str], List[str], Path]:
    """Load FAQ entries and split into aligned question/answer arrays."""
    faq_path = _resolve_faq_path()
    with faq_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError("FAQ JSON must be a list of objects with 'question' and 'answer' keys.")

    entries: List[Dict[str, str]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        q = str(item.get("question", "")).strip()
        a = str(item.get("answer", "")).strip()
        if q and a:
            entries.append({"question": q, "answer": a})
        else:
            # Skip malformed rows safely instead of crashing runtime.
            _ = i

    if not entries:
        raise ValueError("No valid FAQ entries found in JSON file.")

    questions = [e["question"] for e in entries]
    answers = [e["answer"] for e in entries]
    return entries, questions, answers, faq_path


@st.cache_resource(show_spinner=False)
def load_embedder() -> SentenceTransformer:
    """Load the embedding model from local cache/files only."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu", local_files_only=True)


@st.cache_resource(show_spinner=False)
def build_index(questions: List[str]) -> Tuple[faiss.IndexFlatIP, np.ndarray]:
    """Build FAISS IP index on normalized embeddings (IP == cosine for unit vectors)."""
    embedder = load_embedder()
    embeddings = embedder.encode(
        questions,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=32,
    ).astype(np.float32)

    # L2-normalize to make inner product equivalent to cosine similarity.
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index, embeddings


# --------------------------
# LLM: Lazy Loading
# --------------------------
def _llm_model_path() -> Path:
    custom_path = os.getenv("PHI3_MODEL_PATH", "").strip()
    if custom_path:
        return Path(custom_path)
    return Path(LLM_FILE_NAME)


def get_llm() -> Llama:
    """Load Phi-3 only when needed to keep memory use low."""
    global llm
    if llm is not None:
        return llm

    model_path = _llm_model_path()
    if not model_path.exists():
        raise FileNotFoundError(
            f"LLM model file not found at: {model_path}. "
            "Download phi-3-mini-4k-instruct-q4_K_M.gguf once and place it locally."
        )

    llm = Llama(
        model_path=str(model_path),
        n_ctx=2048,
        n_threads=4,
        verbose=False,
    )
    return llm


# --------------------------
# Retrieval + Decision Logic
# --------------------------
def is_compound_query(query: str, top_scores: np.ndarray) -> bool:
    q = f" {query.lower()} "
    markers = [" and ", " or ", " vs ", " versus ", " compare ", " difference ", " both "]
    marker_hit = any(m in q for m in markers)

    # If multiple retrieved items are reasonably strong, treat as multi-part intent.
    multi_hit = int(np.sum(top_scores >= MEDIUM_THRESHOLD)) > 1
    return marker_hit or multi_hit


def build_grounded_prompt(query: str, top_indices: np.ndarray, entries: List[Dict[str, str]]) -> str:
    excerpts = []
    for idx in top_indices:
        faq = entries[int(idx)]
        excerpts.append(f"Q: {faq['question']}\nA: {faq['answer']}")

    joined_excerpts = "\n\n".join(excerpts)
    return (
        "You are a calm, respectful admissions assistant for NUST. "
        "Use ONLY the following FAQ excerpts to answer the student's question. "
        "If the information is not present, say you don't know.\n\n"
        f"FAQ excerpts:\n{joined_excerpts}\n\n"
        f"Student question: {query}\n"
        "Answer:"
    )


def fuzzy_fallback(query: str, questions: List[str], answers: List[str]) -> Tuple[str, Dict[str, str]]:
    best_score = -1
    best_idx = -1

    for i, q in enumerate(questions):
        score = fuzz.token_set_ratio(query, q)
        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx >= 0 and best_score > FUZZY_THRESHOLD:
        return answers[best_idx], {
            "source": "fuzzy_fallback",
            "confidence": f"{best_score:.1f}",
            "matched_question": questions[best_idx],
        }

    return UNKNOWN_MESSAGE, {
        "source": "unknown",
        "confidence": f"{best_score:.1f}" if best_score >= 0 else "0.0",
        "matched_question": "",
    }


def get_answer(
    query: str,
    index: faiss.IndexFlatIP,
    entries: List[Dict[str, str]],
    questions: List[str],
    answers: List[str],
) -> Tuple[str, Dict[str, str]]:
    if not query or not query.strip():
        return "Please enter your admissions question first.", {
            "source": "empty_query",
            "confidence": "0.0",
            "matched_question": "",
        }

    embedder = load_embedder()
    q_vec = embedder.encode([query], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(q_vec)

    # Top-1 for fast path.
    top1_scores, top1_indices = index.search(q_vec, 1)
    top1_score = float(top1_scores[0][0])
    top1_idx = int(top1_indices[0][0])

    if top1_score >= HIGH_THRESHOLD:
        return answers[top1_idx], {
            "source": "direct_faq",
            "confidence": f"{top1_score:.3f}",
            "matched_question": questions[top1_idx],
        }

    # Medium-confidence path for potentially compound queries.
    topk_scores, topk_indices = index.search(q_vec, TOP_K)
    scores_row = topk_scores[0]
    indices_row = topk_indices[0]

    if top1_score >= MEDIUM_THRESHOLD and is_compound_query(query, scores_row):
        try:
            local_llm = get_llm()
            prompt = build_grounded_prompt(query, indices_row, entries)
            output = local_llm(
                prompt,
                temperature=0,
                max_tokens=256,
            )
            generated = output["choices"][0]["text"].strip()
            if generated:
                return generated, {
                    "source": "grounded_ai",
                    "confidence": f"{top1_score:.3f}",
                    "matched_question": questions[top1_idx],
                }
        except Exception:
            # If LLM is unavailable or fails, continue to robust fallback.
            pass

    # Low confidence (or LLM failed): deterministic fuzzy fallback.
    return fuzzy_fallback(query, questions, answers)


# --------------------------
# Streamlit UI
# --------------------------
def source_note(meta: Dict[str, str]) -> str:
    src = meta.get("source", "")
    confidence = meta.get("confidence", "0.0")

    if src == "direct_faq":
        return f"Source: Direct FAQ semantic match (confidence: {confidence})."
    if src == "grounded_ai":
        return (
            "Source: AI-generated summary from retrieved FAQ excerpts only "
            f"(retrieval confidence: {confidence})."
        )
    if src == "fuzzy_fallback":
        return f"Source: Fuzzy FAQ fallback (match score: {confidence})."
    if src == "empty_query":
        return "Source: Input validation."
    return "Source: Not found in official FAQ."


def render_chat_bubble(answer: str) -> None:
    st.markdown(
        f"""
        <div style="
            border: 1px solid #d7e3f4;
            border-radius: 12px;
            padding: 14px 16px;
            background: #f7fbff;
            color: #1f2d3d;
            line-height: 1.5;
            margin-top: 8px;
            margin-bottom: 10px;
        ">
            {answer}
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="NUST Admissions Assistant", page_icon="🎓", layout="centered")

    st.title("NUST Admissions Assistant (Calm & Trusted)")
    st.write(
        "I answer questions based on the official NUST FAQ. "
        "I'm 100% offline and will never guess."
    )

    try:
        entries, questions, answers, faq_path = load_faq_data()
        index, _ = build_index(questions)
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.error(f"Failed to initialize offline retrieval stack: {e}")
        st.stop()

    st.caption(f"Loaded FAQ file: {faq_path}")

    if "history" not in st.session_state:
        st.session_state.history = []

    with st.form("ask_form", clear_on_submit=False):
        query = st.text_input("Ask your admissions question", placeholder="Example: What is NET and how do I apply?")
        submitted = st.form_submit_button("Ask")

    if submitted:
        answer, meta = get_answer(query, index, entries, questions, answers)
        st.session_state.history.append({
            "query": query,
            "answer": answer,
            "meta": meta,
        })

    if st.session_state.history:
        st.subheader("Conversation")
        for turn in reversed(st.session_state.history[-10:]):
            st.markdown(f"**Student:** {turn['query']}")
            render_chat_bubble(turn["answer"])
            st.caption(source_note(turn["meta"]))


if __name__ == "__main__":
    main()
