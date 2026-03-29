# NUST Admissions Assistant (Offline-First)

A complete offline-first admissions chatbot for NUST Islamabad, grounded in official FAQ data.

## What This App Does

- Uses semantic retrieval (`all-MiniLM-L6-v2` + FAISS) for fast answers.
- Uses a lazy-loaded local LLM (`Phi-3 Mini 4K Instruct Q4_K_M`) only for medium-confidence compound queries.
- Uses fuzzy matching fallback (`rapidfuzz`) when semantic confidence is low.
- Never guesses outside the FAQ. If content is missing, it says it does not know.
- Runs fully offline after one-time model setup.

## Project Files

- `app.py`: Streamlit app and hybrid retrieval logic.
- `requirements.txt`: pinned dependencies.
- `data/nust_faq.json`: FAQ data file.

The app searches FAQ files in this order:
1. `faq.json`
2. `data/faq.json`
3. `data/nust_faq.json`

## One-Time Setup

### 1. Create and activate a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the Phi-3 GGUF model once

Download:
- `phi-3-mini-4k-instruct-q4_K_M.gguf`

Typical source:
- https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf

Place the model file in the same folder as `app.py` (or set `PHI3_MODEL_PATH` env var):

```bash
export PHI3_MODEL_PATH="/absolute/path/to/phi-3-mini-4k-instruct-q4_K_M.gguf"
```

## Run

```bash
pip install -r requirements.txt && streamlit run app.py
```

## Hybrid Answering Logic

`get_answer(query)` uses three stages:

1. **Fast path (semantic high confidence)**
- Embed query and retrieve top-1 from FAISS.
- If cosine similarity `>= 0.70`, return FAQ answer directly.

2. **Compound medium-confidence path**
- If top-1 similarity `>= 0.50` and query looks compound (`and`, `or`, `vs`, `compare`, etc.) or multiple retrieved items are strong:
- Lazy-load local Phi-3 model (first time only).
- Retrieve top-3 FAQ entries.
- Prompt model to answer using only retrieved excerpts.
- Generation settings: `temperature=0`, `max_tokens=256`.

3. **Fallback path**
- Run fuzzy matching over all FAQ questions.
- If best score `> 80`, return that FAQ answer.
- Else return:
  - "I'm sorry, I couldn't find an answer in the official NUST FAQ. Please try rephrasing or ask another admissions-related question."

## Why This Works Under 8 GB RAM

- FAISS + MiniLM embeddings are lightweight (typically under 500 MB RAM with this dataset size).
- Phi-3 is loaded only when needed, using around 2.4 GB RAM on CPU.
- Most simple queries stay on the fast semantic route, so memory remains low.

## Reliability and Edge Cases

- Missing JSON file: app shows an error and stops safely.
- Empty query: user-friendly prompt asks for input.
- LLM load/generation failure: automatically falls back to fuzzy retrieval.
- Offline behavior: designed for no internet calls after initial setup files are available locally.

## Notes

- Keep the FAQ data updated if official NUST policy changes.
- For stricter offline enforcement, ensure sentence-transformer model files are already cached locally before runtime.
