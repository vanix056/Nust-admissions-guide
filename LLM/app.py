from pathlib import Path
from datetime import datetime

import streamlit as st

from nustbot import qa_engine as qa
from nustbot.ui import (
    inject_theme,
    render_chat_bubble,
    render_message_time,
    render_header,
    render_suggestions,
    render_scroll_to_latest,
)

# Re-export engine functions
load_faq_data = qa.load_faq_data
build_index = qa.build_index
get_answer = qa.get_answer
source_note = qa.source_note
suggest_followup_queries = qa.suggest_followup_queries


LOGO_CANDIDATES = [
    Path("assets/nust_logo.png"),
    Path("nust_logo.png"),
    Path("../approach 1/nust_logo.png"),
    Path("../approach2/nust_logo.png"),
]


def _resolve_logo_path() -> Path | None:
    for candidate in LOGO_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


def _llm_profile_flags(profile: str) -> tuple[bool, bool, bool]:
    if profile == "Accuracy":
        return False, False, False
    if profile == "Balanced":
        return False, True, False
    return True, True, True


def main() -> None:
    st.set_page_config(
        page_title="NUST Admissions Assistant",
        page_icon="🎓",
        layout="centered",
    )

    logo_path = _resolve_logo_path()

    # inject_theme() also stamps the header into the parent DOM via JS
    inject_theme(logo_path)
    # render_header() is now a no-op — kept for import compatibility
    render_header()

    try:
        entries, questions, _, _ = load_faq_data()
        index, _ = build_index(questions)
    except Exception as exc:
        st.error(f"Initialization failed: {exc}")
        st.stop()

    # ── Session state ────────────────────────────────────────────────────────
    if "history" not in st.session_state:
        st.session_state.history = []
    if "queued_query" not in st.session_state:
        st.session_state.queued_query = ""
    if "response_mode" not in st.session_state:
        st.session_state.response_mode = "Fast (recommended)"
    if "llm_profile" not in st.session_state:
        st.session_state.llm_profile = "Accuracy"
    if "composer_query" not in st.session_state:
        st.session_state.composer_query = ""

    speed_mode = st.session_state.response_mode
    llm_profile = st.session_state.llm_profile
    use_llm_generation = speed_mode == "Accurate (LLM, slower)"
    llm_cpu_fast_mode, llm_bypass_in_llm_mode, llm_enforce_latency_budget = (
        _llm_profile_flags(llm_profile)
    )

    request_query = st.session_state.queued_query.strip()
    if request_query:
        st.session_state.queued_query = ""

    # ── Process queued query ─────────────────────────────────────────────────
    if request_query:
        answer, meta = get_answer(
            request_query,
            index,
            entries,
            questions,
            use_llm_generation=use_llm_generation,
            llm_cpu_fast_mode=llm_cpu_fast_mode,
            llm_bypass_in_llm_mode=llm_bypass_in_llm_mode,
            llm_enforce_latency_budget=llm_enforce_latency_budget,
        )

        suggestions = suggest_followup_queries(
            request_query,
            index,
            entries,
            questions,
            matched_question=meta.get("matched_question", ""),
            max_suggestions=3,
        )

        st.session_state.history.append(
            {
                "query": request_query,
                "answer": answer,
                "meta": meta,
                "suggestions": suggestions,
                "ts": datetime.now().strftime("%I:%M %p"),
            }
        )

    # ── Render conversation history ──────────────────────────────────────────
    history_len = len(st.session_state.history)
    for i, item in enumerate(st.session_state.history):
        render_chat_bubble(item["query"], role="user")
        render_message_time(item.get("ts", ""), role="user")
        render_chat_bubble(item["answer"], role="assistant")
        render_message_time(item.get("ts", ""), role="assistant")
        st.caption(source_note(item["meta"]))

        matched = item["meta"].get("matched_question", "")
        if matched:
            st.caption(f"Matched FAQ: {matched}")

        if i == history_len - 1:
            next_query = render_suggestions(item.get("suggestions", []))
            if next_query:
                st.session_state.queued_query = next_query
                st.rerun()

    # Anchor so JS can scroll to latest
    st.markdown('<div id="chat-bottom-anchor"></div>', unsafe_allow_html=True)

    if st.session_state.history:
        render_scroll_to_latest()

    # ── Fixed composer ───────────────────────────────────────────────────────
    with st.form("ask_form", clear_on_submit=True):
        c1, c2 = st.columns([7, 1])
        with c1:
            query = st.text_input(
                "Ask your admissions question",
                key="composer_query",
                placeholder="Example: What is NET fee for Pakistani students?",
                label_visibility="collapsed",
            )
        with c2:
            submitted = st.form_submit_button("Send", use_container_width=True)

    if submitted and query.strip():
        st.session_state.queued_query = query.strip()
        st.rerun()


if __name__ == "__main__":
    main()