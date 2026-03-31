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
    render_inline_typing,
    render_thinking_banner,
)

load_faq_data            = qa.load_faq_data
build_index              = qa.build_index
get_answer               = qa.get_answer
source_note              = qa.source_note
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


def main() -> None:
    st.set_page_config(
        page_title="NUST Admissions Assistant",
        page_icon="🎓",
        layout="centered",
    )

    logo_path = _resolve_logo_path()
    inject_theme(logo_path)
    render_header()  # no-op, kept for import compatibility

    try:
        entries, questions, _, _ = load_faq_data()
        index, _                 = build_index(questions)
    except Exception as exc:
        st.error(f"Initialization failed: {exc}")
        st.stop()

    # ── Session state ────────────────────────────────────────────────────────
    for key, default in {
        "history":        [],
        "queued_query":   "",
        "composer_query": "",
        "is_processing":  False,   # Track if we're mid-answer
        "pending_query":  "",      # Query being processed right now
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

    request_query = st.session_state.queued_query.strip()
    if request_query:
        st.session_state.queued_query   = ""
        st.session_state.is_processing  = True
        st.session_state.pending_query  = request_query

    # ── Render conversation history (always visible) ─────────────────────────
    history_len = len(st.session_state.history)
    next_query  = None

    for i, item in enumerate(st.session_state.history):
        is_last = (i == history_len - 1) and not st.session_state.is_processing
        is_newest = (i == history_len - 1)  # just arrived after typing indicator

        render_chat_bubble(item["query"], role="user")
        render_message_time(item.get("ts", ""), role="user")
        # Animate the answer bubble only for the newest item (just replaced typing dots)
        render_chat_bubble(item["answer"], role="assistant", animate_in=is_newest)
        render_message_time(item.get("ts", ""), role="assistant")
        st.caption(source_note(item["meta"]))

        matched = item["meta"].get("matched_question", "")
        if matched:
            st.caption(f"Matched FAQ: {matched}")

        if is_last:
            next_query = render_suggestions(item.get("suggestions", []))

    # ── If processing: show user bubble + typing indicator immediately ────────
    if st.session_state.is_processing and st.session_state.pending_query:
        pending = st.session_state.pending_query

        # Show the user's message right away
        render_chat_bubble(pending, role="user")
        render_message_time(datetime.now().strftime("%I:%M %p"), role="user")

        # Show typing animation in a placeholder
        typing_placeholder = st.empty()
        with typing_placeholder:
            render_inline_typing(role="assistant")

        # Scroll down so the animation is visible
        render_scroll_to_latest(anchor_id="chat-bottom-anchor")
        st.markdown('<div id="chat-bottom-anchor"></div>', unsafe_allow_html=True)

        # ── Now actually fetch the answer ─────────────────────────────────
        answer, meta = get_answer(
            pending, index, entries, questions,
        )
        suggestions = suggest_followup_queries(
            pending, index, entries, questions,
            matched_question=meta.get("matched_question", ""),
            max_suggestions=3,
        )

        # Commit to history
        st.session_state.history.append({
            "query":       pending,
            "answer":      answer,
            "meta":        meta,
            "suggestions": suggestions,
            "ts":          datetime.now().strftime("%I:%M %p"),
        })

        # Clear processing flags
        st.session_state.is_processing = False
        st.session_state.pending_query = ""

        # Rerun to render the committed answer cleanly (no typing bubble)
        st.rerun()

    else:
        # Bottom anchor when not processing
        st.markdown('<div id="chat-bottom-anchor"></div>', unsafe_allow_html=True)

        if history_len > 0:
            render_scroll_to_latest()

    if next_query:
        st.session_state.queued_query = next_query
        st.rerun()

    # ── Composer ─────────────────────────────────────────────────────────────
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