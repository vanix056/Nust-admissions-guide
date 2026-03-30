import base64
import html
from pathlib import Path
from typing import TypedDict

import streamlit as st

from main import build_default_chatbot


PRIMARY_BLUE = "#0B3C5D"
SECONDARY_BLUE = "#1D5C8B"
BACKGROUND = "#F4F6F8"
WHITE = "#FFFFFF"
TEXT_PRIMARY = "#1A1A1A"
TEXT_SECONDARY = "#555555"
ACCENT = "#F4B400"
BORDER = "#E0E0E0"
USER_BUBBLE = "#E6F0FA"
NOT_FOUND_MESSAGE = "I\u2019m not able to find this information in the official NUST FAQs."


class ChatMessage(TypedDict):
    role: str
    content: str


def _resolve_logo_path() -> Path | None:
    base = Path(__file__).resolve().parent
    assets_logo = base / "assets" / "nust_logo.png"
    root_logo = base / "nust_logo.png"

    if assets_logo.exists():
        return assets_logo
    if root_logo.exists():
        return root_logo
    return None


def _logo_data_uri(logo_path: Path) -> str:
    data = logo_path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _inject_css() -> None:
    st.markdown(
        f"""
        <style>
            .stApp {{
                background-color: {BACKGROUND};
                color: {TEXT_PRIMARY};
            }}
            .block-container {{
                max-width: 980px;
                padding-top: 1rem;
                padding-bottom: 1.5rem;
            }}
            .nust-header {{
                background: {PRIMARY_BLUE};
                border-radius: 12px;
                padding: 0.85rem 1rem;
                display: flex;
                align-items: center;
                gap: 0.85rem;
                box-shadow: 0 1px 4px rgba(0, 0, 0, 0.08);
                margin-bottom: 0.9rem;
            }}
            .nust-header-title {{
                color: {WHITE};
                font-size: 1.22rem;
                font-weight: 600;
                letter-spacing: 0.1px;
            }}
            .nust-logo {{
                height: 44px;
                width: auto;
                object-fit: contain;
                display: block;
                background: {WHITE};
                border-radius: 8px;
                padding: 4px;
            }}
            .nust-subtitle {{
                color: {TEXT_SECONDARY};
                font-size: 0.92rem;
                margin-bottom: 0.55rem;
            }}
            .chat-bubble {{
                border-radius: 12px;
                padding: 0.75rem 0.9rem;
                line-height: 1.45;
                width: fit-content;
                max-width: 92%;
                box-shadow: 0 1px 2px rgba(0, 0, 0, 0.06);
                border: 1px solid {BORDER};
                word-wrap: break-word;
            }}
            .user-bubble {{
                background: {USER_BUBBLE};
                color: {TEXT_PRIMARY};
                margin-left: auto;
                border-color: #d2e3f4;
            }}
            .assistant-bubble {{
                background: {WHITE};
                color: {TEXT_PRIMARY};
                margin-right: auto;
            }}
            .source-line {{
                margin-top: 0.35rem;
                color: {SECONDARY_BLUE};
                font-size: 0.82rem;
                font-weight: 500;
            }}
            section[data-testid="stSidebar"] {{
                background: {WHITE};
                border-left: 1px solid {BORDER};
            }}
            .sidebar-note {{
                color: {TEXT_SECONDARY};
                font-size: 0.88rem;
                line-height: 1.5;
                background: {BACKGROUND};
                border-left: 3px solid {ACCENT};
                border-radius: 8px;
                padding: 0.6rem 0.7rem;
            }}
            .stButton > button {{
                border: 1px solid {BORDER};
                border-radius: 8px;
                background: {WHITE};
                color: {TEXT_PRIMARY};
            }}
            .stButton > button:hover {{
                border-color: {SECONDARY_BLUE};
                color: {SECONDARY_BLUE};
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_header() -> None:
    logo_path = _resolve_logo_path()
    if logo_path is None:
        st.markdown(
            f'<div class="nust-header"><div class="nust-header-title">NUST Admissions Assistant</div></div>',
            unsafe_allow_html=True,
        )
        st.warning("Logo not found. Place the file at assets/nust_logo.png")
        return

    logo_uri = _logo_data_uri(logo_path)
    st.markdown(
        f"""
        <div class="nust-header">
            <img src="{logo_uri}" alt="NUST logo" class="nust-logo" />
            <div class="nust-header-title">NUST Admissions Assistant</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _init_session() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Welcome. Ask about admissions, deadlines, fees, or eligibility from the official NUST FAQs.",
            }
        ]


@st.cache_resource
def _get_bot():
    return build_default_chatbot()


def chatbot(query: str) -> str:
    text = (query or "").strip()
    if not text:
        return "Please enter a question."

    answer = _get_bot().answer(text)
    if not answer or not answer.strip():
        return NOT_FOUND_MESSAGE
    return answer


def _render_sidebar() -> str | None:
    st.sidebar.markdown("### Quick Questions")

    quick_questions = [
        "What is the deadline?",
        "What is the fee structure?",
        "How to apply?",
    ]

    selected = None
    for idx, question in enumerate(quick_questions):
        if st.sidebar.button(question, use_container_width=True, key=f"quick_{idx}"):
            selected = question

    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown(
        '<div class="sidebar-note">This assistant uses official NUST FAQ data and works fully offline.</div>',
        unsafe_allow_html=True,
    )

    if st.sidebar.button("Clear Chat", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Chat cleared. Please ask your next admissions question.",
            }
        ]
        st.rerun()

    return selected


def _format_for_html(text: str) -> str:
    return html.escape(text).replace("\n", "<br>")


def _render_messages() -> None:
    for msg in st.session_state.messages:
        role = msg["role"]
        bubble_class = "user-bubble" if role == "user" else "assistant-bubble"

        with st.chat_message(role):
            st.markdown(
                f'<div class="chat-bubble {bubble_class}">{_format_for_html(msg["content"])}</div>',
                unsafe_allow_html=True,
            )
            if role == "assistant":
                st.markdown('<div class="source-line">📌 Source: NUST FAQ</div>', unsafe_allow_html=True)


def _handle_prompt(prompt: str) -> None:
    cleaned = (prompt or "").strip()
    st.session_state.messages.append({"role": "user", "content": cleaned})

    try:
        answer = chatbot(cleaned)
    except Exception:
        answer = NOT_FOUND_MESSAGE

    st.session_state.messages.append({"role": "assistant", "content": answer})


def run() -> None:
    st.set_page_config(page_title="NUST Admissions Assistant", page_icon="🎓", layout="centered")
    _inject_css()

    try:
        _get_bot()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    _init_session()
    _render_header()
    st.markdown('<div class="nust-subtitle">Official admissions support powered by local FAQ retrieval.</div>', unsafe_allow_html=True)

    sidebar_prompt = _render_sidebar()
    _render_messages()

    user_prompt = st.chat_input("Ask about admissions, deadlines, fees...")
    next_prompt = user_prompt or sidebar_prompt

    if next_prompt is not None:
        _handle_prompt(next_prompt)
        st.rerun()


if __name__ == "__main__":
    run()

