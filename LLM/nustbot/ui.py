import base64
import html
import re
from pathlib import Path
from typing import List, Optional

import streamlit as st
import streamlit.components.v1 as components


def _logo_base64(logo_path: Optional[Path]) -> str:
    if not logo_path or not logo_path.exists():
        return ""
    data = logo_path.read_bytes()
    return base64.b64encode(data).decode("ascii")


def inject_theme(logo_path: Optional[Path]) -> None:
    logo_b64 = _logo_base64(logo_path)
    logo_css = ""
    if logo_b64:
        logo_css = f"background-image:url('data:image/png;base64,{logo_b64}');"

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&family=Space+Grotesk:wght@500;700&display=swap');
        .stApp {{
            background:
                radial-gradient(circle at 8% 10%, rgba(141, 205, 255, 0.35) 0, rgba(141, 205, 255, 0) 40%),
                radial-gradient(circle at 95% 5%, rgba(198, 224, 255, 0.26) 0, rgba(198, 224, 255, 0) 42%),
                linear-gradient(180deg, #f5f8fd 0%, #eef3fb 100%);
            font-family: 'Manrope', sans-serif;
        }}
        section[data-testid="stSidebar"] {{
            display: none !important;
        }}
        .block-container {{
            max-width: 900px;
            padding-top: 0.9rem;
        }}
        .app-hero {{
            border: 1px solid rgba(8, 56, 105, 0.14);
            border-radius: 16px;
            padding: 14px 16px;
            background: #ffffff;
            box-shadow: 0 8px 22px rgba(10, 33, 79, 0.08);
            margin-bottom: 10px;
        }}
        .app-hero-title {{
            display: flex;
            align-items: center;
            gap: 14px;
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.2rem;
            font-weight: 700;
            color: #163a66;
            margin: 0;
        }}
        .app-logo {{
            width: 44px;
            height: 44px;
            border-radius: 10px;
            background-size: contain;
            background-repeat: no-repeat;
            background-position: center;
            background-color: #ffffff;
            border: 1px solid rgba(15,53,91,0.15);
            {logo_css}
        }}
        .app-hero-sub {{
            margin-top: 6px;
            font-size: 0.9rem;
            color: #557799;
        }}
        h3 {{
            color: #000000 !important;
        }}
        .chat-wrap {{
            border: 1px solid rgba(10, 63, 109, 0.12);
            border-radius: 18px;
            padding: 12px;
            background: #ffffff;
            box-shadow: 0 12px 30px rgba(16, 40, 83, 0.08);
            margin-bottom: 12px;
        }}
        .chat-topbar {{
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px 12px;
            border-radius: 12px;
            background: linear-gradient(135deg, #0a66ff 0%, #3f8cff 100%);
            color: #ffffff;
            font-weight: 700;
            margin-bottom: 12px;
        }}
        .chat-topbar-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #9df4b2;
            box-shadow: 0 0 0 3px rgba(157, 244, 178, 0.25);
        }}
        .msg-row {{
            display: flex;
            margin: 8px 0;
        }}
        .msg-row.user {{
            justify-content: flex-end;
        }}
        .msg-row.assistant {{
            justify-content: flex-start;
        }}
        .msg-avatar {{
            width: 28px;
            height: 28px;
            border-radius: 50%;
            font-size: 0.74rem;
            font-weight: 700;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
        }}
        .msg-row.user .msg-avatar {{
            order: 2;
            margin-left: 8px;
            background: #0a6dff;
            color: #ffffff;
        }}
        .msg-row.assistant .msg-avatar {{
            margin-right: 8px;
            background: #d7e7ff;
            color: #1f4f88;
        }}
        .msg-bubble {{
            max-width: 78%;
            border-radius: 18px;
            padding: 11px 14px;
            line-height: 1.55;
            white-space: pre-wrap;
            box-shadow: 0 4px 10px rgba(8, 52, 96, 0.1);
            animation: riseIn .24s ease-out;
        }}
        .msg-bubble.user {{
            background: linear-gradient(130deg, #0a67ff 0%, #2f83ff 100%);
            color: #f7fbff;
            border-bottom-right-radius: 5px;
        }}
        .msg-bubble.assistant {{
            border: 1px solid #cadeff;
            background: #f6f9ff;
            color: #1f4f88;
            border-bottom-left-radius: 5px;
        }}
        .runtime-panel {{
            border: 1px solid rgba(12, 69, 120, 0.12);
            border-radius: 14px;
            background: #ffffff;
            padding: 10px 12px;
            margin: 8px 0 12px;
        }}
        .runtime-panel p,
        .runtime-panel label,
        .runtime-panel [data-testid="stMarkdownContainer"],
        .runtime-panel [data-testid="stWidgetLabel"],
        .runtime-panel div {{
            color: #000000 !important;
        }}
        .composer-shell {{
            border: 1px solid rgba(12, 69, 120, 0.12);
            border-radius: 16px;
            background: #ffffff;
            padding: 8px 10px 2px;
            margin-bottom: 12px;
            box-shadow: 0 8px 20px rgba(18, 41, 84, 0.06);
            position: sticky;
            bottom: 10px;
            z-index: 20;
        }}
        .composer-shell [data-testid="stTextInput"] input {{
            border-radius: 999px;
            border: 1px solid #c7d9f5;
            padding-left: 14px;
            background: #ffffff !important;
            color: #000000 !important;
            caret-color: #000000 !important;
        }}
        .composer-shell [data-testid="stTextInput"] input::placeholder {{
            color: #000000 !important;
            opacity: 0.9;
        }}
        .composer-shell [data-testid="stFormSubmitButton"] button {{
            border-radius: 999px;
            background: #ffffff !important;
            color: #000000 !important;
            border: 1px solid #9bb7df !important;
            font-weight: 700;
        }}
        .composer-shell [data-testid="stFormSubmitButton"] button:hover {{
            background: #f4f8ff !important;
            color: #000000 !important;
        }}
        @keyframes riseIn {{
            from {{ opacity: 0; transform: translateY(8px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        .thinking-wrap {{
            display:flex;
            align-items:center;
            gap:10px;
            margin: 8px 0 14px;
            padding: 10px 12px;
            border-radius: 12px;
            border: 1px solid #d4e5f7;
            background: rgba(255,255,255,0.85);
            color: #13436b;
            font-weight:600;
        }}
        .thinking-dots span {{
            display:inline-block;
            width:8px;
            height:8px;
            margin-right:4px;
            border-radius:50%;
            background:#2d79b7;
            animation: pulse 1.2s infinite ease-in-out;
        }}
        .thinking-dots span:nth-child(2) {{ animation-delay: .15s; }}
        .thinking-dots span:nth-child(3) {{ animation-delay: .3s; }}
        @keyframes pulse {{
            0%, 80%, 100% {{ transform: scale(0.75); opacity: .45; }}
            40% {{ transform: scale(1.0); opacity: 1; }}
        }}
        .suggest-title {{
            color: #2a4e78;
            font-weight: 700;
            margin: 4px 0 6px;
        }}
        .msg-time {{
            font-size: 0.72rem;
            color: #7b92ad;
            margin: 2px 2px 8px;
        }}
        .msg-time.user {{
            text-align: right;
            margin-right: 40px;
        }}
        .msg-time.assistant {{
            text-align: left;
            margin-left: 40px;
        }}
        [data-testid="stCaptionContainer"] p {{
            color: #000000 !important;
            font-weight: 600;
        }}
        .typing-inline {{
            display: inline-flex;
            gap: 4px;
            align-items: center;
        }}
        .typing-inline span {{
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: #5a7da3;
            animation: dotPulse 1.2s infinite;
        }}
        .typing-inline span:nth-child(2) {{ animation-delay: .15s; }}
        .typing-inline span:nth-child(3) {{ animation-delay: .3s; }}
        @keyframes dotPulse {{
            0%, 80%, 100% {{ transform: scale(0.7); opacity: .45; }}
            40% {{ transform: scale(1.0); opacity: 1; }}
        }}
        div[data-testid="stButton"] > button[kind="secondary"] {{
            border-radius: 999px;
            border: 1px solid #cadeff;
            background: #f6f9ff;
            color: #1f4f88;
            text-align: left;
            font-weight: 600;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    components.html(
        """
        <script>
        const root = window.parent.document;
        if (!root.__nustPulseSet) {
          root.__nustPulseSet = true;
          let t = 0;
          setInterval(() => {
            t += 0.05;
            const app = root.querySelector('.stApp');
            if (!app) return;
            app.style.backgroundPosition = `${Math.sin(t) * 6}px ${Math.cos(t) * 6}px`;
          }, 60);
        }
        </script>
        """,
        height=0,
    )


def render_header() -> None:
    st.markdown(
        """
        <div class="app-hero">
          <div class="app-hero-title">
            <div class="app-logo"></div>
            <div>NUST Admissions Assistant</div>
          </div>
          <div class="app-hero-sub">
            Grounded chatbot for official NUST admissions FAQs with fast retrieval and LLM reasoning modes.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chat_topbar() -> None:
    st.markdown(
        """
        <div class="chat-topbar">
            <div class="chat-topbar-dot"></div>
            <div>NUST Admissions Chat</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_controls_panel(faq_path: str) -> None:
    st.markdown('<div class="runtime-panel">', unsafe_allow_html=True)
    st.markdown("**Chat Settings**")
    st.markdown("Response mode: **Retrieval only**")
    st.caption(f"FAQ source: {faq_path}")
    st.markdown("</div>", unsafe_allow_html=True)
    return None


def render_chat_bubble(content: str, role: str = "assistant") -> None:
    link_pattern = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
    parts = []
    cursor = 0
    for match in link_pattern.finditer(content):
        parts.append(html.escape(content[cursor:match.start()]))
        label = html.escape(match.group(1))
        url = html.escape(match.group(2), quote=True)
        parts.append(f'<a href="{url}" target="_blank" rel="noopener noreferrer">{label}</a>')
        cursor = match.end()
    parts.append(html.escape(content[cursor:]))
    safe_answer_html = "".join(parts)

    safe_role = "user" if role == "user" else "assistant"
    avatar_text = "You" if safe_role == "user" else "NU"
    st.markdown(
        (
            f'<div class="msg-row {safe_role}">'
            f'<div class="msg-avatar">{avatar_text}</div>'
            f'<div class="msg-bubble {safe_role}">{safe_answer_html}</div>'
            f'</div>'
        ),
        unsafe_allow_html=True,
    )


def render_message_time(ts: str, role: str = "assistant") -> None:
    safe_role = "user" if role == "user" else "assistant"
    safe_ts = html.escape(ts)
    st.markdown(f'<div class="msg-time {safe_role}">{safe_ts}</div>', unsafe_allow_html=True)


def render_inline_typing(role: str = "assistant") -> None:
    safe_role = "user" if role == "user" else "assistant"
    avatar_text = "You" if safe_role == "user" else "NU"
    st.markdown(
        (
            f'<div class="msg-row {safe_role}">'
            f'<div class="msg-avatar">{avatar_text}</div>'
            f'<div class="msg-bubble {safe_role}"><div class="typing-inline"><span></span><span></span><span></span></div></div>'
            f'</div>'
        ),
        unsafe_allow_html=True,
    )


def render_composer_start() -> None:
    st.markdown('<div class="composer-shell">', unsafe_allow_html=True)


def render_composer_end() -> None:
    st.markdown('</div>', unsafe_allow_html=True)


def render_thinking_banner(message: str = "Fetching answer from official FAQ context...") -> None:
    st.markdown(
                f"""
        <div class="thinking-wrap">
          <div class="thinking-dots"><span></span><span></span><span></span></div>
                    <div>{html.escape(message)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_suggestions(suggestions: List[str]) -> Optional[str]:
    if not suggestions:
        return None

    st.markdown('<div class="suggest-title">Suggested next questions</div>', unsafe_allow_html=True)
    for i, suggestion in enumerate(suggestions[:3]):
        if st.button(suggestion, key=f"suggest_btn_{i}_{hash(suggestion)}", use_container_width=True):
            return suggestion
    return None


def render_scroll_to_latest(anchor_id: str = "chat-bottom-anchor") -> None:
    components.html(
                f"""
        <script>
        const doc = window.parent.document;
                const anchor = doc.getElementById('{anchor_id}');
                if (anchor) {{
                    requestAnimationFrame(() => {{
                        anchor.scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
                        const input = doc.querySelector('input[aria-label="Ask your admissions question"]');
                        if (input) {{ input.focus(); }}
                    }});
                }}
        </script>
        """,
        height=0,
    )
