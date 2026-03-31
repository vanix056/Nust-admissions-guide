import base64
import html
import re
from pathlib import Path
from typing import List, Optional

import streamlit as st
import streamlit.components.v1 as components


HEADER_H  = 65   # px — header height
COMPOSER_H = 68  # px — composer height


def _logo_base64(logo_path: Optional[Path]) -> str:
    if not logo_path or not logo_path.exists():
        return ""
    data = logo_path.read_bytes()
    return base64.b64encode(data).decode("ascii")


def inject_theme(logo_path: Optional[Path]) -> None:
    logo_b64 = _logo_base64(logo_path)
    logo_css = ""
    if logo_b64:
        logo_css = f"background-image:url('data:image/png;base64,{logo_b64}');background-size:cover;background-position:center 25%;"

    # ── 1. CSS ───────────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        *, *::before, *::after {{ box-sizing: border-box; }}

        html, body {{ overflow-x: hidden; }}

        .stApp {{
            background: #f0f2f5 !important;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }}

        /* Hide Streamlit's own toolbar & footer */
        header[data-testid="stHeader"]  {{ display: none !important; }}
        footer                           {{ display: none !important; }}
        #MainMenu                        {{ display: none !important; }}
        section[data-testid="stSidebar"]{{ display: none !important; }}

        /* Strip ALL default block-container padding; we control spacing ourselves */
        .block-container {{
            max-width: 800px !important;
            padding-top: {HEADER_H}px !important;
            padding-bottom: {COMPOSER_H + 16}px !important;
            padding-left: 0 !important;
            padding-right: 0 !important;
        }}

        /* ── Scrollbar ── */
        ::-webkit-scrollbar {{ width: 4px; }}
        ::-webkit-scrollbar-track {{ background: transparent; }}
        ::-webkit-scrollbar-thumb {{ background: rgba(0,132,255,0.3); border-radius: 4px; }}

        /* ════════════════════════════════════════
           HEADER  (injected via JS into parent doc)
        ════════════════════════════════════════ */
        #nust-header {{
            position: fixed;
            top: 0;
            left: 50%;
            transform: translateX(-50%);
            width: min(800px, 100vw);
            height: {HEADER_H}px;
            z-index: 99999;
            background: #ffffff;
            border-bottom: 1px solid #e4e6eb;
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 0 18px;
            box-shadow: 0 1px 6px rgba(0,0,0,0.08);
            font-family: 'Inter', -apple-system, sans-serif;
        }}

        #nust-header .logo {{
            width: 40px; height: 40px;
            border-radius: 50%;
            background-color: #e4e6eb;
            background-repeat: no-repeat;
            {logo_css}
            border: 2px solid #e4e6eb;
            flex-shrink: 0;
        }}

        #nust-header .info {{ flex: 1; }}

        #nust-header .title {{
            font-size: 0.97rem;
            font-weight: 700;
            color: #050505;
            line-height: 1.2;
        }}

        #nust-header .sub {{
            font-size: 0.73rem;
            color: #65676b;
            margin-top: 1px;
        }}

        #nust-header .pill {{
            display: flex;
            align-items: center;
            gap: 5px;
            color: #31a24c;
            font-size: 0.75rem;
            font-weight: 500;
        }}

        #nust-header .dot {{
            width: 8px; height: 8px;
            border-radius: 50%;
            background: #31a24c;
            animation: ping 1.8s ease-out infinite;
        }}

        @keyframes ping {{
            0%   {{ box-shadow: 0 0 0 0   rgba(49,162,76,0.6); }}
            70%  {{ box-shadow: 0 0 0 6px rgba(49,162,76,0);   }}
            100% {{ box-shadow: 0 0 0 0   rgba(49,162,76,0);   }}
        }}

        /* ════════════════════════════════════════
           CHAT MESSAGES
        ════════════════════════════════════════ */
        .msg-row {{
            display: flex;
            align-items: flex-end;
            margin: 3px 0;
            gap: 8px;
            animation: msgIn 0.22s cubic-bezier(0.22,1,0.36,1) both;
        }}

        @keyframes msgIn {{
            from {{ opacity:0; transform:translateY(8px) scale(0.98); }}
            to   {{ opacity:1; transform:translateY(0)   scale(1);    }}
        }}

        .msg-row.user      {{ justify-content: flex-end;   }}
        .msg-row.assistant {{ justify-content: flex-start; }}

        .msg-avatar {{
            width: 28px; height: 28px;
            border-radius: 50%;
            font-size: 0.6rem; font-weight: 700;
            display: inline-flex;
            align-items: center; justify-content: center;
            flex-shrink: 0;
        }}

        .msg-row.user .msg-avatar {{
            order: 2;
            background: linear-gradient(135deg,#0084ff,#0064d2);
            color: #fff;
        }}

        .msg-row.assistant .msg-avatar {{
            background: #e4e6eb;
            color: #050505;
        }}

        .msg-bubble {{
            max-width: 72%;
            border-radius: 18px;
            padding: 9px 14px;
            line-height: 1.55;
            font-size: 0.915rem;
            white-space: pre-wrap;
            word-break: break-word;
        }}

        .msg-bubble.user {{
            background: #0084ff;
            color: #fff;
            border-bottom-right-radius: 4px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.12);
        }}

        .msg-bubble.assistant {{
            background: #fff;
            color: #050505;
            border-bottom-left-radius: 4px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.08);
        }}

        .msg-bubble a       {{ color:#0084ff; text-decoration:underline; text-underline-offset:2px; }}
        .msg-bubble.user a  {{ color:#cce8ff; }}

        .msg-time {{
            font-size: 0.67rem;
            color: #65676b;
            margin: 2px 0 5px;
        }}
        .msg-time.user      {{ text-align:right; margin-right:38px; }}
        .msg-time.assistant {{ text-align:left;  margin-left:38px;  }}

        /* ── Captions ── */
        [data-testid="stCaptionContainer"] p {{
            color: #65676b !important;
            font-size: 0.7rem !important;
            margin: 0.05rem 0 !important;
        }}

        /* ── Suggestions ── */
        .suggest-title {{
            color: #65676b;
            font-weight: 600;
            font-size: 0.72rem;
            margin: 8px 0 5px 38px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        div[data-testid="stButton"] {{ margin: 0.1rem 0 !important; }}

        div[data-testid="stButton"] > button[kind="secondary"] {{
            border-radius: 18px !important;
            border: 1.5px solid #0084ff !important;
            background: #fff !important;
            color: #0084ff !important;
            font-weight: 500 !important;
            font-size: 0.82rem !important;
            padding: 5px 14px !important;
            margin-left: 38px !important;
            transition: background 0.15s, transform 0.12s !important;
        }}

        div[data-testid="stButton"] > button[kind="secondary"]:hover {{
            background: #e7f3ff !important;
            transform: translateX(2px) !important;
        }}

        /* ════════════════════════════════════════
           COMPOSER  — fixed bottom
        ════════════════════════════════════════ */
        div[data-testid="stForm"] {{
            position: fixed !important;
            left: 50% !important;
            transform: translateX(-50%) !important;
            bottom: 0 !important;
            width: min(800px, 100vw) !important;
            z-index: 99998 !important;
            background: #fff !important;
            border-top: 1px solid #e4e6eb !important;
            border-radius: 0 !important;
            padding: 10px 14px !important;
            box-shadow: 0 -1px 4px rgba(0,0,0,0.06) !important;
        }}

        div[data-testid="stForm"] [data-testid="stTextInput"] input {{
            border-radius: 22px !important;
            border: 1.5px solid #e4e6eb !important;
            padding: 10px 18px !important;
            background: #f0f2f5 !important;
            color: #050505 !important;
            caret-color: #0084ff !important;
            font-family: 'Inter', sans-serif !important;
            font-size: 0.9rem !important;
            transition: border-color 0.18s, background 0.18s !important;
        }}

        div[data-testid="stForm"] [data-testid="stTextInput"] input:focus {{
            border-color: #0084ff !important;
            background: #fff !important;
            box-shadow: 0 0 0 2px rgba(0,132,255,0.12) !important;
            outline: none !important;
        }}

        div[data-testid="stForm"] [data-testid="stTextInput"] input::placeholder {{
            color: #65676b !important;
        }}

        div[data-testid="stForm"] [data-testid="stFormSubmitButton"] button {{
            border-radius: 22px !important;
            background: #0084ff !important;
            color: #fff !important;
            border: none !important;
            font-weight: 700 !important;
            font-family: 'Inter', sans-serif !important;
            min-height: 42px !important;
            font-size: 0.9rem !important;
            transition: background 0.15s, transform 0.12s !important;
            box-shadow: none !important;
        }}

        div[data-testid="stForm"] [data-testid="stFormSubmitButton"] button:hover {{
            background: #0073e6 !important;
            transform: scale(1.03) !important;
        }}

        div[data-testid="stForm"] [data-testid="stHorizontalBlock"] {{
            align-items: center !important;
            gap: 0.5rem !important;
        }}

        /* ── Typing indicator ── */
        .typing-inline {{
            display: inline-flex; gap: 4px; align-items: center; padding: 2px;
        }}
        .typing-inline span {{
            width:7px; height:7px; border-radius:50%; background:#bcc0c4;
            animation: dotBounce 1.3s ease-in-out infinite;
        }}
        .typing-inline span:nth-child(2) {{ animation-delay:.18s; }}
        .typing-inline span:nth-child(3) {{ animation-delay:.36s; }}

        @keyframes dotBounce {{
            0%,80%,100% {{ transform:scale(0.65) translateY(0);   opacity:0.5; }}
            40%          {{ transform:scale(1)    translateY(-5px); opacity:1;   }}
        }}

        /* ── Thinking banner ── */
        .thinking-wrap {{
            display:flex; align-items:center; gap:10px;
            margin:6px 0 10px 38px; padding:7px 14px;
            border-radius:14px; border:1px solid #e4e6eb;
            background:#fff; color:#65676b;
            font-size:0.82rem; font-weight:500;
            animation:fadeUp 0.25s ease-out;
            box-shadow:0 1px 2px rgba(0,0,0,0.06);
        }}
        @keyframes fadeUp {{
            from {{ opacity:0; transform:translateY(5px); }}
            to   {{ opacity:1; transform:translateY(0);   }}
        }}
        .thinking-dots span {{
            display:inline-block; width:5px; height:5px;
            margin-right:3px; border-radius:50%; background:#0084ff;
            animation:dotBounce 1.3s ease-in-out infinite;
        }}
        .thinking-dots span:nth-child(2) {{ animation-delay:.18s; }}
        .thinking-dots span:nth-child(3) {{ animation-delay:.36s; }}

        @media (max-width:760px) {{
            .msg-bubble {{ max-width:82%; font-size:0.88rem; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── 2. Inject header into the PARENT document via JS ────────────────────
    # This sidesteps all of Streamlit's wrapper/overflow issues entirely.
    components.html(
        f"""
        <script>
        (function() {{
            var doc = window.parent.document;

            // Avoid duplicate injection on rerun
            if (doc.getElementById('nust-header')) return;

            // Copy our <style> block into parent head so #nust-header CSS works
            var styles = window.frameElement
                ? window.frameElement.ownerDocument.querySelectorAll('style')
                : [];

            var el = doc.createElement('div');
            el.id = 'nust-header';
            el.innerHTML = `
                <div class="logo"></div>
                <div class="info">
                    <div class="title">NUST Admissions Assistant</div>
                    <div class="sub">Official NUST FAQ &middot; Instant retrieval &middot; LLM reasoning</div>
                </div>
                <div class="pill">
                    <div class="dot"></div>
                    <span>Active now</span>
                </div>
            `;

            // Copy CSS from the iframe's parent <style> tags into parent doc
            var cssText = `
                #nust-header {{
                    position:fixed; top:0; left:50%;
                    transform:translateX(-50%);
                    width:min(800px,100vw);
                    height:{HEADER_H}px;
                    z-index:99999;
                    background:#ffffff;
                    border-bottom:1px solid #e4e6eb;
                    display:flex; align-items:center;
                    gap:12px; padding:0 18px;
                    box-shadow:0 1px 6px rgba(0,0,0,0.08);
                    font-family:'Inter',-apple-system,sans-serif;
                }}
                #nust-header .logo {{
                    width:40px; height:40px; border-radius:50%;
                    background-color:#e4e6eb;
                    {logo_css}
                    border:2px solid #e4e6eb; flex-shrink:0;
                }}
                #nust-header .info {{ flex:1; }}
                #nust-header .title {{
                    font-size:0.97rem; font-weight:700;
                    color:#050505; line-height:1.2;
                }}
                #nust-header .sub {{
                    font-size:0.73rem; color:#65676b; margin-top:1px;
                }}
                #nust-header .pill {{
                    display:flex; align-items:center;
                    gap:5px; color:#31a24c;
                    font-size:0.75rem; font-weight:500;
                }}
                #nust-header .dot {{
                    width:8px; height:8px; border-radius:50%;
                    background:#31a24c;
                    animation:nust-ping 1.8s ease-out infinite;
                }}
                @keyframes nust-ping {{
                    0%   {{ box-shadow:0 0 0 0   rgba(49,162,76,0.6); }}
                    70%  {{ box-shadow:0 0 0 6px rgba(49,162,76,0);   }}
                    100% {{ box-shadow:0 0 0 0   rgba(49,162,76,0);   }}
                }}
                @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
            `;

            var styleEl = doc.createElement('style');
            styleEl.id = 'nust-header-style';
            styleEl.textContent = cssText;

            if (!doc.getElementById('nust-header-style')) {{
                doc.head.appendChild(styleEl);
            }}
            doc.body.appendChild(el);
        }})();
        </script>
        """,
        height=0,
    )


def render_header() -> None:
    """No-op: header is injected into parent DOM by inject_theme()."""
    pass


def render_chat_bubble(content: str, role: str = "assistant") -> None:
    link_pattern = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
    parts = []
    cursor = 0
    for match in link_pattern.finditer(content):
        parts.append(html.escape(content[cursor:match.start()]))
        label = html.escape(match.group(1))
        url   = html.escape(match.group(2), quote=True)
        parts.append(f'<a href="{url}" target="_blank" rel="noopener noreferrer">{label}</a>')
        cursor = match.end()
    parts.append(html.escape(content[cursor:]))
    safe_html = "".join(parts)

    safe_role   = "user" if role == "user" else "assistant"
    avatar_text = "You" if safe_role == "user" else "NU"

    st.markdown(
        f'<div class="msg-row {safe_role}">'
        f'<div class="msg-avatar">{avatar_text}</div>'
        f'<div class="msg-bubble {safe_role}">{safe_html}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_message_time(ts: str, role: str = "assistant") -> None:
    safe_role = "user" if role == "user" else "assistant"
    safe_ts   = html.escape(ts)
    st.markdown(
        f'<div class="msg-time {safe_role}">{safe_ts}</div>',
        unsafe_allow_html=True,
    )


def render_inline_typing(role: str = "assistant") -> None:
    safe_role   = "user" if role == "user" else "assistant"
    avatar_text = "You" if safe_role == "user" else "NU"
    st.markdown(
        f'<div class="msg-row {safe_role}">'
        f'<div class="msg-avatar">{avatar_text}</div>'
        f'<div class="msg-bubble {safe_role}">'
        f'<div class="typing-inline"><span></span><span></span><span></span></div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )


def render_thinking_banner(message: str = "Fetching answer from official FAQ context…") -> None:
    st.markdown(
        f'<div class="thinking-wrap">'
        f'<div class="thinking-dots"><span></span><span></span><span></span></div>'
        f'<div>{html.escape(message)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_suggestions(suggestions: List[str]) -> Optional[str]:
    if not suggestions:
        return None
    st.markdown('<div class="suggest-title">You might also ask</div>', unsafe_allow_html=True)
    for i, suggestion in enumerate(suggestions[:3]):
        if st.button(suggestion, key=f"suggest_btn_{i}_{hash(suggestion)}", use_container_width=False):
            return suggestion
    return None


def render_scroll_to_latest(anchor_id: str = "chat-bottom-anchor") -> None:
    components.html(
        f"""
        <script>
        (function() {{
            var doc = window.parent.document;
            var anchor = doc.getElementById('{anchor_id}');
            if (anchor) anchor.scrollIntoView({{ behavior: 'smooth', block: 'end' }});
        }})();
        </script>
        """,
        height=0,
    )


# ── API compatibility stubs ──────────────────────────────────────────────────

def render_chat_topbar() -> None:
    pass


def render_controls_panel(faq_path: str) -> tuple[str, str]:
    c1, c2 = st.columns(2)
    with c1:
        speed_mode = st.radio(
            "Response mode",
            options=["Fast (recommended)", "Accurate (LLM, slower)"],
            key="response_mode",
        )
    with c2:
        llm_profile = st.radio(
            "LLM Profile",
            options=["Accuracy", "Balanced", "Speed"],
            index=0,
            key="llm_profile",
        )
    st.caption(f"FAQ source: {faq_path}")
    return speed_mode, llm_profile


def render_composer_start() -> None:
    pass


def render_composer_end() -> None:
    pass