import streamlit as st

from main import build_default_chatbot


st.set_page_config(page_title="NUST Admissions Chatbot", page_icon="🎓", layout="centered")
st.title("🎓 NUST Admissions Chatbot")
st.caption("Offline FAQ assistant - answers are only from official NUST FAQs.")


@st.cache_resource
def get_bot():
    return build_default_chatbot()


def _init_messages() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hi! Ask me anything about NUST admissions. I will only reply from the official FAQ dataset.",
            }
        ]


def _render_messages() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def main() -> None:
    try:
        bot = get_bot()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    _init_messages()

    col_left, col_right = st.columns([3, 1])
    with col_right:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "Chat cleared. Ask a new admissions question.",
                }
            ]

    _render_messages()

    user_text = st.chat_input("Ask your admissions question...")
    if not user_text:
        return

    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    response = bot.answer(user_text)
    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        st.markdown(response)


if __name__ == "__main__":
    main()

