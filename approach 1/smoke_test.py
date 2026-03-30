from main import build_default_chatbot


def run() -> None:
    bot = build_default_chatbot()

    samples = [
        "hi",
        "hello",
        "how are you",
        "who are you",
        "what do you do",
        "how can you help me",
        "Hi, what is the deadline?",
        "Is there any age limit for undergraduate admission?",
        "what is fee Structure at SEECS",
        "Whaat is fee at NuST",
        "what is Fee",
        "whats the fee structure for mbbs program",
        "Is there hostel facility and what is fee structure for MBBS programme?",
        "Do you offer aviation pilot admissions?",
        "tell me about scholarship in harvard",
        "good bye",
        "",
    ]

    for q in samples:
        print(f"Q: {q!r}")
        print(bot.answer(q))
        print("-" * 80)


if __name__ == "__main__":
    run()

