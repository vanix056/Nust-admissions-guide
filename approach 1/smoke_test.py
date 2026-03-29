from main import build_default_chatbot


def run() -> None:
    bot = build_default_chatbot()

    samples = [
        "Is there any age limit for undergraduate admission?",
        "Is there hostel facility and what is fee structure for MBBS programme?",
        "Do you offer aviation pilot admissions?",
        "",
    ]

    for q in samples:
        print(f"Q: {q!r}")
        print(bot.answer(q))
        print("-" * 80)


if __name__ == "__main__":
    run()

