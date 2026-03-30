from main import build_default_chatbot

QUERIES = [
    "Is there any age limit to apply at NUST?",
    "Is there negative marking in the NET?",
    "How many MCQs are in the entry test and how long is it?",
    "How do I pay the application processing fee online?",
    "Is the application fee refundable if I don't get selected?",
    "What is the fee for the NET if I'm a Pakistani national?",
    "I have ICS background - can I apply for Engineering programmes?",
    "I studied Pre-Medical but want to apply for BS Computer Science. Am I eligible?",
    "I have Arts/Humanities background - can I apply for computing programmes?",
    "I'm in my final year of A-Levels and results aren't out yet. Can I still apply?",
    "Do I need to pass Urdu and Islamiyat for IBCC equivalence as an overseas Pakistani?",
    "I'm a foreigner - can I apply to NUST?",
    "Which MDCAT is valid for NSHS admission?",
    "Can I pay MBBS tuition fee in installments?",
    "Is hostel available for MBBS students?",
    "I missed my NET session - can I reschedule?",
    "I scored below 60% in FSc Part 1 - am I still eligible to apply?",
    "If I get selected on both NET merit and ACT/SAT merit, what happens?",
]


def main() -> None:
    bot = build_default_chatbot()
    misses = 0
    for idx, query in enumerate(QUERIES, 1):
        answer = bot.answer(query)
        miss = "I'm not able to find" in answer
        misses += int(miss)
        status = "MISS" if miss else "HIT "
        print(f"{idx:02d}. {status} - {query}")

    print(f"Total misses: {misses}")


if __name__ == "__main__":
    main()

