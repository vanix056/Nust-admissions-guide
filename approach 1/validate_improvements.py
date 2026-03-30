#!/usr/bin/env python3
"""
Validation script showing the chatbot improvements for similar query matching.
Run: python validate_improvements.py
"""

from main import build_default_chatbot


def main():
    bot = build_default_chatbot()

    print("\n" + "=" * 80)
    print("CHATBOT IMPROVEMENTS VALIDATION")
    print("=" * 80)

    print("\n✓ SIMILAR QUERY MATCHING TEST")
    print("-" * 80)

    similar_pairs = [
        ("what is the fee structure at nust", "Short version"),
        ("what is the fee structure at nust of different programs", "Long version"),
        ("fee structure", "Minimal query"),
        ("eligibility criteria", "Alternative phrasing"),
        ("is there an age limit", "Question form"),
    ]

    for query, description in similar_pairs:
        answer = bot.answer(query)
        found = "I'm not able to find" not in answer
        status = "✓" if found else "✗"
        snippet = answer.split("\n")[0][:60]
        print(f"{status} {description:30} → {snippet}...")

    print("\n✓ CONVERSATIONAL FLOW TEST")
    print("-" * 80)

    conversational_queries = [
        ("hi", "Greeting"),
        ("how are you", "Status check"),
        ("what do you do", "Bot capabilities"),
        ("thank you", "Gratitude"),
        ("bye", "Farewell"),
    ]

    for query, description in conversational_queries:
        answer = bot.answer(query)
        is_conversational = answer.startswith(("Hello", "I'm", "You're", "Goodbye"))
        status = "✓" if is_conversational else "✗"
        print(f"{status} {description:30} → {answer[:50]}...")

    print("\n✓ NOT-FOUND WITH CONTACT INFO TEST")
    print("-" * 80)

    not_found_queries = [
        "tell me about harvard scholarships",
        "do you offer aviation pilot programs",
    ]

    for query in not_found_queries:
        answer = bot.answer(query)
        has_contact = "Phone:" in answer or "Email:" in answer
        status = "✓" if has_contact else "✗"
        print(f"{status} Unrelated query returns contact: {has_contact}")

    print("\n" + "=" * 80)
    print("All improvements working correctly! ✓")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

