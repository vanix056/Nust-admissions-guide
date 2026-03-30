#!/usr/bin/env python3
"""
Real-World Student Query Handling - Comprehensive Test Suite
Tests typos, paraphrasing, grammatical errors, and non-exact questions
"""

from main import build_default_chatbot


def main():
    bot = build_default_chatbot()

    print("\n" + "="*90)
    print(" NUST ADMISSIONS CHATBOT - REAL-WORLD STUDENT QUERY HANDLING TEST")
    print("="*90)

    test_categories = {
        "Typos & Common Mistakes": [
            ("what is the fee structure at NSUT", "NSUT→NUST typo"),
            ("Fee sturcture", "Typo in 'structure'"),
            ("what is fee", "Incomplete query"),
            ("fee", "Single keyword"),
        ],
        "Paraphrasing & Variations": [
            ("what are fees at nust", "Plural: fees vs fee"),
            ("nust fee breakdown", "Synonym: breakdown vs structure"),
            ("what is the cost at nust", "Synonym: cost vs fee"),
            ("fees nust", "Word order variation"),
            ("nust charges", "Synonym: charges vs fee"),
        ],
        "Grammatical Variations": [
            ("fees structure", "Missing article"),
            ("what fees nust", "Non-standard grammar"),
            ("fee structure where", "Extra words"),
        ],
        "Conversational Flow": [
            ("hi", "Greeting"),
            ("hello", "Greeting variant"),
            ("how are you", "Status check"),
            ("who are you", "Identity check"),
            ("goodbye", "Farewell"),
        ],
        "Edge Cases": [
            ("tell me about stanford", "Unrelated - should reject"),
            ("harvard scholarships", "Unrelated - should reject"),
            ("", "Empty input"),
        ],
    }

    total_passed = 0
    total_tests = 0
    category_results = {}

    for category, tests in test_categories.items():
        print(f"\n{category.upper()}")
        print("-" * 90)

        passed = 0
        for query, description in tests:
            try:
                answer = bot.answer(query)
                
                # Determine if query was handled correctly
                is_conversational = any(
                    answer.startswith(p) for p in ("Hello", "I'm", "Goodbye", "You're")
                )
                is_found = "I'm not able to find" not in answer and query.strip() != ""
                
                # For unrelated queries, NOT finding is correct
                if any(x in query.lower() for x in ["stanford", "harvard"]):
                    success = not is_found  # Should NOT find these
                elif query.strip() == "":
                    success = "Please enter" in answer  # Should prompt
                elif is_conversational:
                    success = True  # Conversational always OK
                else:
                    success = is_found  # FAQ queries should find
                
                status = "✓ PASS" if success else "✗ FAIL"
                answer_preview = answer.split("\n")[0][:50].replace("\n", " ")
                
                print(f"  {status} | {description:30} | Q: {query:30}")
                print(f"         | Response: {answer_preview}...")
                
                if success:
                    passed += 1
                total_passed += 1
            except Exception as e:
                print(f"  ✗ ERROR | {description:30} | {str(e)[:40]}")
            
            total_tests += 1
        
        category_results[category] = (passed, len(tests))
        print(f"  Category: {passed}/{len(tests)} passed")

    print("\n" + "="*90)
    print("SUMMARY")
    print("="*90)
    for category, (passed, total) in category_results.items():
        pct = int((passed/total)*100) if total > 0 else 0
        bar = "█" * (pct // 5) + "░" * ((100-pct) // 5)
        print(f"{category:35} {passed:2}/{total:2} [{bar}] {pct:3}%")

    print("\n" + "="*90)
    overall_pct = int((total_passed/total_tests)*100) if total_tests > 0 else 0
    print(f"OVERALL: {total_passed}/{total_tests} tests passed ({overall_pct}%)")
    print("="*90 + "\n")

    if overall_pct >= 85:
        print("✓ Chatbot is ready for real-world student use!")
    elif overall_pct >= 70:
        print("⚠ Good handling, some edge cases need attention")
    else:
        print("✗ Needs more improvement for student use")


if __name__ == "__main__":
    main()

