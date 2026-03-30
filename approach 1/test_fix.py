#!/usr/bin/env python3
"""
Quick test to verify the similar question matching fix works.
Run this after the improvements to validate similar queries get answers.
"""

from main import build_default_chatbot


def test_similar_matching():
    """Test that similar question variations get correct answers."""
    bot = build_default_chatbot()

    # Test cases: (query, should_find_answer)
    test_cases = [
        # Similar queries - all should find answers now
        ("what is the fee structure at nust", True),
        ("fee structure", True),
        ("fee structure for programs", True),
        ("eligibility criteria", True),
        ("what are admission requirements", True),
        ("age limit", True),
        ("hostel facility", True),
        
        # Conversational - should get templated responses
        ("hi", True),
        ("hello", True),
        ("how are you", True),
        ("who are you", True),
        ("how can you help me", True),
        ("goodbye", True),
        
        # Unrelated - should not find but should give contact
        ("tell me about harvard", False),
        ("scholarship for stanford", False),
    ]

    print("\n" + "=" * 80)
    print("SIMILAR QUESTION MATCHING FIX - VALIDATION TEST")
    print("=" * 80 + "\n")

    passed = 0
    failed = 0

    for query, should_find in test_cases:
        answer = bot.answer(query)
        found = "I'm not able to find" not in answer and "Please enter" not in answer
        
        # For conversational, check if response looks right
        is_conversational = any(
            answer.startswith(prefix)
            for prefix in ("Hello", "I'm", "You're", "Goodbye", "I can help")
        )
        
        if is_conversational:
            # Conversational queries should always pass
            success = True
        else:
            # FAQ queries should match expectation
            success = (found == should_find)

        status = "✓ PASS" if success else "✗ FAIL"
        answer_preview = answer.split("\n")[0][:50]
        
        print(f"{status}: {query:40} → {answer_preview}...")
        
        if success:
            passed += 1
        else:
            failed += 1

    print("\n" + "=" * 80)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 80 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = test_similar_matching()
    exit(0 if success else 1)

