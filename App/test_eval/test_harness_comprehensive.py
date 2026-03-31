#!/usr/bin/env python3
"""
NUST Admissions Chatbot - Comprehensive Test Suite
Generates ~15,000 test cases (~100 per FAQ) across all 144 FAQ topics
"""

import json
import time
import random
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from nustbot import qa_engine as qa


def generate_query_variations(question: str, num_variations: int = 100) -> list:
    """
    Generate diverse query variations for comprehensive testing
    """
    variations = [question]  # Include original
    
    # List of templates for paraphrasing
    templates = [
        "Can you tell me {}",
        "I would like to know {}",
        "Please explain {}",
        "Tell me about {}",
        "What about {}",
        "I'm curious about {}",
        "How is it that {}",
        "Explain to me {}",
        "I need information on {}",
        "What can you tell me about {}",
    ]
    
    # Abbreviations and synonyms
    replacements = {
        'nust': ['NUST', 'nust', 'Nust'],
        'entry test': ['entry test', 'NET', 'NET-H', 'admission test'],
        'act': ['ACT', 'act', 'SAT', 'sat'],
        'mbbs': ['MBBS', 'mbbs', 'medical', 'doctor'],
        'engineering': ['Engineering', 'engineering', 'Engr', 'Engg'],
        'fee': ['fee', 'fees', 'cost', 'charge', 'amount', 'price'],
        'admission': ['admission', 'admissions', 'enrollment', 'registration'],
        'scholarship': ['scholarship', 'financial aid', 'grant', 'sponsorship'],
        'application': ['application', 'apply', 'registration'],
        'test': ['test', 'exam', 'paper', 'assessment'],
        'program': ['program', 'programme', 'degree', 'course'],
        'candidate': ['candidate', 'applicant', 'student', 'person'],
        'hostel': ['hostel', 'dormitory', 'residence hall', 'accommodation'],
    }
    
    # Generate paraphrased variations
    for template in templates[:10]:
        q_lower = question.lower()
        if not q_lower.startswith(tuple("can what how will when where why is are do does".split())):
            variation = template.format(question.lower())
        else:
            variation = question
        variations.append(variation)
    
    # Generate case variations
    variations.append(question.upper())
    variations.append(question.lower())
    
    # Generate abbreviation variations
    for full, short_forms in replacements.items():
        for short in short_forms:
            if full in question.lower():
                variation = question.replace(full, short)
                variations.append(variation)
    
    # Generate typo variations (character swaps, insertions, deletions)
    for _ in range(15):
        q = question
        # Random character swap
        if len(q) > 2:
            i = random.randint(0, len(q) - 2)
            q_list = list(q)
            q_list[i], q_list[i + 1] = q_list[i + 1], q_list[i]
            variations.append(''.join(q_list))
        
        # Random character insertion
        if len(q) > 1:
            i = random.randint(0, len(q) - 1)
            variations.append(q[:i] + random.choice('abcdefghijklmnopqrstuvwxyz') + q[i:])
        
        # Random character deletion
        if len(q) > 2:
            i = random.randint(0, len(q) - 1)
            variations.append(q[:i] + q[i + 1:])
    
    # Generate synonym variations
    synonym_pairs = {
        'apply': ['apply', 'apply for', 'register for', 'submit application for'],
        'eligible': ['eligible', 'qualify', 'qualified', 'can'],
        'available': ['available', 'offered', 'provided'],
        'requirement': ['requirement', 'prerequisite', 'condition', 'criteria'],
    }
    
    for original, synonyms in synonym_pairs.items():
        if original in question.lower():
            for syn in synonyms:
                variation = question.lower().replace(original, syn)
                variations.append(variation)
    
    # Remove duplicates and limit to desired count
    variations = list(set(variations))[:num_variations]
    
    return variations


def run_comprehensive_test_suite():
    """
    Run comprehensive test suite with ~15,000 test cases
    """
    
    print("\n" + "=" * 80)
    print("NUST ADMISSIONS CHATBOT - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print(f"Generating ~15,000 test cases (~100 per FAQ)")
    print("=" * 80 + "\n")
    
    # Load FAQ data
    print("Loading FAQ data...")
    entries, questions, answers, faq_path = qa.load_faq_data()
    index, embedding_model = qa.build_index(questions)
    
    print(f"Loaded {len(questions)} FAQ entries\n")
    print(f"Building indices...\n")
    
    # Initialize tracking
    results = {
        "summary": {
            "total_tests": 0,
            "correct": 0,
            "incorrect": 0,
            "unknown": 0,
            "accuracy_percent": 0.0
        },
        "latency": {
            "avg_ms": 0,
            "min_ms": float('inf'),
            "max_ms": 0,
            "times": []
        },
        "by_question": {}
    }
    
    start_time = time.time()
    total_queries = 0
    
    # Test each FAQ with ~100 variations
    for idx, question in enumerate(questions):
        expected_index = idx
        
        # Generate test variations
        variations = generate_query_variations(question, num_variations=100)
        
        question_results = {
            "total": len(variations),
            "correct": 0,
            "incorrect": 0,
            "unknown": 0,
            "accuracy_percent": 0.0,
            "failed_queries": []
        }
        
        # Test each variation
        for var_idx, variation in enumerate(variations):
            total_queries += 1
            
            # Measure latency
            query_start = time.time()
            try:
                answer, meta = qa.get_answer(variation, index, entries, questions)
                query_time = (time.time() - query_start) * 1000  # Convert to ms
                matched_q = meta.get("matched_question", "")
                source = meta.get("source", "")
            except Exception as e:
                query_time = (time.time() - query_start) * 1000
                answer = ""
                matched_q = ""
                source = "error"
            
            results["latency"]["times"].append(query_time)
            results["latency"]["min_ms"] = min(results["latency"]["min_ms"], query_time)
            results["latency"]["max_ms"] = max(results["latency"]["max_ms"], query_time)
            
            # Check if answer is correct
            is_correct = False
            is_unknown = False
            
            if source == "unknown" or "could not find" in answer.lower():
                is_unknown = True
                question_results["unknown"] += 1
            elif matched_q.lower().strip() == question.lower().strip():
                is_correct = True
                question_results["correct"] += 1
            else:
                question_results["incorrect"] += 1
                if len(question_results["failed_queries"]) < 5:
                    question_results["failed_queries"].append({
                        "query": variation[:60] + "..." if len(variation) > 60 else variation,
                        "status": "incorrect",
                        "matched": matched_q[:50] if matched_q else ""
                    })
            
            # Progress indicator
            if (total_queries) % 500 == 0:
                elapsed = time.time() - start_time
                rate = total_queries / elapsed
                remaining = (15000 - total_queries) / rate if rate > 0 else 0
                accuracy_so_far = (results["summary"]["correct"] / total_queries * 100) if total_queries > 0 else 0
                print(f"[{total_queries:,}/15,000] Accuracy: {accuracy_so_far:.1f}% | "
                      f"Rate: {rate:.0f} q/s | ETA: {remaining/60:.1f} min")
        
        # Calculate accuracy for this question
        if question_results["total"] > 0:
            question_results["accuracy_percent"] = (question_results["correct"] / question_results["total"]) * 100
        
        results["by_question"][question] = question_results
        results["summary"]["correct"] += question_results["correct"]
        results["summary"]["incorrect"] += question_results["incorrect"]
        results["summary"]["unknown"] += question_results["unknown"]
        
        # Progress by FAQ
        if (idx + 1) % 20 == 0:
            print(f"[{idx+1}/{len(questions)}] FAQ #{idx+1}: Tested {len(variations)} variations | "
                  f"Accuracy: {question_results['accuracy_percent']:.1f}%")
    
    # Calculate final statistics
    results["summary"]["total_tests"] = total_queries
    if total_queries > 0:
        results["summary"]["accuracy_percent"] = (results["summary"]["correct"] / total_queries) * 100
    
    results["latency"]["avg_ms"] = sum(results["latency"]["times"]) / len(results["latency"]["times"]) if results["latency"]["times"] else 0
    
    # Calculate percentiles
    sorted_times = sorted(results["latency"]["times"])
    results["latency"]["p50_ms"] = sorted_times[len(sorted_times) // 2] if sorted_times else 0
    results["latency"]["p95_ms"] = sorted_times[int(len(sorted_times) * 0.95)] if sorted_times else 0
    results["latency"]["p99_ms"] = sorted_times[int(len(sorted_times) * 0.99)] if sorted_times else 0
    
    total_time = time.time() - start_time
    
    # Print summary
    print("\n" + "=" * 80)
    print("COMPREHENSIVE TEST RESULTS")
    print("=" * 80)
    print(f"\nOverall Results:")
    print(f"  Total Tests: {results['summary']['total_tests']:,}")
    print(f"  Correct: {results['summary']['correct']:,} ({results['summary']['accuracy_percent']:.2f}%)")
    print(f"  Incorrect: {results['summary']['incorrect']:,} ({results['summary']['incorrect']/total_queries*100:.2f}%)")
    print(f"  Unknown: {results['summary']['unknown']:,} ({results['summary']['unknown']/total_queries*100:.2f}%)")
    print(f"\nLatency Profile:")
    print(f"  Average: {results['latency']['avg_ms']:.2f}ms")
    print(f"  Median (P50): {results['latency']['p50_ms']:.2f}ms")
    print(f"  P95: {results['latency']['p95_ms']:.2f}ms")
    print(f"  P99: {results['latency']['p99_ms']:.2f}ms")
    print(f"  Min: {results['latency']['min_ms']:.2f}ms")
    print(f"  Max: {results['latency']['max_ms']:.2f}ms")
    print(f"\nExecution Time: {total_time:.1f} seconds")
    print("=" * 80 + "\n")
    
    # Save detailed results
    output_path = Path(__file__).parent / "test_results_comprehensive.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Detailed results saved to: {output_path}")
    
    # Generate human-readable report
    report_path = Path(__file__).parent / "TEST_RESULTS_COMPREHENSIVE.txt"
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("NUST ADMISSIONS CHATBOT - COMPREHENSIVE TEST REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total Test Cases: {results['summary']['total_tests']:,}\n")
        f.write(f"Overall Accuracy: {results['summary']['accuracy_percent']:.2f}%\n")
        f.write(f"FAQs Tested: {len(results['by_question'])}\n")
        f.write(f"Variations per FAQ: ~100\n\n")
        
        f.write("PERFORMANCE METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Average Latency: {results['latency']['avg_ms']:.2f}ms\n")
        f.write(f"Median (P50): {results['latency']['p50_ms']:.2f}ms\n")
        f.write(f"P95: {results['latency']['p95_ms']:.2f}ms\n")
        f.write(f"P99: {results['latency']['p99_ms']:.2f}ms\n\n")
        
        f.write("TOP 20 MOST ACCURATE FAQs\n")
        f.write("-" * 80 + "\n")
        sorted_faqs = sorted(results['by_question'].items(), 
                            key=lambda x: x[1]['accuracy_percent'], 
                            reverse=True)[:20]
        for question, result in sorted_faqs:
            f.write(f"{result['accuracy_percent']:.1f}% - {question[:70]}\n")
        
        f.write("\n\nLOWEST ACCURACY FAQs (Need Improvement)\n")
        f.write("-" * 80 + "\n")
        sorted_faqs_low = sorted(results['by_question'].items(), 
                                key=lambda x: x[1]['accuracy_percent'])[:15]
        for question, result in sorted_faqs_low:
            f.write(f"{result['accuracy_percent']:.1f}% - {question[:70]}\n")
            if result['failed_queries']:
                for failed in result['failed_queries'][:2]:
                    f.write(f"    Failed: {failed['query']}\n")
    
    print(f"✓ Human-readable report saved to: {report_path}")
    
    return results


if __name__ == "__main__":
    results = run_comprehensive_test_suite()
