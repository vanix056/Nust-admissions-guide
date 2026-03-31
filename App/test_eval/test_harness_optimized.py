"""
Optimized Comprehensive Test Harness for NUST Admissions Chatbot
Generates ~5000 strategically selected test cases and measures accuracy
"""

import json
import time
import re
from pathlib import Path
from typing import List, Dict, Tuple, Any
from datetime import datetime
from collections import defaultdict
import sys
import random

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from nustbot import qa_engine as qa

# ──────────────────────────────────────────────────────────────────
# Test Case Generators
# ──────────────────────────────────────────────────────────────────

class TestCaseGenerator:
    """Generates diverse test case variations for queries"""
    
    @staticmethod
    def generate_for_question(question: str, test_count: int = 35) -> List[str]:
        """Generate diverse minimal test cases for a single question"""
        test_cases = [question]  # Always include original
        
        question_lower = question.lower()
        words = question.split()
        
        # 1. Typo variations (3-5)
        for i in range(min(3, len(words))):
            if len(words[i]) > 2:
                # Missing character
                typo = words[i][:-1]  # Remove last char
                variant = " ".join(words[:i] + [typo] + words[i+1:])
                test_cases.append(variant)
                
                # Extra character
                typo = words[i] + words[i][0]
                variant = " ".join(words[:i] + [typo] + words[i+1:])
                test_cases.append(variant)
        
        # 2. Case variations (3-4)
        test_cases.extend([
            question.upper(),
            question.lower(),
            question.capitalize(),
        ])
        
        # 3. Paraphrases (5-7)
        paraphrases = [
            question.replace("what", "which").replace("What", "Which"),
            question.replace("can", "could").replace("Can", "Could"),
            question.replace("is", "are").replace("Is", "Are"),
            question.replace("?", "."),
            "Tell me about " + question_lower.strip("?"),
            "Please explain " + question_lower.strip("?"),
        ]
        test_cases.extend([p for p in paraphrases if p])
        
        # 4. Abbreviations (3-5)
        abbrevs = [
            question.replace("entry test", "ET"),
            question.replace("Entry Test", "ET"),
            question.replace("nust", "NUST university"),
            question.replace("NUST", "NUST university"),
        ]
        test_cases.extend([a for a in abbrevs if a])
        
        # 5. Punctuation (2-3)
        test_cases.extend([
            question if question.endswith("?") else question + "?",
            question.replace("?", ".") if "?" in question else question,
            question.replace(",", ""),
        ])
        
        # 6. Prefixes (4-5)
        prefixes = [
            "please " + question_lower,
            "kindly " + question_lower,
            "can you tell me " + question_lower,
            "what about " + question_lower if "what" not in question_lower else None,
        ]
        test_cases.extend([p for p in prefixes if p])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_cases = []
        for case in test_cases:
            normalized = case.lower().strip()
            if normalized not in seen:
                seen.add(normalized)
                unique_cases.append(case)
        
        return unique_cases[:test_count]


# ──────────────────────────────────────────────────────────────────
# Test Runner
# ──────────────────────────────────────────────────────────────────

class OptimizedTestRunner:
    """Runs tests efficiently and tracks results"""
    
    def __init__(self):
        self.results = {
            "total": 0,
            "correct": 0,
            "incorrect": 0,
            "unknown": 0,
            "latencies": [],
            "by_question": defaultdict(lambda: {
                "total": 0,
                "correct": 0,
                "incorrect": 0,
                "unknown": 0,
                "latencies": [],
                "failed_queries": [],
            }),
        }
    
    def run_test(
        self,
        test_query: str,
        expected_question: str,
        entries: List[Dict],
        questions: List[str],
        index: Dict,
    ) -> Tuple[bool, float, str, str]:
        """Run a single test case"""
        start = time.perf_counter()
        try:
            answer, meta = qa.get_answer(test_query, index, entries, questions)
            latency = (time.perf_counter() - start) * 1000
            
            matched_q = meta.get("matched_question", "")
            source = meta.get("source", "")
            
            # Check if answer is correct (exact question match or close fuzzy match)
            is_correct = (
                matched_q.lower().strip() == expected_question.lower().strip()
            )
            
            return is_correct, latency, matched_q, source
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            return False, latency, "", f"error: {str(e)}"
    
    def run_full_test_suite(
        self,
        entries: List[Dict],
        questions: List[str],
        index: Dict,
        test_cases_per_question: int = 35,
    ) -> Dict:
        """Run full test suite efficiently"""
        
        print(f"\n{'='*80}")
        print(f"NUST ADMISSIONS CHATBOT - OPTIMIZED TEST SUITE")
        print(f"{'='*80}")
        print(f"Total FAQs: {len(questions)}")
        print(f"Test cases per FAQ: {test_cases_per_question}")
        print(f"Total test cases: ~{len(questions) * test_cases_per_question:,}")
        print(f"{'='*80}\n")
        
        generator = TestCaseGenerator()
        start_time = time.time()
        
        for q_idx, question in enumerate(questions):
            # Generate test cases
            test_cases = generator.generate_for_question(question, test_cases_per_question)
            
            if (q_idx + 1) % 20 == 0:
                elapsed = time.time() - start_time
                acc = (self.results["correct"] / self.results["total"]) * 100 if self.results["total"] > 0 else 0
                eta = (elapsed / (q_idx + 1)) * (len(questions) - q_idx - 1)
                print(f"[{q_idx+1}/{len(questions)}] Accuracy: {acc:.1f}% | "
                      f"Tests: {self.results['total']:,} | ETA: {eta/60:.1f}min")
            
            for test_query in test_cases:
                is_correct, latency, matched_q, source = self.run_test(
                    test_query, question, entries, questions, index
                )
                
                self.results["total"] += 1
                self.results["latencies"].append(latency)
                
                self.results["by_question"][question]["total"] += 1
                self.results["by_question"][question]["latencies"].append(latency)
                
                if is_correct:
                    self.results["correct"] += 1
                    self.results["by_question"][question]["correct"] += 1
                elif source == "unknown":
                    self.results["unknown"] += 1
                    self.results["by_question"][question]["unknown"] += 1
                    if len(self.results["by_question"][question]["failed_queries"]) < 3:
                        self.results["by_question"][question]["failed_queries"].append({
                            "query": test_query,
                            "type": "unknown",
                        })
                else:
                    self.results["incorrect"] += 1
                    self.results["by_question"][question]["incorrect"] += 1
                    if len(self.results["by_question"][question]["failed_queries"]) < 3:
                        self.results["by_question"][question]["failed_queries"].append({
                            "query": test_query,
                            "matched": matched_q,
                            "type": "incorrect",
                        })
        
        return self.results
    
    def generate_report(self) -> str:
        """Generate comprehensive test report"""
        total = self.results["total"]
        correct = self.results["correct"]
        incorrect = self.results["incorrect"]
        unknown = self.results["unknown"]
        
        accuracy = (correct / total * 100) if total > 0 else 0
        latencies = self.results["latencies"]
        
        avg_latency = sum(latencies) / len(latencies) if latencies else 0
        min_latency = min(latencies) if latencies else 0
        max_latency = max(latencies) if latencies else 0
        
        # Calculate percentiles
        sorted_latencies = sorted(latencies)
        p50 = sorted_latencies[len(sorted_latencies) // 2] if sorted_latencies else 0
        p95 = sorted_latencies[int(len(sorted_latencies) * 0.95)] if sorted_latencies else 0
        p99 = sorted_latencies[int(len(sorted_latencies) * 0.99)] if sorted_latencies else 0
        
        # Sort by accuracy
        by_question_sorted = sorted(
            self.results["by_question"].items(),
            key=lambda x: x[1]["correct"] / x[1]["total"] if x[1]["total"] > 0 else 0,
            reverse=True
        )
        
        report = f"""
{'='*80}
NUST ADMISSIONS CHATBOT - COMPREHENSIVE TEST RESULTS
{'='*80}

EXECUTIVE SUMMARY
{'-'*80}
Total Test Cases Run: {total:,}
Correct Matches: {correct:,} ({accuracy:.2f}%)
Incorrect Matches: {incorrect:,} ({incorrect/total*100:.2f}%)
Unknown Responses: {unknown:,} ({unknown/total*100:.2f}%)
Total FAQs Tested: {len(self.results["by_question"])}

LATENCY PROFILE (milliseconds)
{'-'*80}
Average Latency: {avg_latency:.2f}ms
Minimum Latency: {min_latency:.2f}ms
Maximum Latency: {max_latency:.2f}ms
Median (P50): {p50:.2f}ms
95th Percentile: {p95:.2f}ms
99th Percentile: {p99:.2f}ms

ACCURACY BREAKDOWN
{'-'*80}
Correct (FAQ matched): {correct:,} ({accuracy:.2f}%)
Unknown (Low confidence): {unknown:,} ({unknown/total*100:.2f}%)
Incorrect (Wrong match): {incorrect:,} ({incorrect/total*100:.2f}%)

TOP 15 MOST ACCURATE FAQs (100% Accuracy)
{'-'*80}
"""
        
        perfect_faqs = [item for item in by_question_sorted if item[1]["correct"] == item[1]["total"] and item[1]["total"] > 0]
        for i, (question, stats) in enumerate(perfect_faqs[:15]):
            idx = [q for q in self.results["by_question"]].index(question) + 1
            report += f"{i+1}. [{idx}] {question[:75]}\n"
            report += f"   Tests: {stats['total']} | Avg Latency: {sum(stats['latencies'])/len(stats['latencies']):.1f}ms\n"
        
        report += f"\nTOP 15 HIGHEST ACCURACY FAQs (>95%)\n{'-'*80}\n"
        
        high_acc_faqs = [item for item in by_question_sorted 
                        if item[1]["total"] > 0 and 
                        (item[1]["correct"] / item[1]["total"]) > 0.95 and 
                        (item[1]["correct"] / item[1]["total"]) < 1.0][:15]
        
        for i, (question, stats) in enumerate(high_acc_faqs):
            acc = (stats["correct"] / stats["total"]) * 100
            idx = [q for q in self.results["by_question"]].index(question) + 1
            report += f"{i+1}. [{idx}] {question[:75]}\n"
            report += f"   Accuracy: {acc:.1f}% ({stats['correct']}/{stats['total']})\n"
        
        report += f"\nBOTTOM 15 LOWEST ACCURACY FAQs\n{'-'*80}\n"
        
        low_acc_faqs = [item for item in by_question_sorted[-15:]]
        for i, (question, stats) in enumerate(low_acc_faqs):
            acc = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0
            idx = [q for q in self.results["by_question"]].index(question) + 1
            report += f"{i+1}. [{idx}] {question[:75]}\n"
            report += f"   Accuracy: {acc:.1f}% ({stats['correct']}/{stats['total']})\n"
            report += f"   Failed: {stats['incorrect'] + stats['unknown']}\n"
            if stats["failed_queries"]:
                for failure in stats["failed_queries"][:2]:
                    report += f"     - Query: \"{failure['query'][:50]}...\" ({failure['type']})\n"
        
        report += f"\nFAILURE ANALYSIS\n{'-'*80}\n"
        
        total_failures = incorrect + unknown
        if total_failures > 0:
            report += f"Total Failures: {total_failures:,}\n"
            report += f"Unknown (Low confidence): {unknown} ({unknown/total_failures*100:.1f}%)\n"
            report += f"Incorrect (Wrong match): {incorrect} ({incorrect/total_failures*100:.1f}%)\n"
        else:
            report += "No failures detected!\n"
        
        report += f"\nWHY QUERIES WERE MISSED\n{'-'*80}\n"
        
        report += """
1. UNKNOWN RESPONSES (Low Confidence):
   - Query falls below MIN_SEMANTIC_CONF (0.34) threshold
   - Fuzzy match score below MIN_FUZZY_CONF (62)
   - System correctly identifies insufficient confidence
   - These are DESIRED (prevents false answers)

2. INCORRECT MATCHES (Wrong FAQ Selected):
   - Query semantically similar to multiple FAQs
   - Fuzzy matching picked wrong FAQ due to overlap
   - Some longer FAQs may match better due to word density
   - Paraphrased queries sometimes match different FAQ

3. ROOT CAUSES FOR MISSES:

   a) Semantic Ambiguity:
      - "fee structure" matches both general and MBBS-specific
      - "admission" could be general or program-specific
      - "application" matches multiple FAQs

   b) Fuzzy Matching Limitations:
      - "What is the detail of ACT / SAT Score?"
        vs "What is the duration of test and MCQs?"
      - Both have similar token overlap

   c) Query Variations:
      - Abbreviations (ET vs Entry Test) not always recognized
      - Paraphrases may not preserve intent
      - Very short queries "NET fee?" lacks context

   d) FAQ Similarity:
      - Pick/drop facility has near-duplicate FAQs
      - Fee structure has multiple variants
      - Some programs (BSHND, MBBS, NSHS) share patterns

4. CONFIDENCE THRESHOLD TRADE-OFFS:
   - Higher thresholds = More "unknown", Fewer false answers
   - Lower thresholds = More answers, Risk of errors
   - Current thresholds optimized for reliability over completeness

RECOMMENDATIONS TO IMPROVE ACCURACY:

1. Add synonyms/aliases (SAT↔ACT, entry test↔NET, etc.)
2. Expand intent override routes for ambiguous queries
3. Adjust confidence thresholds if needed
4. Add hierarchical routing (program-specific questions)
5. Consider semantic clustering of similar FAQs
"""
        
        report += f"\n{'='*80}\n"
        report += f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"{'='*80}\n"
        
        return report


# ──────────────────────────────────────────────────────────────────
# Main Execution
# ──────────────────────────────────────────────────────────────────

def main():
    """Main test execution"""
    
    # Load FAQ data
    print("Loading FAQ data...")
    try:
        entries, questions, answers, faq_path = qa.load_faq_data()
        print(f"Building indices...")
        index, _ = qa.build_index(questions)
    except Exception as e:
        print(f"Error loading FAQ data: {e}")
        return
    
    print(f"Loaded {len(questions)} FAQ entries\n")
    
    # Run test suite
    runner = OptimizedTestRunner()
    results = runner.run_full_test_suite(
        entries, questions, index, test_cases_per_question=35
    )
    
    # Generate and save report
    report = runner.generate_report()
    print(report)
    
    # Save report to file
    report_path = Path(__file__).parent / "test_results.txt"
    with open(report_path, "w") as f:
        f.write(report)
    
    # Save detailed results as JSON
    json_results = {
        "summary": {
            "total_tests": results["total"],
            "correct": results["correct"],
            "incorrect": results["incorrect"],
            "unknown": results["unknown"],
            "accuracy_percent": (results["correct"] / results["total"] * 100) if results["total"] > 0 else 0,
        },
        "latency": {
            "avg_ms": sum(results["latencies"]) / len(results["latencies"]) if results["latencies"] else 0,
            "min_ms": min(results["latencies"]) if results["latencies"] else 0,
            "max_ms": max(results["latencies"]) if results["latencies"] else 0,
            "p95_ms": sorted(results["latencies"])[int(len(results["latencies"]) * 0.95)] if results["latencies"] else 0,
            "p99_ms": sorted(results["latencies"])[int(len(results["latencies"]) * 0.99)] if results["latencies"] else 0,
        },
        "by_question": {
            q: {
                "total": stats["total"],
                "correct": stats["correct"],
                "incorrect": stats["incorrect"],
                "unknown": stats["unknown"],
                "accuracy_percent": (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0,
                "failed_queries": stats["failed_queries"],
            }
            for q, stats in results["by_question"].items()
        },
    }
    
    json_path = Path(__file__).parent / "test_results.json"
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nReport saved to: {report_path}")
    print(f"JSON results saved to: {json_path}")


if __name__ == "__main__":
    main()
