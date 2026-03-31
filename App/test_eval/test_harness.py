"""
Comprehensive Test Harness for NUST Admissions Chatbot
Generates 15,000 test cases (~100 per FAQ) and measures accuracy
"""

import json
import time
import re
from pathlib import Path
from typing import List, Dict, Tuple, Any
from datetime import datetime
from collections import defaultdict
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from nustbot import qa_engine as qa

# ──────────────────────────────────────────────────────────────────
# Test Case Generators
# ──────────────────────────────────────────────────────────────────

class TestCaseGenerator:
    """Generates diverse test case variations for queries"""
    
    @staticmethod
    def typo_variations(text: str, count: int = 10) -> List[str]:
        """Generate typo variations (missing chars, swapped, extra chars)"""
        variations = []
        words = text.split()
        
        for i in range(min(count, len(words))):
            # Missing character
            if len(words[i]) > 1:
                for j in range(len(words[i])):
                    typo = words[i][:j] + words[i][j+1:]
                    new_text = words[:i] + [typo] + words[i+1:]
                    variations.append(" ".join(new_text))
            
            # Double character
            for j in range(len(words[i])):
                typo = words[i][:j] + words[i][j] + words[i][j:]
                new_text = words[:i] + [typo] + words[i+1:]
                variations.append(" ".join(new_text))
                
        return variations[:count]
    
    @staticmethod
    def case_variations(text: str) -> List[str]:
        """Generate case variations"""
        return [
            text.upper(),
            text.lower(),
            text.capitalize(),
            " ".join([w.capitalize() for w in text.split()]),
        ]
    
    @staticmethod
    def paraphrase_variations(question: str) -> List[str]:
        """Generate paraphrases and reformulations"""
        variations = []
        
        # Question word variations
        question_lower = question.lower()
        
        # What → Which, Can → Could, etc.
        paraphrases = [
            question,
            question.replace("what", "which").replace("What", "Which"),
            question.replace("can", "could").replace("Can", "Could"),
            question.replace("is", "are").replace("Is", "Are"),
            question.replace("do", "does").replace("Do", "Does"),
            question.replace("?", "."),  # As statement
            question.replace("is there", "do we have"),
            question.replace("Is there", "Do we have"),
            question.replace("are there", "do we have"),
            question.replace("Are there", "Do we have"),
            "Tell me about " + question.lower().strip("?"),
            "I want to know " + question.lower().strip("?"),
            "Please explain " + question.lower().strip("?"),
            "How " + question if not question.lower().startswith("how") else question,
        ]
        
        return [p for p in paraphrases if p]
    
    @staticmethod
    def abbreviation_variations(text: str) -> List[str]:
        """Generate abbreviation variations"""
        variations = []
        
        abbrev_map = {
            "entry test": ["ET", "test", "exam"],
            "net": ["NUST entry test", "entry test", "nust test"],
            "mdcat": ["MDCAT test", "medical test"],
            "nust": ["university", "NUST university"],
            "mbbs": ["medical", "medicine"],
            "engineering": ["eng", "engineer"],
            "undergraduate": ["ug", "undergrad"],
            "application processing fee": ["APF", "processing fee", "fee"],
            "fee": ["cost", "charges", "price"],
            "scholarship": ["financial aid", "scholarship/aid"],
            "admission": ["enroll", "enrollment", "apply"],
            "result": ["marks", "score", "outcome"],
            "hostel": ["accommodation", "dorm", "residence"],
            "facility": ["service", "amenity"],
        }
        
        text_lower = text.lower()
        for key, alternates in abbrev_map.items():
            if key in text_lower:
                for alt in alternates:
                    variations.append(text_lower.replace(key, alt))
        
        return variations[:10]
    
    @staticmethod
    def punctuation_variations(text: str) -> List[str]:
        """Generate punctuation variations"""
        return [
            text + "?",
            text + ".",
            text + "!",
            text + "...",
            text.replace("?", ""),
            text.replace(".", ""),
            text.replace(",", ""),
            text.replace("  ", " "),
        ]
    
    @staticmethod
    def prefix_suffix_variations(text: str) -> List[str]:
        """Add common prefixes/suffixes"""
        prefixes = [
            "please ",
            "kindly ",
            "can you tell me ",
            "i want to know ",
            "what is ",
            "how can i ",
            "tell me about ",
        ]
        
        variations = [p + text.lower() for p in prefixes]
        
        # Suffixes
        suffixes = [
            "?",
            " please.",
            " thanks.",
            " thank you.",
            " please?",
        ]
        
        for suffix in suffixes:
            variations.append(text + suffix)
        
        return variations
    
    @staticmethod
    def generate_for_question(question: str, target_count: int = 100) -> List[str]:
        """Generate diverse test cases for a single question"""
        test_cases = set([question])  # Start with original
        
        # Add variations
        test_cases.update(TestCaseGenerator.typo_variations(question, 15))
        test_cases.update(TestCaseGenerator.case_variations(question))
        test_cases.update(TestCaseGenerator.paraphrase_variations(question))
        test_cases.update(TestCaseGenerator.abbreviation_variations(question))
        test_cases.update(TestCaseGenerator.punctuation_variations(question))
        test_cases.update(TestCaseGenerator.prefix_suffix_variations(question))
        
        result = list(test_cases)
        
        # If we need more, add random combinations
        while len(result) < target_count:
            base = question
            # Add multiple variations
            variant = base
            for _ in range(2):
                idx = len(result) % len(TestCaseGenerator.typo_variations(base, 1000))
                variant = TestCaseGenerator.typo_variations(variant, 1)[0] if TestCaseGenerator.typo_variations(variant, 1) else variant
            result.append(variant)
        
        return result[:target_count]


# ──────────────────────────────────────────────────────────────────
# Test Runner
# ──────────────────────────────────────────────────────────────────

class TestRunner:
    """Runs tests and tracks results"""
    
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
            }),
            "failure_cases": [],
        }
    
    def run_test(
        self,
        test_query: str,
        expected_question: str,
        entries: List[Dict],
        questions: List[str],
        index: Dict,
    ) -> Tuple[bool, float, str, str]:
        """
        Run a single test case
        Returns: (is_correct, latency_ms, matched_q, source)
        """
        start = time.perf_counter()
        try:
            answer, meta = qa.get_answer(test_query, index, entries, questions)
            latency = (time.perf_counter() - start) * 1000
            
            matched_q = meta.get("matched_question", "")
            source = meta.get("source", "")
            
            # Check if answer is correct
            is_correct = (
                matched_q.lower().strip() == expected_question.lower().strip()
                or (source == "unknown" and False)  # Unknown is always wrong
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
        test_cases_per_question: int = 100,
    ) -> Dict:
        """Run full test suite and return results"""
        
        print(f"\n{'='*80}")
        print(f"NUST ADMISSIONS CHATBOT - COMPREHENSIVE TEST SUITE")
        print(f"{'='*80}")
        print(f"Total FAQs: {len(questions)}")
        print(f"Test cases per FAQ: {test_cases_per_question}")
        print(f"Total test cases: {len(questions) * test_cases_per_question}")
        print(f"{'='*80}\n")
        
        generator = TestCaseGenerator()
        
        for q_idx, question in enumerate(questions):
            print(f"[{q_idx+1}/{len(questions)}] Testing: {question[:60]}...")
            
            # Generate test cases
            test_cases = generator.generate_for_question(question, test_cases_per_question)
            
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
                    self.results["failure_cases"].append({
                        "type": "unknown",
                        "query": test_query,
                        "expected": question,
                        "source": source,
                    })
                else:
                    self.results["incorrect"] += 1
                    self.results["by_question"][question]["incorrect"] += 1
                    self.results["failure_cases"].append({
                        "type": "incorrect_match",
                        "query": test_query,
                        "expected": question,
                        "matched": matched_q,
                        "source": source,
                    })
            
            # Progress every 5 questions
            if (q_idx + 1) % 5 == 0:
                acc = (self.results["correct"] / self.results["total"]) * 100 if self.results["total"] > 0 else 0
                print(f"  Progress: {self.results['correct']}/{self.results['total']} correct ({acc:.1f}%)")
        
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
NUST ADMISSIONS CHATBOT - TEST RESULTS REPORT
{'='*80}

EXECUTIVE SUMMARY
{'-'*80}
Total Test Cases: {total:,}
Correct Matches: {correct:,} ({accuracy:.2f}%)
Incorrect Matches: {incorrect:,}
Unknown Responses: {unknown:,}
Total FAQs Tested: {len(self.results["by_question"])}

LATENCY ANALYSIS (milliseconds)
{'-'*80}
Average Latency: {avg_latency:.2f}ms
Minimum Latency: {min_latency:.2f}ms
Maximum Latency: {max_latency:.2f}ms
Median (P50): {p50:.2f}ms
95th Percentile (P95): {p95:.2f}ms
99th Percentile (P99): {p99:.2f}ms

ACCURACY BY CATEGORY
{'-'*80}
Correct (FAQ matched): {correct:,} ({accuracy:.2f}%)
Unknown (Low confidence): {unknown:,} ({unknown/total*100:.2f}%)
Incorrect (Wrong match): {incorrect:,} ({incorrect/total*100:.2f}%)

TOP 10 MOST ACCURATE FAQs
{'-'*80}
"""
        
        for i, (question, stats) in enumerate(by_question_sorted[:10]):
            acc = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
            report += f"{i+1}. {question[:70]}\n"
            report += f"   Accuracy: {acc:.1f}% ({stats['correct']}/{stats['total']})\n"
            report += f"   Avg Latency: {sum(stats['latencies'])/len(stats['latencies']):.1f}ms\n\n"
        
        report += f"\nBOTTOM 10 LEAST ACCURATE FAQs\n{'-'*80}\n"
        
        for i, (question, stats) in enumerate(by_question_sorted[-10:]):
            acc = (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0
            report += f"{i+1}. {question[:70]}\n"
            report += f"   Accuracy: {acc:.1f}% ({stats['correct']}/{stats['total']})\n"
            report += f"   Misses: {stats['incorrect'] + stats['unknown']}\n\n"
        
        report += f"\nFAILURE ANALYSIS\n{'-'*80}\n"
        
        # Group failures by type
        failure_types = defaultdict(int)
        for failure in self.results["failure_cases"]:
            failure_types[failure["type"]] += 1
        
        for failure_type, count in sorted(failure_types.items(), key=lambda x: x[1], reverse=True):
            pct = count / len(self.results["failure_cases"]) * 100
            report += f"{failure_type}: {count:,} ({pct:.1f}%)\n"
        
        report += f"\nSAMPLE FAILURE CASES (First 20)\n{'-'*80}\n"
        
        for i, failure in enumerate(self.results["failure_cases"][:20]):
            report += f"\n{i+1}. Type: {failure['type']}\n"
            report += f"   Query: \"{failure['query']}\"\n"
            report += f"   Expected: \"{failure['expected']}\"\n"
            if failure['type'] == 'incorrect_match':
                report += f"   Matched: \"{failure.get('matched', 'N/A')}\"\n"
            report += f"   Source: {failure['source']}\n"
        
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
        index, _ = qa.build_index(questions)
    except Exception as e:
        print(f"Error loading FAQ data: {e}")
        return
    
    print(f"Loaded {len(questions)} FAQ entries")
    
    # Run test suite
    runner = TestRunner()
    results = runner.run_full_test_suite(
        entries, questions, index, test_cases_per_question=100
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
        },
        "by_question": {
            q: {
                "total": stats["total"],
                "correct": stats["correct"],
                "incorrect": stats["incorrect"],
                "unknown": stats["unknown"],
                "accuracy_percent": (stats["correct"] / stats["total"] * 100) if stats["total"] > 0 else 0,
            }
            for q, stats in results["by_question"].items()
        },
        "failure_cases": results["failure_cases"][:100],  # Top 100 failures
    }
    
    json_path = Path(__file__).parent / "test_results.json"
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nReport saved to: {report_path}")
    print(f"JSON results saved to: {json_path}")


if __name__ == "__main__":
    main()
