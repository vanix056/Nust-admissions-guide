import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from nustbot import qa_engine as qa


@dataclass
class Case:
    expected_question: str
    query: str


def collapse_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def typoify(s: str) -> str:
    s = s.replace("what", "wht").replace("is", "iz").replace("the", "teh")
    s = s.replace("application", "aplication").replace("admission", "admision")
    return s


def make_cases(question: str) -> List[str]:
    q = collapse_spaces(question)
    q_no_q = q[:-1] if q.endswith("?") else q
    return [
        q,
        q.lower(),
        q_no_q,
        f"Please tell me, {q_no_q}?",
        f"Kindly guide: {q_no_q}?",
        f"I need help about this: {q_no_q}?",
        f"Can you clarify {q_no_q}?",
        f"{typoify(q_no_q)}?",
        f"{q_no_q} please",
        f"Could you explain: {q_no_q}?",
    ]


def reset_caches() -> None:
    qa.load_faq_data.clear()
    qa.build_question_vocab.clear()
    qa.load_embedder.clear()
    qa.get_available_embedding_models.clear()
    qa.build_index.clear()
    qa.response_cache.clear()


def run_eval(model_candidates: List[str], out_path: Path) -> Tuple[int, int, float, List[dict], List[str]]:
    qa.EMBEDDING_MODEL_CANDIDATES = model_candidates
    reset_caches()

    entries, questions, _answers, _faq_path = qa.load_faq_data()
    indices, _vectors = qa.build_index(questions)
    active_models = list(indices.keys())

    cases: List[Case] = []
    for q in questions:
        for v in make_cases(q):
            cases.append(Case(expected_question=q, query=v))

    correct = 0
    wrong = 0
    failures = []

    for c in cases:
        ans, meta = qa.get_answer(c.query, indices, entries, questions)
        matched = meta.get("matched_question", "")
        ok = c.expected_question in matched
        if ok:
            correct += 1
        else:
            wrong += 1
            failures.append(
                {
                    "expected": c.expected_question,
                    "query": c.query,
                    "matched": matched,
                    "source": meta.get("source", ""),
                    "answer_preview": ans[:220].replace("\n", " "),
                }
            )

    total = len(cases)
    acc = (correct / total) * 100 if total else 0.0

    report = {
        "models_requested": model_candidates,
        "models_active": active_models,
        "total": total,
        "correct": correct,
        "wrong": wrong,
        "accuracy_percent": round(acc, 2),
        "failures": failures,
    }
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return total, correct, acc, failures, active_models


def apply_auto_fix_rules() -> int:
    """Add small high-impact intent rules based on observed failures."""
    file_path = Path("nustbot/qa_engine.py")
    text = file_path.read_text(encoding="utf-8")

    marker = "    # Hostel availability for MBBS students (boys/girls)."
    if "# Explicit compare/advice wording for test-route selection." in text:
        return 0

    patch_block = '''
    # Explicit compare/advice wording for test-route selection.
    if tokens.intersection({"net", "sat", "act"}) and tokens.intersection({"better", "best", "choose", "comparison", "compare", "difference", "route"}):
        idx = find_best(lambda q, a: "taken act / sat tests" in q and "entry test" in q)
        if idx >= 0:
            return idx

    # Generic fee + structure/charges wording should prefer fee-structure FAQ.
    if tokens.intersection({"fee", "charges", "cost", "tuition"}) and tokens.intersection({"structure", "details", "breakdown", "overview"}):
        idx = find_best(lambda q, a: "fee structure of different ug programmes" in q)
        if idx >= 0:
            return idx

'''

    text = text.replace(marker, patch_block + marker)
    file_path.write_text(text, encoding="utf-8")
    return 1


def main() -> None:
    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)

    l6_report = out_dir / "bulk_eval_l6.json"
    l12_report = out_dir / "bulk_eval_l6_l12.json"

    t1, c1, a1, f1, m1 = run_eval(["all-MiniLM-L6-v2"], l6_report)
    t2, c2, a2, f2, m2 = run_eval(["all-MiniLM-L6-v2", "all-MiniLM-L12-v2"], l12_report)

    print("L6_ONLY_ACTIVE_MODELS", m1)
    print("L6_ONLY", f"total={t1} correct={c1} wrong={t1-c1} acc={a1:.2f}%")
    print("L6_L12_ACTIVE_MODELS", m2)
    print("L6_L12", f"total={t2} correct={c2} wrong={t2-c2} acc={a2:.2f}%")

    # If failures exist on blended mode, apply one auto-fix pass and rerun once.
    if f2:
        changed = apply_auto_fix_rules()
        if changed:
            print("AUTO_FIX_RULES_APPLIED", changed)
            # Reload module-level caches and rerun blended eval.
            t3, c3, a3, f3, m3 = run_eval(["all-MiniLM-L6-v2", "all-MiniLM-L12-v2"], out_dir / "bulk_eval_l6_l12_after_fix.json")
            print("L6_L12_AFTER_FIX_ACTIVE_MODELS", m3)
            print("L6_L12_AFTER_FIX", f"total={t3} correct={c3} wrong={t3-c3} acc={a3:.2f}%")
        else:
            print("AUTO_FIX_RULES_APPLIED", 0)


if __name__ == "__main__":
    main()
