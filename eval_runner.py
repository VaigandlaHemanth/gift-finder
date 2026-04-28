"""
Evaluation runner for Gift Finder.
Runs all test cases and scores them against rubric.
Outputs structured eval report.
"""
import json
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from gift_finder import find_gifts
from catalog import load_products_to_chromadb
from schema import GiftFinderResponse


def run_eval(test_cases_path: str = "evals/eval_cases.json"):
    """Run all evaluation test cases and generate report."""

    # Load catalog first
    print("Initializing catalog...")
    load_products_to_chromadb("data/products.json")

    # Load test cases
    with open(test_cases_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    results = []
    total_score = 0
    max_score = 0

    print(f"\nRunning {len(test_cases)} evaluation cases...\n")

    for case in test_cases:
        print(f"Running {case['id']}: {case['query'][:50]}...")

        try:
            response = find_gifts(case["query"])
            scores = score_case(case, response)
            results.append({
                "case_id": case["id"],
                "query": case["query"],
                "language": case["language"],
                "expected_behavior": case["expected_behavior"],
                "actual_behavior": "refuse" if response.out_of_scope else "recommend",
                "scores": scores,
                "total_score": sum(s["score"] for s in scores),
                "max_score": sum(s["max"] for s in scores),
                "passed": all(s["score"] == s["max"] for s in scores),
                "response": response.model_dump()
            })
            total_score += sum(s["score"] for s in scores)
            max_score += sum(s["max"] for s in scores)
        except Exception as e:
            print(f"  ERROR: {e}")
            total_score += 0
            max_score += 5
            results.append({
                "case_id": case["id"],
                "query": case["query"],
                "error": str(e),
                "passed": False,
                "total_score": 0,
                "max_score": 5
            })

    # Generate report
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_cases": len(test_cases),
        "passed_cases": sum(1 for r in results if r.get("passed")),
        "failed_cases": sum(1 for r in results if not r.get("passed")),
        "total_score": total_score,
        "max_possible": max_score,
        "percentage": (total_score / max_score * 100) if max_score > 0 else 0,
        "results": results
    }

    # Save report
    with open("evals/eval_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total Cases: {report['total_cases']}")
    print(f"Passed: {report['passed_cases']}")
    print(f"Failed: {report['failed_cases']}")
    print(f"Score: {report['total_score']}/{report['max_possible']} ({report['percentage']:.1f}%)")
    print("="*60)

    # Print failures
    failures = [r for r in results if not r.get("passed")]
    if failures:
        print("\nFAILED CASES:")
        for f in failures:
            print(f"  {f['case_id']}: {f['query'][:60]}")
            if "scores" in f:
                for s in f["scores"]:
                    if s["score"] != s["max"]:
                        print(f"    - {s['criterion']}: {s['score']}/{s['max']} - {s['reason']}")

    return report


def score_case(case: dict, response: GiftFinderResponse) -> list:
    """Score a single test case against rubric."""
    scores = []

    # Criterion 1: Correct behavior (recommend vs refuse)
    expected = case["expected_behavior"]
    actual = "refuse" if response.out_of_scope else "recommend"
    correct_behavior = expected == actual
    scores.append({
        "criterion": "Correct Behavior",
        "score": 1 if correct_behavior else 0,
        "max": 1,
        "reason": f"Expected {expected}, got {actual}" + (" PASS" if correct_behavior else " FAIL")
    })

    # Criterion 2: Language consistency
    lang_match = response.language_detected == case["language"]
    scores.append({
        "criterion": "Language Match",
        "score": 1 if lang_match else 0,
        "max": 1,
        "reason": f"Expected {case['language']}, detected {response.language_detected}" + (" PASS" if lang_match else " FAIL")
    })

    # Criterion 3: Schema validity (always checked)
    try:
        # Re-validate to catch any issues
        GiftFinderResponse(**response.model_dump())
        schema_valid = True
        schema_reason = "Valid schema PASS"
    except Exception as e:
        schema_valid = False
        schema_reason = f"Schema error: {e} FAIL"

    scores.append({
        "criterion": "Schema Validity",
        "score": 1 if schema_valid else 0,
        "max": 1,
        "reason": schema_reason
    })

    # Criterion 4: Reasoning quality (for recommendations)
    if not response.out_of_scope and response.recommendations:
        reasons_good = all(
            len(r.reason_en) > 20 and len(r.reason_ar) > 20 
            for r in response.recommendations
        )
        expects_low_confidence = "low confidence" in case.get("notes", "").lower()
        uncertainty_good = (
            response.uncertainty_note_en is not None
            and min(r.confidence for r in response.recommendations) < 0.7
        )
        reasoning_good = uncertainty_good if expects_low_confidence else reasons_good
        scores.append({
            "criterion": "Reasoning Quality",
            "score": 1 if reasoning_good else 0,
            "max": 1,
            "reason": (
                "Low-confidence case includes uncertainty note and reduced confidence"
                if expects_low_confidence and uncertainty_good
                else "Expected low-confidence uncertainty note FAIL"
                if expects_low_confidence
                else "Reasons are detailed and specific"
                if reasons_good
                else "Reasons too short or generic FAIL"
            )
        })

        # Criterion 5: Category accuracy (if specified)
        if case.get("expected_categories"):
            actual_cats = [r.category_en for r in response.recommendations]
            matches = any(cat in actual_cats for cat in case["expected_categories"])
            scores.append({
                "criterion": "Category Accuracy",
                "score": 1 if matches else 0,
                "max": 1,
                "reason": f"Expected categories {case['expected_categories']}, got {actual_cats}" + (" PASS" if matches else " FAIL")
            })
        else:
            scores.append({
                "criterion": "Category Accuracy",
                "score": 1,
                "max": 1,
                "reason": "No category constraint specified PASS"
            })
    else:
        # For refusal cases, check refusal quality
        has_explanation = response.uncertainty_note_en is not None
        scores.append({
            "criterion": "Reasoning Quality",
            "score": 1 if has_explanation else 0,
            "max": 1,
            "reason": "Refusal has explanation" if has_explanation else "Refusal lacks explanation FAIL"
        })
        scores.append({
            "criterion": "Category Accuracy",
            "score": 1,
            "max": 1,
            "reason": "N/A for refusal case PASS"
        })

    return scores


if __name__ == "__main__":
    run_eval()
