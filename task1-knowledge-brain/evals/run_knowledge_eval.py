import json
from pathlib import Path

from app.main import search_articles

BASE = Path(__file__).resolve().parent
cases = json.loads((BASE / "knowledge_eval.json").read_text(encoding="utf-8-sig"))

results = []
for case in cases:
    try:
        hits = search_articles(case["question"], top_k=5)
        titles = [hit.get("title", "") for hit in hits]
        expected_title = case["expected_title"]

        if case["expected"] == "supported":
            passed = expected_title in titles
            observed = titles[0] if titles else "NO RESULTS"
        else:
            # Search alone cannot prove a question is unsupported.
            # These cases require a separate API-level escalation check.
            passed = None
            observed = titles[0] if titles else "NO RESULTS"

        results.append({
            "id": case["id"],
            "expected": case["expected"],
            "question": case["question"],
            "expected_title": expected_title,
            "top_title": observed,
            "expected_article_in_top_5": passed
        })
    except Exception as exc:
        results.append({
            "id": case["id"],
            "question": case["question"],
            "error": str(exc),
            "expected_article_in_top_5": False
        })

supported = [r for r in results if r.get("expected") == "supported"]
passed = sum(r.get("expected_article_in_top_5") is True for r in supported)

output = {
    "total_cases": len(cases),
    "supported_cases": len(supported),
    "supported_retrieval_passes": passed,
    "supported_retrieval_accuracy": round(passed / len(supported), 4) if supported else 0,
    "note": "Unsupported cases are recorded for manual/API-level escalation evaluation; retrieval results alone do not establish whether evidence is sufficient.",
    "results": results
}

(BASE / "knowledge_eval_results.json").write_text(
    json.dumps(output, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print(f"Cases loaded: {len(cases)}")
print(f"Supported retrieval: {passed}/{len(supported)}")
print(f"Accuracy: {output['supported_retrieval_accuracy']:.1%}")
print(f"Results saved to: {BASE / 'knowledge_eval_results.json'}")
print("\nCase results:")
for row in results:
    status = row.get("expected_article_in_top_5")
    label = "PASS" if status is True else ("FAIL" if status is False else "REVIEW")
    print(f"{row['id']} | {label} | {row.get('top_title', row.get('error', ''))}")
