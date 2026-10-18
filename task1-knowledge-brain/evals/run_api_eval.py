import json
from pathlib import Path
from fastapi.testclient import TestClient
import app.main as main

BASE = Path(__file__).resolve().parent
cases = json.loads(
    (BASE / "knowledge_eval.json").read_text(encoding="utf-8-sig")
)

# Prevent real LLM calls during evaluation.
main.generate_answer = lambda question, hits: "MOCK EVALUATION ANSWER [1]"

client = TestClient(main.app)
results = []

for case in cases:
    response = client.post(
        "/query",
        json={"question": case["question"], "top_k": 5},
    )

    if response.status_code != 200:
        results.append({
            "id": case["id"],
            "passed": False,
            "error": f"HTTP {response.status_code}",
        })
        continue

    data = response.json()
    if case["expected"] == "supported":
        passed = data["status"] == "answered" and data["needs_human"] is False
    else:
        passed = (
            data["status"] == "insufficient_evidence"
            and data["needs_human"] is True
        )

    results.append({
        "id": case["id"],
        "expected": case["expected"],
        "actual_status": data["status"],
        "needs_human": data["needs_human"],
        "passed": passed,
    })

passed_count = sum(r.get("passed", False) for r in results)
output = {
    "total": len(results),
    "passed": passed_count,
    "failed": len(results) - passed_count,
    "accuracy": round(passed_count / len(results), 4) if results else 0,
    "results": results,
}

(BASE / "knowledge_api_eval_results.json").write_text(
    json.dumps(output, indent=2),
    encoding="utf-8",
)

print(f"API evaluation: {passed_count}/{len(results)} passed")
print(f"Accuracy: {output['accuracy']:.1%}")
for row in results:
    print(
        f"{row['id']} | "
        f"{'PASS' if row.get('passed') else 'FAIL'} | "
        f"{row.get('actual_status', row.get('error'))} | "
        f"needs_human={row.get('needs_human', '-')}"
    )
print(f"Saved: {BASE / 'knowledge_api_eval_results.json'}")
