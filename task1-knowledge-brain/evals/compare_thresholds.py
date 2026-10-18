import json
import time
from pathlib import Path
from fastapi.testclient import TestClient
import app.main as main

BASE = Path(__file__).resolve().parent
cases = json.loads(
    (BASE / "knowledge_eval.json").read_text(encoding="utf-8-sig")
)

# Avoid real model calls and API charges.
main.generate_answer = lambda question, hits: "MOCK EVALUATION ANSWER [1]"
client = TestClient(main.app)

def run_configuration(threshold):
    results = []
    total_seconds = 0.0

    for case in cases:
        start = time.perf_counter()

        # Apply threshold in memory only; main.py is not edited.
        original_source = Path(main.__file__).read_text(encoding="utf-8")
        marker = "MIN_RETRIEVAL_SCORE = 0.35"
        replacement = f"MIN_RETRIEVAL_SCORE = {threshold}"

        # The application currently defines this value inside query().
        # Temporarily replace the function's constant via source-free
        # function wrapper, leaving the original source file untouched.
        original_query = main.query

        def query_with_threshold(request):
            import inspect
            import textwrap
            import ast

            source = textwrap.dedent(inspect.getsource(original_query))
            source = source.replace(marker, replacement)
            namespace = dict(main.__dict__)
            exec(compile(source, main.__file__, "exec"), namespace)
            return namespace["query"](request)

        main.app.dependency_overrides = {}
        main.app.router.routes
        # Install a temporary endpoint function for this test run.
        route = next(
            r for r in main.app.routes
            if getattr(r, "path", None) == "/query"
            and "POST" in getattr(r, "methods", set())
        )
        saved_endpoint = route.endpoint
        route.endpoint = query_with_threshold

        try:
            response = client.post(
                "/query",
                json={"question": case["question"], "top_k": 5},
            )
        finally:
            route.endpoint = saved_endpoint

        elapsed = time.perf_counter() - start
        total_seconds += elapsed

        if response.status_code != 200:
            results.append({
                "id": case["id"],
                "expected": case["expected"],
                "passed": False,
                "error": f"HTTP {response.status_code}",
                "elapsed_ms": round(elapsed * 1000, 2),
            })
            continue

        data = response.json()
        if case["expected"] == "supported":
            passed = (
                data["status"] == "answered"
                and data["needs_human"] is False
            )
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
            "elapsed_ms": round(elapsed * 1000, 2),
        })

    passed_count = sum(r.get("passed", False) for r in results)
    return {
        "threshold": threshold,
        "total": len(results),
        "passed": passed_count,
        "failed": len(results) - passed_count,
        "accuracy": round(passed_count / len(results), 4) if results else 0,
        "total_elapsed_ms": round(total_seconds * 1000, 2),
        "average_elapsed_ms": round(
            total_seconds * 1000 / len(results), 2
        ) if results else 0,
        "llm_calls": 0,
        "results": results,
    }

report = {
    "evaluation_type": "mocked API threshold comparison",
    "note": (
        "Synthetic 20-case dataset. LLM generation is mocked; "
        "latency measures local API evaluation, not live model latency."
    ),
    "configurations": [
        run_configuration(0.35),
        run_configuration(0.45),
    ],
}

out = BASE / "threshold_comparison_results.json"
out.write_text(json.dumps(report, indent=2), encoding="utf-8")
for config in report["configurations"]:
    print(
        f"Threshold {config['threshold']}: "
        f"{config['passed']}/{config['total']} passed, "
        f"accuracy={config['accuracy']:.1%}, "
        f"avg={config['average_elapsed_ms']} ms"
    )
print(f"Saved: {out}")
