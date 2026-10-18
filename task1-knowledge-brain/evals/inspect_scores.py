import json
from pathlib import Path
from app.main import search_articles

path = Path("evals/knowledge_eval.json")
cases = json.loads(path.read_text(encoding="utf-8-sig"))

for case in cases:
    hits = search_articles(case["question"], top_k=1)
    if hits:
        hit = hits[0]
        print(
            f'{case["id"]} | {case["expected"]} | '
            f'score={hit["score"]} | '
            f'semantic={hit.get("semantic_score", "n/a")} | '
            f'keywords={hit.get("keyword_score", "n/a")} | '
            f'{hit["title"]}'
        )
    else:
        print(f'{case["id"]} | {case["expected"]} | NO RESULTS')
