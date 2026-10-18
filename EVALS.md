# Evaluations

Only record results that were actually run. A successful smoke test is not equivalent to a complete evaluation.

## Current recorded checks

| Task | Check | Observed result |
|---|---|---|
| Knowledge Brain | `pytest -q` | 15 passed in 48.48 seconds |
| Knowledge Brain | `GET /health` | `status: ok`; database exists; Anthropic configured |
| Inbox Agent | `npm run typecheck` | Passed with no TypeScript errors |
| Inbox Agent | `npm run dev` | One hard-coded sample email produced a draft with source links and human approval required |
| Meeting-to-Scope | `python -m app.main` | Generated `scope_output.json` |
| Meeting-to-Scope | Output inspection | Four requirement candidates and three provisional estimate ranges |

## Required evaluation work

| Task | Target | Status |
|---|---|---|
| Knowledge Brain | At least 20 fixed test cases | Not yet completed |
| Inbox Agent | At least 20 cases and 50 realistic email inputs | Not yet completed |
| Meeting-to-Scope | At least 20 fixed test cases | Not yet completed |
| Model/configuration comparison | At least two choices on the same fixed cases | Not yet completed |

For each evaluation case, record the case ID, input, expected behavior, actual behavior, model/configuration, pass/fail, latency, cost when measurable, and failure reason.

Use fixed test inputs when comparing models or configurations. Do not invent evaluation scores, latency measurements, cost figures, or results for cases that have not been run.
