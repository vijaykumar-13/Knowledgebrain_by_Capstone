# AI Coding Log

This log records development and verification actions observed during the capstone work. It is not a verbatim transcript of every prompt.

## 2026-10-17 to 2026-10-18 — Environment setup and verification

- Installed/configured Node.js for the Windows environment. Node.js v24.22.0 and npm v11.19.1 were verified in a terminal where PATH was configured.
- Installed Task 2 dependencies using `npm install`. The audit reported zero vulnerabilities. Approved the esbuild install script after npm reported it needed approval.
- Ran `npm run typecheck`; it completed successfully.
- Started Task 1 and checked `/health`. The response reported `status: ok`, database present, and Anthropic configured.
- Tested the Task 1 `/query` endpoint with a website-logo question. The endpoint returned `insufficient_evidence` in that direct test.
- Ran Task 2 with its hard-coded sample email. It produced a draft with source links, required human approval, and reported that no email was sent. The draft was not a confirmed logo-specific answer.
- Corrected Task 3 `requirements.txt` to contain separate dependency lines. Confirmed Pydantic 2.14.1 was installed in the existing Python environment.
- Ran `python -m app.main` in Task 3. It generated `scope_output.json` with four requirement candidates, three provisional estimate ranges, a feasibility response, and baseline comparison.
- Ran Task 1 regression tests using the existing Python environment. Result: 15 passed in 48.48 seconds.
- Reviewed README.md, DESIGN.md, EVALS.md, and AI_LOG.md and updated the documentation to distinguish verified results from remaining work.

## Known issues and next actions

- Task 2 currently uses one hard-coded email rather than a 50-email batch.
- Task 3 may treat consultant notes as requirements; improve speaker-aware extraction.
- Feasibility output must not be interpreted as proof that a feature is supported.
- Build fixed evaluation datasets and record actual results before claiming evaluation completion.
