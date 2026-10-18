# AI Tech Consultant Capstone

A three-task prototype combining a knowledge-base service, support-email drafting, and meeting-to-scope generation.

## Requirements

- Python 3.11 or later
- Node.js 20 or later
- Python dependencies listed in each task's `requirements.txt`
- Node dependencies listed in `task2-inbox-agent/package.json`
- Optional Anthropic API credentials configured locally

Keep API keys and credentials in local environment files. Never commit secrets.

## Task 1: Knowledge Brain

A FastAPI service that stores help-center content in SQLite and exposes a query API. It retrieves relevant evidence and uses the configured answer-generation provider when available.

From a PowerShell terminal:

```powershell
$root = "C:\Users\SASHIKUMAR\OneDrive\Desktop\KnowledgeBrain\ai-tech-consultant-starter\ai-tech-consultant-starter"
$py = "$HOME\OneDrive\Desktop\KnowledgeBrain\.venv\Scripts\python.exe"

Set-Location "$root\task1-knowledge-brain"
& $py -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Check health at `http://127.0.0.1:8000/health` and API documentation at `http://127.0.0.1:8000/docs`.

In another terminal, run tests from the Task 1 directory:

```powershell
$root = "C:\Users\SASHIKUMAR\OneDrive\Desktop\KnowledgeBrain\ai-tech-consultant-starter\ai-tech-consultant-starter"
$py = "$HOME\OneDrive\Desktop\KnowledgeBrain\.venv\Scripts\python.exe"

Set-Location "$root\task1-knowledge-brain"
& $py -m pytest -q
```

## Task 2: Inbox Agent

A TypeScript CLI prototype that classifies emails, queries Task 1, and prepares drafts with supporting sources. It does not send emails.

Start Task 1 first. In a separate PowerShell terminal:

```powershell
$root = "C:\Users\SASHIKUMAR\OneDrive\Desktop\KnowledgeBrain\ai-tech-consultant-starter\ai-tech-consultant-starter"

$env:Path = "C:\Program Files\nodejs;" + $env:Path
Set-Location "$root\task2-inbox-agent"

npm.cmd install
if ($LASTEXITCODE -ne 0) { throw "npm install failed" }

npm.cmd run typecheck
if ($LASTEXITCODE -ne 0) { throw "TypeScript type-check failed" }

npm.cmd run dev
```

Run commands from the specified task directory. Use absolute paths when switching between task folders.

## Task 3: Meeting-to-Scope

Reads `transcript.txt` and generates `scope_output.json`. The output includes client-requested requirement candidates, consultant actions, a knowledge-base feasibility check, provisional estimates, open questions, and a baseline comparison.

Run it from PowerShell:

```powershell
$root = "C:\Users\SASHIKUMAR\OneDrive\Desktop\KnowledgeBrain\ai-tech-consultant-starter\ai-tech-consultant-starter"
$py = "$HOME\OneDrive\Desktop\KnowledgeBrain\.venv\Scripts\python.exe"

Set-Location "$root\task3-meeting-to-scope"
& $py -m app.main
if ($LASTEXITCODE -ne 0) { throw "Task 3 failed" }

Get-Content ".\scope_output.json"
```

Run Task 3 tests:

```powershell
& $py -m pytest -q
```

Requirements remain unconfirmed until approved. Estimates are planning placeholders, and technical feasibility and scope changes require human review.

## Recorded Verification Results

| Component | Recorded result | Limitation |
|---|---|---|
| Task 1 tests | 15 passed | Automated tests cover implemented behavior |
| Task 1 API evaluation | 20/20 passed | Synthetic cases; answer generation was mocked in the evaluation |
| Task 1 threshold comparison | 20/20 passed at thresholds 0.35 and 0.45 | Does not compare two LLM models; latency is not a controlled benchmark |
| Task 2 type-check | Passed | Type-checking does not prove runtime correctness |
| Task 2 batch evaluation | 50/50 synthetic cases passed | 25 supported and 25 unsupported scenarios; not a real-world benchmark |
| Task 3 tests | 8 passed | Automated tests cover implemented behavior |
| Task 3 scope evaluation | 20/20 synthetic cases passed | Variations of one core scenario, not 20 diverse real meetings |

These results are recorded from the development and test runs. Re-run the relevant commands to verify the current checkout.

## Known Limitations and Remaining Work

- Expand evaluations with diverse, realistic support emails and meeting transcripts.
- Improve requirement extraction, evidence-sufficiency checks, and baseline comparison.
- Compare at least two model or configuration choices on the same cases, with actual pass/fail outcomes.
- Measure latency under controlled conditions and record model costs where measurable.
- Review generated evidence, estimates, and scope proposals with a human before relying on them.
- Keep evaluation scripts, result files, and documentation synchronized.

## Safety and Approval

The Inbox Agent creates drafts but does not send emails. Meeting-to-Scope creates a draft proposal, not an approved scope or delivery commitment. Knowledge-base evidence may be incomplete or unrelated to a particular project configuration, so feasibility must be independently confirmed.
