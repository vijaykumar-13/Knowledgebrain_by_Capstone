import json
import os
import re
import urllib.request
import urllib.error
from pydantic import BaseModel, Field

KNOWLEDGE_URL = os.getenv(
    "KNOWLEDGE_BRAIN_URL", "http://127.0.0.1:8000"
)


class MeetingRequest(BaseModel):
    project_id: str = Field(min_length=1)
    transcript: str = Field(min_length=20)
    baseline_requirements: list[str] = Field(default_factory=list)


def check_knowledge(question: str) -> dict:
    data = json.dumps({"question": question, "top_k": 4}).encode()
    request = urllib.request.Request(
        f"{KNOWLEDGE_URL}/query",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode())
    except (urllib.error.URLError, TimeoutError) as exc:
        return {
            "status": "unavailable",
            "answer": str(exc),
            "sources": [],
            "needs_human": True,
        }


def build_scope(transcript: str, baseline: list[str]) -> dict:
    lines = [
        (number, line.strip())
        for number, line in enumerate(transcript.splitlines(), 1)
        if line.strip()
    ]

    requirements = []
    consultant_actions = []
    open_questions = []
    client_lines = []

    for line_number, line in lines:
        match = re.match(
            r"^\s*(Client|Consultant)\s*:\s*(.+)$",
            line,
            re.IGNORECASE,
        )
        if not match:
            open_questions.append(
                f"Confirm speaker and meaning of transcript line {line_number}: {line}"
            )
            continue

        speaker = match.group(1).lower()
        statement = match.group(2).strip()
        lower = statement.lower()

        if speaker == "consultant":
            consultant_actions.append({
                "description": statement,
                "source": f"Transcript line {line_number}",
                "status": "proposed_action_not_client_requirement",
            })
            if any(word in lower for word in (
                "check", "confirm", "verify", "investigate", "review"
            )):
                open_questions.append(
                    f"Complete and review consultant action: {statement}"
                )
            continue

        # Capture client statements separately from consultant suggestions.
        if any(word in lower for word in (
            "acceptance criteria", "before confirming the timeline",
            "confirm the timeline", "agree on"
        )):
            open_questions.append(
                f"Agree with the client on: {statement}"
            )
            continue

        if any(word in lower for word in (
            "need", "want", "must", "would like", "require"
        )):
            description = statement
            requirements.append({
                "requirement_id": f"REQ-{len(requirements) + 1:03d}",
                "description": description,
                "status": "client_requested_needs_confirmation",
                "source": f"Transcript line {line_number}",
                "confirmed": False,
            })
            client_lines.append(description)

    if not any("platform" in question.lower() for question in open_questions):
        open_questions.append(
            "Verify whether the existing platform supports the requested changes."
        )

    if not any("filter" in question.lower() for question in open_questions):
        open_questions.append(
            "Confirm the product filter fields and expected filter behaviour."
        )

    if not any("acceptance criteria" in question.lower() for question in open_questions):
        open_questions.append(
            "Agree measurable acceptance criteria with the client."
        )

    open_questions.extend([
        "Confirm the scope of mobile improvements and target devices.",
        "Confirm dependencies, budget, and desired delivery date.",
    ])

    # Knowledge retrieval is advisory, not proof that the project is feasible.
    knowledge = check_knowledge(
        "Find documented platform capabilities relevant to website search, "
        "product filters, and mobile experience improvements. "
        "Identify limitations and missing evidence."
    )

    estimates = [
        {
            "task": "Clarification and technical investigation",
            "low_hours": 2,
            "likely_hours": 4,
            "high_hours": 8,
            "assumption": "Planning placeholder; confirm platform and requirements.",
        },
        {
            "task": "Implementation and integration",
            "low_hours": 4,
            "likely_hours": 10,
            "high_hours": 20,
            "assumption": "Provisional; depends on platform capability and agreed filters.",
        },
        {
            "task": "Testing and acceptance",
            "low_hours": 2,
            "likely_hours": 4,
            "high_hours": 8,
            "assumption": "Depends on agreed acceptance criteria and supported devices.",
        },
    ]

    total = {
        "low_hours": sum(item["low_hours"] for item in estimates),
        "likely_hours": sum(item["likely_hours"] for item in estimates),
        "high_hours": sum(item["high_hours"] for item in estimates),
        "status": "provisional_not_a_commitment",
    }

    current = [item["description"] for item in requirements]

    return {
        "requirements": requirements,
        "consultant_actions": consultant_actions,
        "feasibility_check": {
            "status": "provisional_needs_technical_review",
            "needs_human": True,
            "reason": (
                "Retrieved documentation may describe related features but does "
                "not establish that all requested changes are supported by this "
                "project's existing configuration."
            ),
            "knowledge_lookup": knowledge,
            "confirmed_feasible": False,
        },
        "estimates": estimates,
        "total_estimate": total,
        "scope": {
            "objective": (
                "Improve website search and mobile product discovery, "
                "subject to client confirmation and technical feasibility."
            ),
            "deliverables": current,
            "exclusions": [
                "Unconfirmed features or filter fields",
                "Platform changes not yet assessed",
                "Any work not approved in the final scope",
            ],
            "assumptions": [
                "Client requests are proposals, not approved requirements.",
                "Technical feasibility has not been established.",
                "Estimates are planning placeholders, not a delivery commitment.",
            ],
            "open_questions": list(dict.fromkeys(open_questions)),
        },
        "baseline_changes": {
            "added_or_new": [
                item for item in current if item not in baseline
            ],
            "possibly_removed": [
                item for item in baseline if item not in current
            ],
            "note": (
                "Text comparison only. A human must verify changes against "
                "the approved baseline before accepting any scope change."
            ),
        },
        "proposal_status": "DRAFT - HUMAN APPROVAL REQUIRED",
        "approval_required": True,
    }


def main():
    try:
        with open("transcript.txt", encoding="utf-8") as file:
            transcript = file.read()
    except FileNotFoundError:
        print("Create transcript.txt in this folder first.")
        return

    result = build_scope(transcript, baseline=[])
    result["project_id"] = "demo-project"

    with open("scope_output.json", "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)

    print("Created scope_output.json.")
    print(f"Client-requested requirements: {len(result['requirements'])}")
    print(f"Consultant actions: {len(result['consultant_actions'])}")
    print(f"Open questions: {len(result['scope']['open_questions'])}")
    print(
        "Provisional estimate: "
        f"{result['total_estimate']['low_hours']}-"
        f"{result['total_estimate']['high_hours']} hours "
        f"(likely {result['total_estimate']['likely_hours']} hours)"
    )
    print("Feasibility requires human review.")
    print("Proposal remains a draft; no approval or commitment was made.")


if __name__ == "__main__":
    main()
