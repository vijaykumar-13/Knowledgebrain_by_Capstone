import unittest
from unittest.mock import patch

from app.main import build_scope


TRANSCRIPT = """Client: We need to improve website search and the mobile experience.
Client: We want filters so customers can find products faster.
Consultant: We need to check what the existing platform supports and confirm the filter fields.
Client: We should agree on acceptance criteria before confirming the timeline.
"""


MOCK_KNOWLEDGE = {
    "status": "answered",
    "answer": "Some related documentation was found.",
    "sources": [{"title": "Predictive Keyword Search", "relevance": 1.0}],
    "needs_human": False,
}


class ScopeTests(unittest.TestCase):
    @patch("app.main.check_knowledge", return_value=MOCK_KNOWLEDGE)
    def test_client_requests_separated_from_consultant_actions(self, _mock):
        result = build_scope(TRANSCRIPT, [])

        self.assertEqual(len(result["requirements"]), 2)
        self.assertEqual(len(result["consultant_actions"]), 1)
        self.assertTrue(all(
            item["confirmed"] is False
            for item in result["requirements"]
        ))

    @patch("app.main.check_knowledge", return_value=MOCK_KNOWLEDGE)
    def test_feasibility_never_auto_approved(self, _mock):
        result = build_scope(TRANSCRIPT, [])

        self.assertTrue(result["feasibility_check"]["needs_human"])
        self.assertFalse(result["feasibility_check"]["confirmed_feasible"])
        self.assertTrue(result["approval_required"])

    @patch("app.main.check_knowledge", return_value=MOCK_KNOWLEDGE)
    def test_estimate_totals(self, _mock):
        result = build_scope(TRANSCRIPT, [])

        self.assertEqual(result["total_estimate"]["low_hours"], 8)
        self.assertEqual(result["total_estimate"]["likely_hours"], 18)
        self.assertEqual(result["total_estimate"]["high_hours"], 36)
        self.assertEqual(
            result["total_estimate"]["status"],
            "provisional_not_a_commitment",
        )

    @patch("app.main.check_knowledge", return_value=MOCK_KNOWLEDGE)
    def test_acceptance_criteria_remain_open(self, _mock):
        result = build_scope(TRANSCRIPT, [])

        questions = " ".join(result["scope"]["open_questions"]).lower()
        self.assertIn("acceptance criteria", questions)
        self.assertTrue(any(
            "platform" in question.lower()
            for question in result["scope"]["open_questions"]
        ))

    @patch("app.main.check_knowledge", return_value=MOCK_KNOWLEDGE)
    def test_baseline_comparison(self, _mock):
        baseline = [
            "We need to improve website search and the mobile experience."
        ]
        result = build_scope(TRANSCRIPT, baseline)

        self.assertEqual(
            result["baseline_changes"]["possibly_removed"], []
        )
        self.assertEqual(
            result["baseline_changes"]["added_or_new"],
            ["We want filters so customers can find products faster."],
        )

    @patch("app.main.check_knowledge", return_value=MOCK_KNOWLEDGE)
    def test_consultant_only_statement_is_not_a_requirement(self, _mock):
        transcript = (
            "Consultant: We need to investigate the platform "
            "and confirm technical limitations."
        )
        result = build_scope(transcript, [])

        self.assertEqual(result["requirements"], [])
        self.assertEqual(len(result["consultant_actions"]), 1)
        self.assertTrue(result["feasibility_check"]["needs_human"])

    @patch("app.main.check_knowledge", return_value=MOCK_KNOWLEDGE)
    def test_unknown_speaker_is_flagged(self, _mock):
        transcript = "Unknown: We need a new checkout page."
        result = build_scope(transcript, [])

        self.assertEqual(result["requirements"], [])
        self.assertTrue(any(
            "confirm speaker" in question.lower()
            for question in result["scope"]["open_questions"]
        ))

    @patch("app.main.check_knowledge", return_value=MOCK_KNOWLEDGE)
    def test_proposal_remains_draft(self, _mock):
        result = build_scope(TRANSCRIPT, [])

        self.assertEqual(
            result["proposal_status"],
            "DRAFT - HUMAN APPROVAL REQUIRED",
        )
        self.assertTrue(result["approval_required"])


if __name__ == "__main__":
    unittest.main()
