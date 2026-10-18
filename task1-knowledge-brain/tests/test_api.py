import pytest
from fastapi.testclient import TestClient

import app.main as main

client = TestClient(main.app)


@pytest.fixture(autouse=True)
def mock_claude(monkeypatch):
    """Prevent tests from making real, potentially billable Claude requests."""
    monkeypatch.setattr(
        main,
        "generate_answer",
        lambda question, hits: "Mock support answer based on the retrieved articles. [1]",
    )


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_query_contract():
    response = client.post(
        "/query",
        json={"question": "How do I configure a setting?"},
    )

    assert response.status_code == 200

    data = response.json()
    assert data["status"] in {"answered", "insufficient_evidence"}
    assert isinstance(data["answer"], str)
    assert isinstance(data["sources"], list)
    assert isinstance(data["needs_human"], bool)


def test_unsupported_question_escalates():
    response = client.post(
        "/query",
        json={
            "question": "How can I build a rocket to travel to Mars?",
            "top_k": 5,
        },
    )

    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "insufficient_evidence"
    assert data["answer"] == (
        "I could not find sufficient evidence in the knowledge base "
        "to answer this question. Please escalate to a human expert."
    )
    assert data["sources"] == []
    assert data["needs_human"] is True

def test_search_handles_alternative_wording():
    from app.main import search_articles

    results = search_articles(
        "How do I upload users using a spreadsheet?",
        top_k=5,
    )

    assert results, "Search should return relevant articles"
    assert results[0]["title"] == "How To Import Members Via CSV File"



def test_search_relevance_for_member_csv():
    results = main.search_articles(
        "How do I import members via CSV file?",
        top_k=5,
    )

    assert results, "Search should return relevant articles"
    assert results[0]["title"] == "How To Import Members Via CSV File"

    titles = [item["title"] for item in results]
    assert "Settings » General Settings » Social Media Tab" not in titles





def test_search_results_are_sorted_by_score():
    results = main.search_articles(
        "How do I import members via CSV file?",
        top_k=5,
    )

    scores = [item["score"] for item in results]
    assert scores == sorted(scores, reverse=True)


def test_query_rejects_question_that_is_too_short():
    response = client.post(
        "/query",
        json={"question": "Hi"},
    )

    assert response.status_code == 422


def test_query_rejects_invalid_top_k():
    response = client.post(
        "/query",
        json={
            "question": "How do I import members?",
            "top_k": 11,
        },
    )

    assert response.status_code == 422


def test_supported_question_uses_mock_answer():
    response = client.post(
        "/query",
        json={"question": "How do I import members via CSV file?"},
    )

    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "answered"
    assert data["needs_human"] is False
    assert "Mock support answer" in data["answer"]
    assert len(data["sources"]) > 0