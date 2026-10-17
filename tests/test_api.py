"""
Tests for the API. They replace the database, embedding model and LLM with fakes,
so they run in seconds, need no API key, and can run in CI (GitHub Actions) later.

Run:  python -m pytest -q
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import api  # noqa: E402
import generate  # noqa: E402

FAKE_CHUNKS = [
    {"id": "en_29", "text": "Article (29) Annual leave ... thirty (30) days ...", "article_number": 29,
     "part": 1, "total_parts": 1, "language": "en", "page": 20, "similarity": 0.74},
    {"id": "ar_29", "text": "المادة (29) الإجازة السنوية ...", "article_number": 29,
     "part": 1, "total_parts": 1, "language": "ar", "page": 15, "similarity": 0.70},
    {"id": "en_32", "text": "Article (32) Various holidays ...", "article_number": 32,
     "part": 1, "total_parts": 1, "language": "en", "page": 22, "similarity": 0.64},
]


class FakeProvider:
    name = "fake"

    def __init__(self, answer):
        self.answer = answer

    def generate(self, system, user):
        return self.answer


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(generate, "retrieve", lambda question, top_k=6: FAKE_CHUNKS)
    monkeypatch.setattr(api, "warm_up", lambda: None)

    def make(answer):
        monkeypatch.setattr(api, "get_provider", lambda: FakeProvider(answer))
        return TestClient(api.app)
    return make


def test_answer_with_citation(client):
    answer = generate.Answer(answer="30 days per year.", cited_articles=[29], found=True)
    with client(answer) as c:
        r = c.post("/ask", json={"question": "How many days of annual leave?"})
    assert r.status_code == 200
    body = r.json()
    assert body["found"] is True
    assert body["cited_articles"] == [29]
    assert body["language"] == "en"
    assert body["sources"][0]["language"] == "en"          # question's language first
    assert {s["article_number"] for s in body["sources"]} == {29}


def test_invented_citation_is_dropped(client):
    answer = generate.Answer(answer="30 days.", cited_articles=[29, 99], found=True)
    with client(answer) as c:
        body = c.post("/ask", json={"question": "Annual leave?"}).json()
    assert body["cited_articles"] == [29]


def test_out_of_scope_question_is_refused(client):
    answer = generate.Answer(answer="Not in the law.", cited_articles=[], found=False)
    with client(answer) as c:
        body = c.post("/ask", json={"question": "What is the best restaurant in Dubai?"}).json()
    assert body["found"] is False
    assert body["cited_articles"] == []
    assert body["sources"] == []
    assert "couldn't find" in body["answer"]


def test_arabic_question_gets_arabic_refusal(client):
    answer = generate.Answer(answer="x", cited_articles=[], found=False)
    with client(answer) as c:
        body = c.post("/ask", json={"question": "ما هو أفضل مطعم في دبي؟"}).json()
    assert body["language"] == "ar"
    assert body["answer"] == generate.NOT_FOUND["ar"]


def test_empty_question_is_rejected(client):
    answer = generate.Answer(answer="x", cited_articles=[], found=False)
    with client(answer) as c:
        r = c.post("/ask", json={"question": "  "})
    assert r.status_code == 422


def test_health_reports_database_down(client, monkeypatch):
    monkeypatch.setattr(api, "DATABASE_URL", "postgresql://nobody:x@127.0.0.1:1/none")
    answer = generate.Answer(answer="x", cited_articles=[], found=False)
    with client(answer) as c:
        body = c.get("/health").json()
    assert body["status"] == "degraded"
    assert body["database"] == "unreachable"