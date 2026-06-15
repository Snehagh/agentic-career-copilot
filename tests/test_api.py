"""Integration tests for all FastAPI routes (no API keys required)."""
from unittest.mock import patch
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_root():
    assert client.get("/").status_code == 200


def test_search_endpoint():
    with patch("app.api.routes.query", return_value=["Python engineer, 5 years"]):
        r = client.post("/api/v1/search", json={"query": "Python skills", "collection": "resumes"})
    assert r.status_code == 200
    assert r.json()["results"][0] == "Python engineer, 5 years"


def test_analyze_endpoint():
    with patch("app.api.routes.run_career_crew", return_value={
        "resume_analysis": "Strong Python skills",
        "job_analysis": "Needs RAG experience",
        "match_score": '{"score": 91, "matched_skills": ["Python"], "gaps": []}',
        "recommendations": "1. Add Docker to resume",
    }):
        r = client.post("/api/v1/analyze", json={"job_title": "AI Engineer"})
    assert r.status_code == 200
    data = r.json()
    assert "match_score" in data
    assert "recommendations" in data


def test_evaluate_endpoint():
    r = client.post("/api/v1/evaluate", json={
        "query": "What are the candidate's Python skills?",
        "actual_output": "The candidate has 5 years of Python experience with FastAPI.",
        "retrieval_context": ["Python (5 years): FastAPI, LangChain, CrewAI"],
    })
    assert r.status_code == 200
    data = r.json()
    assert 0.0 <= data["answer_relevancy"] <= 1.0
    assert 0.0 <= data["faithfulness"] <= 1.0


def test_evaluate_scores_are_reasonable():
    """Highly relevant response should score above 0.5."""
    r = client.post("/api/v1/evaluate", json={
        "query": "Python machine learning AWS skills",
        "actual_output": "Strong Python machine learning skills with AWS experience.",
        "retrieval_context": ["Python machine learning AWS SageMaker experience."],
    })
    data = r.json()
    assert data["answer_relevancy"] > 0.5
    assert data["faithfulness"] > 0.5
