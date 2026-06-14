"""Test FastAPI endpoints."""

from fastapi.testclient import TestClient
from exchange_connectivity_hub.api.main import app


def test_health_endpoint():
    """Health check should return status and collection count."""
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"


def test_query_endpoint_missing_question():
    """Should return 422 for missing question."""
    client = TestClient(app)
    response = client.post("/query", json={})

    assert response.status_code == 422


def test_query_endpoint_valid_request():
    """Valid request should return answer with sources."""
    import os
    import pytest

    if not os.getenv("VOYAGE_API_KEY") or not os.getenv("ANTHROPIC_API_KEY"):
        pytest.skip("API keys not set")

    client = TestClient(app)
    response = client.post(
        "/query",
        json={"question": "What is lot size?", "exchange_filter": "SGX"},
    )

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
