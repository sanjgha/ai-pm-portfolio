"""Test the FastAPI layer (graph + store mocked)."""

from fastapi.testclient import TestClient

from pre_trade_risk_rules_assistant.api import main


def _client():
    return TestClient(main.app)


def test_draft_ok_persists_and_returns_rule(monkeypatch):
    monkeypatch.setattr(
        main,
        "run_graph",
        lambda req: {
            "status": "ok",
            "attempts": 0,
            "validated_rule": {"rule_type": "price_collar", "collar_pct": 5},
            "readback": "5% collar",
        },
    )
    monkeypatch.setattr(main.store, "save_rule", lambda rule: "rule-123")
    resp = _client().post("/rules/draft", json={"request": "5% collar on SGX"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["rule_id"] == "rule-123"
    assert body["readback"] == "5% collar"


def test_draft_escalated_returns_errors(monkeypatch):
    monkeypatch.setattr(
        main,
        "run_graph",
        lambda req: {
            "status": "escalated",
            "attempts": 2,
            "errors": ["collar bad"],
            "escalation_reason": "collar bad",
        },
    )
    resp = _client().post("/rules/draft", json={"request": "bad"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "escalated"
    assert body["errors"] == ["collar bad"]
    assert "rule_id" not in body


def test_get_rule_found_and_missing(monkeypatch):
    monkeypatch.setattr(
        main.store, "get_rule", lambda rid: {"rule_id": rid, "config": {}} if rid == "x" else None
    )
    client = _client()
    assert client.get("/rules/x").status_code == 200
    assert client.get("/rules/missing").status_code == 404
