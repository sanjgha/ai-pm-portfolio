"""Test the SQLite rules store and JSONL audit log."""

import json

from pre_trade_risk_rules_assistant import store


def _rule_dict():
    return {
        "rule_type": "price_collar",
        "description": "5% collar on SGX",
        "scope": {"exchange": "SGX", "symbols": [], "segment": None},
        "side": "both",
        "action": "warn",
        "collar_pct": 5.0,
        "reference_price": "last",
    }


def test_save_then_get_roundtrip(tmp_path):
    db = tmp_path / "rules.db"
    audit = tmp_path / "audit.jsonl"
    rule_id = store.save_rule(_rule_dict(), db_path=db, audit_path=audit)
    assert rule_id
    fetched = store.get_rule(rule_id, db_path=db)
    assert fetched is not None
    assert fetched["config"]["collar_pct"] == 5.0
    assert fetched["rule_type"] == "price_collar"


def test_get_missing_returns_none(tmp_path):
    db = tmp_path / "rules.db"
    assert store.get_rule("nope", db_path=db) is None


def test_audit_log_appended(tmp_path):
    db = tmp_path / "rules.db"
    audit = tmp_path / "audit.jsonl"
    rule_id = store.save_rule(_rule_dict(), db_path=db, audit_path=audit)
    lines = audit.read_text().strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["event"] == "rule_saved"
    assert entry["rule_id"] == rule_id
    assert "timestamp" in entry
