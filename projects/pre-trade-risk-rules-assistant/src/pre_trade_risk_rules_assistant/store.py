"""Persistence: SQLite rules store + append-only JSONL audit log."""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pre_trade_risk_rules_assistant.config import get_config


def _default_paths() -> tuple[Path, Path]:
    cfg = get_config()["storage"]
    return Path(cfg["db_path"]), Path(cfg["audit_path"])


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS rules (
            id TEXT PRIMARY KEY,
            rule_type TEXT NOT NULL,
            config_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    return conn


def _append_audit(audit_path: Path, entry: dict[str, Any]) -> None:
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with open(audit_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def save_rule(
    rule: dict[str, Any],
    db_path: Path | None = None,
    audit_path: Path | None = None,
) -> str:
    """Persist an approved rule and append an audit entry. Returns the new rule id."""
    dflt_db, dflt_audit = _default_paths()
    db_path = db_path or dflt_db
    audit_path = audit_path or dflt_audit

    rule_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    conn = _connect(db_path)
    try:
        conn.execute(
            "INSERT INTO rules (id, rule_type, config_json, created_at) VALUES (?, ?, ?, ?)",
            (rule_id, rule["rule_type"], json.dumps(rule), created_at),
        )
        conn.commit()
    finally:
        conn.close()

    _append_audit(
        audit_path,
        {
            "event": "rule_saved",
            "rule_id": rule_id,
            "rule_type": rule["rule_type"],
            "timestamp": created_at,
        },
    )
    return rule_id


def get_rule(rule_id: str, db_path: Path | None = None) -> dict[str, Any] | None:
    """Fetch a stored rule by id, or None."""
    dflt_db, _ = _default_paths()
    db_path = db_path or dflt_db
    if not db_path.exists():
        return None
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT id, rule_type, config_json, created_at FROM rules WHERE id = ?",
            (rule_id,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return {
        "rule_id": row[0],
        "rule_type": row[1],
        "config": json.loads(row[2]),
        "created_at": row[3],
    }
