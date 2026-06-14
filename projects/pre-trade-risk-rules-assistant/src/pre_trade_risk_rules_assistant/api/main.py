"""FastAPI endpoints: draft a rule and fetch a stored rule."""

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from pre_trade_risk_rules_assistant import store
from pre_trade_risk_rules_assistant.graph import run_graph

app = FastAPI(title="RuleForge — Pre-Trade Risk Rules Assistant")


class DraftRequest(BaseModel):
    """Request body for drafting a rule."""

    request: str


@app.post("/rules/draft")
def draft_rule(body: DraftRequest) -> dict[str, Any]:
    """Run the agent on the NL request; persist + return on success, else escalate."""
    state = run_graph(body.request)
    if state.get("status") == "ok":
        rule_id = store.save_rule(state["validated_rule"])
        return {
            "status": "ok",
            "rule_id": rule_id,
            "config": state["validated_rule"],
            "readback": state.get("readback", ""),
            "attempts": state.get("attempts", 0),
        }
    return {
        "status": "escalated",
        "errors": state.get("errors", []),
        "escalation_reason": state.get("escalation_reason", ""),
        "attempts": state.get("attempts", 0),
    }


@app.get("/rules/{rule_id}")
def get_rule(rule_id: str) -> dict[str, Any]:
    """Fetch a stored rule by id, or 404."""
    rule = store.get_rule(rule_id)
    if rule is None:
        raise HTTPException(status_code=404, detail="rule not found")
    return rule
