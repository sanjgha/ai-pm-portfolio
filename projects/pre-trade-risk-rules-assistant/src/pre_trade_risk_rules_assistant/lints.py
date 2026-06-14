"""Domain lint checks: semantic validation beyond what JSON schema can express."""

import re

from pre_trade_risk_rules_assistant.config import get_config
from pre_trade_risk_rules_assistant.schemas.rules import (
    OrderNotionalLimitRule,
    PriceCollarRule,
    RestrictedListRule,
)

_TICKER_RE = re.compile(r"^[A-Z0-9]{1,6}$")


def _malformed(symbols: list[str]) -> list[str]:
    """Return the subset of symbols that are NOT well-formed uppercase tickers."""
    return [s for s in symbols if not _TICKER_RE.match(s)]


def run_lints(rule: object) -> list[str]:
    """Run all applicable domain lints; return human-readable error strings ([] = pass)."""
    cfg = get_config()
    lints_cfg = cfg["lints"]
    errors: list[str] = []

    # L5 (all rules): scope.symbols must be well-formed tickers.
    bad_scope = _malformed(getattr(rule, "scope").symbols)
    if bad_scope:
        errors.append(f"scope.symbols not well-formed tickers: {bad_scope}")

    if isinstance(rule, OrderNotionalLimitRule):
        # L2: notional within sane band.
        if rule.max_notional > lints_cfg["max_notional"]:
            errors.append(
                f"max_notional {rule.max_notional} exceeds ceiling {lints_cfg['max_notional']}"
            )
        if rule.max_notional < lints_cfg["min_notional"]:
            errors.append(
                f"max_notional {rule.max_notional} below floor {lints_cfg['min_notional']}"
            )
        # L3: currency must match the exchange's expected currency (units consistency).
        expected = cfg["exchange_currency"].get(rule.scope.exchange.value)
        if expected and rule.currency.upper() != expected:
            errors.append(
                f"currency {rule.currency} inconsistent with {rule.scope.exchange.value} "
                f"(expected {expected})"
            )

    elif isinstance(rule, PriceCollarRule):
        # L1: collar within the sane band.
        if not (lints_cfg["min_collar_pct"] <= rule.collar_pct <= lints_cfg["max_collar_pct"]):
            errors.append(
                f"collar_pct {rule.collar_pct} outside sane band "
                f"[{lints_cfg['min_collar_pct']}, {lints_cfg['max_collar_pct']}]"
            )

    elif isinstance(rule, RestrictedListRule):
        # L4: restricted symbols well-formed and unique.
        bad = _malformed(rule.symbols)
        if bad:
            errors.append(f"restricted symbols not well-formed: {bad}")
        if len(set(rule.symbols)) != len(rule.symbols):
            errors.append(f"restricted symbols contain duplicates: {rule.symbols}")

    return errors
