"""Pydantic v2 schemas for pre-trade risk rules (the deterministic risk gateway)."""

from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class RuleType(str, Enum):
    """Discriminator enum for the three supported pre-trade rule types."""

    ORDER_NOTIONAL_LIMIT = "order_notional_limit"
    PRICE_COLLAR = "price_collar"
    RESTRICTED_LIST = "restricted_list"


class Side(str, Enum):
    """Order side applicability."""

    BUY = "buy"
    SELL = "sell"
    BOTH = "both"


class Action(str, Enum):
    """Risk action to take when a rule is triggered."""

    BLOCK = "block"
    WARN = "warn"
    REVIEW = "review"


class Exchange(str, Enum):
    """Supported APAC exchanges."""

    SGX = "SGX"
    HKEX = "HKEX"
    ASX = "ASX"
    TSE = "TSE"


class Scope(BaseModel):
    """Exchange + optional symbol/segment scope for a rule."""

    model_config = ConfigDict(extra="forbid")
    exchange: Exchange
    symbols: list[str] = Field(default_factory=list)  # empty = all symbols on exchange
    segment: str | None = None  # e.g. "small_cap"


class BaseRule(BaseModel):
    """Common fields shared by all rule types."""

    model_config = ConfigDict(extra="forbid")
    description: str = Field(min_length=3)
    scope: Scope
    side: Side = Side.BOTH
    action: Action = Action.BLOCK


class OrderNotionalLimitRule(BaseRule):
    """Block or warn when order notional exceeds a configured ceiling."""

    rule_type: Literal[RuleType.ORDER_NOTIONAL_LIMIT]
    max_notional: float = Field(gt=0)
    currency: str = "USD"


class PriceCollarRule(BaseRule):
    """Block or warn when order price deviates beyond a collar percentage."""

    rule_type: Literal[RuleType.PRICE_COLLAR]
    collar_pct: float = Field(gt=0, le=100)
    reference_price: Literal["last", "open", "vwap"] = "last"


class RestrictedListRule(BaseRule):
    """Block trading on a named list of restricted ticker symbols."""

    rule_type: Literal[RuleType.RESTRICTED_LIST]
    symbols: list[str] = Field(min_length=1)  # restricted tickers


Rule = Annotated[
    Union[OrderNotionalLimitRule, PriceCollarRule, RestrictedListRule],
    Field(discriminator="rule_type"),
]

RuleAdapter: TypeAdapter[Rule] = TypeAdapter(Rule)
