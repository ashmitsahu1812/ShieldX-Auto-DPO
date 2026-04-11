from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class TradeAction(BaseModel):
    """Action issued by an agent to execute a market order."""

    symbol: str = Field(..., description="Ticker symbol to trade.")
    decision: Literal["buy", "sell", "hold"] = Field(
        ..., description="Trading decision for current market tick."
    )
    quantity: int = Field(
        default=0,
        ge=0,
        le=1000,
        description="Order size in shares. Must be 0 for hold.",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Model confidence in this decision. Affects reward scaling.",
    )
    rationale: str = Field(default="", description="Short justification for auditability.")


class MarketObservation(BaseModel):
    """Observation returned after each step."""

    task_id: str
    task_name: str
    difficulty: Literal["easy", "medium", "hard"]
    symbol: str
    objective: str

    # Time
    day_index: int
    max_days: int

    # Price signals
    current_price: float
    next_price: float
    price_window: List[float] = Field(description="Last 5 prices including current.")
    momentum_1d: float = Field(description="1-day price return.")
    momentum_3d: float = Field(description="3-day price return.")
    volatility: float = Field(default=0.0, description="Current step volatility (std of recent returns).")
    market_regime: str = Field(default="unknown", description="Qualitative regime label for the task.")

    # Portfolio
    cash: float
    position: int
    avg_entry_price: float
    portfolio_value: float
    peak_portfolio_value: float
    drawdown: float
    last_decision: str

    # Risk limits (surfaced so the agent can reason about constraints)
    max_drawdown_limit: float = Field(default=0.1, description="Max allowed drawdown for this task.")
    max_position_ratio: float = Field(default=0.7, description="Max allowed position/portfolio ratio.")
    target_position_ratio: Optional[float] = Field(default=None, description="Target ratio for rebalancing tasks.")
    min_cash_ratio: Optional[float] = Field(default=None, description="Minimum cash ratio for rebalancing tasks.")

    metadata: Dict[str, float | str] = Field(default_factory=dict)


class MarketState(BaseModel):
    """Internal state snapshot exposed via /state."""

    task_id: str
    symbol: str
    day_index: int
    step_count: int
    max_steps: int
    cash: float
    position: int
    avg_entry_price: float
    portfolio_value: float
    peak_portfolio_value: float
    drawdown: float
    cumulative_reward: float
    task_score: float
    done: bool
    market_regime: str = "unknown"
    volatility: float = 0.0


class MarketReward(BaseModel):
    """Structured reward explanation for grader transparency."""

    reward: float = Field(..., description="Step reward in strict (0.05, 0.95).")
    task_score: float = Field(..., description="Current task score in strict (0.05, 0.95).")
    action_alignment: float
    return_component: float
    risk_component: float
    confidence_multiplier: float = Field(default=1.0, description="Confidence-weighted reward scaling.")
    explanation: str
    done: bool


class TaskGrade(BaseModel):
    """Task-level deterministic grade."""

    task_id: str
    score: float
    task_score: float
