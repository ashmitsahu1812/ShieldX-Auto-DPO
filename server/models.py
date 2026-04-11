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
        description="Order size in shares. Ignored for hold.",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Model confidence in this decision.",
    )
    rationale: str = Field(default="", description="Short justification for auditability.")


class MarketObservation(BaseModel):
    """Observation returned after each step."""

    task_id: str
    task_name: str
    difficulty: Literal["easy", "medium", "hard"]
    symbol: str
    objective: str
    day_index: int
    max_days: int
    current_price: float
    next_price: float
    price_window: List[float]
    momentum_1d: float
    momentum_3d: float
    cash: float
    position: int
    avg_entry_price: float
    portfolio_value: float
    peak_portfolio_value: float
    drawdown: float
    last_decision: str
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


class MarketReward(BaseModel):
    """Structured reward explanation for grader transparency."""

    reward: float = Field(..., description="Step reward normalized to strict (0,1).")
    task_score: float = Field(..., description="Current task score in strict (0,1).")
    action_alignment: float
    return_component: float
    risk_component: float
    explanation: str
    done: bool


class TaskGrade(BaseModel):
    """Task-level deterministic grade."""

    task_id: str
    score: float
    task_score: float
