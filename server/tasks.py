from typing import Any, Dict, List


TASKS: List[Dict[str, Any]] = [
    {
        "id": "task-001-trend-following",
        "name": "Trend-Following Execution",
        "difficulty": "easy",
        "symbol": "NOVA",
        "objective": (
            "Capture an intraday uptrend while avoiding overtrading. "
            "Buy early in the trend, hold through momentum, exit before reversal."
        ),
        "prices": [100.0, 102.0, 104.0, 107.0, 109.0, 112.0],
        "ideal_actions": ["buy", "buy", "hold", "hold", "sell"],
        "initial_cash": 10000.0,
        "max_steps": 5,
        "target_return": 0.06,
        "min_return": -0.02,
        "max_drawdown": 0.06,
        "max_position_ratio": 0.85,
        "market_regime": "trending_up",
        # volatility per step (std of daily returns)
        "volatility": [0.0, 0.020, 0.019, 0.028, 0.018, 0.027],
        "description": (
            "NOVA is in a clean uptrend. The agent should accumulate early, "
            "ride the trend, and exit near the peak. Penalizes selling too early "
            "or holding through the top."
        ),
    },
    {
        "id": "task-002-mean-reversion",
        "name": "Mean-Reversion Swing",
        "difficulty": "medium",
        "symbol": "KITE",
        "objective": (
            "Buy into weakness during a pullback and reduce exposure into recovery spikes. "
            "Manage position size carefully — do not over-concentrate during the dip."
        ),
        "prices": [100.0, 96.0, 92.0, 95.0, 99.0, 103.0, 97.0],
        "ideal_actions": ["buy", "buy", "hold", "sell", "sell", "buy"],
        "initial_cash": 12000.0,
        "max_steps": 6,
        "target_return": 0.08,
        "min_return": -0.04,
        "max_drawdown": 0.12,
        "max_position_ratio": 0.75,
        "market_regime": "mean_reverting",
        "volatility": [0.0, 0.040, 0.041, 0.032, 0.041, 0.040, 0.061],
        "description": (
            "KITE drops sharply then recovers. The agent must buy the dip in two tranches, "
            "hold through the trough, trim into the recovery, then re-enter on the final dip. "
            "Tests patience and position sizing under drawdown pressure."
        ),
    },
    {
        "id": "task-003-risk-managed-hedge",
        "name": "Risk-Managed Event Trading",
        "difficulty": "hard",
        "symbol": "ORCA",
        "objective": (
            "Navigate a volatile event-driven price path with disciplined de-risking. "
            "Preserve capital during the drawdown phase, re-enter selectively, "
            "and exit before the final leg down. Max drawdown constraint is strict (8%)."
        ),
        "prices": [120.0, 118.0, 111.0, 106.0, 114.0, 109.0, 103.0, 99.0],
        "ideal_actions": ["hold", "sell", "hold", "buy", "sell", "sell", "hold"],
        "initial_cash": 15000.0,
        "max_steps": 7,
        "target_return": 0.04,
        "min_return": -0.08,
        "max_drawdown": 0.08,
        "max_position_ratio": 0.55,
        "market_regime": "volatile",
        "volatility": [0.0, 0.017, 0.059, 0.046, 0.075, 0.044, 0.055, 0.039],
        "description": (
            "ORCA undergoes a sharp event-driven selloff followed by a dead-cat bounce and "
            "continued decline. The agent must avoid holding through the crash, exploit the "
            "bounce with a small re-entry, and exit before the second leg down. "
            "The tight drawdown cap (8%) punishes passive holding severely."
        ),
    },
    {
        "id": "task-004-portfolio-rebalance",
        "name": "Dynamic Portfolio Rebalancing",
        "difficulty": "hard",
        "symbol": "APEX",
        "objective": (
            "Rebalance a concentrated position back toward a 40% equity target "
            "while minimizing market impact. Sell into strength, avoid panic selling, "
            "and maintain a minimum cash buffer of 30% at all times."
        ),
        "prices": [200.0, 204.0, 201.0, 207.0, 203.0, 210.0, 206.0, 212.0, 208.0, 215.0],
        "ideal_actions": ["sell", "hold", "sell", "hold", "sell", "hold", "sell", "hold", "sell"],
        "initial_cash": 5000.0,
        "initial_position": 40,  # starts with 40 shares — over-concentrated
        "max_steps": 9,
        "target_return": 0.03,
        "min_return": -0.05,
        "max_drawdown": 0.05,
        "max_position_ratio": 0.60,
        "target_position_ratio": 0.40,
        "min_cash_ratio": 0.30,
        "market_regime": "gradual_uptrend",
        "volatility": [0.0, 0.020, 0.015, 0.029, 0.019, 0.034, 0.019, 0.029, 0.019, 0.034],
        "description": (
            "APEX is in a slow uptrend. The agent starts over-concentrated in equity (40 shares, "
            "only $5k cash). It must systematically sell into strength every other step to reach "
            "the 40% equity target, while never letting cash drop below 30%. "
            "Tests disciplined rebalancing — not panic selling, not greedy holding."
        ),
    },
]


def get_task(task_id: str) -> Dict[str, Any]:
    for task in TASKS:
        if task["id"] == task_id:
            return task
    return TASKS[0]
