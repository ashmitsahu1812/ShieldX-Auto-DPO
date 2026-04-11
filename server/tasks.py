from typing import Any, Dict, List


TASKS: List[Dict[str, Any]] = [
    {
        "id": "task-001-trend-following",
        "name": "Trend-Following Execution",
        "difficulty": "easy",
        "symbol": "NOVA",
        "objective": "Capture an intraday uptrend while avoiding overtrading.",
        "prices": [100.0, 102.0, 104.0, 107.0, 109.0, 112.0],
        "ideal_actions": ["buy", "buy", "hold", "hold", "sell"],
        "initial_cash": 10000.0,
        "max_steps": 5,
        "target_return": 0.06,
        "min_return": -0.02,
        "max_drawdown": 0.06,
        "max_position_ratio": 0.85,
    },
    {
        "id": "task-002-mean-reversion",
        "name": "Mean-Reversion Swing",
        "difficulty": "medium",
        "symbol": "KITE",
        "objective": "Buy weakness and reduce exposure into recovery spikes.",
        "prices": [100.0, 96.0, 92.0, 95.0, 99.0, 103.0, 97.0],
        "ideal_actions": ["buy", "buy", "hold", "sell", "sell", "buy"],
        "initial_cash": 12000.0,
        "max_steps": 6,
        "target_return": 0.08,
        "min_return": -0.04,
        "max_drawdown": 0.12,
        "max_position_ratio": 0.75,
    },
    {
        "id": "task-003-risk-managed-hedge",
        "name": "Risk-Managed Event Trading",
        "difficulty": "hard",
        "symbol": "ORCA",
        "objective": "Navigate volatility with disciplined de-risking and selective re-entry.",
        "prices": [120.0, 118.0, 111.0, 106.0, 114.0, 109.0, 103.0, 99.0],
        "ideal_actions": ["hold", "sell", "hold", "buy", "sell", "sell", "hold"],
        "initial_cash": 15000.0,
        "max_steps": 7,
        "target_return": 0.04,
        "min_return": -0.08,
        "max_drawdown": 0.08,
        "max_position_ratio": 0.55,
    },
]


def get_task(task_id: str) -> Dict[str, Any]:
    for task in TASKS:
        if task["id"] == task_id:
            return task
    return TASKS[0]
