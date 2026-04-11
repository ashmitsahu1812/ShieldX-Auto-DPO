from typing import Any, Dict, List


MIN_STRICT_SCORE = 0.11
MAX_STRICT_SCORE = 0.89


def strict_score(value: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return MIN_STRICT_SCORE
    if v <= 0.0:
        return MIN_STRICT_SCORE
    if v >= 1.0:
        return MAX_STRICT_SCORE
    return max(MIN_STRICT_SCORE, min(MAX_STRICT_SCORE, v))


def action_alignment(decision: str, ideal: str) -> float:
    if decision == ideal:
        return 1.0
    if decision == "hold" and ideal in {"buy", "sell"}:
        return 0.45
    if ideal == "hold" and decision in {"buy", "sell"}:
        return 0.35
    return 0.1


def normalize_return(task: Dict[str, Any], final_return: float) -> float:
    lo = float(task.get("min_return", -0.1))
    hi = float(task.get("target_return", 0.05))
    if hi <= lo:
        return 0.5
    ratio = (final_return - lo) / (hi - lo)
    return max(0.0, min(1.0, ratio))


def grade_episode(
    task: Dict[str, Any],
    history: List[Dict[str, Any]],
    initial_cash: float,
    final_portfolio_value: float,
    drawdown: float,
    done: bool,
) -> float:
    if not history:
        return MIN_STRICT_SCORE

    alignments = [float(item.get("alignment", 0.0)) for item in history]
    avg_alignment = sum(alignments) / float(max(len(alignments), 1))

    final_return = (final_portfolio_value - initial_cash) / max(initial_cash, 1.0)
    return_component = normalize_return(task, final_return)

    max_drawdown = float(task.get("max_drawdown", 0.1))
    if max_drawdown <= 0.0:
        risk_component = 0.5
    else:
        risk_component = 1.0 - max(0.0, min(1.0, drawdown / max_drawdown))

    completion = 1.0 if done else min(len(history) / float(max(task.get("max_steps", 1), 1)), 1.0)

    raw = (0.5 * avg_alignment) + (0.25 * return_component) + (0.2 * risk_component) + (0.05 * completion)
    scaled = MIN_STRICT_SCORE + (MAX_STRICT_SCORE - MIN_STRICT_SCORE) * raw
    return strict_score(scaled)
