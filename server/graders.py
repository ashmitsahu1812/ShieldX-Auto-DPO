from typing import Any, Dict, List


MIN_STRICT_SCORE = 0.01
MAX_STRICT_SCORE = 0.99


def strict_score(value: float) -> float:
    """Clamp a raw score into the strict open interval (0.05, 0.95)."""
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
    """
    Score how well the agent's decision matches the ideal action.
    Perfect match = 1.0, opposite direction = 0.05, partial = 0.4.
    """
    if decision == ideal:
        return 1.0
    # Holding when action was needed — partial credit
    if decision == "hold" and ideal in {"buy", "sell"}:
        return 0.40
    # Acting when hold was ideal — small penalty
    if ideal == "hold" and decision in {"buy", "sell"}:
        return 0.30
    # buy vs sell or sell vs buy — worst case
    return 0.05


def confidence_multiplier(confidence: float, alignment: float) -> float:
    """
    Reward high confidence when correct, penalize high confidence when wrong.
    Neutral at confidence=0.5.
    """
    if alignment >= 0.9:
        # Correct decision: bonus for high confidence
        return 1.0 + 0.15 * (confidence - 0.5)
    elif alignment <= 0.1:
        # Wrong decision: penalty for high confidence
        return 1.0 - 0.20 * (confidence - 0.5)
    return 1.0


def normalize_return(task: Dict[str, Any], final_return: float) -> float:
    lo = float(task.get("min_return", -0.1))
    hi = float(task.get("target_return", 0.05))
    if hi <= lo:
        return 0.5
    ratio = (final_return - lo) / (hi - lo)
    return max(0.0, min(1.0, ratio))


def rebalance_component(task: Dict[str, Any], history: List[Dict[str, Any]]) -> float:
    """
    Extra grading component for rebalancing tasks.
    Measures how close the final position ratio is to the target.
    Returns 0.0 for tasks without a target_position_ratio.
    """
    target = task.get("target_position_ratio")
    if target is None or not history:
        return 0.0
    last = history[-1]
    portfolio_value = float(last.get("portfolio_value", 1.0))
    position = float(last.get("executed_qty", 0))  # approximate
    price = float(last.get("price", 1.0))
    position_value = position * price
    actual_ratio = position_value / max(portfolio_value, 1.0)
    distance = abs(actual_ratio - float(target))
    return max(0.0, 1.0 - distance * 5.0)


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

    # Optional rebalancing bonus for task-004
    rebal = rebalance_component(task, history)
    has_rebal = task.get("target_position_ratio") is not None

    if has_rebal:
        raw = (
            (0.40 * avg_alignment)
            + (0.20 * return_component)
            + (0.20 * risk_component)
            + (0.15 * rebal)
            + (0.05 * completion)
        )
    else:
        raw = (
            (0.50 * avg_alignment)
            + (0.25 * return_component)
            + (0.20 * risk_component)
            + (0.05 * completion)
        )

    scaled = MIN_STRICT_SCORE + (MAX_STRICT_SCORE - MIN_STRICT_SCORE) * raw
    return strict_score(scaled)
