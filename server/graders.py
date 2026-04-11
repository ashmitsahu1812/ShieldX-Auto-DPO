from typing import Dict, Any

from .models import PrivacyAction, PrivacyReward

CORRECT_SCORE = 0.85
INCORRECT_SCORE = 0.15
DUPLICATE_SCORE = 0.05
IRRELEVANT_SCORE = 0.02
MIN_STRICT_SCORE = 0.11
MAX_STRICT_SCORE = 0.89


def _strict_score(value: float) -> float:
    """Force scores into strict open interval required by validator."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return IRRELEVANT_SCORE
    return max(MIN_STRICT_SCORE, min(MAX_STRICT_SCORE, number))


def _build_reward(value: float, explanation: str, done: bool) -> PrivacyReward:
    safe = _strict_score(value)
    return PrivacyReward(
        value=safe,
        partial_score=safe,
        logic_explanation=explanation,
        done=done,
    )


def evaluate_pii_redaction(action: PrivacyAction, task: Dict[str, Any]) -> PrivacyReward:
    """Task 1: PII Scrubber"""
    target = action.target
    ground_truth = task.get("ground_truth", [])
    if target in ground_truth:
        return _build_reward(CORRECT_SCORE, f"Correctly identified PII: {target}", done=False)
    return _build_reward(INCORRECT_SCORE, "Incorrect field flagged as PII", done=False)


def evaluate_erasure(action: PrivacyAction, task: Dict[str, Any]) -> PrivacyReward:
    """Task 3: Selective Erasure"""
    ground_truth_delete = task.get("ground_truth_delete", [])
    ground_truth_retain = task.get("ground_truth_retain", [])

    if action.operation == "delete" and action.target in ground_truth_delete:
        return _build_reward(CORRECT_SCORE, "Successfully erased profile data.", done=False)
    if action.operation == "retain" and action.target in ground_truth_retain:
        return _build_reward(
            CORRECT_SCORE,
            "Successfully retained billing history for tax compliance.",
            done=True,
        )
    return _build_reward(INCORRECT_SCORE, "Violation: Incorrect data handling.", done=False)


def evaluate_breach(action: PrivacyAction, task: Dict[str, Any]) -> PrivacyReward:
    """Task 5: Breach Assessment"""
    ground_truth = task.get("ground_truth", [])
    if action.target in ground_truth:
        return _build_reward(CORRECT_SCORE, f"Identified affected user ID: {action.target}", done=False)
    return _build_reward(INCORRECT_SCORE, "False detection of affected ID.", done=False)


def grade_action(action: PrivacyAction, task: Dict[str, Any], state: Dict[str, Any]) -> PrivacyReward:
    t_id = str(task.get("id", ""))
    history = state.get("history", [])

    already_done = any(
        isinstance(h, dict)
        and isinstance(h.get("action"), dict)
        and h["action"].get("target") == action.target
        and float(h.get("reward", 0.0)) > 0.0
        for h in history
    )
    if already_done:
        return _build_reward(
            DUPLICATE_SCORE,
            f"Duplicate action: Redundancy identified for {action.target}",
            done=False,
        )

    try:
        if "pii-scrubber" in t_id:
            return evaluate_pii_redaction(action, task)
        if "erasure" in t_id:
            return evaluate_erasure(action, task)
        if "breach" in t_id:
            return evaluate_breach(action, task)

        if action.target in task.get("ground_truth", []):
            return _build_reward(CORRECT_SCORE, "Target correctly processed.", done=True)

        return _build_reward(IRRELEVANT_SCORE, "Action irrelevant to task goal.", done=False)
    except Exception as exc:
        return _build_reward(IRRELEVANT_SCORE, f"Grader fallback used due to internal error: {exc}", done=False)
