from typing import Dict, Any, List, Set

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


def _task_targets(task: Dict[str, Any]) -> Set[str]:
    targets: Set[str] = set()
    for key in ("ground_truth", "ground_truth_delete", "ground_truth_retain"):
        values = task.get(key, [])
        if isinstance(values, list):
            for value in values:
                targets.add(str(value))
    return targets


def _is_positive_logic(logic: str) -> bool:
    text = str(logic or "").lower()
    positive_markers = (
        "correctly",
        "successfully",
        "identified",
        "target correctly processed",
        "retained billing",
    )
    return any(marker in text for marker in positive_markers)


def grade_task_score(
    task: Dict[str, Any],
    history: List[Dict[str, Any]],
    done: bool,
    max_steps: int,
) -> float:
    """
    Deterministic task-level grader.
    Returns a strict score in (0,1), weighted by coverage, correctness, and completion.
    """
    if not history:
        return MIN_STRICT_SCORE

    targets = _task_targets(task)
    acted_targets: List[str] = []
    positive_steps = 0
    for entry in history:
        action = entry.get("action", {})
        target = str(action.get("target", ""))
        acted_targets.append(target)
        if _is_positive_logic(str(entry.get("logic", ""))):
            positive_steps += 1

    unique_acted = set(acted_targets)
    hit_targets = unique_acted.intersection(targets) if targets else set()

    coverage = float(len(hit_targets)) / float(max(len(targets), 1))
    correctness = float(positive_steps) / float(max(len(history), 1))
    efficiency = float(len(unique_acted)) / float(max(len(history), 1))
    progress = min(float(len(history)) / float(max(max_steps, 1)), 1.0)
    completion = 1.0 if done else (0.5 * progress)

    raw = (0.45 * coverage) + (0.25 * correctness) + (0.15 * efficiency) + (0.15 * completion)
    scaled = MIN_STRICT_SCORE + (MAX_STRICT_SCORE - MIN_STRICT_SCORE) * raw
    return _strict_score(scaled)


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
