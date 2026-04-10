from typing import List, Dict, Any
from .models import PrivacyAction, PrivacyReward

def evaluate_pii_redaction(action: PrivacyAction, task: Dict[str, Any]) -> PrivacyReward:
    """Task 1: PII Scrubber"""
    target = action.target
    if target in task["ground_truth"]:
        return PrivacyReward(value=0.25, partial_score=0.25, logic_explanation=f"Correctly identified PII: {target}", done=False)
    else:
        return PrivacyReward(value=-0.1, partial_score=-0.1, logic_explanation="Incorrect field flagged as PII", done=False)

def evaluate_erasure(action: PrivacyAction, task: Dict[str, Any]) -> PrivacyReward:
    """Task 3: Selective Erasure"""
    if action.operation == "delete" and action.target in task["ground_truth_delete"]:
        return PrivacyReward(value=0.5, partial_score=0.5, logic_explanation="Successfully erased profile data.", done=False)
    elif action.operation == "retain" and action.target in task["ground_truth_retain"]:
        return PrivacyReward(value=0.5, partial_score=0.5, logic_explanation="Successfully retained billing history for tax compliance.", done=True)
    else:
        return PrivacyReward(value=-0.2, partial_score=-0.2, logic_explanation="Violation: Incorrect data handling.", done=False)

def evaluate_breach(action: PrivacyAction, task: Dict[str, Any]) -> PrivacyReward:
    """Task 5: Breach Assessment"""
    if action.target in task["ground_truth"]:
        return PrivacyReward(value=0.25, partial_score=0.25, logic_explanation=f"Identified affected user ID: {action.target}", done=False)
    else:
        return PrivacyReward(value=-0.05, partial_score=-0.05, logic_explanation="False detection of affected ID.", done=False)

def grade_action(action: PrivacyAction, task: Dict[str, Any], state: Dict[str, Any]) -> PrivacyReward:
    t_id = task["id"]
    if "pii-scrubber" in t_id:
        return evaluate_pii_redaction(action, task)
    elif "erasure" in t_id:
        return evaluate_erasure(action, task)
    elif "breach" in t_id:
        return evaluate_breach(action, task)
    if action.target in task.get("ground_truth", []):
        return PrivacyReward(value=1.0, partial_score=1.0, logic_explanation="Correct target identified.", done=True)
    return PrivacyReward(value=0.0, partial_score=0.0, logic_explanation="Action recorded but no progress made.", done=False)
