import os
import json
from typing import Dict, Any, Tuple
from .models import PrivacyAction, PrivacyObservation, PrivacyReward
from .tasks import get_task, TASKS
from .graders import grade_action

class ShieldXEnv:
    """
    OpenEnv Environment for Autonomous Data Privacy Governance.
    """
    TASKS = TASKS
    def __init__(self, task_id: str = "task-001-pii-scrubber", max_steps: int = 5):
        self.task_id = task_id
        self.task = get_task(task_id)
        self.max_steps = max_steps
        self.step_count = 0
        self.total_reward = 0.0
        self.history = []
        self.done = False

    def reset(self) -> PrivacyObservation:
        self.step_count = 0
        self.total_reward = 0.0
        self.current_score = 0.01
        self.history = []
        self.done = False
        return self.state()

    def state(self) -> PrivacyObservation:
        return PrivacyObservation(
            task_id=self.task["id"],
            instruction=self.task["instruction"],
            data_buffer=self.task["data"],
            policy_context=self.task["policy"],
            region=self.task["region"],
            step_count=self.step_count,
            max_steps=self.max_steps
        )

    def step(self, action: PrivacyAction) -> Tuple[PrivacyObservation, float, bool, Dict[str, Any]]:
        # Ensure current_score is initialized
        if not hasattr(self, 'current_score'):
            self.current_score = 0.01
            
        if self.done:
            return self.state(), 0.01, True, {"msg": "Episode already finished."}
            
        self.step_count += 1
        reward_obj = grade_action(action, self.task, {"history": self.history})
        
        # In this 'Strictly Positive' mode, we treat raw rewards as progress points.
        # Mistakes give 0 extra points, but never subtract.
        raw_val = max(0, reward_obj.value)
        self.total_reward += raw_val
        
        # Calculate new score in (0.01, 0.99)
        # We use a mapping that ensures even 0 points -> 0.01 and many points -> 0.95
        new_score = 0.01 + (0.94 * (1.0 - pow(0.5, self.total_reward / 0.5)))
        new_score = max(0.011, min(0.989, new_score))
        
        # Every step MUST have a positive reward > 0
        step_reward = max(0.01, new_score - self.current_score)
        
        # Update trackers
        self.current_score += step_reward
        
        self.history.append({
            "step": self.step_count,
            "action": action.dict(),
            "reward": float(step_reward),
            "score_at_step": float(self.current_score),
            "logic": reward_obj.logic_explanation
        })
        
        if reward_obj.done or self.step_count >= self.max_steps:
            self.done = True
        
        # Double check: current_score MUST be < 1.0
        final_out_score = min(0.99, self.current_score)
        
        return self.state(), float(step_reward), self.done, {
            "cumulative_reward": float(final_out_score),
            "score": float(final_out_score),
            "explanation": reward_obj.logic_explanation
        }
