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
        # Initialize internal score tracking if not present (safety for resets)
        if not hasattr(self, 'current_score'):
            self.current_score = 0.01
            
        if self.done:
            return self.state(), 0.01, True, {"msg": "Episode already finished."}
            
        self.step_count += 1
        reward_obj = grade_action(action, self.task, {"history": self.history})
        
        # Add to raw points
        self.total_reward += reward_obj.value
        
        # Map accumulated raw points to a [0.01, 0.99] score space using a sigmoid
        # This keeps the total cumulative reward strictly between 0 and 1
        # Logistic function: L / (1 + e^-k(x-x0))
        # Here we use a shifted sigmoid so 0 points -> ~0.01
        new_score = 0.01 + (0.98 * (1.0 / (1.0 + pow(2.71828, -self.total_reward))))
        
        # Ensure it is strictly within (0.011, 0.989)
        new_score = max(0.011, min(0.989, new_score))
        
        # The reward for THIS step is the delta that moves the agent toward the high score
        # Sum of deltas will be (Final Score - Starting Score)
        step_reward = new_score - self.current_score
        
        # Safety: Every step should have a non-zero reward for RL stability
        # We use 0.01 instead of 0.001 to avoid '0.00' rounding in logs
        if abs(step_reward) < 0.005:
            step_reward = 0.01 if reward_obj.value >= 0 else -0.01
            
        # Update current score for next step
        self.current_score = new_score
        
        self.history.append({
            "step": self.step_count,
            "action": action.dict(),
            "reward": float(step_reward),
            "score_at_step": float(new_score),
            "logic": reward_obj.logic_explanation
        })
        
        if reward_obj.done or self.step_count >= self.max_steps:
            self.done = True
        
        return self.state(), float(step_reward), self.done, {
            "cumulative_reward": float(new_score),
            "score": float(new_score),
            "explanation": reward_obj.logic_explanation
        }
