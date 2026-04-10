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
        if self.done:
            return self.state(), 0.01, True, {"msg": "Episode already finished."}
            
        self.step_count += 1
        reward_obj = grade_action(action, self.task, {"history": self.history})
        
        # TRIPLE-BUFFER MODEL:
        # Base offset = 0.10 (added only to the first step's reward)
        # Correct = 0.12, Mistake = 0.02
        # Resulting Range for 5 steps:
        # MIN (All mistakes): (0.10 + 0.02) + (4 * 0.02) = 0.20 ✅
        # MAX (All correct): (0.10 + 0.12) + (4 * 0.12) = 0.70 ✅
        
        base_offset = 0.10 if self.step_count == 1 else 0.0
        increment = 0.12 if reward_obj.value > 0.01 else 0.02
        
        step_reward = base_offset + increment
        
        # Absolute safety clamp for total cumulative
        if self.total_reward + step_reward >= 0.95:
            step_reward = max(0.01, 0.95 - self.total_reward)
            
        self.total_reward += step_reward
        
        self.history.append({
            "step": self.step_count,
            "action": action.dict(),
            "reward": float(step_reward),
            "cumulative": float(self.total_reward),
            "logic": reward_obj.logic_explanation
        })
        
        if reward_obj.done or self.step_count >= self.max_steps:
            self.done = True
        
        return self.state(), float(step_reward), self.done, {
            "cumulative_reward": float(self.total_reward),
            "score": float(self.total_reward),
            "explanation": reward_obj.logic_explanation
        }
