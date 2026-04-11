import os
import json
import math
from typing import Dict, Any, Tuple
from .models import PrivacyAction, PrivacyObservation, PrivacyReward
from .tasks import get_task, TASKS
from .graders import grade_action

class ShieldXEnv:
    """
    OpenEnv Environment for Autonomous Data Privacy Governance.
    """
    TASKS = TASKS
    MIN_STRICT_SCORE = 0.01
    MAX_STRICT_SCORE = 0.99
    CORRECT_THRESHOLD = 0.5

    def __init__(self, task_id: str = "task-001-pii-scrubber", max_steps: int = 5):
        self.task_id = task_id
        self.task = get_task(task_id)
        self.max_steps = max_steps
        self.step_count = 0
        self.total_reward = self.MIN_STRICT_SCORE
        self.history = []
        self.done = False

    def _strict_unit_clamp(self, value: float) -> float:
        """Clamp scores to strict open interval semantics used by evaluators."""
        try:
            number = float(value)
        except (TypeError, ValueError):
            return self.MIN_STRICT_SCORE
        if not math.isfinite(number):
            return self.MIN_STRICT_SCORE
        return max(self.MIN_STRICT_SCORE, min(self.MAX_STRICT_SCORE, number))

    def reset(self) -> PrivacyObservation:
        self.step_count = 0
        self.total_reward = self.MIN_STRICT_SCORE
        self.current_score = self.MIN_STRICT_SCORE
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
            safe_score = self._strict_unit_clamp(self.total_reward)
            return self.state(), self.MIN_STRICT_SCORE, True, {
                "msg": "Episode already finished.",
                "cumulative_reward": safe_score,
                "score": safe_score,
                "explanation": "No-op: episode already finished.",
            }
            
        self.step_count += 1
        reward_obj = grade_action(action, self.task, {"history": self.history})
        
        # TRIPLE-BUFFER MODEL:
        # Base offset = 0.10 (added only to the first step's reward)
        # Correct = 0.12, Mistake = 0.02
        # Resulting Range for 5 steps:
        # MIN (All mistakes): (0.10 + 0.02) + (4 * 0.02) = 0.20 ✅
        # MAX (All correct): (0.10 + 0.12) + (4 * 0.12) = 0.70 ✅
        
        base_offset = 0.10 if self.step_count == 1 else 0.0
        increment = 0.12 if reward_obj.value >= self.CORRECT_THRESHOLD else 0.02
        
        step_reward = self._strict_unit_clamp(base_offset + increment)
        
        # Absolute safety clamp for total cumulative
        if self.total_reward + step_reward >= 0.95:
            step_reward = self._strict_unit_clamp(0.95 - self.total_reward)
            
        self.total_reward = self._strict_unit_clamp(self.total_reward + step_reward)
        
        self.history.append({
            "step": self.step_count,
            "action": action.dict(),
            "reward": float(self._strict_unit_clamp(step_reward)),
            "cumulative": float(self._strict_unit_clamp(self.total_reward)),
            "logic": reward_obj.logic_explanation
        })
        
        if reward_obj.done or self.step_count >= self.max_steps:
            self.done = True
        
        safe_step_reward = float(self._strict_unit_clamp(step_reward))
        safe_total_reward = float(self._strict_unit_clamp(self.total_reward))
        return self.state(), safe_step_reward, self.done, {
            "cumulative_reward": safe_total_reward,
            "score": safe_total_reward,
            "explanation": reward_obj.logic_explanation
        }
