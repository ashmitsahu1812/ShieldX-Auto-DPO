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
            return self.state(), 0.0, True, {"msg": "Episode already finished."}
        self.step_count += 1
        reward_obj = grade_action(action, self.task, {"history": self.history})
        self.total_reward += reward_obj.value
        self.history.append({
            "step": self.step_count,
            "action": action.dict(),
            "reward": reward_obj.value,
            "logic": reward_obj.logic_explanation
        })
        if reward_obj.done or self.step_count >= self.max_steps:
            self.done = True
        return self.state(), reward_obj.value, self.done, {
            "cumulative_reward": self.total_reward,
            "score": min(max(self.total_reward, 0.0), 1.0),
            "explanation": reward_obj.logic_explanation
        }
