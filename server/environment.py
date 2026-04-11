import math
from typing import Dict, Any, Tuple, Optional
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
    # Conservative cap so even multi-task aggregation stays strictly below 1.0.
    # With 5 steps and current shaping, each task ends in roughly [0.09, 0.19].
    MAX_TASK_SCORE = 0.19

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

    def _coerce_action(self, action_input: Any) -> PrivacyAction:
        """Best-effort action coercion so malformed payloads do not crash scoring."""
        default_payload = {
            "operation": "retain",
            "target": "unknown",
            "legal_basis": "fallback",
            "reasoning": "fallback action used",
        }
        if isinstance(action_input, PrivacyAction):
            return action_input
        if isinstance(action_input, dict):
            payload = {
                "operation": action_input.get("operation", default_payload["operation"]),
                "target": str(action_input.get("target", default_payload["target"])),
                "legal_basis": str(action_input.get("legal_basis", default_payload["legal_basis"])),
                "reasoning": str(action_input.get("reasoning", default_payload["reasoning"])),
            }
            try:
                return PrivacyAction(**payload)
            except Exception:
                return PrivacyAction(**default_payload)
        return PrivacyAction(**default_payload)

    def reset(self, task_id: Optional[str] = None) -> PrivacyObservation:
        # Allow evaluators/clients to select tasks via reset(task_id=...).
        if task_id:
            self.task_id = str(task_id)
            self.task = get_task(self.task_id)
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

    def step(self, action: Any) -> Tuple[PrivacyObservation, float, bool, Dict[str, Any]]:
        if self.done:
            safe_score = self._strict_unit_clamp(self.total_reward)
            return self.state(), self.MIN_STRICT_SCORE, True, {
                "msg": "Episode already finished.",
                "cumulative_reward": safe_score,
                "score": safe_score,
                "explanation": "No-op: episode already finished.",
            }
            
        self.step_count += 1
        safe_action = self._coerce_action(action)
        try:
            reward_obj = grade_action(safe_action, self.task, {"history": self.history})
        except Exception as exc:
            reward_obj = PrivacyReward(
                value=self.MIN_STRICT_SCORE,
                partial_score=self.MIN_STRICT_SCORE,
                logic_explanation=f"Grading fallback due to internal error: {exc}",
                done=False,
            )
        
        # Dense strict-bounds shaping:
        # first step gets a tiny bootstrap offset; subsequent steps encode progress.
        # This keeps task scores informative while guaranteeing strict (0,1) margins.
        base_offset = 0.03 if self.step_count == 1 else 0.0
        increment = 0.03 if reward_obj.value >= self.CORRECT_THRESHOLD else 0.01
        
        step_reward = self._strict_unit_clamp(base_offset + increment)
        
        # Absolute safety clamp for total cumulative task score.
        if self.total_reward + step_reward >= self.MAX_TASK_SCORE:
            step_reward = self._strict_unit_clamp(self.MAX_TASK_SCORE - self.total_reward)
            
        self.total_reward = self._strict_unit_clamp(self.total_reward + step_reward)
        
        self.history.append({
            "step": self.step_count,
            "action": safe_action.dict(),
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
