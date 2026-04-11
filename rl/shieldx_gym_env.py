import random
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from server.environment import ShieldXEnv
from server.tasks import TASKS


def _collect_task_ids() -> List[str]:
    return [task["id"] for task in TASKS]


def _collect_candidate_targets() -> List[str]:
    targets = set()
    for task in TASKS:
        for key in ("ground_truth", "ground_truth_delete", "ground_truth_retain"):
            for target in task.get(key, []):
                targets.add(str(target))

    # Add a few generic distractors/fallbacks to keep policy robust.
    targets.update({"unknown", "all", "none", "profile", "billing"})
    return sorted(targets)


class ShieldXGymEnv(gym.Env):
    """Gymnasium wrapper for ShieldX so it can be used with RL algorithms.

    Observation: compact feature vector [task one-hot | last-op one-hot | scalars]
    Action: discrete index mapped to (operation, target)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        task_id: Optional[str] = None,
        random_task: bool = True,
        max_steps: int = 5,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.task_ids = _collect_task_ids()
        self.fixed_task_id = task_id
        self.random_task = random_task and task_id is None
        self.max_steps = max_steps

        self.operations = ["redact", "delete", "export", "retain", "notify"]
        self.candidate_targets = _collect_candidate_targets()

        self.n_tasks = len(self.task_ids)
        self.n_operations = len(self.operations)
        self.n_targets = len(self.candidate_targets)

        self.action_space = spaces.Discrete(self.n_operations * self.n_targets)

        # [task one-hot] + [last operation one-hot] +
        # [step_ratio, remaining_ratio, cumulative_score, target_coverage, last_reward, last_target_ratio]
        obs_dim = self.n_tasks + self.n_operations + 6
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self._rng = random.Random(seed)
        self._episode_step = 0
        self._last_operation_idx = 0
        self._last_target_idx = 0
        self.current_task_id = task_id or self.task_ids[0]
        self.env = ShieldXEnv(task_id=self.current_task_id, max_steps=self.max_steps)

    def _select_task_id(self, requested_task_id: Optional[str] = None) -> str:
        if requested_task_id and requested_task_id in self.task_ids:
            return requested_task_id
        if self.fixed_task_id:
            return self.fixed_task_id
        if self.random_task:
            return self._rng.choice(self.task_ids)
        return self.task_ids[0]

    def _decode_action(self, action: int) -> Tuple[str, str, int, int]:
        operation_idx = int(action) // self.n_targets
        target_idx = int(action) % self.n_targets
        operation_idx = max(0, min(self.n_operations - 1, operation_idx))
        target_idx = max(0, min(self.n_targets - 1, target_idx))
        return (
            self.operations[operation_idx],
            self.candidate_targets[target_idx],
            operation_idx,
            target_idx,
        )

    def _task_index(self) -> int:
        return self.task_ids.index(self.current_task_id)

    def _target_coverage(self) -> float:
        task = self.env.task
        expected_targets = []
        if task.get("ground_truth"):
            expected_targets = list(task.get("ground_truth", []))
        else:
            expected_targets.extend(task.get("ground_truth_delete", []))
            expected_targets.extend(task.get("ground_truth_retain", []))

        if not expected_targets:
            return 0.0

        expected_set = {str(value) for value in expected_targets}
        seen = {
            entry.get("action", {}).get("target")
            for entry in self.env.history
            if isinstance(entry, dict)
        }
        matched = len(expected_set.intersection(seen))
        return float(matched) / float(len(expected_set))

    def _build_observation(self, last_reward: float) -> np.ndarray:
        task_one_hot = np.zeros(self.n_tasks, dtype=np.float32)
        task_one_hot[self._task_index()] = 1.0

        last_op_one_hot = np.zeros(self.n_operations, dtype=np.float32)
        last_op_one_hot[self._last_operation_idx] = 1.0

        step_ratio = float(self.env.step_count) / float(max(self.max_steps, 1))
        remaining_ratio = float(max(self.max_steps - self.env.step_count, 0)) / float(max(self.max_steps, 1))
        cumulative_score = float(self.env._strict_unit_clamp(self.env.total_reward))
        coverage = float(max(0.0, min(1.0, self._target_coverage())))
        last_reward = float(self.env._strict_unit_clamp(last_reward))

        if self.n_targets > 1:
            last_target_ratio = float(self._last_target_idx) / float(self.n_targets - 1)
        else:
            last_target_ratio = 0.0

        scalar_features = np.array(
            [
                step_ratio,
                remaining_ratio,
                cumulative_score,
                coverage,
                last_reward,
                last_target_ratio,
            ],
            dtype=np.float32,
        )

        return np.concatenate([task_one_hot, last_op_one_hot, scalar_features]).astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng.seed(seed)

        options = options or {}
        requested_task = options.get("task_id")
        self.current_task_id = self._select_task_id(requested_task)
        self.env = ShieldXEnv(task_id=self.current_task_id, max_steps=self.max_steps)
        self.env.reset()

        self._episode_step = 0
        self._last_operation_idx = 0
        self._last_target_idx = 0

        obs = self._build_observation(last_reward=self.env.MIN_STRICT_SCORE)
        info = {"task_id": self.current_task_id}
        return obs, info

    def step(self, action: int):
        operation, target, operation_idx, target_idx = self._decode_action(int(action))

        self._last_operation_idx = operation_idx
        self._last_target_idx = target_idx

        payload = {
            "operation": operation,
            "target": target,
            "legal_basis": "RL policy",
            "reasoning": "Q-learning action",
        }

        _, reward, done, info = self.env.step(payload)
        self._episode_step += 1

        terminated = bool(done)
        truncated = bool(self._episode_step >= self.max_steps and not terminated)

        obs = self._build_observation(last_reward=reward)

        step_info = dict(info)
        step_info["task_id"] = self.current_task_id
        step_info["decoded_action"] = {"operation": operation, "target": target}

        return obs, float(reward), terminated, truncated, step_info

    def render(self):
        return None

    def close(self):
        return None
