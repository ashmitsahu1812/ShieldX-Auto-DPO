from __future__ import annotations

import random
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from server.environment import StockExchangeEnv


class ShieldXGymEnv(gym.Env):
    """Gym wrapper for the stock exchange environment."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        task_id: Optional[str] = None,
        random_task: bool = True,
        max_steps: int = 7,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.task_ids = [task["id"] for task in StockExchangeEnv.TASKS]
        self.random_task = bool(random_task and task_id is None)
        self.fixed_task_id = task_id
        self.max_steps = max_steps

        self.decisions = ["buy", "sell", "hold"]
        self.qty_buckets = [0, 5, 10]
        self.action_space = spaces.Discrete(len(self.decisions) * len(self.qty_buckets))

        # [task one-hot] + [cash_ratio, position_ratio, momentum_1d, momentum_3d, drawdown, progress]
        self.n_tasks = len(self.task_ids)
        self.n_operations = len(self.decisions)
        self.observation_space = spaces.Box(
            low=np.array([0.0] * (self.n_tasks + 6), dtype=np.float32),
            high=np.array([1.0] * self.n_tasks + [2.0, 2.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._rng = random.Random(seed)
        self.current_task_id = task_id or self.task_ids[0]
        self.env = StockExchangeEnv(task_id=self.current_task_id)
        self.episode_step = 0

    def _pick_task(self, requested: Optional[str]) -> str:
        if requested and requested in self.task_ids:
            return requested
        if self.fixed_task_id:
            return self.fixed_task_id
        if self.random_task:
            return self._rng.choice(self.task_ids)
        return self.task_ids[0]

    def _decode_action(self, action: int) -> Tuple[str, int]:
        n_qty = len(self.qty_buckets)
        decision_idx = int(action) // n_qty
        qty_idx = int(action) % n_qty
        decision_idx = max(0, min(len(self.decisions) - 1, decision_idx))
        qty_idx = max(0, min(len(self.qty_buckets) - 1, qty_idx))
        return self.decisions[decision_idx], self.qty_buckets[qty_idx]

    def _task_index(self) -> int:
        return self.task_ids.index(self.current_task_id)

    def _to_obs(self, obs: Dict[str, Any]) -> np.ndarray:
        one_hot = np.zeros(self.n_tasks, dtype=np.float32)
        one_hot[self._task_index()] = 1.0

        initial_cash = float(self.env.task.get("initial_cash", 10000.0))
        cash_ratio = float(obs.get("cash", 0.0)) / max(initial_cash, 1.0)
        portfolio_value = float(obs.get("portfolio_value", 0.0))
        position_value = float(obs.get("position", 0.0)) * float(obs.get("current_price", 0.0))
        position_ratio = position_value / max(portfolio_value, 1.0)

        max_days = max(int(obs.get("max_days", 1)), 1)
        progress = float(obs.get("day_index", 0)) / float(max_days)

        features = np.array(
            [
                cash_ratio,
                position_ratio,
                float(obs.get("momentum_1d", 0.0)),
                float(obs.get("momentum_3d", 0.0)),
                float(obs.get("drawdown", 0.0)),
                progress,
            ],
            dtype=np.float32,
        )
        return np.concatenate([one_hot, features]).astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng.seed(seed)

        options = options or {}
        self.current_task_id = self._pick_task(options.get("task_id"))
        self.env = StockExchangeEnv(task_id=self.current_task_id)
        obs = self.env.reset(task_id=self.current_task_id)
        self.episode_step = 0

        return self._to_obs(obs.model_dump()), {"task_id": self.current_task_id}

    def step(self, action: int):
        decision, quantity = self._decode_action(int(action))

        payload = {
            "symbol": self.env.task["symbol"],
            "decision": decision,
            "quantity": quantity,
            "confidence": 0.6,
            "rationale": "q_learning",
        }

        obs, reward, done, info = self.env.step(payload)
        self.episode_step += 1

        terminated = bool(done)
        truncated = bool(self.episode_step >= self.max_steps and not terminated)

        return self._to_obs(obs.model_dump()), float(reward), terminated, truncated, info

    def render(self):
        return None

    def close(self):
        return None
