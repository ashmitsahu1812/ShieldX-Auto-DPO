import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple, Optional

from .environment import CodeReviewEnv
from .models import Action as CodeAction

class CodeReviewGymEnv(gym.Env):
    """
    Gymnasium wrapper for CodeReviewEnv to enable standard RL integration.
    """
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, task_type: str = "syntax_review", max_steps: int = 8):
        super().__init__()
        self.internal_env = CodeReviewEnv(task_type=task_type, max_steps=max_steps)
        
        # Action space: [ActionType, FileIndex, LineIndex]
        # ActionType: 0=comment, 1=approve, 2=request_changes
        # Assuming max 10 files and max 500 lines for simulation mapping
        self.action_space = spaces.MultiDiscrete([3, 10, 500])
        
        # Observation space: 
        # For simplicity in standard RL, we provide a flat vector for indices
        # but the actual LLM-based RL often uses the raw text observation.
        # We'll expose the raw observation in the info dict and a placeholder vector here.
        self.observation_space = spaces.Dict({
            "step_count": spaces.Box(low=0, high=max_steps, shape=(1,), dtype=np.int32),
            "files_count": spaces.Box(low=0, high=10, shape=(1,), dtype=np.int32),
        })

    def _get_obs(self, internal_obs):
        return {
            "step_count": np.array([internal_obs.step_count], dtype=np.int32),
            "files_count": np.array([len(internal_obs.files_changed)], dtype=np.int32),
        }

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed)
        internal_obs = self.internal_env.reset()
        obs = self._get_obs(internal_obs)
        info = {
            "raw_observation": internal_obs.model_dump(),
            "message": "Environment reset"
        }
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        action_type_idx = action[0]
        file_idx = action[1]
        line_num = action[2]
        
        # Map indices back to names
        action_map = ["comment", "approve", "request_changes"]
        action_type = action_map[action_type_idx]
        
        filename = "unknown"
        internal_obs = self.internal_env.state()
        if file_idx < len(internal_obs.files_changed):
            filename = internal_obs.files_changed[file_idx].filename
            
        # Create domain action
        domain_action = CodeAction(
            action_type=action_type,
            file=filename if action_type == "comment" else None,
            line=line_num if action_type == "comment" else None,
            comment="RL Agent Feedback"
        )
        
        internal_obs, reward, done, internal_info = self.internal_env.step(domain_action)
        
        obs = self._get_obs(internal_obs)
        terminated = done
        truncated = False # Manual truncation not defined here
        
        info = {
            "internal_info": internal_info.model_dump(),
            "raw_observation": internal_obs.model_dump(),
            "action_taken": domain_action.model_dump()
        }
        
        return obs, float(reward), terminated, truncated, info

    def render(self):
        obs = self.internal_env.state()
        print(f"--- CodeReviewEnv Step {obs.step_count}/{obs.max_steps} ---")
        print(f"Feedback: {obs.last_action_feedback}")
