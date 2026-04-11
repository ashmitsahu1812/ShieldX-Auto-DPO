from __future__ import annotations

import argparse
from typing import Dict, Tuple

import numpy as np

from rl.qlearning_utils import discretize_observation, get_q_row, load_q_table
from rl.shieldx_gym_env import ShieldXGymEnv


MIN_STRICT_SCORE = 0.11


def evaluate_task(
    task_id: str,
    q_table: Dict[Tuple[int, ...], np.ndarray],
    episodes: int,
    max_steps: int,
) -> Dict[str, float]:
    env = ShieldXGymEnv(task_id=task_id, random_task=False, max_steps=max_steps)

    total_reward = 0.0
    total_score = 0.0
    success_count = 0

    for _ in range(episodes):
        obs, _ = env.reset(options={"task_id": task_id})
        done = False
        truncated = False
        final_score = MIN_STRICT_SCORE

        while not done and not truncated:
            state = discretize_observation(obs, env.n_tasks, env.n_operations)
            row = get_q_row(q_table, state, env.action_space.n)
            action = int(np.argmax(row))

            obs, reward, done, truncated, info = env.step(action)
            total_reward += float(reward)
            final_score = float(info.get("score", MIN_STRICT_SCORE))

        total_score += final_score
        if final_score >= 0.5:
            success_count += 1

    return {
        "avg_reward": total_reward / float(max(episodes, 1)),
        "avg_final_score": total_score / float(max(episodes, 1)),
        "success_rate": success_count / float(max(episodes, 1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Q-learning policy on stock exchange tasks")
    parser.add_argument("--policy", type=str, required=True, help="Path to qlearning policy json")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=7)
    args = parser.parse_args()

    q_table, _, _, _ = load_q_table(args.policy)

    task_ids = [
        "task-001-trend-following",
        "task-002-mean-reversion",
        "task-003-risk-managed-hedge",
    ]

    all_metrics = []
    for task_id in task_ids:
        metrics = evaluate_task(task_id, q_table=q_table, episodes=args.episodes, max_steps=args.max_steps)
        all_metrics.append(metrics)
        print(
            f"[EVAL] task={task_id} avg_reward={metrics['avg_reward']:.4f} "
            f"avg_final_score={metrics['avg_final_score']:.4f} success_rate={metrics['success_rate']:.2%}"
        )

    overall_reward = sum(m["avg_reward"] for m in all_metrics) / len(all_metrics)
    overall_score = sum(m["avg_final_score"] for m in all_metrics) / len(all_metrics)
    overall_success = sum(m["success_rate"] for m in all_metrics) / len(all_metrics)

    print(
        f"[SUMMARY] avg_reward={overall_reward:.4f} "
        f"avg_final_score={overall_score:.4f} avg_success_rate={overall_success:.2%}"
    )


if __name__ == "__main__":
    main()
