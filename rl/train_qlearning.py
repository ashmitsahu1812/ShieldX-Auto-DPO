from __future__ import annotations

import argparse
from typing import Dict, Tuple

import numpy as np

from rl.qlearning_utils import discretize_observation, get_q_row, save_q_table
from rl.shieldx_gym_env import ShieldXGymEnv


def evaluate_policy(env: ShieldXGymEnv, q_table: Dict[Tuple[int, ...], np.ndarray], episodes: int = 50) -> float:
    total_reward = 0.0

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        truncated = False

        while not done and not truncated:
            state = discretize_observation(obs, env.n_tasks, env.n_operations)
            row = get_q_row(q_table, state, env.action_space.n)
            action = int(np.argmax(row))
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += float(reward)

    return total_reward / float(max(episodes, 1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Q-learning on stock exchange OpenEnv")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.998)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=200)
    parser.add_argument("--output", type=str, default="artifacts/qlearning_policy.json")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    env = ShieldXGymEnv(random_task=True, max_steps=args.max_steps, seed=args.seed)

    q_table: Dict[Tuple[int, ...], np.ndarray] = {}
    epsilon = args.epsilon
    returns = []

    for episode in range(1, args.episodes + 1):
        obs, _ = env.reset()
        state = discretize_observation(obs, env.n_tasks, env.n_operations)

        done = False
        truncated = False
        total_reward = 0.0

        while not done and not truncated:
            row = get_q_row(q_table, state, env.action_space.n)
            if rng.random() < epsilon:
                action = int(env.action_space.sample())
            else:
                action = int(np.argmax(row))

            next_obs, reward, done, truncated, _ = env.step(action)
            next_state = discretize_observation(next_obs, env.n_tasks, env.n_operations)
            next_row = get_q_row(q_table, next_state, env.action_space.n)

            bootstrap = 0.0 if (done or truncated) else float(np.max(next_row))
            td_target = float(reward) + args.gamma * bootstrap
            row[action] += args.alpha * (td_target - row[action])

            state = next_state
            total_reward += float(reward)

        epsilon = max(args.epsilon_min, epsilon * args.epsilon_decay)
        returns.append(total_reward)

        if episode % args.log_interval == 0:
            avg_return = sum(returns[-args.log_interval :]) / float(args.log_interval)
            print(
                f"[TRAIN] episode={episode} avg_return={avg_return:.4f} "
                f"epsilon={epsilon:.4f} states={len(q_table)}"
            )

    eval_env = ShieldXGymEnv(random_task=True, max_steps=args.max_steps, seed=args.seed + 7)
    avg_eval_reward = evaluate_policy(eval_env, q_table, episodes=200)

    save_q_table(
        path=args.output,
        q_table=q_table,
        n_tasks=env.n_tasks,
        n_operations=env.n_operations,
        n_actions=env.action_space.n,
    )

    print(f"[RESULT] saved_policy={args.output}")
    print(f"[RESULT] eval_avg_reward={avg_eval_reward:.4f}")


if __name__ == "__main__":
    main()
