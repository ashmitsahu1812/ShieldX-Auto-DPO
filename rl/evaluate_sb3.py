import argparse
import json
from pathlib import Path
from typing import Dict

from rl.shieldx_gym_env import ShieldXGymEnv

MIN_STRICT_SCORE = 0.01


def evaluate_task(task_id: str, model, episodes: int, max_steps: int) -> Dict[str, float]:
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
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(int(action))
            total_reward += float(reward)
            final_score = float(info.get("score", MIN_STRICT_SCORE))

        total_score += final_score
        if final_score >= 0.2:
            success_count += 1

    env.close()
    return {
        "avg_reward": total_reward / float(max(episodes, 1)),
        "avg_final_score": total_score / float(max(episodes, 1)),
        "success_rate": float(success_count) / float(max(episodes, 1)),
    }


def main() -> None:
    try:
        from stable_baselines3 import DQN, PPO
    except Exception as exc:
        raise RuntimeError(
            "stable-baselines3 is not installed. Install it with: pip install -r requirements-rl.txt"
        ) from exc

    parser = argparse.ArgumentParser(description="Evaluate ShieldX SB3 policy")
    parser.add_argument("--algo", choices=["ppo", "dqn"], default="ppo")
    parser.add_argument("--model", type=str, default="artifacts/sb3_model.zip")
    parser.add_argument("--meta", type=str, default="artifacts/sb3_model_meta.json")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=5)
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if args.algo == "ppo":
        model = PPO.load(str(model_path))
    else:
        model = DQN.load(str(model_path))

    _default_task_ids = [
        "task-001-trend-following",
        "task-002-mean-reversion",
        "task-003-risk-managed-hedge",
    ]
    if Path(args.meta).exists():
        meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))
        task_ids = meta.get("task_ids", _default_task_ids)
    else:
        task_ids = _default_task_ids

    all_metrics = []
    for task_id in task_ids:
        metrics = evaluate_task(
            task_id=task_id,
            model=model,
            episodes=args.episodes,
            max_steps=args.max_steps,
        )
        all_metrics.append(metrics)

        print(
            f"[EVAL_SB3] task={task_id} avg_reward={metrics['avg_reward']:.4f} "
            f"avg_final_score={metrics['avg_final_score']:.4f} "
            f"success_rate={metrics['success_rate']:.2%}"
        )

    overall_reward = sum(metric["avg_reward"] for metric in all_metrics) / len(all_metrics)
    overall_score = sum(metric["avg_final_score"] for metric in all_metrics) / len(all_metrics)
    overall_success = sum(metric["success_rate"] for metric in all_metrics) / len(all_metrics)

    print(f"[SUMMARY_SB3] avg_reward={overall_reward:.4f} avg_final_score={overall_score:.4f} avg_success_rate={overall_success:.2%}")


if __name__ == "__main__":
    main()
