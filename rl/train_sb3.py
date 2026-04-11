import argparse
import json
from pathlib import Path
from typing import List

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from rl.shieldx_gym_env import ShieldXGymEnv


def make_env(seed: int, max_steps: int):
    def _init():
        return ShieldXGymEnv(random_task=True, max_steps=max_steps, seed=seed)

    return _init


def parse_layers(value: str) -> List[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ShieldX with SB3 PPO or DQN")
    parser.add_argument("--algo", choices=["ppo", "dqn"], default="ppo")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--layers", type=str, default="128,128")
    parser.add_argument("--verbose", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--model-out", type=str, default="artifacts/sb3_model.zip")
    parser.add_argument("--meta-out", type=str, default="artifacts/sb3_model_meta.json")
    args = parser.parse_args()

    net_arch = parse_layers(args.layers)
    policy_kwargs = {"net_arch": net_arch}

    n_envs = max(1, args.n_envs)
    if args.algo == "dqn" and n_envs != 1:
        print("[TRAIN_SB3] DQN works best with a single environment; forcing n_envs=1.")
        n_envs = 1
    env_fns = [make_env(args.seed + idx, args.max_steps) for idx in range(n_envs)]
    vec_env = VecMonitor(DummyVecEnv(env_fns))

    if args.algo == "ppo":
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            n_steps=256,
            batch_size=128,
            n_epochs=10,
            ent_coef=0.01,
            policy_kwargs=policy_kwargs,
            verbose=args.verbose,
            seed=args.seed,
        )
    else:
        model = DQN(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            buffer_size=50_000,
            learning_starts=1_000,
            batch_size=128,
            train_freq=4,
            gradient_steps=1,
            exploration_fraction=0.2,
            exploration_final_eps=0.05,
            target_update_interval=500,
            policy_kwargs=policy_kwargs,
            verbose=args.verbose,
            seed=args.seed,
        )

    print(f"[TRAIN_SB3] algo={args.algo} timesteps={args.timesteps} n_envs={n_envs} net_arch={net_arch}")
    model.learn(total_timesteps=args.timesteps, progress_bar=False)

    out_path = Path(args.model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))

    template_env = ShieldXGymEnv(random_task=True, max_steps=args.max_steps, seed=args.seed)
    meta = {
        "algo": args.algo,
        "timesteps": int(args.timesteps),
        "max_steps": int(args.max_steps),
        "seed": int(args.seed),
        "n_envs": int(n_envs),
        "learning_rate": float(args.learning_rate),
        "gamma": float(args.gamma),
        "net_arch": net_arch,
        "operations": template_env.operations,
        "candidate_targets": template_env.candidate_targets,
        "task_ids": template_env.task_ids,
    }
    template_env.close()

    meta_path = Path(args.meta_out)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    vec_env.close()
    print(f"[RESULT] model_saved={out_path}")
    print(f"[RESULT] metadata_saved={meta_path}")


if __name__ == "__main__":
    main()
