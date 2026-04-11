from server.environment import StockExchangeEnv


def run_rollout(task_id: str) -> float:
    env = StockExchangeEnv(task_id=task_id)
    env.reset(task_id=task_id)
    task = env.task

    total = 0.0
    for decision in task["ideal_actions"]:
        if env.done:
            break
        qty = 10 if decision in {"buy", "sell"} else 0
        _, reward, _, _ = env.step(
            {
                "symbol": task["symbol"],
                "decision": decision,
                "quantity": qty,
                "confidence": 0.9,
                "rationale": "verify",
            }
        )
        total += reward
    return total


if __name__ == "__main__":
    for task in StockExchangeEnv.TASKS:
        score = run_rollout(task["id"])
        print(f"task={task['id']} total_reward={score:.4f}")
        assert 0.0 < score < 10.0
