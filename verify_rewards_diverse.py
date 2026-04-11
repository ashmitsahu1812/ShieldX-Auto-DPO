from server.environment import StockExchangeEnv


if __name__ == "__main__":
    for task in StockExchangeEnv.TASKS:
        env = StockExchangeEnv(task_id=task["id"])
        env.reset(task_id=task["id"])

        actions = ["hold", "buy", "sell", "hold", "buy", "sell", "hold", "sell", "hold"]
        info = {"score": 0.01}
        done = False
        for decision in actions:
            if env.done:
                break
            qty = 5 if decision in {"buy", "sell"} else 0
            _, reward, done, info = env.step(
                {
                    "symbol": task["symbol"],
                    "decision": decision,
                    "quantity": qty,
                    "confidence": 0.4,
                    "rationale": "diverse",
                }
            )
        print(f"task={task['id']} score={info['score']:.4f} done={done}")
        assert 0.0 < info["score"] < 1.0, f"Score out of range for {task['id']}: {info['score']}"
    print("All diverse reward checks passed.")
