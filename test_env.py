from server.environment import StockExchangeEnv


def smoke_test() -> None:
    env = StockExchangeEnv(task_id="task-001-trend-following")
    obs = env.reset()
    print(f"reset task={obs.task_id} price={obs.current_price}")

    action = {
        "symbol": env.task["symbol"],
        "decision": "buy",
        "quantity": 10,
        "confidence": 0.9,
        "rationale": "smoke",
    }
    _, reward, done, info = env.step(action)
    print(f"step reward={reward:.4f} done={done} score={info['score']:.4f}")

    assert 0.0 < reward < 1.0
    assert 0.0 < info["score"] < 1.0


if __name__ == "__main__":
    smoke_test()
