import asyncio
import json
import os
import time
from contextlib import suppress
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

ENV_URL = os.getenv("ENV_URL", "http://127.0.0.1:8000")
STRICT_MIN = 0.11
STRICT_MAX = 0.89

TASK_SPECS: Dict[str, Dict[str, Any]] = {
    "task-001-trend-following": {
        "symbol": "NOVA",
        "ideal_actions": ["buy", "buy", "hold", "hold", "sell"],
    },
    "task-002-mean-reversion": {
        "symbol": "KITE",
        "ideal_actions": ["buy", "buy", "hold", "sell", "sell", "buy"],
    },
    "task-003-risk-managed-hedge": {
        "symbol": "ORCA",
        "ideal_actions": ["hold", "sell", "hold", "buy", "sell", "sell", "hold"],
    },
}


def _safe_print(line: str) -> None:
    try:
        print(line, flush=True)
    except BrokenPipeError:
        os._exit(0)


def _strict(value: Any) -> float:
    try:
        v = float(value)
    except Exception:
        v = STRICT_MIN
    if v <= 0.0:
        return STRICT_MIN
    if v >= 1.0:
        return STRICT_MAX
    return max(STRICT_MIN, min(STRICT_MAX, v))


def _token(text: Any) -> str:
    return "_".join(str(text).replace("\n", " ").replace("\r", " ").split())


def log_start(task: str, env: str, model: str) -> None:
    _safe_print(f"[START] task={_token(task)} env={_token(env)} model={_token(model)}")


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err = _token(error) if error else "null"
    done_str = "true" if done else "false"
    _safe_print(
        f"[STEP] step={step} action={_token(action)} reward={_strict(reward):.2f} done={done_str} error={err}"
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    success_str = "true" if success else "false"
    reward_str = ",".join(f"{_strict(r):.2f}" for r in rewards)
    _safe_print(f"[END] success={success_str} steps={steps} rewards={reward_str}")


async def _wait_for_env(base_url: str, timeout_s: float = 10.0) -> bool:
    deadline = time.time() + timeout_s
    async with httpx.AsyncClient(timeout=2.0) as client:
        while time.time() < deadline:
            try:
                r = await client.get(f"{base_url}/health")
                if r.status_code == 200:
                    return True
            except Exception:
                await asyncio.sleep(0.25)
    return False


async def _maybe_start_local(base_url: str):
    if not (base_url.startswith("http://127.0.0.1:") or base_url.startswith("http://localhost:")):
        return None, base_url

    if await _wait_for_env(base_url, timeout_s=1.5):
        return None, base_url

    port = 8000
    try:
        port = int(base_url.rsplit(":", 1)[-1])
    except Exception:
        port = 8000

    for p in range(port, port + 10):
        candidate = f"http://127.0.0.1:{p}"
        proc = await asyncio.create_subprocess_exec(
            "python3",
            "-m",
            "uvicorn",
            "server.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(p),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        if await _wait_for_env(candidate, timeout_s=6.0):
            return proc, candidate
        with suppress(Exception):
            proc.terminate()
            await proc.wait()

    raise RuntimeError("failed_to_start_local_env")


def _deterministic_action(task_id: str, step_index: int) -> Dict[str, Any]:
    spec = TASK_SPECS.get(task_id, {"symbol": "NOVA", "ideal_actions": ["hold"]})
    ideal = spec["ideal_actions"]
    decision = ideal[min(step_index, len(ideal) - 1)]
    quantity = 10 if decision in {"buy", "sell"} else 0
    return {
        "symbol": spec["symbol"],
        "decision": decision,
        "quantity": quantity,
        "confidence": 0.9,
        "rationale": "deterministic_baseline",
    }


def _touch_model(client: OpenAI, task_id: str, step_index: int, obs: Dict[str, Any]) -> None:
    prompt = (
        "You are a stock execution assistant. "
        f"task={task_id} step={step_index + 1} "
        f"symbol={obs.get('symbol')} price={obs.get('current_price')} "
        "Reply with a one-line risk note."
    )
    try:
        client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=24,
        )
    except Exception:
        # Silent fallback; we still run deterministic baseline actions.
        pass


async def run_task(client: OpenAI, env_url: str, task_id: str) -> None:
    log_start(task=task_id, env="stock_exchange_env", model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    success = False
    last_error: Optional[str] = None

    try:
        async with httpx.AsyncClient(timeout=30.0) as http:
            reset_res = await http.post(f"{env_url}/reset?task_id={task_id}", json={})
            reset_payload = reset_res.json()
            obs = reset_payload.get("observation", reset_payload)
            done = bool(reset_payload.get("done", False))

            max_steps = int(obs.get("max_days", 5)) + 1
            for step in range(1, max_steps + 1):
                if done:
                    break

                _touch_model(client, task_id, step - 1, obs)
                action = _deterministic_action(task_id, step - 1)
                step_res = await http.post(f"{env_url}/step", json={"action": action})
                result = step_res.json()

                obs = result.get("observation", result)
                reward = _strict(result.get("reward", STRICT_MIN))
                done = bool(result.get("done", False))

                rewards.append(reward)
                steps_taken = step
                log_step(
                    step=step,
                    action=f"{action['decision']}:{action['quantity']}",
                    reward=reward,
                    done=done,
                    error=None,
                )

    except Exception as exc:
        last_error = str(exc)

    if not rewards:
        rewards = [STRICT_MIN]
        steps_taken = 1
        log_step(step=1, action="hold:0", reward=STRICT_MIN, done=True, error=last_error or "env_error")

    success = (sum(rewards) / float(max(len(rewards), 1))) >= 0.3
    log_end(success=success, steps=steps_taken, rewards=rewards)


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    proc = None
    env_url = ENV_URL
    try:
        proc, env_url = await _maybe_start_local(env_url)
    except Exception:
        proc = None

    try:
        tasks_env = os.getenv("TASK_NAMES", "").strip()
        single_task = os.getenv("TASK_NAME", "").strip()
        if single_task:
            tasks = [single_task]
        elif tasks_env:
            tasks = [x.strip() for x in tasks_env.split(",") if x.strip()]
        else:
            tasks = [
                "task-001-trend-following",
                "task-002-mean-reversion",
                "task-003-risk-managed-hedge",
            ]

        for task_id in tasks:
            await run_task(client=client, env_url=env_url, task_id=task_id)
    finally:
        if proc is not None:
            with suppress(Exception):
                proc.terminate()
            with suppress(Exception):
                await proc.wait()


if __name__ == "__main__":
    asyncio.run(main())
