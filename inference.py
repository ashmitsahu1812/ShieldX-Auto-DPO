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
# Support both HF_TOKEN and OPENAI_API_KEY; HF_TOKEN takes precedence
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN or OPENAI_API_KEY environment variable is required")

ENV_URL = os.getenv("ENV_URL", "http://127.0.0.1:8000")
STRICT_MIN = 0.05
STRICT_MAX = 0.95

SYSTEM_PROMPT = """You are an expert quantitative trader. Given market observations, decide the best action.

You MUST respond with valid JSON only — no prose, no markdown fences. Schema:
{
  "decision": "buy" | "sell" | "hold",
  "quantity": <integer 0-50>,
  "confidence": <float 0.0-1.0>,
  "rationale": "<one sentence>"
}

Rules:
- quantity must be 0 when decision is "hold"
- confidence reflects how certain you are (0.9+ = very confident)
- rationale must reference the price trend or risk signal
"""


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


def _build_user_prompt(obs: Dict[str, Any], step: int, history: List[str]) -> str:
    """Build a rich prompt from the current market observation."""
    price_window = obs.get("price_window", [obs.get("current_price", 0)])
    price_str = " → ".join(f"{p:.2f}" for p in price_window)
    history_str = "\n".join(history[-3:]) if history else "No prior steps."

    return (
        f"=== Step {step} | Task: {obs.get('task_id', '?')} ===\n"
        f"Objective: {obs.get('objective', '')}\n"
        f"Symbol: {obs.get('symbol', '?')} | Regime: {obs.get('market_regime', 'unknown')} | "
        f"Volatility: {obs.get('volatility', 0.0):.4f}\n"
        f"Price path: {price_str}\n"
        f"Current: ${obs.get('current_price', 0):.2f} | Next hint: ${obs.get('next_price', 0):.2f}\n"
        f"Momentum 1d: {obs.get('momentum_1d', 0):.4f} | 3d: {obs.get('momentum_3d', 0):.4f}\n"
        f"Portfolio: ${obs.get('portfolio_value', 0):.2f} | Cash: ${obs.get('cash', 0):.2f} | "
        f"Position: {obs.get('position', 0)} shares\n"
        f"Drawdown: {obs.get('drawdown', 0):.4f} | Max allowed: {obs.get('max_drawdown_limit', 0.1):.2f}\n"
        f"Progress: day {obs.get('day_index', 0)}/{obs.get('max_days', 0)}\n"
        f"Recent actions:\n{history_str}\n\n"
        f"Respond with JSON only."
    )


def _llm_action(
    client: OpenAI,
    obs: Dict[str, Any],
    step: int,
    history: List[str],
) -> Dict[str, Any]:
    """Ask the LLM for a trading decision. Falls back to hold on any failure."""
    symbol = obs.get("symbol", "NOVA")
    fallback = {"symbol": symbol, "decision": "hold", "quantity": 0, "confidence": 0.5, "rationale": "llm_fallback"}

    try:
        prompt = _build_user_prompt(obs, step, history)
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=120,
        )
        raw = response.choices[0].message.content or ""
        # Strip markdown fences if model wraps anyway
        raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        parsed = json.loads(raw)

        decision = str(parsed.get("decision", "hold")).lower()
        if decision not in {"buy", "sell", "hold"}:
            decision = "hold"
        quantity = int(parsed.get("quantity", 0))
        if decision == "hold":
            quantity = 0
        confidence = float(parsed.get("confidence", 0.5))
        rationale = str(parsed.get("rationale", "llm_decision"))

        return {
            "symbol": symbol,
            "decision": decision,
            "quantity": max(0, min(50, quantity)),
            "confidence": max(0.0, min(1.0, confidence)),
            "rationale": rationale,
        }
    except Exception as exc:
        _safe_print(f"[DEBUG] LLM parse error: {exc}")
        return fallback


async def run_task(client: OpenAI, env_url: str, task_id: str) -> None:
    log_start(task=task_id, env="stock_exchange_env", model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    last_error: Optional[str] = None
    action_history: List[str] = []

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

                # LLM drives the decision
                action = _llm_action(client, obs, step, action_history)

                step_res = await http.post(f"{env_url}/step", json={"action": action})
                result = step_res.json()

                obs = result.get("observation", result)
                reward = _strict(result.get("reward", STRICT_MIN))
                done = bool(result.get("done", False))

                rewards.append(reward)
                steps_taken = step
                action_summary = f"{action['decision']}:{action['quantity']}@conf={action['confidence']:.2f}"
                action_history.append(f"Step {step}: {action_summary} → reward {reward:.2f}")

                log_step(
                    step=step,
                    action=action_summary,
                    reward=reward,
                    done=done,
                    error=None,
                )

    except Exception as exc:
        last_error = str(exc)
        if not rewards:
            rewards = [STRICT_MIN]
            steps_taken = 1
            log_step(step=1, action="hold:0", reward=STRICT_MIN, done=True, error=last_error)

    finally:
        if not rewards:
            rewards = [STRICT_MIN]
            steps_taken = 1
            log_step(step=1, action="hold:0", reward=STRICT_MIN, done=True, error=last_error or "env_error")

        avg_score = sum(rewards) / float(max(len(rewards), 1))
        success = avg_score >= 0.3
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
                "task-004-portfolio-rebalance",
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
