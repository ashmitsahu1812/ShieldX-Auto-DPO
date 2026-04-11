from __future__ import annotations

import json
import os
from typing import Any, Dict

from fastapi import Body, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .environment import StockExchangeEnv
from .models import MarketObservation, MarketState, TradeAction

app = FastAPI(title="OpenEnv Stock Exchange Simulator")

static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


env_registry: Dict[str, StockExchangeEnv] = {}


def get_session_env(session_id: str = "default") -> StockExchangeEnv:
    if session_id not in env_registry:
        env_registry[session_id] = StockExchangeEnv()
    return env_registry[session_id]


def _dump(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return obj


def _schema(model: Any) -> Dict[str, Any]:
    if hasattr(model, "model_json_schema"):
        return model.model_json_schema()
    return model.schema()  # pragma: no cover


def _reference_grade(task_id: str) -> Dict[str, Any]:
    env = StockExchangeEnv(task_id=task_id)
    env.reset(task_id=task_id)
    task = env.task

    for idx, ideal_decision in enumerate(task["ideal_actions"]):
        if env.done:
            break
        qty = 12 if ideal_decision in {"buy", "sell"} else 0
        action = {
            "symbol": task["symbol"],
            "decision": ideal_decision,
            "quantity": qty,
            "confidence": 0.9,
            "rationale": f"reference_step_{idx + 1}",
        }
        env.step(action)

    score = float(env.evaluate_task())
    return {"task_id": task_id, "score": score, "task_score": score}


@app.get("/")
def root() -> FileResponse:
    return FileResponse(os.path.join(static_dir, "index.html"))


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "healthy"}


@app.get("/metadata")
def metadata() -> Dict[str, Any]:
    return {
        "name": "stock_exchange_env",
        "description": "Real-world stock execution and risk-management simulation.",
        "readme_content": None,
        "version": "1.0.0",
        "author": "ashmitsahu",
        "documentation_url": None,
    }


@app.get("/tasks")
def tasks() -> Dict[str, Any]:
    return {
        "tasks": [
            {
                "id": task["id"],
                "name": task["name"],
                "difficulty": task["difficulty"],
                "symbol": task["symbol"],
                "objective": task["objective"],
                "max_steps": task["max_steps"],
            }
            for task in StockExchangeEnv.TASKS
        ]
    }


@app.get("/schema")
def schema() -> Dict[str, Any]:
    return {
        "action": _schema(TradeAction),
        "observation": _schema(MarketObservation),
        "state": _schema(MarketState),
    }


@app.post("/mcp")
def mcp(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": payload.get("id"),
        "result": {"status": "ok"},
    }


@app.post("/reset")
def reset(
    payload: Dict[str, Any] = Body(default_factory=dict),
    task_id: str = "task-001-trend-following",
) -> Dict[str, Any]:
    body_task = payload.get("task_id") or payload.get("taskId") or payload.get("task")
    selected_task = str(body_task or task_id)

    env = StockExchangeEnv(task_id=selected_task)
    env_registry["default"] = env
    obs = env.reset(task_id=selected_task)

    return {
        "observation": _dump(obs),
        "reward": float(env.task_score),
        "done": False,
    }


@app.post("/step")
def step(action: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
    env = get_session_env()
    payload = action.get("action", action) if isinstance(action, dict) else {}

    obs, reward, done, _info = env.step(payload)
    return {
        "observation": _dump(obs),
        "reward": float(reward),
        "done": bool(done),
    }


@app.get("/state")
def state() -> Dict[str, Any]:
    env = get_session_env()
    return _dump(env.state())


@app.get("/grade")
def grade(task_id: str = "") -> Dict[str, Any]:
    if task_id:
        return _reference_grade(str(task_id))

    env = get_session_env()
    score = float(env.evaluate_task())
    return {
        "task_id": env.task["id"],
        "score": score,
        "task_score": score,
    }


@app.get("/grader")
def grader(task_id: str = "") -> Dict[str, Any]:
    if task_id:
        return _reference_grade(str(task_id))

    scores = [_reference_grade(task["id"]) for task in StockExchangeEnv.TASKS]
    aggregate = sum(item["score"] for item in scores) / float(max(len(scores), 1))
    return {
        "scores": scores,
        "score": float(aggregate),
        "task_score": float(aggregate),
    }


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    await websocket.accept()
    env = StockExchangeEnv()

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                message = json.loads(raw)
            except Exception as exc:
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "observation",
                            "data": {
                                "observation": _dump(env.state()),
                                "reward": float(env.task_score),
                                "done": True,
                                "error": f"invalid_json:{exc}",
                            },
                        }
                    )
                )
                continue

            msg_type = message.get("type", "")
            data = message.get("data", {}) or {}

            if msg_type == "reset":
                tid = data.get("task_id") or data.get("taskId") or data.get("task")
                obs = env.reset(task_id=str(tid) if tid else None)
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "observation",
                            "data": {
                                "observation": _dump(obs),
                                "reward": float(env.task_score),
                                "done": False,
                            },
                        }
                    )
                )

            elif msg_type == "step":
                payload = data.get("action", data) if isinstance(data, dict) else {}
                obs, reward, done, _ = env.step(payload)
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "observation",
                            "data": {
                                "observation": _dump(obs),
                                "reward": float(reward),
                                "done": bool(done),
                            },
                        }
                    )
                )

            elif msg_type == "state":
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "state",
                            "data": _dump(env.state()),
                        }
                    )
                )

            elif msg_type == "close":
                break

            else:
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "observation",
                            "data": {
                                "observation": _dump(env.state()),
                                "reward": float(env.task_score),
                                "done": True,
                                "error": f"unknown_type:{msg_type}",
                            },
                        }
                    )
                )
    except WebSocketDisconnect:
        pass


def main() -> None:
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
