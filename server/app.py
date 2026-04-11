from fastapi import Body, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from .environment import ShieldXEnv
from typing import Dict, Any
import os
import json

app = FastAPI(title="ShieldX: Autonomous DPO Dashboard")

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")

env_registry: Dict[str, ShieldXEnv] = {}

def get_session_env(session_id: str = "default") -> ShieldXEnv:
    if session_id not in env_registry:
        env_registry[session_id] = ShieldXEnv()
    return env_registry[session_id]

@app.get("/")
def read_index():
    return FileResponse(os.path.join(static_dir, "index.html"))

@app.get("/health")
def health():
    # openenv validate (runtime) expects status="healthy"
    return {"status": "healthy", "env": "ShieldX"}

@app.get("/metadata")
def metadata():
    # Minimal OpenEnv metadata payload used by runtime validators.
    return {
        "name": "shieldx_privacy_env",
        "description": "Autonomous Data Privacy Officer (DPO) Environment.",
        "version": "1.0.0",
        "tasks": [
            {"id": t["id"], "name": t.get("name", ""), "difficulty": t.get("difficulty", "")}
            for t in ShieldXEnv.TASKS
        ],
    }

@app.get("/schema")
def schema():
    # Return JSON schemas for action/observation/state.
    # We keep this light and compatible with both Pydantic v1 and v2.
    from .models import PrivacyAction, PrivacyObservation

    def _schema(model):
        if hasattr(model, "model_json_schema"):
            return model.model_json_schema()
        return model.schema()

    obs_schema = _schema(PrivacyObservation)
    return {
        "action": _schema(PrivacyAction),
        "observation": obs_schema,
        "state": obs_schema,
    }

@app.post("/mcp")
def mcp(payload: Dict[str, Any]):
    # Minimal JSON-RPC compatible response for runtime validators.
    return {
        "jsonrpc": "2.0",
        "id": payload.get("id"),
        "result": {"status": "ok"},
    }

@app.post("/reset")
def reset(task_id: str = "task-001-pii-scrubber"):
    env = ShieldXEnv(task_id=task_id)
    env_registry["default"] = env
    obs = env.reset()
    # Some validators expect reset() to return the same result envelope as step().
    # Keep reward/score strictly inside (0, 1).
    initial = float(env.MIN_STRICT_SCORE)
    return {
        "observation": obs,
        "reward": initial,
        "done": False,
        "info": {
            "score": initial,
            "cumulative_reward": initial,
            "explanation": "Environment reset.",
        },
    }

@app.get("/state")
def state():
    env = get_session_env()
    return env.state()

@app.post("/step")
def step(action: Dict[str, Any] = Body(default_factory=dict)):
    env = get_session_env()
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info
    }

def _model_dump(obj: Any) -> Any:
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return obj

@app.websocket("/ws")
async def ws(websocket: WebSocket):
    """
    Minimal OpenEnv WebSocket protocol compatible with openenv.core.EnvClient:
    - Client sends {"type": "reset"|"step"|"state"|"close", "data": {...}}
    - Server responds {"type": "observation"|"state"|"error", "data": {...}}
    """
    await websocket.accept()
    env = ShieldXEnv()

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except Exception as exc:
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "error",
                            "data": {"message": f"Invalid JSON: {exc}", "code": "INVALID_JSON"},
                        }
                    )
                )
                continue

            msg_type = msg.get("type", "")
            data = msg.get("data", {}) or {}

            try:
                if msg_type == "reset":
                    # Allow task selection via reset kwargs.
                    task_id = data.get("task_id") or data.get("taskId") or data.get("task")
                    obs = env.reset(task_id=task_id)
                    initial = float(env.MIN_STRICT_SCORE)
                    payload = {
                        "observation": _model_dump(obs),
                        "reward": initial,
                        "done": False,
                        "info": {"score": initial, "cumulative_reward": initial, "explanation": "Environment reset."},
                    }
                    await websocket.send_text(json.dumps({"type": "observation", "data": payload}))

                elif msg_type == "step":
                    obs, reward, done, info = env.step(data)
                    payload = {
                        "observation": _model_dump(obs),
                        "reward": float(reward),
                        "done": bool(done),
                        "info": info,
                    }
                    await websocket.send_text(json.dumps({"type": "observation", "data": payload}))

                elif msg_type == "state":
                    state_obj = env.state()
                    await websocket.send_text(json.dumps({"type": "state", "data": _model_dump(state_obj)}))

                elif msg_type == "close":
                    break

                else:
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "error",
                                "data": {"message": f"Unknown message type: {msg_type}", "code": "INVALID_MESSAGE"},
                            }
                        )
                    )
            except Exception as exc:
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "error",
                            "data": {"message": str(exc), "code": "INTERNAL_ERROR"},
                        }
                    )
                )
    except WebSocketDisconnect:
        pass

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()
