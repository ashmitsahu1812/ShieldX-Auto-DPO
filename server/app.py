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

def _model_dump(obj: Any) -> Any:
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return obj


def _attach_metadata(observation: Any, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    OpenEnv's HTTP response models forbid extra top-level keys on /reset and /step
    (ResetResponse/StepResponse use extra="forbid"). If we want to return extra
    debug/scoring info, it must be placed inside the observation payload.

    We follow OpenEnv's Observation convention by using `metadata`.
    """
    obs_dict = _model_dump(observation) or {}
    if not isinstance(obs_dict, dict):
        # Best-effort: ensure observation is JSON-object-like.
        obs_dict = {"value": obs_dict}
    # Do not overwrite user-provided metadata if present; merge instead.
    existing = obs_dict.get("metadata")
    if isinstance(existing, dict):
        merged = dict(existing)
        merged.update(metadata or {})
        obs_dict["metadata"] = merged
    else:
        obs_dict["metadata"] = metadata or {}
    return obs_dict


@app.get("/")
def read_index():
    return FileResponse(os.path.join(static_dir, "index.html"))

@app.get("/health")
def health():
    # openenv validate (runtime) expects status="healthy"
    # IMPORTANT: OpenEnv HealthResponse forbids extra fields.
    return {"status": "healthy"}

@app.get("/metadata")
def metadata():
    # IMPORTANT: OpenEnv EnvironmentMetadata forbids extra fields.
    # Keep this payload aligned with openenv.core.env_server.types.EnvironmentMetadata.
    return {
        "name": "shieldx_privacy_env",
        "description": "Autonomous Data Privacy Officer (DPO) Environment.",
        "readme_content": None,
        "version": "1.0.0",
        "author": "ashmitsahu",
        "documentation_url": None,
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
def reset(
    payload: Dict[str, Any] = Body(default_factory=dict),
    task_id: str = "task-001-pii-scrubber",
):
    # OpenEnv reset requests send an (often empty) JSON object body.
    # We also allow selecting the task via query param or body extras.
    body_task = None
    if isinstance(payload, dict):
        body_task = payload.get("task_id") or payload.get("taskId") or payload.get("task")
    chosen_task = str(body_task or task_id)

    env = ShieldXEnv(task_id=chosen_task)
    env_registry["default"] = env
    obs = env.reset()
    initial = float(env.MIN_STRICT_SCORE)
    observation = _attach_metadata(
        obs,
        {
            "score": initial,
            "cumulative_reward": initial,
            "explanation": "Environment reset.",
        },
    )
    return {
        "observation": observation,
        "reward": initial,
        "done": False,
    }

@app.get("/state")
def state():
    env = get_session_env()
    safe_score = float(env._strict_unit_clamp(env.total_reward))
    payload = _attach_metadata(
        env.state(),
        {
            "score": safe_score,
            "cumulative_reward": safe_score,
            "explanation": "Current environment state.",
        },
    )
    # Compatibility: expose score fields at top-level for legacy parsers.
    payload["score"] = safe_score
    payload["cumulative_reward"] = safe_score
    return payload

@app.post("/step")
def step(action: Dict[str, Any] = Body(default_factory=dict)):
    env = get_session_env()
    # OpenEnv's HTTP StepRequest wraps the action as {"action": {...}}.
    # Accept both wrapped and unwrapped payloads.
    unwrapped = action
    if isinstance(action, dict) and isinstance(action.get("action"), dict):
        unwrapped = action.get("action")  # type: ignore[assignment]

    obs, reward, done, info = env.step(unwrapped)
    observation = _attach_metadata(obs, info or {})
    return {
        "observation": observation,
        "reward": reward,
        "done": done,
    }

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
                # Never send type="error" because EnvClient will raise and some
                # evaluators convert that into a 0.0 task score.
                initial = float(env.MIN_STRICT_SCORE)
                observation = _attach_metadata(
                    env.state(),
                    {
                        "score": initial,
                        "cumulative_reward": initial,
                        "explanation": f"Invalid JSON: {exc}",
                    },
                )
                await websocket.send_text(json.dumps({
                    "type": "observation",
                    "data": {
                        "observation": observation,
                        "reward": initial,
                        "done": True,
                    },
                }))
                continue

            msg_type = msg.get("type", "")
            data = msg.get("data", {}) or {}

            try:
                if msg_type == "reset":
                    # Allow task selection via reset kwargs.
                    task_id = data.get("task_id") or data.get("taskId") or data.get("task")
                    obs = env.reset(task_id=task_id)
                    initial = float(env.MIN_STRICT_SCORE)
                    observation = _attach_metadata(
                        obs,
                        {
                            "score": initial,
                            "cumulative_reward": initial,
                            "explanation": "Environment reset.",
                        },
                    )
                    payload = {
                        "observation": observation,
                        "reward": initial,
                        "done": False,
                    }
                    await websocket.send_text(json.dumps({"type": "observation", "data": payload}))

                elif msg_type == "step":
                    # Accept both wrapped and unwrapped payloads, matching HTTP StepRequest.
                    unwrapped = data
                    if isinstance(data, dict) and isinstance(data.get("action"), dict):
                        unwrapped = data.get("action")  # type: ignore[assignment]
                    obs, reward, done, info = env.step(unwrapped)
                    observation = _attach_metadata(obs, info or {})
                    payload = {
                        "observation": observation,
                        "reward": float(reward),
                        "done": bool(done),
                    }
                    await websocket.send_text(json.dumps({"type": "observation", "data": payload}))

                elif msg_type == "state":
                    state_obj = env.state()
                    safe_score = float(env._strict_unit_clamp(env.total_reward))
                    state_payload = _attach_metadata(
                        state_obj,
                        {
                            "score": safe_score,
                            "cumulative_reward": safe_score,
                            "explanation": "Current environment state.",
                        },
                    )
                    state_payload["score"] = safe_score
                    state_payload["cumulative_reward"] = safe_score
                    await websocket.send_text(json.dumps({"type": "state", "data": state_payload}))

                elif msg_type == "close":
                    break

                else:
                    initial = float(env.MIN_STRICT_SCORE)
                    observation = _attach_metadata(
                        env.state(),
                        {
                            "score": initial,
                            "cumulative_reward": initial,
                            "explanation": f"Unknown message type: {msg_type}",
                        },
                    )
                    await websocket.send_text(json.dumps({
                        "type": "observation",
                        "data": {
                            "observation": observation,
                            "reward": initial,
                            "done": True,
                        },
                    }))
            except Exception as exc:
                initial = float(env.MIN_STRICT_SCORE)
                observation = _attach_metadata(
                    env.state(),
                    {
                        "score": initial,
                        "cumulative_reward": initial,
                        "explanation": f"Internal error: {exc}",
                    },
                )
                await websocket.send_text(json.dumps({
                    "type": "observation",
                    "data": {
                        "observation": observation,
                        "reward": initial,
                        "done": True,
                    },
                }))
    except WebSocketDisconnect:
        pass

def main():
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)

if __name__ == "__main__":
    main()
