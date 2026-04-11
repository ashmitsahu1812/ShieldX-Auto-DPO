from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from .environment import ShieldXEnv
from typing import Dict, Any
import os

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
    return {"status": "ok", "env": "ShieldX"}

@app.post("/reset")
def reset(task_id: str = "task-001-pii-scrubber"):
    env = ShieldXEnv(task_id=task_id)
    env_registry["default"] = env
    return env.reset()

@app.get("/state")
def state():
    env = get_session_env()
    return env.state()

@app.post("/step")
def step(action: Dict[str, Any]):
    env = get_session_env()
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info
    }

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()
