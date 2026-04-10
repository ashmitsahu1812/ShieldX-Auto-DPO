from fastapi import FastAPI, HTTPException
from .models import PrivacyAction, PrivacyObservation, PrivacyReward
from .environment import ShieldXEnv
from .gradio_ui import create_shieldx_demo
import gradio as gr
from typing import Dict, Any

app = FastAPI(title="ShieldX: Autonomous DPO Environment")

env_registry: Dict[str, ShieldXEnv] = {}

def get_session_env(session_id: str = "default") -> ShieldXEnv:
    if session_id not in env_registry:
        env_registry[session_id] = ShieldXEnv()
    return env_registry[session_id]

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
def step(action: PrivacyAction):
    env = get_session_env()
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info
    }

# Mount Gradio for HF Spaces UI
app = gr.mount_gradio_app(app, create_shieldx_demo(), path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
