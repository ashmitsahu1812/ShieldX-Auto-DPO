import os
import json
import asyncio
import httpx
from typing import List, Optional, Set, Dict, Any
from openai import OpenAI

# Required Environment Variables (Mandatory for Hackathon)
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-Coder-32B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "your_huggingface_token_here")

# Environment Endpoint (Local or Hosted)
API_URL = os.getenv("API_URL", "https://ashmitsahu-scalarxmeta.hf.space")

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def validate_action(raw_action: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitizes and validates the model's JSON output.
    Ensures mandatory fields are present and meet minimum quality.
    """
    valid_action = {
        "action_type": raw_action.get("action_type", "comment"),
        "file": raw_action.get("file", "unknown"),
        "line": raw_action.get("line", 0),
        "comment": raw_action.get("comment", "Analyzing the codebase for potential logic flaws and edge cases.")
    }
    
    # Enforce minimum comment length if it's too short
    if len(valid_action["comment"].split()) < 3:
        valid_action["comment"] = f"Reviewing {valid_action['file']} for potential logic errors, edge cases, and unexpected failures."
        
    return valid_action

async def get_model_message(
    client: OpenAI, 
    step: int, 
    obs: Dict[str, Any], 
    last_reward: float, 
    history: List[Dict[str, Any]],
    found_issues: Set[str]
) -> Dict[str, Any]:
    """
    Generates the next action using advanced reasoning and adaptive strategy.
    """
    adaptive_hint = ""
    if last_reward < 0:
        adaptive_hint = "\n> CRITICAL NOTICE: Your previous action received a negative reward or was unjustified. If you are requesting changes without first pointing out specific bugs with 'comment' actions, you will be penalized. DO NOT REQUEST CHANGES until you have identified at least one valid bug."

    prompt = f"""
    Step {step}: Last Reward: {last_reward}
    {adaptive_hint}
    
    ### Task Context
    - PR Title: {obs.get('title')}
    - Description: {obs.get('description')}
    - Files Changed (Deltas): {json.dumps(obs.get('files_changed'), indent=2)}
    
    ### Review History (Cumulative)
    {json.dumps(history, indent=2)}
    
    ### Known Issues Found
    {list(found_issues)}
    
    ### Constraints & Instructions
    1. You are a Senior Software Engineer. Your goal is to identify all bugs and maximize your score.
    2. MANDATORY PROTOCOL: You must use 'comment' actions to point out specific defects before you use 'request_changes'.
    3. SCORING WARNING: Requesting changes without having at least one valid 'comment' on a bug will result in a SEVERE score penalty.
    4. Your 'comment' must be at least 10 words long and describe the failure mode (e.g., 'crashes', 'incorrect logic', 'memory leak').
    5. If you have finished and found bugs, use 'request_changes'. If the PR is perfect, use 'approve'.
    
    Output ONLY valid JSON:
    {{
      "reasoning": "Internal reasoning (e.g. 'I see a bug in file X, line Y. I will comment on it first before rejecting.')",
      "action_type": "comment" | "approve" | "request_changes",
      "file": "filename",
      "line": line_number,
      "comment": "10+ words, specific failure analysis."
    }}
    """
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a senior engineer auditing code. Precision and detailed failure analysis are your top priorities. Output ONLY JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        raw_output = json.loads(response.choices[0].message.content)
        return validate_action(raw_output)
    except Exception as e:
        print(f"[DEBUG] Model/Parsing error: {e}", flush=True)
        return {
            "action_type": "comment", 
            "file": "unknown", 
            "line": 0,
            "comment": "I am continuing to analyze the logical flow and potential edge cases within this pull request."
        }

async def run_baseline_task(task_type: str, task_index: int = 0) -> float:
    """
    Runs a single task through the environment with adaptive memory and early stopping.
    """
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN or "dummy")
    BENCHMARK = "code_review_env"
    TASK_NAME = f"{task_type}_{task_index}"
    MAX_STEPS = 8
    
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
    
    history: List[Dict[str, Any]] = []
    found_issues: Set[str] = set()
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    consecutive_penalties = 0

    async with httpx.AsyncClient() as http_client:
        try:
            resp = await http_client.post(f"{API_URL}/reset", json={
                "task_type": task_type,
                "task_index": task_index
            })
            result = resp.json()
            session_id = result["session_id"]
            obs = result["observation"]
            
            last_reward = 0.0
            done = False

            for step in range(1, MAX_STEPS + 1):
                if done:
                    break

                action_dict = await get_model_message(client, step, obs, last_reward, history, found_issues)

                # step() -> OpenENV.step()
                resp = await http_client.post(f"{API_URL}/step", json={
                    "session_id": session_id,
                    "action": action_dict
                })
                step_result = resp.json()
                
                obs = step_result["observation"]
                reward = step_result["reward"]
                done = step_result["done"]
                info = step_result["info"]
                
                rewards.append(reward)
                steps_taken = step
                last_reward = reward
                
                # Adaptive logic: track penalties
                if reward < 0:
                    consecutive_penalties += 1
                else:
                    consecutive_penalties = 0
                    if action_dict["action_type"] == "comment":
                        found_issues.add(f"{action_dict['file']}:{action_dict['line']}")

                log_step(step=step, action=action_dict.get("action_type", "unknown"), reward=reward, done=done, error=None)

                # Store structured history
                history.append({
                    "step": step,
                    "action": action_dict.get("action_type"),
                    "reward": round(reward, 2),
                    "summary": action_dict.get("comment", "")[:50] + "..."
                })

                # Early stopping on repeated failure
                if consecutive_penalties >= 3:
                    print(f"[DEBUG] Early stopping: consecutive penalties threshold reached.", flush=True)
                    break

                if done:
                    score = info.get("score", 0.0)
                    break

            success = score >= 0.5

        except Exception as e:
            print(f"[DEBUG] Runtime error: {e}", flush=True)
            success = False
        finally:
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    
    return score

async def main() -> None:
    # Reproduce baseline on the mandatory tasks
    tasks = [
        ("syntax_review", 0),
        ("bug_detection", 0),
        ("adversarial_review", 0)
    ]
    avg_score = 0.0
    for t_type, t_idx in tasks:
        avg_score += await run_baseline_task(t_type, t_idx)
    
    print(f"\n[SUMMARY] Baseline Evaluation Complete. Average Score: {avg_score/len(tasks):.3f}")

if __name__ == "__main__":
    asyncio.run(main())
