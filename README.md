---
title: ScalarX Meta — AI Code Review Simulator
emoji: 🚀
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
---

# 🚀 ScalarX Meta: AI Code Review Simulator

![OpenEnv Compliant](https://img.shields.io/badge/OpenEnv-Compliant-green?style=for-the-badge)
![Docker Ready](https://img.shields.io/badge/Docker-Ready-blue?style=for-the-badge)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Spaces-yellow?style=for-the-badge)

**ScalarX Meta** is a production-grade, OpenEnv-compliant simulation environment designed to benchmark and train AI agents in the high-stakes cognitive task of **Pull Request (PR) review**. It challenges agents to identify subtle bugs, architectural flaws, and security vulnerabilities across multi-file diffs.

---

## 🕹️ Action Space
The agent interacts using structured JSON actions via the `step()` API:
*   `comment`: Provide inline feedback on a specific `file` and `line`.
*   `approve`: Finalize the review and accept the PR.
*   `request_changes`: Reject the PR due to identified defects.

## 📊 Observation Space
At each step, the agent receives:
*   **PR Metadata**: Title, Description, and Author.
*   **Diff Context**: Unified diffs for all changed files.
*   **History**: A transcript of previous comments and rewards.
*   **Budget**: Current step count vs. `max_steps`.

---

## 📉 Self-Learning Flywheel
Unique to ScalarX Meta is a **Self-Learning Flywheel** (accessible via the Web UI). It tracks agent performance patterns and stores them in `flywheel_store.json`. This data is used to:
1.  Identify common agent failure modes in bug detection.
2.  Provide human-in-the-loop (HITL) feedback to refine the rewards.
3.  Generate better synthetic task data over time.

---

## 📈 Evaluation Tasks
| Task ID | Difficulty | Focus |
| :--- | :--- | :--- |
| `syntax_review` | **Easy** | Syntax errors, naming conventions, and basic best practices. |
| `bug_detection` | **Medium** | Logical errors, incorrect loop boundaries, and memory leaks. |
| `full_review` | **Hard** | Complex multi-file regressions and architectural flaws. |
| `adversarial_review` | **Expert** | **Deceptive code** designed to bypass basic static analysis. |

---

## 🚀 Getting Started (Hackathon Setup)

### 1. Requirements
Ensure you have the following environment variables defined:
*   `API_BASE_URL`: The LLM inference endpoint (Default: `https://router.huggingface.co/v1`).
*   `MODEL_NAME`: The model identifier (Default: `Qwen/Qwen2.5-Coder-32B-Instruct`).
*   `HF_TOKEN`: Your Hugging Face "Read" token.

### 2. Local Setup
```bash
# Clone and install dependencies
git clone <your-repo-url>
cd scalarxmeta
pip install -r requirements.txt

# Create a .env file
echo "MODEL_NAME=Qwen/Qwen2.5-Coder-32B-Instruct" > .env
echo "API_BASE_URL=https://router.huggingface.co/v1" >> .env
echo "HF_TOKEN=your_token_here" >> .env

# Start the environment server
uvicorn server.app:app --port 7860

# In a new terminal, run the baseline inference
python3 inference.py
```

### 3. Docker Deployment
```bash
docker build -t openenv-scalarx .
docker run -p 7860:7860 -e HF_TOKEN="your_token" openenv-scalarx
```

### 4. Pre-Submission Validation
Run these checks from the repository root before submitting:

```bash
docker build .
openenv validate
bash validate.sh https://ashmit1812-scalarxmeta.hf.space .
```

Notes:
- Use your live Hugging Face Space URL, not the `huggingface.co/spaces/...` repo page URL.
- The validator expects `POST /reset` on the running Space to return HTTP `200`.

---

## 📝 Mandatory STDOUT Logging
The `inference.py` script emits structured logs strictly following the OpenEnv requirement (note usage of double-spaces after `[STEP]` for alignment parsing):

```text
[START] task=<task_name> env=code_review_env model=<model_name>
[STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
```

---

## 🏆 Baseline Notes
The submitted baseline is intentionally conservative and validator-safe:
- It uses the required OpenAI client interface and emits strict `[START]`, `[STEP]`, and `[END]` logs.
- It can run against the local API, the deployed Space, or an in-process fallback for local verification.
- Final task scores are kept strictly inside `(0, 1)` to satisfy validator requirements.
- The fallback reviewer is heuristic and observation-only; it does not use hidden labels or oracle metadata.

This repository is optimized first for OpenEnv compliance, reproducibility, and deployability, and second for raw benchmark score.

---
**Built for the OpenEnv Hackathon** • **Engineered for Precision** • **Deployable on HF Spaces**
