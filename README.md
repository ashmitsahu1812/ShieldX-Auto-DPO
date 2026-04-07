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
![Bulletproof API](https://img.shields.io/badge/API-Bulletproof-red?style=for-the-badge)

**ScalarX Meta** is a production-grade, OpenEnv-compliant simulation environment designed to benchmark and train AI agents in the high-stakes cognitive task of **Pull Request (PR) review**. It challenges agents to identify subtle bugs, architectural flaws, and security vulnerabilities across multi-file diffs.

---

## 🛡️ Bulletproof Tiered Intelligence
Unique to this submission is a **3-Tiered Intelligence Strategy** in the `inference.py` script, ensuring a high score even if API limits are hit:
1.  **Tier 1: Hugging Face (Quality)**: Uses the high-precision `Qwen2.5-Coder-32B` model for elite bug detection.
2.  **Tier 2: Pollinations AI (Unlimited)**: Automatic zero-delay fallback to a free, limitless API if HF credits are depleted.
3.  **Tier 3: Heuristic Guard (Safety Net)**: A local, rule-based reviewer that takes over if all internet/API connections fail, ensuring the agent always successfully finishes the task.

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
ScalarX Meta includes a **Self-Learning Flywheel** (accessible via the Web UI). It tracks agent performance patterns and stores them in `flywheel_store.json`. This data is used to:
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
*   `API_BASE_URL`: The LLM inference endpoint.
*   `MODEL_NAME`: The model identifier.
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
bash validate.sh http://localhost:7860 .
```

---

## 📝 Mandatory STDOUT Logging
The `inference.py` script emits structured logs strictly following the OpenEnv requirement:

```text
[START] task=<task_name> env=code_review_env model=<model_name>
[STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
```

---

## 🏆 Baseline Notes
The baseline implemented in `inference.py` uses an **Elite Reviewer Strategy**:
*   **State-Awareness**: It tracks reported bugs to avoid loop-commenting and duplicate penalties.
*   **Scoring Optimization**: It follows a "Comment-First" protocol, ensuring every bug is logged in the `Observation` history before submitting a final decision, maximizing precision rewards.
*   **Fault-Tolerance**: Pass the benchmarks even if the API is down using the Tier 3 Heuristic Guard.

---
**Built for the OpenEnv Hackathon** • **Engineered for Precision** • **Deployable on HF Spaces**

