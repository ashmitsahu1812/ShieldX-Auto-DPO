---
title: OpenEnv Stock Exchange RL
emoji: 📈
colorFrom: green
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
tags:
  - openenv
  - finance
  - rl
---

# OpenEnv Stock Exchange Environment

This project is a real-world OpenEnv environment for **equity execution and risk management**.  
An agent acts as a junior trader that must place buy/sell/hold orders while balancing return, drawdown, and position concentration.

## Why this is real-world

This environment models work that humans actually do:
- Tactical order execution
- Risk budgeting (drawdown + concentration)
- Decision logging with rationale for auditability

## OpenEnv API

Implemented endpoints:
- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /health`
- `GET /schema`
- `GET /metadata`
- `GET /grade`
- `GET /grader`

OpenEnv manifest: [openenv.yaml](/Users/ashmitsahu/Desktop/scalarxmeta/openenv.yaml)

## Typed Models

- Action: [TradeAction](/Users/ashmitsahu/Desktop/scalarxmeta/server/models.py)
- Observation: [MarketObservation](/Users/ashmitsahu/Desktop/scalarxmeta/server/models.py)
- Reward: [MarketReward](/Users/ashmitsahu/Desktop/scalarxmeta/server/models.py)
- State: [MarketState](/Users/ashmitsahu/Desktop/scalarxmeta/server/models.py)

## Tasks (Easy → Medium → Hard)

1. `task-001-trend-following` (easy)
2. `task-002-mean-reversion` (medium)
3. `task-003-risk-managed-hedge` (hard)

Each task has:
- deterministic price path
- objective constraints
- ideal action sequence used by the agent grader

Task definitions: [tasks.py](/Users/ashmitsahu/Desktop/scalarxmeta/server/tasks.py)

## Grading + Reward Design

Step reward uses weighted components:
- action alignment vs task objective
- per-step portfolio return component
- risk component (position concentration + drawdown penalties)

Task score uses deterministic trajectory grading:
- average action alignment
- final return vs task target/min range
- max drawdown adherence
- completion signal

Grader implementation: [graders.py](/Users/ashmitsahu/Desktop/scalarxmeta/server/graders.py)

All step rewards and task scores are kept strictly in `(0, 1)` using safe normalization (`0.11` to `0.89`) to satisfy strict validators.

## Baseline Inference Script

Root script: [inference.py](/Users/ashmitsahu/Desktop/scalarxmeta/inference.py)

Requirements satisfied:
- filename is exactly `inference.py` in repo root
- uses OpenAI client for LLM calls
- reads env vars:
  - `API_BASE_URL` (default present)
  - `MODEL_NAME` (default present)
  - `HF_TOKEN` (required)
- emits strict stdout format:
  - `[START]`
  - `[STEP]`
  - `[END]`

## Reproducible Baseline Scores

Deterministic baseline policy scores (reference grader):
- `task-001-trend-following`: `0.7993`
- `task-002-mean-reversion`: `0.7366`
- `task-003-risk-managed-hedge`: `0.8354`

You can verify with:
```bash
curl -s http://127.0.0.1:8000/grader
```

## RL Component

A lightweight RL pipeline is included in `rl/`:
- env wrapper: [shieldx_gym_env.py](/Users/ashmitsahu/Desktop/scalarxmeta/rl/shieldx_gym_env.py)
- tabular Q-learning trainer: [train_qlearning.py](/Users/ashmitsahu/Desktop/scalarxmeta/rl/train_qlearning.py)
- evaluator: [evaluate_qlearning.py](/Users/ashmitsahu/Desktop/scalarxmeta/rl/evaluate_qlearning.py)
- discretization helpers: [qlearning_utils.py](/Users/ashmitsahu/Desktop/scalarxmeta/rl/qlearning_utils.py)

## Local Setup

Install dependencies:
```bash
python3 -m pip install -r requirements.txt
```

Run API server:
```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Run inference:
```bash
export HF_TOKEN="your_token"
python3 inference.py
```

Run OpenEnv validation:
```bash
openenv validate
```

## Docker

Build:
```bash
docker build -t stock-openenv .
```

Run:
```bash
docker run -p 8000:8000 stock-openenv
```

## Hugging Face Space

This repository is configured for Docker Spaces with `app_port: 8000` and OpenEnv tags.
