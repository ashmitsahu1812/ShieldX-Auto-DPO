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
  - trading
  - risk-management
---

# OpenEnv Stock Exchange Environment

A real-world **equity execution and risk management** simulation for training and evaluating trading agents.  
An agent acts as a junior trader placing buy/sell/hold orders while balancing return, drawdown, position concentration, and portfolio rebalancing objectives.

## Why This Is Real-World

This environment models work that humans actually do every day:

- **Tactical order execution** — deciding when to enter and exit positions
- **Risk budgeting** — respecting drawdown limits and concentration caps
- **Portfolio rebalancing** — systematically reducing over-concentration
- **Decision logging with rationale** — every action requires a justification for auditability
- **Confidence-calibrated decisions** — high confidence on wrong calls is penalized; high confidence on correct calls is rewarded

## OpenEnv API

All endpoints follow the OpenEnv spec:

| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | Start or restart an episode. Accepts `task_id`. |
| `/step` | POST | Execute a trade action. Returns observation, reward, done. |
| `/state` | GET | Current internal state snapshot. |
| `/health` | GET | Liveness check. |
| `/schema` | GET | JSON schemas for action, observation, state. |
| `/metadata` | GET | Environment metadata. |
| `/tasks` | GET | List all tasks with metadata. |
| `/tasks/{task_id}` | GET | Full detail for a specific task. |
| `/grade` | GET | Score the current session. |
| `/grader` | GET | Run reference grader on all tasks. |
| `/ws` | WebSocket | Real-time interaction. |

OpenEnv manifest: [openenv.yaml](./openenv.yaml)

## Typed Models

All models are Pydantic v2 with full field documentation.

| Model | Description |
|---|---|
| [`TradeAction`](./server/models.py) | Agent action: decision, quantity, confidence, rationale |
| [`MarketObservation`](./server/models.py) | Full market state returned to agent each step |
| [`MarketReward`](./server/models.py) | Structured reward breakdown with explanation |
| [`MarketState`](./server/models.py) | Internal state snapshot via `/state` |

## Action Space

```json
{
  "symbol": "NOVA",
  "decision": "buy | sell | hold",
  "quantity": 10,
  "confidence": 0.85,
  "rationale": "Price momentum is positive and drawdown is within limits."
}
```

- `decision`: one of `buy`, `sell`, `hold`
- `quantity`: integer shares (0 for hold)
- `confidence`: float 0.0–1.0 — **affects reward scaling** (calibrated confidence is rewarded)
- `rationale`: free-text justification logged for auditability

## Observation Space

Each step returns a rich observation including:

| Field | Type | Description |
|---|---|---|
| `current_price` | float | Current market price |
| `price_window` | list[float] | Last 5 prices |
| `momentum_1d` / `momentum_3d` | float | Short-term price momentum |
| `volatility` | float | Current step volatility |
| `market_regime` | string | `trending_up`, `mean_reverting`, `volatile`, `gradual_uptrend` |
| `cash` | float | Available cash |
| `position` | int | Current share count |
| `portfolio_value` | float | Total portfolio value |
| `drawdown` | float | Current drawdown from peak |
| `max_drawdown_limit` | float | Task's max allowed drawdown |
| `max_position_ratio` | float | Task's max position/portfolio ratio |
| `target_position_ratio` | float? | Target ratio for rebalancing tasks |
| `min_cash_ratio` | float? | Minimum cash ratio for rebalancing tasks |
| `last_decision` | string | Agent's previous action |

## Tasks (Easy → Medium → Hard)

### task-001-trend-following (Easy)
- **Symbol**: NOVA | **Steps**: 5 | **Regime**: trending_up
- **Objective**: Capture an intraday uptrend. Buy early, hold through momentum, exit before reversal.
- **Challenge**: Timing entries and exits on a clean uptrend. Penalizes overtrading.

### task-002-mean-reversion (Medium)
- **Symbol**: KITE | **Steps**: 6 | **Regime**: mean_reverting
- **Objective**: Buy weakness during a pullback, reduce exposure into recovery spikes.
- **Challenge**: Patience during the dip, position sizing under drawdown pressure, re-entry timing.

### task-003-risk-managed-hedge (Hard)
- **Symbol**: ORCA | **Steps**: 7 | **Regime**: volatile
- **Objective**: Navigate a volatile event-driven selloff. De-risk early, exploit the bounce, exit before the second leg down.
- **Challenge**: Tight 8% drawdown cap. Passive holding is severely penalized. Requires precise timing.

### task-004-portfolio-rebalance (Hard)
- **Symbol**: APEX | **Steps**: 9 | **Regime**: gradual_uptrend
- **Objective**: Rebalance a concentrated position (40 shares, only $5k cash) toward a 40% equity target while maintaining ≥30% cash buffer.
- **Challenge**: Systematic selling into strength without panic-selling. Violating the cash floor is penalized. Tests disciplined rebalancing over a long horizon.

## Reward Design

Step reward uses three weighted components:

```
reward = strict(conf_mult × (0.45 × alignment + 0.35 × return + 0.20 × risk))
```

- **Action alignment** (45%): How well the decision matches the task's ideal action sequence
- **Return component** (35%): Portfolio return this step, normalized around 0
- **Risk component** (20%): Penalizes drawdown violations and position over-concentration
- **Confidence multiplier**: Scales reward up for high-confidence correct calls, down for high-confidence wrong calls

Task score (episode-level) uses:
- Average action alignment (50%)
- Final return vs target range (25%)
- Max drawdown adherence (20%)
- Episode completion (5%)
- Rebalancing accuracy bonus for task-004 (15%, replaces some alignment weight)

All rewards and episode **task scores** are clamped to the strict open interval **(0.01, 0.99)** (compatible with validators that require scores strictly inside `(0, 1)`).

## Baseline Inference Script

[`inference.py`](./inference.py) — the LLM agent drives all decisions via structured JSON output.

The LLM receives a rich prompt with:
- Task objective and market regime
- Full price window and momentum signals
- Portfolio state, drawdown, and risk limits
- Recent action history

It responds with `{"decision": ..., "quantity": ..., "confidence": ..., "rationale": ...}`.

### Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `HF_TOKEN` | Yes* | — | Hugging Face API token |
| `OPENAI_API_KEY` | Yes* | — | OpenAI API key (fallback if HF_TOKEN not set) |
| `API_BASE_URL` | No | `https://api.openai.com/v1` | LLM API endpoint |
| `MODEL_NAME` | No | `gpt-4.1-mini` | Model identifier |
| `ENV_URL` | No | `http://127.0.0.1:8000` | Environment server URL |

*One of `HF_TOKEN` or `OPENAI_API_KEY` is required.

### Stdout Format

```
[START] task=task-001-trend-following env=stock_exchange_env model=Qwen/Qwen2.5-Coder-32B-Instruct
[STEP] step=1 action=buy:10@conf=0.90 reward=0.82 done=false error=null
[STEP] step=2 action=buy:10@conf=0.85 reward=0.79 done=false error=null
...
[END] success=true steps=5 score=0.7993 rewards=0.82,0.79,0.71,0.68,0.83
```

## Reproducible Baseline Scores

Reference grader scores (deterministic ideal-action policy):

| Task | Score |
|---|---|
| task-001-trend-following | ~0.80 |
| task-002-mean-reversion | ~0.74 |
| task-003-risk-managed-hedge | ~0.84 |
| task-004-portfolio-rebalance | ~0.72 |

Verify with:
```bash
curl -s http://127.0.0.1:8000/grader
```

## RL Component

A lightweight RL pipeline is included in [`rl/`](./rl/):

| File | Description |
|---|---|
| [`shieldx_gym_env.py`](./rl/shieldx_gym_env.py) | Gymnasium wrapper (discrete 9-action space) |
| [`train_qlearning.py`](./rl/train_qlearning.py) | Tabular Q-learning trainer |
| [`evaluate_qlearning.py`](./rl/evaluate_qlearning.py) | Q-table policy evaluator |
| [`qlearning_utils.py`](./rl/qlearning_utils.py) | Discretization and Q-table save/load |
| [`train_sb3.py`](./rl/train_sb3.py) | Stable-Baselines3 PPO/DQN trainer |
| [`evaluate_sb3.py`](./rl/evaluate_sb3.py) | SB3 model evaluator |

## Local Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and configure environment variables
cp .env.example .env
# Edit .env and set HF_TOKEN or OPENAI_API_KEY

# Run the API server
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Run inference (in a separate terminal)
python3 inference.py

# Run smoke tests
python3 test_env.py
python3 verify_rewards.py
python3 verify_rewards_diverse.py

# Validate OpenEnv spec
openenv validate
```

## Docker

```bash
# Build
docker build -t stock-openenv .

# Run
docker run -p 8000:8000 -e HF_TOKEN=your_token stock-openenv
```

## Hugging Face Space

Configured for Docker Spaces with `app_port: 8000` and `openenv` tags.  
The environment auto-starts via `uvicorn` and responds to all OpenEnv endpoints immediately.
