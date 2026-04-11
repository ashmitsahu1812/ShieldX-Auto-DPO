from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .graders import action_alignment, grade_episode, strict_score, MIN_STRICT_SCORE
from .models import MarketObservation, MarketReward, MarketState, TradeAction
from .tasks import TASKS, get_task


class StockExchangeEnv:
    """Deterministic stock-exchange training environment for OpenEnv."""

    TASKS = TASKS

    def __init__(self, task_id: str = "task-001-trend-following"):
        self.task = get_task(task_id)
        self.task_id = self.task["id"]
        self._init_runtime()

    def _init_runtime(self) -> None:
        self.prices: List[float] = [float(x) for x in self.task["prices"]]
        self.max_steps: int = int(self.task.get("max_steps", len(self.prices) - 1))
        self.step_count = 0
        self.day_index = 0
        self.cash = float(self.task.get("initial_cash", 10000.0))
        self.initial_cash = self.cash
        self.position = 0
        self.avg_entry_price = 0.0
        self.done = False
        self.last_decision = "hold"
        self.cumulative_reward = MIN_STRICT_SCORE
        self.task_score = MIN_STRICT_SCORE
        self.history: List[Dict[str, Any]] = []
        self.peak_portfolio_value = self._portfolio_value(self.current_price)

    @property
    def current_price(self) -> float:
        return float(self.prices[self.day_index])

    @property
    def next_price(self) -> float:
        next_idx = min(self.day_index + 1, len(self.prices) - 1)
        return float(self.prices[next_idx])

    def _portfolio_value(self, mark_price: float) -> float:
        return float(self.cash + (self.position * mark_price))

    def _drawdown(self, portfolio_value: float) -> float:
        self.peak_portfolio_value = max(self.peak_portfolio_value, portfolio_value)
        if self.peak_portfolio_value <= 0.0:
            return 0.0
        return max(0.0, (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value)

    def _momentum(self, lookback: int) -> float:
        start = max(0, self.day_index - lookback)
        base = self.prices[start]
        if base == 0.0:
            return 0.0
        return (self.current_price - base) / base

    def _price_window(self, window: int = 5) -> List[float]:
        start = max(0, self.day_index - window + 1)
        return [float(x) for x in self.prices[start : self.day_index + 1]]

    def _coerce_action(self, action_input: Any) -> TradeAction:
        default = {
            "symbol": self.task["symbol"],
            "decision": "hold",
            "quantity": 0,
            "confidence": 0.5,
            "rationale": "fallback",
        }
        if isinstance(action_input, TradeAction):
            return action_input
        if isinstance(action_input, dict):
            payload = {
                "symbol": str(action_input.get("symbol", default["symbol"])),
                "decision": str(action_input.get("decision", default["decision"])),
                "quantity": int(action_input.get("quantity", default["quantity"])),
                "confidence": float(action_input.get("confidence", default["confidence"])),
                "rationale": str(action_input.get("rationale", default["rationale"])),
            }
            try:
                return TradeAction(**payload)
            except Exception:
                return TradeAction(**default)
        return TradeAction(**default)

    def _build_observation(self) -> MarketObservation:
        portfolio_value = self._portfolio_value(self.current_price)
        drawdown = self._drawdown(portfolio_value)
        obs = MarketObservation(
            task_id=self.task["id"],
            task_name=self.task["name"],
            difficulty=self.task["difficulty"],
            symbol=self.task["symbol"],
            objective=self.task["objective"],
            day_index=self.day_index,
            max_days=len(self.prices) - 1,
            current_price=self.current_price,
            next_price=self.next_price,
            price_window=self._price_window(),
            momentum_1d=self._momentum(1),
            momentum_3d=self._momentum(3),
            cash=self.cash,
            position=self.position,
            avg_entry_price=self.avg_entry_price,
            portfolio_value=portfolio_value,
            peak_portfolio_value=self.peak_portfolio_value,
            drawdown=drawdown,
            last_decision=self.last_decision,
            metadata={
                "score": float(self.task_score),
                "task_score": float(self.task_score),
                "cumulative_reward": float(self.cumulative_reward),
            },
        )
        return obs

    def _state(self) -> MarketState:
        portfolio_value = self._portfolio_value(self.current_price)
        drawdown = self._drawdown(portfolio_value)
        return MarketState(
            task_id=self.task_id,
            symbol=self.task["symbol"],
            day_index=self.day_index,
            step_count=self.step_count,
            max_steps=self.max_steps,
            cash=self.cash,
            position=self.position,
            avg_entry_price=self.avg_entry_price,
            portfolio_value=portfolio_value,
            peak_portfolio_value=self.peak_portfolio_value,
            drawdown=drawdown,
            cumulative_reward=float(self.cumulative_reward),
            task_score=float(self.task_score),
            done=self.done,
        )

    def reset(self, task_id: str | None = None) -> MarketObservation:
        if task_id:
            self.task = get_task(str(task_id))
            self.task_id = self.task["id"]
        self._init_runtime()
        return self._build_observation()

    def state(self) -> MarketState:
        return self._state()

    def _execute_trade(self, action: TradeAction, price: float) -> int:
        executed = 0
        if action.symbol != self.task["symbol"]:
            return 0

        qty = max(int(action.quantity), 0)
        if action.decision == "buy" and qty > 0:
            affordable = int(self.cash // price)
            executed = min(qty, affordable)
            if executed > 0:
                prev_cost = self.avg_entry_price * self.position
                trade_cost = executed * price
                self.cash -= trade_cost
                self.position += executed
                self.avg_entry_price = (prev_cost + trade_cost) / max(self.position, 1)

        elif action.decision == "sell" and qty > 0:
            executed = min(qty, self.position)
            if executed > 0:
                self.cash += executed * price
                self.position -= executed
                if self.position == 0:
                    self.avg_entry_price = 0.0

        return executed

    def _step_reward(
        self,
        action: TradeAction,
        ideal_decision: str,
        prev_value: float,
        next_value: float,
        drawdown: float,
    ) -> MarketReward:
        align = action_alignment(action.decision, ideal_decision)
        ret_pct = (next_value - prev_value) / max(prev_value, 1.0)
        return_component = max(0.0, min(1.0, 0.5 + (ret_pct * 8.0)))

        position_value = self.position * self.next_price
        position_ratio = position_value / max(next_value, 1.0)
        max_ratio = float(self.task.get("max_position_ratio", 0.7))
        concentration_penalty = max(0.0, (position_ratio - max_ratio) * 2.0)

        max_drawdown = float(self.task.get("max_drawdown", 0.1))
        drawdown_penalty = max(0.0, (drawdown - max_drawdown) * 3.0)
        risk_component = max(0.0, 1.0 - concentration_penalty - drawdown_penalty)

        raw = (0.45 * align) + (0.35 * return_component) + (0.2 * risk_component)
        reward = strict_score(raw)

        explanation = (
            f"decision={action.decision} ideal={ideal_decision} align={align:.2f} "
            f"ret={ret_pct:.4f} risk={risk_component:.2f}"
        )

        return MarketReward(
            reward=reward,
            task_score=self.task_score,
            action_alignment=align,
            return_component=return_component,
            risk_component=risk_component,
            explanation=explanation,
            done=self.done,
        )

    def evaluate_task(self) -> float:
        score = grade_episode(
            task=self.task,
            history=self.history,
            initial_cash=self.initial_cash,
            final_portfolio_value=self._portfolio_value(self.current_price),
            drawdown=self._drawdown(self._portfolio_value(self.current_price)),
            done=self.done,
        )
        self.task_score = strict_score(score)
        return self.task_score

    def step(self, action_input: Any) -> Tuple[MarketObservation, float, bool, Dict[str, Any]]:
        if self.done:
            safe = strict_score(self.task_score)
            return self._build_observation(), MIN_STRICT_SCORE, True, {
                "score": safe,
                "task_score": safe,
                "cumulative_reward": float(self.cumulative_reward),
                "explanation": "Episode already completed.",
            }

        action = self._coerce_action(action_input)
        current_price = self.current_price
        prev_value = self._portfolio_value(current_price)

        executed_qty = self._execute_trade(action, current_price)

        transition_idx = min(self.day_index, len(self.task["ideal_actions"]) - 1)
        ideal_decision = str(self.task["ideal_actions"][transition_idx])

        self.step_count += 1
        self.day_index = min(self.day_index + 1, len(self.prices) - 1)

        next_value = self._portfolio_value(self.current_price)
        drawdown = self._drawdown(next_value)

        if self.step_count >= self.max_steps or self.day_index >= len(self.prices) - 1:
            self.done = True

        reward_obj = self._step_reward(action, ideal_decision, prev_value, next_value, drawdown)
        alignment = action_alignment(action.decision, ideal_decision)

        self.history.append(
            {
                "step": self.step_count,
                "day_index": self.day_index,
                "decision": action.decision,
                "symbol": action.symbol,
                "requested_qty": int(action.quantity),
                "executed_qty": int(executed_qty),
                "price": float(current_price),
                "ideal_decision": ideal_decision,
                "alignment": alignment,
                "reward": float(reward_obj.reward),
                "portfolio_value": float(next_value),
            }
        )

        self.cumulative_reward = strict_score(
            (self.cumulative_reward * 0.5) + (reward_obj.reward * 0.5)
        )
        self.task_score = self.evaluate_task()

        obs = self._build_observation()
        info = {
            "score": float(self.task_score),
            "task_score": float(self.task_score),
            "cumulative_reward": float(self.cumulative_reward),
            "action_alignment": float(reward_obj.action_alignment),
            "return_component": float(reward_obj.return_component),
            "risk_component": float(reward_obj.risk_component),
            "executed_qty": int(executed_qty),
            "ideal_decision": ideal_decision,
            "explanation": reward_obj.explanation,
        }
        return obs, float(reward_obj.reward), bool(self.done), info
