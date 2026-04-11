"""
Contract tests: HTTP API shape, strict (0, 1) scores, Pydantic state validation.
Run: python3 -m unittest tests.test_openenv_contract -v
"""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from server.app import app
from server.models import MarketState


def _assert_open_score(name: str, value: float) -> None:
    assert 0.0 < value < 1.0, f"{name} must be strictly inside (0, 1), got {value!r}"


class TestOpenEnvContract(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_health(self) -> None:
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json().get("status"), "healthy")

    def test_reset_returns_observation_reward_done_score_info(self) -> None:
        r = self.client.post("/reset?task_id=task-001-trend-following", json={})
        self.assertEqual(r.status_code, 200)
        j = r.json()
        for key in ("observation", "reward", "done", "score", "task_score", "info"):
            self.assertIn(key, j, msg=f"missing {key}")
        self.assertFalse(j["done"])
        _assert_open_score("reset.score", float(j["score"]))

    def test_step_returns_info_and_strict_scores(self) -> None:
        self.client.post("/reset?task_id=task-001-trend-following", json={})
        r = self.client.post(
            "/step",
            json={
                "symbol": "NOVA",
                "decision": "hold",
                "quantity": 0,
                "confidence": 0.5,
                "rationale": "contract_test",
            },
        )
        self.assertEqual(r.status_code, 200)
        j = r.json()
        self.assertIn("info", j)
        _assert_open_score("step.score", float(j["score"]))

    def test_grader_reference_scores_strict(self) -> None:
        r = self.client.get("/grader")
        self.assertEqual(r.status_code, 200)
        j = r.json()
        _assert_open_score("grader.aggregate", float(j["score"]))
        self.assertIn("scores", j)
        self.assertGreaterEqual(len(j["scores"]), 4)
        for row in j["scores"]:
            _assert_open_score(f"grader.{row.get('task_id')}", float(row["score"]))

    def test_state_validates_against_market_state(self) -> None:
        self.client.post("/reset?task_id=task-001-trend-following", json={})
        r = self.client.get("/state")
        self.assertEqual(r.status_code, 200)
        MarketState(**r.json())

    def test_tasks_lists_at_least_four(self) -> None:
        r = self.client.get("/tasks")
        self.assertEqual(r.status_code, 200)
        tasks = r.json().get("tasks", [])
        self.assertGreaterEqual(len(tasks), 4)


if __name__ == "__main__":
    unittest.main()
