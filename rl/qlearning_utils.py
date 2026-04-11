from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def discretize_observation(obs: np.ndarray, n_tasks: int, n_operations: int) -> Tuple[int, ...]:
    """Convert continuous observation into compact tabular bins."""
    task_idx = int(np.argmax(obs[:n_tasks]))
    scalars = obs[n_tasks:]
    bins = np.clip((scalars * 10.0).astype(np.int32), 0, 20)
    return (task_idx, *bins.tolist())


def get_q_row(q_table: Dict[Tuple[int, ...], np.ndarray], state: Tuple[int, ...], n_actions: int) -> np.ndarray:
    row = q_table.get(state)
    if row is None:
        row = np.zeros(n_actions, dtype=np.float32)
        q_table[state] = row
    return row


def save_q_table(
    path: str,
    q_table: Dict[Tuple[int, ...], np.ndarray],
    n_tasks: int,
    n_operations: int,
    n_actions: int,
) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    serializable = {
        ",".join(map(str, key)): value.tolist()
        for key, value in q_table.items()
    }

    payload = {
        "n_tasks": n_tasks,
        "n_operations": n_operations,
        "n_actions": n_actions,
        "q_table": serializable,
    }
    p.write_text(json.dumps(payload))


def load_q_table(path: str) -> Tuple[Dict[Tuple[int, ...], np.ndarray], int, int, int]:
    payload = json.loads(Path(path).read_text())
    q_table = {
        tuple(int(x) for x in key.split(",")): np.array(values, dtype=np.float32)
        for key, values in payload["q_table"].items()
    }
    return q_table, int(payload["n_tasks"]), int(payload["n_operations"]), int(payload["n_actions"])
