import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def discretize_observation(obs: np.ndarray, n_tasks: int, n_operations: int) -> Tuple[int, ...]:
    """Convert continuous observation into a compact tabular state key."""
    obs = np.asarray(obs, dtype=np.float32)

    task_idx = int(np.argmax(obs[:n_tasks]))
    op_start = n_tasks
    op_end = n_tasks + n_operations
    last_op_idx = int(np.argmax(obs[op_start:op_end]))

    scalars = obs[op_end:]
    bins = np.clip((scalars * 10.0).astype(np.int32), 0, 10)

    return (task_idx, last_op_idx, *bins.tolist())


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
    serialized_q = {}
    for key, value in q_table.items():
        state_key = "|".join(map(str, key))
        serialized_q[state_key] = [float(x) for x in np.asarray(value, dtype=np.float32).tolist()]

    output = {
        "meta": {
            "n_tasks": int(n_tasks),
            "n_operations": int(n_operations),
            "n_actions": int(n_actions),
        },
        "q_table": serialized_q,
    }

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2), encoding="utf-8")


def load_q_table(path: str):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    meta = payload["meta"]

    q_table = {}
    for key, values in payload["q_table"].items():
        state = tuple(int(part) for part in key.split("|"))
        q_table[state] = np.array(values, dtype=np.float32)

    return q_table, meta
