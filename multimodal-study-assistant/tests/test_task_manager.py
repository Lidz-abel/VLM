from __future__ import annotations

import time

from src.task_manager import TaskManager


def test_task_manager_runs_job() -> None:
    manager = TaskManager(max_workers=1)
    task_id = manager.submit("unit-test", lambda context: {"ok": True})

    deadline = time.time() + 5
    state = manager.get(task_id)
    while state and state["status"] not in {"succeeded", "failed"} and time.time() < deadline:
        time.sleep(0.05)
        state = manager.get(task_id)

    assert state is not None
    assert state["status"] == "succeeded"
    assert state["progress"] == 1.0
    assert state["result"] == {"ok": True}
