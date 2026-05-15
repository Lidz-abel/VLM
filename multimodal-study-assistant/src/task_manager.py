from __future__ import annotations

import time
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Callable

from .run_logger import log_event


TaskFn = Callable[["TaskContext"], Any]


@dataclass
class TaskState:
    task_id: str
    name: str
    status: str = "pending"
    progress: float = 0.0
    message: str = ""
    result: Any = None
    error: str | None = None
    traceback: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "name": self.name,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "result": self.result,
            "error": self.error,
            "traceback": self.traceback,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class TaskContext:
    def __init__(self, manager: "TaskManager", task_id: str) -> None:
        self.manager = manager
        self.task_id = task_id

    def update(self, progress: float | None = None, message: str | None = None, **extra: Any) -> None:
        self.manager.update(self.task_id, progress=progress, message=message, **extra)


class TaskManager:
    def __init__(self, max_workers: int = 1) -> None:
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.tasks: dict[str, TaskState] = {}
        self.futures: dict[str, Future] = {}
        self.lock = Lock()

    def submit(self, name: str, fn: TaskFn) -> str:
        task_id = str(uuid.uuid4())
        state = TaskState(task_id=task_id, name=name)
        with self.lock:
            self.tasks[task_id] = state
        future = self.executor.submit(self._run, task_id, fn)
        with self.lock:
            self.futures[task_id] = future
        log_event("task_submitted", {"task_id": task_id, "name": name})
        return task_id

    def _run(self, task_id: str, fn: TaskFn) -> None:
        self.update(task_id, status="running", progress=0.0, message="running")
        context = TaskContext(self, task_id)
        try:
            result = fn(context)
            self.update(task_id, status="succeeded", progress=1.0, message="succeeded", result=result)
            log_event("task_succeeded", {"task_id": task_id})
        except Exception as exc:
            tb = traceback.format_exc()
            self.update(task_id, status="failed", message="failed", error=str(exc), traceback=tb)
            log_event("task_failed", {"task_id": task_id, "error": str(exc)})

    def update(
        self,
        task_id: str,
        progress: float | None = None,
        message: str | None = None,
        status: str | None = None,
        result: Any = None,
        error: str | None = None,
        traceback: str | None = None,
    ) -> None:
        with self.lock:
            state = self.tasks[task_id]
            if progress is not None:
                state.progress = max(0.0, min(1.0, float(progress)))
            if message is not None:
                state.message = message
            if status is not None:
                state.status = status
            if result is not None:
                state.result = result
            if error is not None:
                state.error = error
            if traceback is not None:
                state.traceback = traceback
            state.updated_at = time.time()

    def get(self, task_id: str) -> dict[str, Any] | None:
        with self.lock:
            state = self.tasks.get(task_id)
            return state.to_dict() if state else None

    def list(self) -> list[dict[str, Any]]:
        with self.lock:
            return [state.to_dict() for state in sorted(self.tasks.values(), key=lambda item: item.created_at, reverse=True)]


default_task_manager = TaskManager(max_workers=1)
