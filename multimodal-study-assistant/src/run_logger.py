from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .utils import PROJECT_ROOT, ensure_dir, get_bool_env, get_gpu_memory_summary


DEFAULT_LOG_PATH = PROJECT_ROOT / "outputs" / "run_logs.jsonl"


def log_event(event: str, payload: dict[str, Any] | None = None, log_path: str | Path | None = None) -> Path:
    target = Path(log_path) if log_path else DEFAULT_LOG_PATH
    ensure_dir(target.parent)
    record = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "event": event,
        "mock_vlm": get_bool_env("MOCK_VLM", False),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
        "gpu_memory": get_gpu_memory_summary(),
        "payload": payload or {},
    }
    with target.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return target


class Timer:
    def __init__(self) -> None:
        self.start = time.time()

    @property
    def elapsed(self) -> float:
        return time.time() - self.start
