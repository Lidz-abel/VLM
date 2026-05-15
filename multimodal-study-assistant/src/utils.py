from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: Any, path: str | Path) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return target


def write_text(text: str, path: str | Path) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(text, encoding="utf-8")
    return target


def safe_stem(path_or_name: str | Path) -> str:
    stem = Path(path_or_name).stem
    stem = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._-]+", "_", stem).strip("._")
    return stem or "document"


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def extract_section(raw_answer: str, section_name: str) -> str:
    pattern = rf"【{re.escape(section_name)}】\s*(.*?)(?=\n【|$)"
    match = re.search(pattern, raw_answer, flags=re.S)
    return match.group(1).strip() if match else ""


def extract_bullets(text: str, limit: int = 12) -> list[str]:
    concepts: list[str] = []
    for line in text.splitlines():
        item = line.strip().lstrip("-*0123456789.、) ")
        if item:
            concepts.append(item)
        if len(concepts) >= limit:
            break
    return concepts


def page_record_from_answer(page: int, image_path: str, raw_answer: str) -> dict[str, Any]:
    ocr_text = extract_section(raw_answer, "识别内容") or extract_section(raw_answer, "OCR 转写内容")
    concepts_text = extract_section(raw_answer, "核心知识点") or extract_section(raw_answer, "关键概念")
    summary = (
        extract_section(raw_answer, "本页摘要")
        or extract_section(raw_answer, "最终答案")
        or extract_section(raw_answer, "分析过程")
        or raw_answer[:600]
    )
    return {
        "page": page,
        "image_path": image_path,
        "summary": summary,
        "ocr_text": ocr_text,
        "key_concepts": extract_bullets(concepts_text),
        "raw_answer": raw_answer,
    }


def get_gpu_memory_summary() -> str:
    try:
        import torch

        if not torch.cuda.is_available():
            return "CUDA unavailable"
        current = torch.cuda.current_device()
        used = torch.cuda.memory_allocated(current) / 1024**3
        reserved = torch.cuda.memory_reserved(current) / 1024**3
        total = torch.cuda.get_device_properties(current).total_memory / 1024**3
        return f"GPU {current}: allocated={used:.2f}GB, reserved={reserved:.2f}GB, total={total:.2f}GB"
    except Exception as exc:
        return f"GPU memory unavailable: {exc}"


def get_bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def visible_cuda_devices() -> list[str]:
    raw = os.getenv("CUDA_VISIBLE_DEVICES")
    if raw is None or not raw.strip():
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def get_nvidia_smi_memory() -> list[dict[str, Any]]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []

    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        rows.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "memory_used_mb": int(parts[2]),
                "memory_free_mb": int(parts[3]),
                "utilization_gpu": int(parts[4]),
            }
        )
    return rows


def assert_gpu_ready(min_free_gb: float = 18.0, require_single_visible: bool = True) -> str:
    if get_bool_env("MOCK_VLM", False):
        return "MOCK_VLM=1, skip GPU readiness check."

    visible = visible_cuda_devices()
    if require_single_visible and len(visible) != 1:
        raise RuntimeError(
            "为避免误用多卡，请先设置单卡可见，例如：CUDA_VISIBLE_DEVICES=0 bash scripts/run_demo.sh。"
        )

    gpu_rows = get_nvidia_smi_memory()
    if not gpu_rows:
        raise RuntimeError("无法读取 nvidia-smi，请确认 GPU 驱动可用。")

    target_index = int(visible[0]) if visible and visible[0].isdigit() else gpu_rows[0]["index"]
    row = next((item for item in gpu_rows if item["index"] == target_index), None)
    if row is None:
        raise RuntimeError(f"没有找到目标 GPU {target_index} 的 nvidia-smi 信息。")

    free_gb = row["memory_free_mb"] / 1024
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"GPU {target_index} 空闲显存只有 {free_gb:.1f}GB，低于阈值 {min_free_gb:.1f}GB。"
            "请等待任务结束、换一张空闲卡，或使用 MOCK_VLM=1 先跑无 GPU 流程。"
        )
    return f"GPU {target_index} ready: free={free_gb:.1f}GB"
