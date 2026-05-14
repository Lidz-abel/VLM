from __future__ import annotations

import json
import os
import re
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
