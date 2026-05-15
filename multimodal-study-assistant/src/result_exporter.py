from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import read_json, safe_stem, write_text


def results_to_markdown(records: list[dict[str, Any]], title: str = "PDF 页面解析结果") -> str:
    lines = [f"# {title}", ""]
    total = len(records)
    failed = sum(1 for item in records if item.get("error"))
    lines.extend([f"- 页面数：{total}", f"- 失败页：{failed}", ""])

    for record in sorted(records, key=lambda item: int(item.get("page", 0) or 0)):
        page = record.get("page", "?")
        lines.extend([f"## 第 {page} 页", ""])
        if record.get("image_path"):
            lines.extend([f"- 图片：`{record['image_path']}`", ""])
        if record.get("error"):
            lines.extend(["### 错误", "", str(record["error"]), ""])
            continue
        lines.extend(["### 摘要", "", record.get("summary", "") or "暂无摘要", ""])
        lines.extend(["### OCR", "", record.get("ocr_text", "") or "暂无 OCR 内容", ""])
        concepts = record.get("key_concepts", [])
        lines.extend(["### 核心知识点", ""])
        if concepts:
            lines.extend(f"- {item}" for item in concepts)
        else:
            lines.append("- 暂无核心知识点")
        lines.extend(["", "### 原始回答", "", record.get("raw_answer", "") or "暂无原始回答", ""])
    return "\n".join(lines).strip() + "\n"


def export_results_markdown(results_path: str | Path, output_path: str | Path | None = None) -> Path:
    results_path = Path(results_path)
    records = read_json(results_path)
    if output_path is None:
        output_path = Path("outputs") / f"{safe_stem(results_path)}_pages.md"
    markdown = results_to_markdown(records, title=f"{safe_stem(results_path)} 页面解析结果")
    return write_text(markdown, output_path)


def failed_pages(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "page": record.get("page"),
            "image_path": record.get("image_path", ""),
            "error": record.get("error", ""),
        }
        for record in records
        if record.get("error")
    ]


def failed_pages_markdown(results_path: str | Path) -> str:
    records = read_json(results_path)
    failed = failed_pages(records)
    if not failed:
        return "没有失败页。\n"
    lines = ["# 失败页列表", ""]
    for item in failed:
        lines.append(f"- 第 {item['page']} 页：`{item['image_path']}`")
        lines.append(f"  - 错误：{item['error']}")
    return "\n".join(lines) + "\n"
