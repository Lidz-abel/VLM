from __future__ import annotations

from pathlib import Path
from typing import Any

from .prompts import NOTE_GENERATION_PROMPT, REVIEW_QUESTION_PROMPT
from .utils import read_json, safe_stem, write_text


def combine_records(records: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for record in records:
        chunks.append(
            "\n".join(
                [
                    f"## 第 {record.get('page', '?')} 页",
                    record.get("summary", ""),
                    record.get("ocr_text", ""),
                    "\n".join(record.get("key_concepts", [])),
                    record.get("raw_answer", ""),
                ]
            ).strip()
        )
    return "\n\n".join(chunk for chunk in chunks if chunk)


def fallback_markdown_note(records: list[dict[str, Any]], title: str = "学习资料复习笔记") -> str:
    content = combine_records(records)
    return f"""# {title}

## 1. 核心概念

{_collect_concepts(records)}

## 2. 主要内容

{_collect_summaries(records)}

## 3. 公式 / 图表解释

请结合各页解析中的公式、图表和代码说明复习。

## 4. 典型题目

如果资料中包含考试题或作业题，请重点复习题干、条件、推理过程和最终答案。

## 5. 易错点

- 不要忽略图片中标注、坐标轴、图例和题目限定条件。
- 对无法识别的文字需要回到原图确认。

## 6. 总结

以下为页面级解析内容：

{content}
"""


def _collect_concepts(records: list[dict[str, Any]]) -> str:
    concepts: list[str] = []
    for record in records:
        for item in record.get("key_concepts", []):
            if item not in concepts:
                concepts.append(item)
    return "\n".join(f"- {item}" for item in concepts) or "- 暂无明确核心概念"


def _collect_summaries(records: list[dict[str, Any]]) -> str:
    lines = []
    for record in records:
        summary = record.get("summary", "").strip()
        if summary:
            lines.append(f"- 第 {record.get('page', '?')} 页：{summary}")
    return "\n".join(lines) or "- 暂无页面摘要"


def generate_note_from_records(
    records: list[dict[str, Any]],
    output_path: str | Path | None = None,
    title: str = "学习资料复习笔记",
    llm_text_func=None,
) -> str:
    source_text = combine_records(records)
    if llm_text_func:
        prompt = f"{NOTE_GENERATION_PROMPT}\n\n【资料解析内容】\n{source_text}"
        note = llm_text_func(prompt)
    else:
        note = fallback_markdown_note(records, title=title)
    if output_path:
        write_text(note, output_path)
    return note


def generate_note_from_results_file(
    results_path: str | Path,
    output_path: str | Path | None = None,
    llm_text_func=None,
) -> str:
    records = read_json(results_path)
    title = safe_stem(results_path)
    if output_path is None:
        output_path = Path("outputs") / f"{title}_notes.md"
    return generate_note_from_records(records, output_path=output_path, title=title, llm_text_func=llm_text_func)


def generate_review_questions(records: list[dict[str, Any]], llm_text_func=None) -> str:
    source_text = combine_records(records)
    if llm_text_func:
        return llm_text_func(f"{REVIEW_QUESTION_PROMPT}\n\n【资料解析内容】\n{source_text}")
    return """# 复习题

## 简答题

1. 请概括本资料的核心概念，并说明它们之间的关系。

## 应用题

1. 选取一页包含题目或案例的内容，复述解题条件并给出推理过程。

## 参考答案

请以页面解析内容为依据作答，无法确认的信息需要回到原图或 PDF 核验。
"""
