from __future__ import annotations

from .utils import extract_section


def parse_image_analysis(raw_answer: str) -> dict[str, str]:
    return {
        "image_type": extract_section(raw_answer, "图片类型"),
        "recognized_content": extract_section(raw_answer, "识别内容"),
        "key_concepts": extract_section(raw_answer, "核心知识点"),
        "analysis": extract_section(raw_answer, "分析过程"),
        "final_answer": extract_section(raw_answer, "最终答案"),
        "raw_answer": raw_answer,
    }
