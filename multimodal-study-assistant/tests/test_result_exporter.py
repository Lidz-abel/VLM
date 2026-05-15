from __future__ import annotations

from src.result_exporter import failed_pages, results_to_markdown


def test_results_to_markdown_and_failed_pages() -> None:
    records = [
        {
            "page": 1,
            "image_path": "page_1.png",
            "summary": "summary",
            "ocr_text": "ocr",
            "key_concepts": ["concept"],
            "raw_answer": "raw",
        },
        {
            "page": 2,
            "image_path": "page_2.png",
            "error": "boom",
        },
    ]

    markdown = results_to_markdown(records)
    failed = failed_pages(records)

    assert "第 1 页" in markdown
    assert "summary" in markdown
    assert len(failed) == 1
    assert failed[0]["page"] == 2
