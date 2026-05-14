from __future__ import annotations

from pathlib import Path
from typing import Callable

import fitz

from .prompts import PDF_PAGE_ANALYSIS_PROMPT
from .utils import ensure_dir, page_record_from_answer, safe_stem, write_json
from .vlm_infer import VLMInferencer


def pdf_to_images(pdf_path: str | Path, output_dir: str | Path, zoom: float = 2.0) -> list[Path]:
    pdf_path = Path(pdf_path)
    output_dir = ensure_dir(output_dir)
    doc = fitz.open(pdf_path)
    image_paths: list[Path] = []

    try:
        matrix = fitz.Matrix(zoom, zoom)
        for page_index in range(doc.page_count):
            page = doc.load_page(page_index)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            image_path = output_dir / f"page_{page_index + 1}.png"
            pix.save(image_path)
            image_paths.append(image_path)
    finally:
        doc.close()

    return image_paths


def parse_pdf(
    pdf_path: str | Path,
    output_dir: str | Path | None = None,
    inferencer: VLMInferencer | None = None,
    question: str | None = None,
    zoom: float = 2.0,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[dict]:
    pdf_path = Path(pdf_path)
    if output_dir is None:
        output_dir = Path("data") / "parsed_pages" / safe_stem(pdf_path)
    output_dir = ensure_dir(output_dir)

    inferencer = inferencer or VLMInferencer()
    images = pdf_to_images(pdf_path, output_dir, zoom=zoom)
    records: list[dict] = []
    prompt = question or PDF_PAGE_ANALYSIS_PROMPT

    for index, image_path in enumerate(images, start=1):
        raw_answer = inferencer.answer_image(image_path=image_path, prompt_override=prompt)
        record = page_record_from_answer(index, str(image_path), raw_answer)
        records.append(record)
        write_json(record, output_dir / f"page_{index}.json")
        write_json(records, output_dir / "results.json")
        if progress_callback:
            progress_callback(index, len(images))

    return records
