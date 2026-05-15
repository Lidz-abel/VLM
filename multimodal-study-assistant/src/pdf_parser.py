from __future__ import annotations

from pathlib import Path
from typing import Callable

import fitz

from .prompts import PDF_PAGE_ANALYSIS_PROMPT
from .run_logger import Timer, log_event
from .utils import ensure_dir, page_record_from_answer, read_json, safe_stem, write_json
from .vlm_infer import VLMInferencer


def _normalize_page_range(total_pages: int, start_page: int | None = None, end_page: int | None = None) -> tuple[int, int]:
    start = max(1, start_page or 1)
    end = min(total_pages, end_page or total_pages)
    if start > end:
        raise ValueError(f"Invalid page range: start_page={start_page}, end_page={end_page}, total_pages={total_pages}")
    return start, end


def pdf_to_images(
    pdf_path: str | Path,
    output_dir: str | Path,
    zoom: float = 2.0,
    start_page: int | None = None,
    end_page: int | None = None,
    overwrite: bool = False,
) -> list[Path]:
    pdf_path = Path(pdf_path)
    output_dir = ensure_dir(output_dir)
    doc = fitz.open(pdf_path)
    image_paths: list[Path] = []

    try:
        start, end = _normalize_page_range(doc.page_count, start_page=start_page, end_page=end_page)
        matrix = fitz.Matrix(zoom, zoom)
        for page_number in range(start, end + 1):
            page_index = page_number - 1
            image_path = output_dir / f"page_{page_number}.png"
            if image_path.exists() and not overwrite:
                image_paths.append(image_path)
                continue
            page = doc.load_page(page_index)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            pix.save(image_path)
            image_paths.append(image_path)
    finally:
        doc.close()

    return image_paths


def parse_page_images(
    image_paths: list[str | Path],
    output_dir: str | Path,
    inferencer: VLMInferencer | None = None,
    question: str | None = None,
    resume: bool = True,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[dict]:
    output_dir = ensure_dir(output_dir)
    inferencer = inferencer or VLMInferencer()
    prompt = question or PDF_PAGE_ANALYSIS_PROMPT

    image_items = sorted((Path(path) for path in image_paths), key=_page_number_from_path)
    records_by_page: dict[int, dict] = {}
    for record_path in sorted(output_dir.glob("page_*.json"), key=_page_number_from_path):
        try:
            record = read_json(record_path)
        except Exception:
            continue
        page = int(record.get("page") or _page_number_from_path(record_path))
        records_by_page[page] = record

    total = len(image_items)
    for done, image_path in enumerate(image_items, start=1):
        timer = Timer()
        page = _page_number_from_path(image_path)
        record_path = output_dir / f"page_{page}.json"
        if resume and record_path.exists():
            records_by_page[page] = read_json(record_path)
            log_event(
                "pdf_page_skipped",
                {"page": page, "image_path": str(image_path), "record_path": str(record_path), "reason": "resume"},
            )
            if progress_callback:
                progress_callback(done, total)
            continue

        try:
            raw_answer = inferencer.answer_image(image_path=image_path, prompt_override=prompt)
            record = page_record_from_answer(page, str(image_path), raw_answer)
        except Exception as exc:
            record = {
                "page": page,
                "image_path": str(image_path),
                "summary": "",
                "ocr_text": "",
                "key_concepts": [],
                "raw_answer": "",
                "error": str(exc),
            }
        log_event(
            "pdf_page_parsed",
            {
                "page": page,
                "image_path": str(image_path),
                "record_path": str(record_path),
                "success": not bool(record.get("error")),
                "error": record.get("error", ""),
                "elapsed_sec": round(timer.elapsed, 3),
            },
        )
        records_by_page[page] = record
        write_json(record, record_path)
        write_json(_sorted_records(records_by_page), output_dir / "results.json")
        if progress_callback:
            progress_callback(done, total)

    records = _sorted_records(records_by_page)
    write_json(records, output_dir / "results.json")
    return records


def parse_pdf(
    pdf_path: str | Path,
    output_dir: str | Path | None = None,
    inferencer: VLMInferencer | None = None,
    question: str | None = None,
    zoom: float = 2.0,
    start_page: int | None = None,
    end_page: int | None = None,
    resume: bool = True,
    overwrite_images: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[dict]:
    pdf_path = Path(pdf_path)
    if output_dir is None:
        output_dir = Path("data") / "parsed_pages" / safe_stem(pdf_path)
    output_dir = ensure_dir(output_dir)

    inferencer = inferencer or VLMInferencer()
    images = pdf_to_images(
        pdf_path,
        output_dir,
        zoom=zoom,
        start_page=start_page,
        end_page=end_page,
        overwrite=overwrite_images,
    )
    return parse_page_images(
        images,
        output_dir,
        inferencer=inferencer,
        question=question,
        resume=resume,
        progress_callback=progress_callback,
    )


def _page_number_from_path(path: str | Path) -> int:
    match = __import__("re").search(r"page_(\d+)", Path(path).stem)
    return int(match.group(1)) if match else 0


def _sorted_records(records_by_page: dict[int, dict]) -> list[dict]:
    return [records_by_page[page] for page in sorted(records_by_page)]
