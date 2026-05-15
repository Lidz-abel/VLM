from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.note_generator import generate_note_from_records
from src.pdf_parser import parse_page_images, pdf_to_images
from src.prompts import build_rag_prompt
from src.rag import PdfRagIndex, build_index_from_results
from src.run_logger import Timer, log_event
from src.utils import get_bool_env, get_nvidia_smi_memory, read_json
from src.vlm_infer import MockVLMInferencer, VLMInferencer
from src.workspace import WorkspaceManager, default_workspace


def make_inferencer(mock: bool = False, model_name: str | None = None) -> VLMInferencer:
    if mock:
        os.environ["MOCK_VLM"] = "1"
        return MockVLMInferencer(model_name=model_name)
    os.environ.pop("MOCK_VLM", None)
    return VLMInferencer(model_name=model_name)


def analyze_image(image_path: str | Path, question: str, structured: bool = True, mock: bool = False, model_name: str | None = None) -> dict[str, Any]:
    timer = Timer()
    inferencer = make_inferencer(mock=mock, model_name=model_name)
    answer = inferencer.answer_image(image_path=image_path, question=question, structured=structured)
    log_event(
        "api_image_analyzed",
        {
            "image_path": str(image_path),
            "question": question,
            "structured": structured,
            "mock": mock,
            "model": inferencer.model_name,
            "elapsed_sec": round(timer.elapsed, 3),
        },
    )
    return {"answer": answer, "model": inferencer.model_name, "mock": mock}


def parse_pdf_document(
    doc_id: str,
    workspace: WorkspaceManager = default_workspace,
    start_page: int | None = None,
    end_page: int | None = None,
    resume: bool = True,
    overwrite_images: bool = False,
    mock: bool = False,
    model_name: str | None = None,
    progress_callback=None,
) -> dict[str, Any]:
    metadata = workspace.get_document(doc_id)
    inferencer = make_inferencer(mock=mock, model_name=model_name)
    images = pdf_to_images(
        pdf_path=metadata["source_path"],
        output_dir=workspace.pages_dir(doc_id),
        start_page=start_page,
        end_page=end_page,
        overwrite=overwrite_images,
    )
    records = parse_page_images(
        images,
        output_dir=workspace.parsed_dir(doc_id),
        inferencer=inferencer,
        resume=resume,
        progress_callback=progress_callback,
    )
    return {
        "doc_id": doc_id,
        "results_path": str(workspace.results_path(doc_id)),
        "pages": len(records),
        "errors": sum(1 for item in records if item.get("error")),
    }


def build_rag_for_document(
    doc_id: str,
    workspace: WorkspaceManager = default_workspace,
    embedding_model: str = "BAAI/bge-small-zh-v1.5",
    device: str = "cpu",
) -> dict[str, Any]:
    index = build_index_from_results(workspace.results_path(doc_id), workspace.vector_dir(doc_id), embedding_model=embedding_model, device=device)
    return {"doc_id": doc_id, "index_dir": str(workspace.vector_dir(doc_id)), "pages": len(index.docs)}


def query_document(
    doc_id: str,
    question: str,
    top_k: int = 4,
    mock: bool = False,
    model_name: str | None = None,
    workspace: WorkspaceManager = default_workspace,
) -> dict[str, Any]:
    index = PdfRagIndex.load(workspace.vector_dir(doc_id))
    contexts = index.search(question, top_k=top_k)
    inferencer = make_inferencer(mock=mock, model_name=model_name)
    prompt = build_rag_prompt(question, contexts)
    answer = inferencer.answer_text(prompt)
    return {"answer": answer, "contexts": contexts, "doc_id": doc_id}


def generate_notes_for_document(doc_id: str, workspace: WorkspaceManager = default_workspace) -> dict[str, Any]:
    records = read_json(workspace.results_path(doc_id))
    note = generate_note_from_records(records, output_path=workspace.notes_path(doc_id))
    return {"output_path": str(workspace.notes_path(doc_id)), "preview": note[:1000]}


def health_payload() -> dict[str, Any]:
    return {
        "status": "ok",
        "mock_vlm": get_bool_env("MOCK_VLM", False),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
        "gpu": get_nvidia_smi_memory(),
    }
