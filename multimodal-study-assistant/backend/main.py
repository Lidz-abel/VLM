from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from backend.schemas import (
    HealthResponse,
    ImageAnalyzeResponse,
    NoteResponse,
    PdfParseRequest,
    RagBuildRequest,
    RagQueryRequest,
    TaskResponse,
)
from backend.services import (
    analyze_image,
    build_rag_for_document,
    generate_notes_for_document,
    health_payload,
    parse_pdf_document,
    query_document,
)
from src.task_manager import TaskContext, default_task_manager
from src.workspace import default_workspace


app = FastAPI(title="Multimodal Study Assistant API", version="0.1.0")


@app.get("/health", response_model=HealthResponse)
def health() -> dict:
    return health_payload()


@app.get("/api/tasks")
def list_tasks() -> list[dict]:
    return default_task_manager.list()


@app.get("/api/tasks/{task_id}")
def get_task(task_id: str) -> dict:
    task = default_task_manager.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@app.post("/api/image/analyze", response_model=ImageAnalyzeResponse)
async def analyze_uploaded_image(
    image: UploadFile = File(...),
    question: str = Form("请解释这张图片"),
    structured: bool = Form(True),
    mock: bool = Form(False),
    model_name: str | None = Form(None),
) -> dict:
    doc = default_workspace.create_document(image.filename or "image.png", fileobj=image.file)
    result = analyze_image(doc["source_path"], question=question, structured=structured, mock=mock, model_name=model_name)
    return result


@app.post("/api/documents")
async def upload_document(file: UploadFile = File(...)) -> dict:
    return default_workspace.create_document(file.filename or "document.pdf", fileobj=file.file)


@app.get("/api/documents")
def list_documents() -> list[dict]:
    return default_workspace.list_documents()


@app.get("/api/documents/{doc_id}")
def get_document(doc_id: str) -> dict:
    try:
        return default_workspace.get_document(doc_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/documents/{doc_id}/parse", response_model=TaskResponse)
def parse_document(doc_id: str, request: PdfParseRequest) -> dict:
    def job(context: TaskContext) -> dict:
        def progress(done: int, total: int) -> None:
            context.update(progress=done / total if total else 0.0, message=f"parsed {done}/{total}")

        return parse_pdf_document(
            doc_id,
            start_page=request.start_page,
            end_page=request.end_page,
            resume=request.resume,
            overwrite_images=request.overwrite_images,
            mock=request.mock,
            model_name=request.model_name,
            progress_callback=progress,
        )

    return {"task_id": default_task_manager.submit(f"parse_pdf:{doc_id}", job)}


@app.get("/api/documents/{doc_id}/results")
def get_results(doc_id: str) -> dict:
    path = default_workspace.results_path(doc_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Results not found")
    return {"doc_id": doc_id, "results_path": str(path), "results": __import__("json").loads(path.read_text(encoding="utf-8"))}


@app.post("/api/documents/{doc_id}/rag/build", response_model=TaskResponse)
def build_rag(doc_id: str, request: RagBuildRequest) -> dict:
    def job(context: TaskContext) -> dict:
        context.update(progress=0.1, message="building rag index")
        result = build_rag_for_document(doc_id, embedding_model=request.embedding_model, device=request.device)
        context.update(progress=1.0, message="rag index built")
        return result

    return {"task_id": default_task_manager.submit(f"build_rag:{doc_id}", job)}


@app.post("/api/documents/{doc_id}/rag/query")
def query_rag(doc_id: str, request: RagQueryRequest) -> dict:
    return query_document(
        doc_id,
        question=request.question,
        top_k=request.top_k,
        mock=request.mock,
        model_name=request.model_name,
    )


@app.post("/api/documents/{doc_id}/notes", response_model=NoteResponse)
def generate_notes(doc_id: str) -> dict:
    if not default_workspace.results_path(doc_id).exists():
        raise HTTPException(status_code=404, detail="Results not found")
    return generate_notes_for_document(doc_id)


@app.get("/api/documents/{doc_id}/files/{kind}")
def get_document_file_path(doc_id: str, kind: str) -> dict:
    mapping = {
        "source": Path(default_workspace.get_document(doc_id)["source_path"]),
        "results": default_workspace.results_path(doc_id),
        "notes": default_workspace.notes_path(doc_id),
    }
    if kind not in mapping:
        raise HTTPException(status_code=404, detail="Unknown file kind")
    path = mapping[kind]
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return {"path": str(path)}
