from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TaskResponse(BaseModel):
    task_id: str


class ImageAnalyzeResponse(BaseModel):
    answer: str
    model: str
    mock: bool


class PdfParseRequest(BaseModel):
    start_page: int | None = None
    end_page: int | None = None
    resume: bool = True
    overwrite_images: bool = False
    mock: bool = False
    model_name: str | None = None


class RagBuildRequest(BaseModel):
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    device: str = "cpu"


class RagQueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=4, ge=1, le=20)
    mock: bool = False
    model_name: str | None = None


class NoteResponse(BaseModel):
    output_path: str
    preview: str


class HealthResponse(BaseModel):
    status: str
    mock_vlm: bool
    cuda_visible_devices: str
    gpu: Any
