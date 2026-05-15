from __future__ import annotations

import shutil
import time
import uuid
from pathlib import Path
from typing import Any, BinaryIO

from .utils import PROJECT_ROOT, ensure_dir, read_json, write_json


class WorkspaceManager:
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = ensure_dir(root or PROJECT_ROOT / "workspace" / "documents")

    def create_document(self, filename: str, fileobj: BinaryIO | None = None, content: bytes | None = None) -> dict[str, Any]:
        doc_id = str(uuid.uuid4())
        doc_dir = ensure_dir(self.root / doc_id)
        source_path = doc_dir / self._safe_filename(filename)
        if fileobj is not None:
            with source_path.open("wb") as f:
                shutil.copyfileobj(fileobj, f)
        elif content is not None:
            source_path.write_bytes(content)
        else:
            source_path.touch()

        metadata = {
            "doc_id": doc_id,
            "filename": filename,
            "source_path": str(source_path),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        write_json(metadata, doc_dir / "metadata.json")
        return metadata

    def get_document(self, doc_id: str) -> dict[str, Any]:
        doc_dir = self.doc_dir(doc_id)
        metadata_path = doc_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Document not found: {doc_id}")
        metadata = read_json(metadata_path)
        metadata.update(
            {
                "doc_dir": str(doc_dir),
                "pages_dir": str(self.pages_dir(doc_id)),
                "parsed_dir": str(self.parsed_dir(doc_id)),
                "results_path": str(self.results_path(doc_id)),
                "notes_path": str(self.notes_path(doc_id)),
                "vector_dir": str(self.vector_dir(doc_id)),
            }
        )
        return metadata

    def list_documents(self) -> list[dict[str, Any]]:
        docs = []
        for metadata_path in sorted(self.root.glob("*/metadata.json")):
            try:
                docs.append(self.get_document(metadata_path.parent.name))
            except Exception:
                continue
        return docs

    def doc_dir(self, doc_id: str) -> Path:
        return self.root / doc_id

    def pages_dir(self, doc_id: str) -> Path:
        return ensure_dir(self.doc_dir(doc_id) / "pages")

    def parsed_dir(self, doc_id: str) -> Path:
        return ensure_dir(self.doc_dir(doc_id) / "parsed")

    def results_path(self, doc_id: str) -> Path:
        return self.parsed_dir(doc_id) / "results.json"

    def notes_path(self, doc_id: str) -> Path:
        return self.doc_dir(doc_id) / "notes.md"

    def vector_dir(self, doc_id: str) -> Path:
        return ensure_dir(self.doc_dir(doc_id) / "vector_db")

    @staticmethod
    def _safe_filename(filename: str) -> str:
        return Path(filename).name or "uploaded_file"


default_workspace = WorkspaceManager()
