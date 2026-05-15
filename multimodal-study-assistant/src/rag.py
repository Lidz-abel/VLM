from __future__ import annotations

from pathlib import Path
from typing import Any

import faiss
import numpy as np

from .prompts import build_rag_prompt
from .utils import ensure_dir, read_json, write_json


def record_to_document(record: dict[str, Any]) -> dict[str, Any]:
    parts = [
        record.get("summary", ""),
        record.get("ocr_text", ""),
        "\n".join(record.get("key_concepts", [])),
        record.get("raw_answer", ""),
    ]
    content = "\n".join(part for part in parts if part).strip()
    return {
        "page": record.get("page"),
        "content": content,
        "image_path": record.get("image_path", ""),
    }


class PdfRagIndex:
    def __init__(self, embedding_model: str = "BAAI/bge-small-zh-v1.5", device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer

        self.embedding_model_name = embedding_model
        self.device = device
        self.encoder = SentenceTransformer(embedding_model, device=device)
        self.index: faiss.Index | None = None
        self.docs: list[dict[str, Any]] = []

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.encoder.encode(
            texts,
            batch_size=16,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype="float32")

    def build_from_records(self, records: list[dict[str, Any]]) -> None:
        self.docs = [record_to_document(record) for record in records]
        if not self.docs:
            raise ValueError("No documents found for indexing.")
        vectors = self.encode([doc["content"] for doc in self.docs])
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def save(self, output_dir: str | Path) -> None:
        if self.index is None:
            raise ValueError("Index is empty. Call build_from_records first.")
        output_dir = ensure_dir(output_dir)
        faiss.write_index(self.index, str(output_dir / "index.faiss"))
        write_json(
            {
                "embedding_model": self.embedding_model_name,
                "device": self.device,
                "docs": self.docs,
            },
            output_dir / "docs.json",
        )

    @classmethod
    def load(cls, index_dir: str | Path) -> "PdfRagIndex":
        index_dir = Path(index_dir)
        metadata = read_json(index_dir / "docs.json")
        instance = cls(embedding_model=metadata["embedding_model"], device=metadata.get("device", "cpu"))
        instance.docs = metadata["docs"]
        instance.index = faiss.read_index(str(index_dir / "index.faiss"))
        return instance

    def search(self, question: str, top_k: int = 4) -> list[dict[str, Any]]:
        if self.index is None:
            raise ValueError("Index is empty.")
        query = self.encode([question])
        scores, ids = self.index.search(query, top_k)
        results: list[dict[str, Any]] = []
        for score, doc_id in zip(scores[0], ids[0]):
            if doc_id < 0:
                continue
            doc = self.docs[int(doc_id)]
            results.append(
                {
                    "page": doc.get("page"),
                    "score": float(score),
                    "content": doc.get("content", ""),
                    "image_path": doc.get("image_path", ""),
                }
            )
        return results


def build_index_from_results(
    parsed_results_path: str | Path,
    output_dir: str | Path,
    embedding_model: str = "BAAI/bge-small-zh-v1.5",
    device: str = "cpu",
) -> PdfRagIndex:
    records = read_json(parsed_results_path)
    index = PdfRagIndex(embedding_model=embedding_model, device=device)
    index.build_from_records(records)
    index.save(output_dir)
    return index


def answer_pdf_question(
    question: str,
    index_dir: str | Path,
    llm_answer_func,
    top_k: int = 4,
) -> tuple[str, list[dict[str, Any]]]:
    index = PdfRagIndex.load(index_dir)
    contexts = index.search(question, top_k=top_k)
    prompt = build_rag_prompt(question, contexts)
    answer = llm_answer_func(prompt, contexts)
    return answer, contexts
