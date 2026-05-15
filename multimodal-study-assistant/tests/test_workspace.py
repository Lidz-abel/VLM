from __future__ import annotations

from src.workspace import WorkspaceManager


def test_workspace_creates_doc_id_layout(tmp_path) -> None:
    workspace = WorkspaceManager(root=tmp_path)
    metadata = workspace.create_document("demo.pdf", content=b"%PDF-1.4")
    doc_id = metadata["doc_id"]

    loaded = workspace.get_document(doc_id)

    assert loaded["doc_id"] == doc_id
    assert loaded["filename"] == "demo.pdf"
    assert workspace.doc_dir(doc_id).exists()
    assert workspace.pages_dir(doc_id).name == "pages"
    assert workspace.parsed_dir(doc_id).name == "parsed"
    assert workspace.results_path(doc_id).name == "results.json"
