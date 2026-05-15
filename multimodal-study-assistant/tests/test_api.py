from __future__ import annotations

from fastapi.testclient import TestClient

from backend.main import app


def test_health_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_image_analyze_mock_endpoint() -> None:
    client = TestClient(app)
    response = client.post(
        "/api/image/analyze",
        data={"question": "请解释这张图片", "structured": "true", "mock": "true"},
        files={"image": ("demo.png", b"fake image bytes", "image/png")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["mock"] is True
    assert "Mock" in payload["answer"] or "MOCK" in payload["answer"]
