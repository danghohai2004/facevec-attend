from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.modules.employees import api as employees_api
from src.modules.recognition import extractor


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(employees_api.router)

    async def override_get_db():
        yield object()

    app.dependency_overrides[employees_api.get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client


def test_register_rejects_upload_larger_than_five_megabytes(client, monkeypatch):
    extract_embeddings = AsyncMock(return_value=[[0.1, 0.2]])
    register_employee = AsyncMock(
        return_value=(
            SimpleNamespace(emp_id=1, name="Alice", emp_code="EMP-1"),
            None,
        )
    )
    monkeypatch.setattr(extractor, "extract_embeddings_from_bytes", extract_embeddings)
    monkeypatch.setattr(employees_api, "register_employee", register_employee)
    monkeypatch.setattr(employees_api, "get_qdrant_client", lambda: object())

    response = client.post(
        "/api/employees",
        data={"name": "Alice", "emp_code": "EMP-1"},
        files={"file": ("face.jpg", b"x" * (5 * 1024 * 1024 + 1), "image/jpeg")},
    )

    assert response.status_code == 413
    assert response.json() == {"detail": "Ảnh quá lớn (tối đa 5MB)."}
    extract_embeddings.assert_not_awaited()
    register_employee.assert_not_awaited()


def test_register_rejects_image_with_multiple_faces(client, monkeypatch):
    extract_embeddings = AsyncMock(return_value=[[0.1, 0.2], [0.3, 0.4]])
    register_employee = AsyncMock(
        return_value=(
            SimpleNamespace(emp_id=1, name="Alice", emp_code="EMP-1"),
            None,
        )
    )
    monkeypatch.setattr(extractor, "extract_embeddings_from_bytes", extract_embeddings)
    monkeypatch.setattr(employees_api, "register_employee", register_employee)
    monkeypatch.setattr(employees_api, "get_qdrant_client", lambda: object())

    response = client.post(
        "/api/employees",
        data={"name": "Alice", "emp_code": "EMP-1"},
        files={"file": ("faces.jpg", b"image", "image/jpeg")},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Ảnh phải có đúng 1 khuôn mặt."}
    register_employee.assert_not_awaited()


def test_register_accepts_image_with_one_face(client, monkeypatch):
    extract_embeddings = AsyncMock(return_value=[[0.1, 0.2]])
    register_employee = AsyncMock(
        return_value=(
            SimpleNamespace(emp_id=1, name="Alice", emp_code="EMP-1"),
            None,
        )
    )
    monkeypatch.setattr(extractor, "extract_embeddings_from_bytes", extract_embeddings)
    monkeypatch.setattr(employees_api, "register_employee", register_employee)
    monkeypatch.setattr(employees_api, "get_qdrant_client", lambda: object())

    response = client.post(
        "/api/employees",
        data={"name": "Alice", "emp_code": "EMP-1"},
        files={"file": ("face.jpg", b"image", "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "message": "Registered Alice (EMP-1)",
        "employee": {"emp_id": 1, "name": "Alice", "emp_code": "EMP-1"},
    }
    register_employee.assert_awaited_once()
