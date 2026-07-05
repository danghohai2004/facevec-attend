from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

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


@pytest.fixture
def api_key_headers(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-key")
    return {"X-API-Key": "test-key"}


def test_register_requires_api_key(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "test-key")
    extract_embeddings = AsyncMock()
    monkeypatch.setattr(extractor, "extract_embeddings_from_bytes", extract_embeddings)

    response = client.post(
        "/api/employees",
        data={"name": "Alice", "emp_code": "EMP-1"},
        files={"files": ("face.jpg", b"image", "image/jpeg")},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid or missing API key"}
    extract_embeddings.assert_not_awaited()


def test_register_rejects_upload_larger_than_five_megabytes(
    client,
    monkeypatch,
    api_key_headers,
):
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
        files={"files": ("face.jpg", b"x" * (5 * 1024 * 1024 + 1), "image/jpeg")},
        headers=api_key_headers,
    )

    assert response.status_code == 413
    assert response.json() == {"detail": "Ảnh quá lớn (tối đa 5MB)."}
    extract_embeddings.assert_not_awaited()
    register_employee.assert_not_awaited()


def test_register_rejects_when_no_frame_has_exactly_one_face(
    client, monkeypatch, api_key_headers
):
    # A frame with 0 or 2+ faces is skipped, not fatal. If NONE of the frames
    # yield exactly one face, there's nothing to enroll → 400.
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
        files={"files": ("faces.jpg", b"image", "image/jpeg")},
        headers=api_key_headers,
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Không nhận được khuôn mặt hợp lệ nào. Vui lòng thử lại."
    }
    register_employee.assert_not_awaited()


def test_register_keeps_only_frames_with_one_face(client, monkeypatch, api_key_headers):
    # Three frames: one face, no face, one face → the two usable embeddings are
    # collected and passed to register_employee; the empty frame is dropped.
    extract_embeddings = AsyncMock(side_effect=[[[0.1, 0.2]], [], [[0.3, 0.4]]])
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
        files=[
            ("files", ("a.jpg", b"image", "image/jpeg")),
            ("files", ("b.jpg", b"image", "image/jpeg")),
            ("files", ("c.jpg", b"image", "image/jpeg")),
        ],
        headers=api_key_headers,
    )

    assert response.status_code == 200
    register_employee.assert_awaited_once()
    # embeddings is the last positional arg to register_employee(db, qdrant, name, emp_code, embeddings)
    assert register_employee.await_args.args[-1] == [[0.1, 0.2], [0.3, 0.4]]


def test_register_accepts_image_with_one_face(client, monkeypatch, api_key_headers):
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
        files={"files": ("face.jpg", b"image", "image/jpeg")},
        headers=api_key_headers,
    )

    assert response.status_code == 200
    assert response.json() == {
        "message": "Registered Alice (EMP-1)",
        "employee": {"emp_id": 1, "name": "Alice", "emp_code": "EMP-1"},
    }
    register_employee.assert_awaited_once()


def test_register_returns_conflict_for_duplicate_emp_code(
    client,
    monkeypatch,
    api_key_headers,
):
    extract_embeddings = AsyncMock(return_value=[[0.1, 0.2]])
    register_employee = AsyncMock(return_value=(None, "EMPLOYEE_DUPLICATE"))
    monkeypatch.setattr(extractor, "extract_embeddings_from_bytes", extract_embeddings)
    monkeypatch.setattr(employees_api, "register_employee", register_employee)
    monkeypatch.setattr(employees_api, "get_qdrant_client", lambda: object())

    response = client.post(
        "/api/employees",
        data={"name": "Alice", "emp_code": "EMP-1"},
        files={"files": ("face.jpg", b"image", "image/jpeg")},
        headers=api_key_headers,
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "Employee code already exists."}
    register_employee.assert_awaited_once()


def test_list_hides_internal_database_error(client, caplog):
    db = MagicMock()
    db.execute = AsyncMock(
        side_effect=RuntimeError("postgresql://admin:secret@db/internal")
    )

    async def override_get_db():
        yield db

    client.app.dependency_overrides[employees_api.get_db] = override_get_db

    with caplog.at_level("ERROR", logger="src.modules.employees.service"):
        response = client.get("/api/employees")

    assert response.status_code == 500
    assert response.json() == {"detail": "Lỗi hệ thống"}
    assert any(record.exc_info is not None for record in caplog.records)
