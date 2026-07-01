from datetime import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from src.modules.attendance import api as attendance_api
from src.modules.employees import api as employees_api
from src.modules.recognition.ws_ingress import make_ws_router
from src.platform.auth import require_api_key
from src.platform.realtime.manager import ConnectionManager


@pytest.fixture
def protected_client():
    app = FastAPI()

    @app.get("/protected", dependencies=[Depends(require_api_key)])
    async def protected():
        return {"status": "ok"}

    with TestClient(app) as client:
        yield client


def test_unconfigured_api_key_fails_closed(monkeypatch, protected_client):
    monkeypatch.delenv("API_KEY", raising=False)

    response = protected_client.get("/protected")

    assert response.status_code == 503
    assert response.json() == {"detail": "Server auth not configured"}


def test_empty_api_key_fails_closed(monkeypatch, protected_client):
    monkeypatch.setenv("API_KEY", "")

    response = protected_client.get("/protected")

    assert response.status_code == 503
    assert response.json() == {"detail": "Server auth not configured"}


def test_missing_api_key_is_rejected(monkeypatch, protected_client):
    monkeypatch.setenv("API_KEY", "test-key")

    response = protected_client.get("/protected")

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid or missing API key"}


def test_incorrect_api_key_is_rejected(monkeypatch, protected_client):
    monkeypatch.setenv("API_KEY", "test-key")

    response = protected_client.get(
        "/protected",
        headers={"X-API-Key": "wrong-key"},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid or missing API key"}


def test_non_ascii_incorrect_api_key_is_rejected(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-key")

    with pytest.raises(HTTPException) as exc_info:
        require_api_key("khóa-sai")

    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "Invalid or missing API key"


def test_correct_api_key_is_accepted(monkeypatch, protected_client):
    monkeypatch.setenv("API_KEY", "test-key")

    response = protected_client.get(
        "/protected",
        headers={"X-API-Key": "test-key"},
    )

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.parametrize(
    ("router", "method", "path"),
    [
        (employees_api.router, "POST", "/api/employees"),
        (employees_api.router, "DELETE", "/api/employees"),
        (attendance_api.router, "PUT", "/api/shift-settings"),
        (attendance_api.router, "POST", "/api/attendance/checkin"),
        (attendance_api.router, "POST", "/api/attendance/checkout"),
    ],
)
def test_write_route_requires_api_key(router, method, path):
    route = next(
        route
        for route in router.routes
        if isinstance(route, APIRoute)
        and route.path == path
        and method in route.methods
    )

    assert require_api_key in [
        dependency.call for dependency in route.dependant.dependencies
    ]


def test_public_get_endpoints_work_without_api_key(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-key")
    employee = SimpleNamespace(emp_id=1, name="Alice", emp_code="EMP-1")
    shifts = SimpleNamespace(
        check_in_start=time(8),
        check_in_end=time(9),
        check_out_start=time(17),
        check_out_end=time(18),
    )
    monkeypatch.setattr(
        employees_api,
        "list_employees",
        AsyncMock(return_value=([], 0, None)),
    )
    monkeypatch.setattr(
        employees_api,
        "get_employee",
        AsyncMock(return_value=(employee, None)),
    )
    monkeypatch.setattr(
        attendance_api,
        "get_employee",
        AsyncMock(return_value=(employee, None)),
    )
    monkeypatch.setattr(
        attendance_api,
        "list_attendance_logs",
        AsyncMock(return_value=([], 0, None)),
    )
    monkeypatch.setattr(
        attendance_api,
        "get_shift_settings",
        AsyncMock(return_value=(shifts, None)),
    )

    app = FastAPI()
    app.include_router(employees_api.router)
    app.include_router(attendance_api.router)

    async def override_get_db():
        yield object()

    app.dependency_overrides[employees_api.get_db] = override_get_db

    with TestClient(app) as client:
        responses = [
            client.get("/api/employees"),
            client.get("/api/employees/1"),
            client.get("/api/attendance", params={"emp_id": 1}),
            client.get("/api/shift-settings"),
        ]

    assert [response.status_code for response in responses] == [200, 200, 200, 200]


def test_recognition_websocket_stays_public():
    class CaptureQueue:
        def __init__(self):
            self.items = []

        async def put(self, item):
            self.items.append(item)

    queue = CaptureQueue()
    app = FastAPI()
    app.include_router(make_ws_router(queue, ConnectionManager()))

    with TestClient(app) as client:
        with client.websocket_connect("/ws/recognition/browser-1") as websocket:
            websocket.send_bytes(b"frame")

    assert len(queue.items) == 1
    assert queue.items[0].client_id == "browser-1"
    assert queue.items[0].frame == b"frame"
