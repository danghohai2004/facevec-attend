from datetime import date, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.modules.attendance import api as attendance_api


def make_attendance_log():
    return SimpleNamespace(
        log_id=10,
        emp_id=1,
        working_date=date(2026, 7, 2),
        checkin_time=datetime(2026, 7, 2, 8, 30),
        checkout_time=None,
        working_duration=None,
    )


def test_history_rejects_reversed_date_range_before_database_work(monkeypatch):
    app = FastAPI()
    app.include_router(attendance_api.router)

    async def override_get_db():
        yield object()

    app.dependency_overrides[attendance_api.get_db] = override_get_db
    get_employee = AsyncMock(return_value=(object(), None))
    list_attendance_logs = AsyncMock(return_value=([], 0, None))
    monkeypatch.setattr(attendance_api, "get_employee", get_employee)
    monkeypatch.setattr(
        attendance_api,
        "list_attendance_logs",
        list_attendance_logs,
    )

    with TestClient(app) as client:
        response = client.get(
            "/api/attendance",
            params={
                "emp_id": 1,
                "from_date": "2026-07-02",
                "to_date": "2026-07-01",
            },
        )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "from_date must be before or equal to to_date."
    }
    get_employee.assert_not_awaited()
    list_attendance_logs.assert_not_awaited()


@pytest.mark.parametrize(
    ("path", "wrapper_name", "check_type", "message"),
    [
        (
            "/api/attendance/checkin",
            "manual_check_in",
            "check_in",
            "Check in successful",
        ),
        (
            "/api/attendance/checkout",
            "manual_check_out",
            "check_out",
            "Check out successful",
        ),
    ],
)
def test_attendance_write_preserves_log_response(
    monkeypatch,
    path,
    wrapper_name,
    check_type,
    message,
):
    app = FastAPI()
    app.include_router(attendance_api.router)

    async def override_get_db():
        yield object()

    app.dependency_overrides[attendance_api.get_db] = override_get_db
    app.dependency_overrides[attendance_api.require_api_key] = lambda: None
    monkeypatch.setattr(
        attendance_api,
        "get_employee",
        AsyncMock(return_value=(object(), None)),
    )
    wrapper = AsyncMock(return_value=(make_attendance_log(), None))
    monkeypatch.setattr(attendance_api, wrapper_name, wrapper)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(path, params={"emp_id": 1})

    assert response.status_code == 200
    assert response.json() == {
        "message": message,
        "check_type": check_type,
        "log": {
            "log_id": 10,
            "emp_id": 1,
            "working_date": "2026-07-02",
            "checkin_time": "2026-07-02T08:30:00",
            "checkout_time": None,
            "working_duration": None,
        },
    }
    wrapper.assert_awaited_once()


@pytest.mark.parametrize(
    ("path", "api_key"),
    [
        ("/api/attendance/checkin", None),
        ("/api/attendance/checkin", "wrong-key"),
        ("/api/attendance/checkout", None),
        ("/api/attendance/checkout", "wrong-key"),
    ],
)
def test_attendance_write_rejects_missing_or_wrong_api_key(
    monkeypatch,
    path,
    api_key,
):
    monkeypatch.setenv("API_KEY", "test-key")
    app = FastAPI()
    app.include_router(attendance_api.router)

    headers = {"X-API-Key": api_key} if api_key else {}
    with TestClient(app) as client:
        response = client.post(path, params={"emp_id": 1}, headers=headers)

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid or missing API key"}
