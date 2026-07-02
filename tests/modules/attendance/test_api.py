from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.modules.attendance import api as attendance_api


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
