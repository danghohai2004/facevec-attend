from unittest.mock import AsyncMock, MagicMock

import pytest

import src.modules.attendance.models  # noqa: F401 — ensure AttendanceLog is in SQLAlchemy registry
from src.modules.employees.models import Employee
from src.modules.employees.service import register_employee


@pytest.mark.asyncio
async def test_register_employee_rejects_duplicate_without_overwriting_identity():
    existing_employee = Employee(emp_id=1, name="Alice", emp_code="EMP-1")
    query_result = MagicMock()
    query_result.scalar_one_or_none.return_value = existing_employee

    db = MagicMock()
    db.execute = AsyncMock(return_value=query_result)
    db.flush = AsyncMock()
    db.commit = AsyncMock()
    db.refresh = AsyncMock()
    db.rollback = AsyncMock()

    qdrant = MagicMock()
    qdrant.delete = AsyncMock()
    qdrant.upsert = AsyncMock()

    employee, err = await register_employee(
        db,
        qdrant,
        name="Mallory",
        emp_code="EMP-1",
        embeddings=[[0.1, 0.2]],
    )

    assert employee is None
    assert err == "EMPLOYEE_DUPLICATE"
    assert existing_employee.name == "Alice"
    db.add.assert_not_called()
    db.flush.assert_not_awaited()
    db.commit.assert_not_awaited()
    db.refresh.assert_not_awaited()
    qdrant.delete.assert_not_awaited()
    qdrant.upsert.assert_not_awaited()
