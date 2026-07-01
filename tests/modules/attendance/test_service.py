from datetime import datetime, date, time
from unittest.mock import AsyncMock, MagicMock
import pytest
import src.modules.employees.models  # noqa: F401 — ensure Employee in SQLAlchemy registry before AttendanceLog
from src.modules.attendance.service import (
    _is_time_in_range, _normalize_shifts_time, check_in, check_out,
)


# --- Test _is_time_in_range ---

def test_time_in_range_normal():
    assert _is_time_in_range(time(9, 0), time(8, 0), time(10, 0)) is True


def test_time_outside_range():
    assert _is_time_in_range(time(11, 0), time(8, 0), time(10, 0)) is False


def test_time_in_range_overnight():
    # overnight shift: 22:00 - 06:00
    assert _is_time_in_range(time(23, 0), time(22, 0), time(6, 0)) is True
    assert _is_time_in_range(time(5, 0), time(22, 0), time(6, 0)) is True
    assert _is_time_in_range(time(12, 0), time(22, 0), time(6, 0)) is False


# --- Test working_date scope (fix #1 & #2) ---

@pytest.mark.asyncio
async def test_check_in_scopes_by_working_date():
    """check_in must filter by working_date=today, not only checkout_time IS NULL"""
    db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.first.return_value = None  # no record today
    db.execute = AsyncMock(return_value=mock_result)
    db.add = MagicMock()
    db.commit = AsyncMock()
    db.refresh = AsyncMock()

    today = date.today()
    now = datetime.combine(today, time(9, 0))
    log, err = await check_in(db, emp_id=1, now=now)

    assert err is None
    # Verify query includes working_date filter — compile Select to SQL text
    call_args = str(db.execute.call_args[0][0])
    assert "working_date" in call_args


@pytest.mark.asyncio
async def test_check_out_scopes_by_working_date():
    """check_out must filter by working_date=today"""
    db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None  # no open log today
    db.execute = AsyncMock(return_value=mock_result)

    today = date.today()
    now = datetime.combine(today, time(17, 30))
    log, err = await check_out(db, emp_id=1, now=now)

    assert log is None
    assert err == "Check in not found to check out"
    # Verify query includes working_date filter — compile Select to SQL text
    call_args = str(db.execute.call_args[0][0])
    assert "working_date" in call_args
