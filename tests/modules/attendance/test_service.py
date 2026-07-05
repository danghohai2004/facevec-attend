from datetime import datetime, date, time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from zoneinfo import ZoneInfo

import pytest
import src.modules.employees.models  # noqa: F401 — ensure Employee in SQLAlchemy registry before AttendanceLog
from src.modules.attendance import service as attendance_service
from src.modules.attendance.service import (
    _is_time_in_range, _normalize_shifts_time, check_in, check_out,
    get_current_time, get_daily_stats, get_monthly_stats, get_summary_stats,
    log_attendance,
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


@pytest.mark.asyncio
async def test_log_attendance_hides_internal_database_error(caplog):
    db = MagicMock()
    db.execute = AsyncMock(
        side_effect=RuntimeError("postgresql://admin:secret@db/internal")
    )

    with caplog.at_level("ERROR", logger="src.modules.attendance.service"):
        result = await log_attendance(db, emp_id=1)

    assert result == "Lỗi hệ thống"
    assert any(record.exc_info is not None for record in caplog.records)


def test_get_current_time_uses_aware_business_timezone_for_overnight_shift(
    monkeypatch,
):
    business_timezone = ZoneInfo("Asia/Ho_Chi_Minh")
    fixed_now = datetime(2026, 7, 1, 23, 30, tzinfo=business_timezone)

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            assert tz == business_timezone
            return fixed_now

    monkeypatch.setattr(attendance_service, "datetime", FixedDateTime)
    shifts = {
        "check_in_start": time(22, 0),
        "check_in_end": time(6, 0),
        "check_out_start": time(7, 0),
        "check_out_end": time(8, 0),
    }

    within, now, check_type = get_current_time(shifts)

    assert within is True
    assert check_type == "check_in"
    assert now == fixed_now
    assert now.utcoffset().total_seconds() == 7 * 60 * 60


@pytest.mark.asyncio
async def test_manual_check_in_inside_window_preserves_log_and_timestamp(monkeypatch):
    db = MagicMock()
    shifts = object()
    now = datetime(2026, 7, 2, 8, 30, tzinfo=ZoneInfo("Asia/Ho_Chi_Minh"))
    log = object()
    monkeypatch.setattr(
        attendance_service,
        "get_shift_settings",
        AsyncMock(return_value=(shifts, None)),
    )
    get_current_time_mock = MagicMock(return_value=(True, now, "check_in"))
    monkeypatch.setattr(
        attendance_service,
        "get_current_time",
        get_current_time_mock,
    )
    check_in_mock = AsyncMock(return_value=(log, None))
    monkeypatch.setattr(attendance_service, "check_in", check_in_mock)

    result = await attendance_service.manual_check_in(db, emp_id=1)

    assert result == (log, None)
    get_current_time_mock.assert_called_once_with(shifts)
    check_in_mock.assert_awaited_once_with(db, 1, now=now)


@pytest.mark.asyncio
async def test_manual_check_in_outside_check_in_window_is_rejected(monkeypatch):
    db = MagicMock()
    db.add = MagicMock()
    monkeypatch.setattr(
        attendance_service,
        "get_shift_settings",
        AsyncMock(return_value=(object(), None)),
    )
    monkeypatch.setattr(
        attendance_service,
        "get_current_time",
        MagicMock(return_value=(True, datetime.now(), "check_out")),
    )
    check_in_mock = AsyncMock()
    monkeypatch.setattr(attendance_service, "check_in", check_in_mock)

    result = await attendance_service.manual_check_in(db, emp_id=1)

    assert result == (None, "Ngoài khung giờ check-in.")
    check_in_mock.assert_not_awaited()
    db.add.assert_not_called()


@pytest.mark.asyncio
async def test_manual_check_out_inside_window_preserves_log_and_timestamp(monkeypatch):
    db = MagicMock()
    shifts = object()
    now = datetime(2026, 7, 2, 17, 30, tzinfo=ZoneInfo("Asia/Ho_Chi_Minh"))
    log = object()
    monkeypatch.setattr(
        attendance_service,
        "get_shift_settings",
        AsyncMock(return_value=(shifts, None)),
    )
    get_current_time_mock = MagicMock(return_value=(True, now, "check_out"))
    monkeypatch.setattr(
        attendance_service,
        "get_current_time",
        get_current_time_mock,
    )
    check_out_mock = AsyncMock(return_value=(log, None))
    monkeypatch.setattr(attendance_service, "check_out", check_out_mock)

    result = await attendance_service.manual_check_out(db, emp_id=1)

    assert result == (log, None)
    get_current_time_mock.assert_called_once_with(shifts)
    check_out_mock.assert_awaited_once_with(db, 1, now=now)


@pytest.mark.asyncio
async def test_manual_check_out_outside_check_out_window_is_rejected(monkeypatch):
    db = MagicMock()
    monkeypatch.setattr(
        attendance_service,
        "get_shift_settings",
        AsyncMock(return_value=(object(), None)),
    )
    monkeypatch.setattr(
        attendance_service,
        "get_current_time",
        MagicMock(return_value=(False, datetime.now(), None)),
    )
    check_out_mock = AsyncMock()
    monkeypatch.setattr(attendance_service, "check_out", check_out_mock)

    result = await attendance_service.manual_check_out(db, emp_id=1)

    assert result == (None, "Ngoài khung giờ check-out.")
    check_out_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_check_in_persists_naive_local_time_and_business_date():
    db = MagicMock()
    result = MagicMock()
    result.scalars.return_value.first.return_value = None
    db.execute = AsyncMock(return_value=result)
    db.commit = AsyncMock()
    db.refresh = AsyncMock()

    utc_now = datetime(2026, 6, 30, 17, 30, tzinfo=ZoneInfo("UTC"))
    log, err = await check_in(db, emp_id=1, now=utc_now)

    assert err is None
    assert log.working_date == date(2026, 7, 1)
    assert log.checkin_time == datetime(2026, 7, 1, 0, 30)
    assert log.checkin_time.tzinfo is None


@pytest.mark.asyncio
async def test_check_out_persists_naive_local_time_for_comparison():
    log = MagicMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = log
    db = MagicMock()
    db.execute = AsyncMock(return_value=result)
    db.commit = AsyncMock()
    db.refresh = AsyncMock()

    utc_now = datetime(2026, 7, 1, 11, 0, tzinfo=ZoneInfo("UTC"))
    returned_log, err = await check_out(db, emp_id=1, now=utc_now)

    assert err is None
    assert returned_log is log
    assert log.checkout_time == datetime(2026, 7, 1, 18, 0)
    assert log.checkout_time.tzinfo is None


@pytest.mark.asyncio
async def test_get_summary_stats_calculates_today_and_day_over_day_deltas(
    monkeypatch,
):
    fixed_now = datetime(
        2026,
        7,
        3,
        9,
        0,
        tzinfo=attendance_service.BUSINESS_TIMEZONE,
    )

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            assert tz == attendance_service.BUSINESS_TIMEZONE
            return fixed_now

    monkeypatch.setattr(attendance_service, "datetime", FixedDateTime)
    monkeypatch.setattr(
        attendance_service,
        "get_shift_settings",
        AsyncMock(
            return_value=(
                {
                    "check_in_start": time(8, 0),
                    "check_in_end": time(9, 0),
                    "check_out_start": time(17, 0),
                    "check_out_end": time(19, 0),
                },
                None,
            )
        ),
    )

    employee_count_result = MagicMock()
    employee_count_result.scalar_one.return_value = 3
    aggregate_result = MagicMock()
    aggregate_result.all.return_value = [
        SimpleNamespace(
            working_date=date(2026, 7, 2),
            attendance=2,
            average_hours=8.0,
            on_time=2,
        ),
        SimpleNamespace(
            working_date=date(2026, 7, 3),
            attendance=3,
            average_hours=7.5,
            on_time=2,
        ),
    ]
    db = MagicMock()
    db.execute = AsyncMock(
        side_effect=[employee_count_result, aggregate_result]
    )

    response, err = await get_summary_stats(db)

    assert err is None
    assert response.total_employees == 3
    assert response.todays_attendance == 3
    assert response.average_working_hours == pytest.approx(7.5)
    assert response.on_time_rate == pytest.approx(200 / 3)
    assert response.deltas.todays_attendance == pytest.approx(50.0)
    assert response.deltas.average_working_hours == pytest.approx(-6.25)
    assert response.deltas.on_time_rate == pytest.approx(-100 / 3)

    employee_sql = str(db.execute.await_args_list[0].args[0])
    aggregate_sql = str(db.execute.await_args_list[1].args[0])
    assert "employees" in employee_sql
    assert "attendance_logs.working_date" in aggregate_sql
    assert "CAST(attendance_logs.checkin_time AS TIME)" in aggregate_sql


@pytest.mark.asyncio
async def test_get_summary_stats_uses_zero_values_and_null_deltas_without_yesterday(
    monkeypatch,
):
    fixed_now = datetime(
        2026,
        7,
        3,
        9,
        0,
        tzinfo=attendance_service.BUSINESS_TIMEZONE,
    )

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now

    monkeypatch.setattr(attendance_service, "datetime", FixedDateTime)
    monkeypatch.setattr(
        attendance_service,
        "get_shift_settings",
        AsyncMock(return_value=(attendance_service._DEFAULT_SHIFT, None)),
    )

    employee_count_result = MagicMock()
    employee_count_result.scalar_one.return_value = 0
    aggregate_result = MagicMock()
    aggregate_result.all.return_value = []
    db = MagicMock()
    db.execute = AsyncMock(
        side_effect=[employee_count_result, aggregate_result]
    )

    response, err = await get_summary_stats(db)

    assert err is None
    assert response.model_dump() == {
        "total_employees": 0,
        "todays_attendance": 0,
        "average_working_hours": 0.0,
        "on_time_rate": 0.0,
        "deltas": {
            "todays_attendance": None,
            "average_working_hours": None,
            "on_time_rate": None,
        },
    }


@pytest.mark.asyncio
async def test_get_monthly_stats_returns_available_years_and_sparse_months():
    years_result = MagicMock()
    years_result.scalars.return_value.all.return_value = [2025, 2026]
    monthly_result = MagicMock()
    monthly_result.all.return_value = [
        SimpleNamespace(
            month=1,
            attendance=2,
            working_hours=15.25,
            average_hours=7.625,
        ),
        SimpleNamespace(
            month=7,
            attendance=1,
            working_hours=None,
            average_hours=None,
        ),
    ]
    db = MagicMock()
    db.execute = AsyncMock(side_effect=[years_result, monthly_result])

    response, err = await get_monthly_stats(db, year=2026)

    assert err is None
    assert response.model_dump() == {
        "available_years": [2025, 2026],
        "items": [
            {
                "month": 1,
                "attendance": 2,
                "working_hours": 15.25,
                "average_hours": 7.6,
            },
            {
                "month": 7,
                "attendance": 1,
                "working_hours": 0.0,
                "average_hours": 0.0,
            },
        ],
    }

    years_sql = str(db.execute.await_args_list[0].args[0])
    monthly_sql = str(db.execute.await_args_list[1].args[0])
    assert "DISTINCT" in years_sql
    assert "EXTRACT(year FROM attendance_logs.working_date)" in years_sql
    assert "EXTRACT(epoch FROM attendance_logs.working_duration)" in monthly_sql
    assert "GROUP BY EXTRACT(month FROM attendance_logs.working_date)" in monthly_sql


@pytest.mark.asyncio
async def test_get_daily_stats_returns_sparse_days_and_ignores_open_log_hours():
    daily_result = MagicMock()
    daily_result.all.return_value = [
        SimpleNamespace(day=1, average_hours=7.556),
        SimpleNamespace(day=3, average_hours=None),
    ]
    db = MagicMock()
    db.execute = AsyncMock(return_value=daily_result)

    response, err = await get_daily_stats(db, year=2026, month=7)

    assert err is None
    assert response.model_dump() == {
        "items": [
            {"day": 1, "average_hours": 7.56},
            {"day": 3, "average_hours": 0.0},
        ]
    }

    daily_sql = str(db.execute.await_args.args[0])
    assert "EXTRACT(epoch FROM attendance_logs.working_duration)" in daily_sql
    assert "EXTRACT(year FROM attendance_logs.working_date)" in daily_sql
    assert "EXTRACT(month FROM attendance_logs.working_date)" in daily_sql
    assert "GROUP BY EXTRACT(day FROM attendance_logs.working_date)" in daily_sql
    assert "working_duration IS NOT NULL" not in daily_sql
