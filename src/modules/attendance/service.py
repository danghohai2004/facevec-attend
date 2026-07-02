import logging
from datetime import datetime, time, date
from zoneinfo import ZoneInfo

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.modules.attendance.models import AttendanceLog, ShiftSettings

logger = logging.getLogger(__name__)

INTERNAL_ERROR = "Lỗi hệ thống"
BUSINESS_TIMEZONE = ZoneInfo("Asia/Ho_Chi_Minh")
_DEFAULT_SHIFT = {
    "check_in_start": time(8, 0),
    "check_in_end": time(10, 0),
    "check_out_start": time(17, 0),
    "check_out_end": time(19, 0),
}


def _as_business_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=BUSINESS_TIMEZONE)
    return value.astimezone(BUSINESS_TIMEZONE)


def _to_database_datetime(value: datetime) -> datetime:
    return _as_business_datetime(value).replace(tzinfo=None)


def _is_time_in_range(current: time, start: time, end: time) -> bool:
    if start <= end:
        return start <= current <= end
    return current >= start or current <= end  # overnight shift


def _normalize_shifts_time(shifts) -> dict:
    """Accept ShiftSettings ORM object or plain dict."""
    if isinstance(shifts, dict):
        return shifts
    return {
        "check_in_start": shifts.check_in_start,
        "check_in_end": shifts.check_in_end,
        "check_out_start": shifts.check_out_start,
        "check_out_end": shifts.check_out_end,
    }


async def get_shift_settings(db: AsyncSession) -> tuple[ShiftSettings | dict, str | None]:
    try:
        result = await db.execute(select(ShiftSettings).order_by(ShiftSettings.id).limit(1))
        settings = result.scalar_one_or_none()
        return settings if settings else _DEFAULT_SHIFT, None
    except Exception:
        logger.exception("Loading shift settings failed")
        return None, INTERNAL_ERROR


async def upsert_shift_settings(db: AsyncSession, data: dict) -> tuple[ShiftSettings, str | None]:
    try:
        result = await db.execute(select(ShiftSettings).order_by(ShiftSettings.id).limit(1))
        settings = result.scalar_one_or_none()
        if settings is None:
            settings = ShiftSettings(**data)
            db.add(settings)
        else:
            for k, v in data.items():
                setattr(settings, k, v)
        await db.commit()
        await db.refresh(settings)
        return settings, None
    except Exception:
        logger.exception("Updating shift settings failed")
        await db.rollback()
        return None, INTERNAL_ERROR


def get_current_time(shifts) -> tuple[bool, datetime, str | None]:
    shifts = _normalize_shifts_time(shifts)
    now = datetime.now(BUSINESS_TIMEZONE)
    t = now.time()
    if _is_time_in_range(t, shifts["check_in_start"], shifts["check_in_end"]):
        return True, now, "check_in"
    if _is_time_in_range(t, shifts["check_out_start"], shifts["check_out_end"]):
        return True, now, "check_out"
    return False, now, None


async def check_in(
    db: AsyncSession,
    emp_id: int,
    now: datetime | None = None,
) -> tuple[AttendanceLog | None, str | None]:
    business_now = _as_business_datetime(
        now or datetime.now(BUSINESS_TIMEZONE)
    )
    working_date = business_now.date()
    database_now = _to_database_datetime(business_now)

    # Fix #1: scope by working_date — yesterday's unclosed log must not block today
    result = await db.execute(
        select(AttendanceLog).filter(
            AttendanceLog.emp_id == emp_id,
            AttendanceLog.working_date == working_date,
            AttendanceLog.checkout_time.is_(None),
        ).limit(1)
    )
    if result.scalars().first():
        return None, "Already checked in"

    log = AttendanceLog(
        emp_id=emp_id,
        working_date=working_date,
        checkin_time=database_now,
    )
    db.add(log)
    await db.commit()
    await db.refresh(log)
    return log, None


async def check_out(
    db: AsyncSession,
    emp_id: int,
    now: datetime | None = None,
) -> tuple[AttendanceLog | None, str | None]:
    business_now = _as_business_datetime(
        now or datetime.now(BUSINESS_TIMEZONE)
    )
    working_date = business_now.date()
    database_now = _to_database_datetime(business_now)

    # Fix #2: scope by working_date — must not close yesterday's log with today's timestamp
    result = await db.execute(
        select(AttendanceLog).filter(
            AttendanceLog.emp_id == emp_id,
            AttendanceLog.working_date == working_date,
            AttendanceLog.checkout_time.is_(None),
        ).order_by(AttendanceLog.checkin_time.desc()).limit(1)
    )
    log = result.scalar_one_or_none()
    if log is None:
        return None, "Check in not found to check out"

    log.checkout_time = database_now
    await db.commit()
    await db.refresh(log)
    return log, None


async def log_attendance(db: AsyncSession, emp_id: int) -> str:
    """Fix #3: loads shifts from DB internally — caller must NOT pass shifts_time."""
    shifts, err = await get_shift_settings(db)
    if err:
        return err

    within, now, check_type = get_current_time(shifts)
    if not within:
        return "Not during working hours"

    if check_type == "check_in":
        _, err = await check_in(db, emp_id, now=now)
    else:
        _, err = await check_out(db, emp_id, now=now)

    if err:
        return err
    return "Check in successful" if check_type == "check_in" else "Check out successful"


async def list_attendance_logs(
    db: AsyncSession,
    emp_id: int,
    page: int,
    page_size: int,
    from_date: date | None = None,
    to_date: date | None = None,
) -> tuple[list[AttendanceLog], int, str | None]:
    try:
        filters = [AttendanceLog.emp_id == emp_id]
        if from_date:
            filters.append(AttendanceLog.working_date >= from_date)
        if to_date:
            filters.append(AttendanceLog.working_date <= to_date)

        total = (await db.execute(
            select(func.count()).select_from(AttendanceLog).filter(*filters)
        )).scalar_one()

        result = await db.execute(
            select(AttendanceLog).filter(*filters)
            .order_by(AttendanceLog.checkin_time.desc())
            .offset((page - 1) * page_size).limit(page_size)
        )
        return result.scalars().all(), total, None
    except Exception:
        logger.exception("Listing attendance logs failed")
        return [], 0, INTERNAL_ERROR
