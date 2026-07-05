import logging
from datetime import date, datetime, time, timedelta
from io import BytesIO
from zoneinfo import ZoneInfo

from openpyxl import Workbook
from sqlalchemy import Time, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.modules.attendance.models import AttendanceLog, ShiftSettings
from src.modules.attendance.schemas import (
    DailyStatItem,
    DailyStatsResponse,
    MonthlyStatItem,
    MonthlyStatsResponse,
    SummaryDeltas,
    SummaryStatsResponse,
)
from src.modules.employees.models import Employee

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


def _working_hours_expression():
    return func.extract("epoch", AttendanceLog.working_duration) / 3600.0


def _percent_change(current: float, previous: float) -> float | None:
    if previous == 0:
        return None
    return (current - previous) / previous * 100


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


async def manual_check_in(
    db: AsyncSession,
    emp_id: int,
) -> tuple[AttendanceLog | None, str | None]:
    shifts, err = await get_shift_settings(db)
    if err:
        return None, err

    within, now, check_type = get_current_time(shifts)
    if not within or check_type != "check_in":
        return None, "Ngoài khung giờ check-in."
    return await check_in(db, emp_id, now=now)


async def manual_check_out(
    db: AsyncSession,
    emp_id: int,
) -> tuple[AttendanceLog | None, str | None]:
    shifts, err = await get_shift_settings(db)
    if err:
        return None, err

    within, now, check_type = get_current_time(shifts)
    if not within or check_type != "check_out":
        return None, "Ngoài khung giờ check-out."
    return await check_out(db, emp_id, now=now)


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


async def get_summary_stats(
    db: AsyncSession,
) -> tuple[SummaryStatsResponse | None, str | None]:
    try:
        shifts, err = await get_shift_settings(db)
        if err:
            return None, err
        shift = _normalize_shifts_time(shifts)

        today = datetime.now(BUSINESS_TIMEZONE).date()
        yesterday = today - timedelta(days=1)

        total_employees = (
            await db.execute(select(func.count()).select_from(Employee))
        ).scalar_one()

        hours = _working_hours_expression()
        result = await db.execute(
            select(
                AttendanceLog.working_date,
                func.count().label("attendance"),
                func.avg(hours).label("average_hours"),
                func.count()
                .filter(
                    cast(AttendanceLog.checkin_time, Time)
                    <= shift["check_in_end"]
                )
                .label("on_time"),
            )
            .where(AttendanceLog.working_date.in_((yesterday, today)))
            .group_by(AttendanceLog.working_date)
        )
        stats_by_date = {row.working_date: row for row in result.all()}

        def values_for(day: date) -> tuple[int, float, float]:
            row = stats_by_date.get(day)
            if row is None:
                return 0, 0.0, 0.0
            attendance = int(row.attendance)
            average_hours = float(row.average_hours or 0.0)
            on_time_rate = (
                float(row.on_time) / attendance * 100
                if attendance
                else 0.0
            )
            return attendance, average_hours, on_time_rate

        today_attendance, today_average, today_on_time = values_for(today)
        yesterday_attendance, yesterday_average, yesterday_on_time = (
            values_for(yesterday)
        )

        return SummaryStatsResponse(
            total_employees=total_employees,
            todays_attendance=today_attendance,
            average_working_hours=today_average,
            on_time_rate=today_on_time,
            deltas=SummaryDeltas(
                todays_attendance=_percent_change(
                    today_attendance,
                    yesterday_attendance,
                ),
                average_working_hours=_percent_change(
                    today_average,
                    yesterday_average,
                ),
                on_time_rate=_percent_change(
                    today_on_time,
                    yesterday_on_time,
                ),
            ),
        ), None
    except Exception:
        logger.exception("Loading attendance summary statistics failed")
        return None, INTERNAL_ERROR


async def get_monthly_stats(
    db: AsyncSession,
    year: int,
) -> tuple[MonthlyStatsResponse | None, str | None]:
    try:
        year_expression = func.extract("year", AttendanceLog.working_date)
        years_result = await db.execute(
            select(year_expression.label("year"))
            .distinct()
            .order_by(year_expression)
        )
        available_years = [
            int(value) for value in years_result.scalars().all()
        ]

        month_expression = func.extract("month", AttendanceLog.working_date)
        hours = _working_hours_expression()
        result = await db.execute(
            select(
                month_expression.label("month"),
                func.count().label("attendance"),
                func.sum(hours).label("working_hours"),
                func.avg(hours).label("average_hours"),
            )
            .where(year_expression == year)
            .group_by(month_expression)
            .order_by(month_expression)
        )
        items = [
            MonthlyStatItem(
                month=int(row.month),
                attendance=int(row.attendance),
                working_hours=float(row.working_hours or 0.0),
                average_hours=round(float(row.average_hours or 0.0), 1),
            )
            for row in result.all()
        ]
        return MonthlyStatsResponse(
            available_years=available_years,
            items=items,
        ), None
    except Exception:
        logger.exception("Loading monthly attendance statistics failed")
        return None, INTERNAL_ERROR


def build_report_workbook(employees, logs, check_in_end: time) -> bytes:
    """Pure workbook builder — Summary sheet (one row per employee, including
    zero-attendance ones) + Detail sheet (one row per log)."""
    by_emp = {e.emp_id: e for e in employees}
    stats = {e.emp_id: {"days": set(), "hours": 0.0, "late": 0} for e in employees}
    for log in logs:
        s = stats.get(log.emp_id)
        if s is None:
            continue
        s["days"].add(log.working_date)
        if log.working_duration is not None:
            s["hours"] += log.working_duration.total_seconds() / 3600
        if log.checkin_time.time() > check_in_end:
            s["late"] += 1

    wb = Workbook()
    summary = wb.active
    summary.title = "Summary"
    summary.append(["Employee Code", "Name", "Days Worked", "Total Hours", "Late Count"])
    for e in employees:
        s = stats[e.emp_id]
        summary.append([e.emp_code, e.name, len(s["days"]), round(s["hours"], 2), s["late"]])

    detail = wb.create_sheet("Detail")
    detail.append(["Employee Code", "Name", "Date", "Check-in", "Check-out", "Hours"])
    for log in logs:
        e = by_emp.get(log.emp_id)
        if e is None:
            continue
        detail.append([
            e.emp_code,
            e.name,
            log.working_date.isoformat(),
            log.checkin_time.strftime("%H:%M:%S"),
            log.checkout_time.strftime("%H:%M:%S") if log.checkout_time else "",
            round(log.working_duration.total_seconds() / 3600, 2)
            if log.working_duration is not None
            else "",
        ])

    buffer = BytesIO()
    wb.save(buffer)
    return buffer.getvalue()


async def get_monthly_report(
    db: AsyncSession,
    year: int,
    month: int,
) -> tuple[bytes | None, str | None]:
    try:
        shifts, err = await get_shift_settings(db)
        if err:
            return None, err
        shift = _normalize_shifts_time(shifts)

        employees = (
            await db.execute(select(Employee).order_by(Employee.emp_code))
        ).scalars().all()
        # ponytail: no pagination — kiosk-scale data, hundreds of logs/month
        logs = (
            await db.execute(
                select(AttendanceLog)
                .where(
                    func.extract("year", AttendanceLog.working_date) == year,
                    func.extract("month", AttendanceLog.working_date) == month,
                )
                .order_by(AttendanceLog.emp_id, AttendanceLog.working_date)
            )
        ).scalars().all()

        return build_report_workbook(employees, logs, shift["check_in_end"]), None
    except Exception:
        logger.exception("Building monthly attendance report failed")
        return None, INTERNAL_ERROR


async def get_daily_stats(
    db: AsyncSession,
    year: int,
    month: int,
) -> tuple[DailyStatsResponse | None, str | None]:
    try:
        year_expression = func.extract("year", AttendanceLog.working_date)
        month_expression = func.extract("month", AttendanceLog.working_date)
        day_expression = func.extract("day", AttendanceLog.working_date)
        hours = _working_hours_expression()
        result = await db.execute(
            select(
                day_expression.label("day"),
                func.avg(hours).label("average_hours"),
            )
            .where(
                year_expression == year,
                month_expression == month,
            )
            .group_by(day_expression)
            .order_by(day_expression)
        )
        items = [
            DailyStatItem(
                day=int(row.day),
                average_hours=round(float(row.average_hours or 0.0), 2),
            )
            for row in result.all()
        ]
        return DailyStatsResponse(items=items), None
    except Exception:
        logger.exception("Loading daily attendance statistics failed")
        return None, INTERNAL_ERROR
