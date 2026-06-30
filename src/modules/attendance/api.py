from datetime import date
from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from src.platform.db.session import get_db
from src.modules.attendance.schemas import (
    ShiftsTime, AttendanceLogOut, AttendanceHistoryResponse, AttendanceCheckResponse,
)
from src.modules.attendance.service import (
    get_shift_settings, upsert_shift_settings, check_in, check_out, list_attendance_logs,
)
from src.modules.employees.service import get_employee, ERR_NOT_FOUND

router = APIRouter(prefix="/api", tags=["Attendance"])


@router.post("/attendance/checkin", response_model=AttendanceCheckResponse)
async def api_checkin(emp_id: int = Query(...), db: AsyncSession = Depends(get_db)):
    # Fix #3: no shifts_time from client — service loads from DB
    employee, err = await get_employee(db, emp_id)
    if err:
        raise HTTPException(404 if err == ERR_NOT_FOUND else 500, err)
    log, err = await check_in(db, emp_id)
    if err:
        raise HTTPException(400, err)
    return AttendanceCheckResponse(
        message="Check in successful", check_type="check_in",
        log=AttendanceLogOut.model_validate(log),
    )


@router.post("/attendance/checkout", response_model=AttendanceCheckResponse)
async def api_checkout(emp_id: int = Query(...), db: AsyncSession = Depends(get_db)):
    employee, err = await get_employee(db, emp_id)
    if err:
        raise HTTPException(404 if err == ERR_NOT_FOUND else 500, err)
    log, err = await check_out(db, emp_id)
    if err:
        raise HTTPException(400, err)
    return AttendanceCheckResponse(
        message="Check out successful", check_type="check_out",
        log=AttendanceLogOut.model_validate(log),
    )


@router.get("/attendance", response_model=AttendanceHistoryResponse)
async def api_history(
    emp_id: int = Query(...),
    from_date: date | None = Query(None),
    to_date: date | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    employee, err = await get_employee(db, emp_id)
    if err:
        raise HTTPException(404 if err == ERR_NOT_FOUND else 500, err)
    logs, total, err = await list_attendance_logs(db, emp_id, page, page_size, from_date, to_date)
    if err:
        raise HTTPException(500, err)
    return AttendanceHistoryResponse(
        items=[AttendanceLogOut.model_validate(l) for l in logs],
        page=page, page_size=page_size, total=total,
    )


@router.get("/shift-settings", response_model=ShiftsTime)
async def api_shift_get(db: AsyncSession = Depends(get_db)):
    settings, err = await get_shift_settings(db)
    if err:
        raise HTTPException(500, err)
    return ShiftsTime.model_validate(settings)


@router.put("/shift-settings", response_model=ShiftsTime)
async def api_shift_update(payload: ShiftsTime, db: AsyncSession = Depends(get_db)):
    settings, err = await upsert_shift_settings(db, payload.model_dump())
    if err:
        raise HTTPException(500, err)
    return ShiftsTime.model_validate(settings)
