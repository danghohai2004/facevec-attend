from datetime import date, datetime, time, timedelta
from pydantic import BaseModel, ConfigDict


class ShiftsTime(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    check_in_start: time
    check_in_end: time
    check_out_start: time
    check_out_end: time


class AttendanceLogOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    log_id: int
    emp_id: int
    working_date: date
    checkin_time: datetime
    checkout_time: datetime | None = None
    working_duration: timedelta | None = None


class AttendanceHistoryResponse(BaseModel):
    items: list[AttendanceLogOut]
    page: int
    page_size: int
    total: int


class AttendanceCheckResponse(BaseModel):
    message: str
    check_type: str
    log: AttendanceLogOut | None = None
