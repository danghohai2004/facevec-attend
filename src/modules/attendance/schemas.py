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


class SummaryDeltas(BaseModel):
    todays_attendance: float | None
    average_working_hours: float | None
    on_time_rate: float | None


class SummaryStatsResponse(BaseModel):
    total_employees: int
    todays_attendance: int
    average_working_hours: float
    on_time_rate: float
    deltas: SummaryDeltas


class MonthlyStatItem(BaseModel):
    month: int
    attendance: int
    working_hours: float
    average_hours: float


class MonthlyStatsResponse(BaseModel):
    available_years: list[int]
    items: list[MonthlyStatItem]


class DailyStatItem(BaseModel):
    day: int
    average_hours: float


class DailyStatsResponse(BaseModel):
    items: list[DailyStatItem]
