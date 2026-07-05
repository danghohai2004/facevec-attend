from sqlalchemy import Column, Computed, Integer, Date, DateTime, Interval, Time, ForeignKey
from sqlalchemy.orm import relationship

from src.platform.db.base import Base


class AttendanceLog(Base):
    __tablename__ = "attendance_logs"

    log_id           = Column(Integer, primary_key=True, autoincrement=True)
    emp_id           = Column(Integer, ForeignKey("employees.emp_id", ondelete="CASCADE"), nullable=False)
    working_date     = Column(Date, nullable=False)
    checkin_time     = Column(DateTime, nullable=False)
    checkout_time    = Column(DateTime)
    # Matches initdb/init.sql's `GENERATED ALWAYS AS (...) STORED` column.
    # Computed(...) tells SQLAlchemy to omit this from INSERT/UPDATE — without
    # it, the ORM sends an explicit NULL for this column and Postgres rejects
    # any value (even NULL) in the column list of a generated column.
    working_duration = Column(Interval, Computed("checkout_time - checkin_time", persisted=True))

    employee = relationship("Employee", back_populates="attendance_logs")


class ShiftSettings(Base):
    __tablename__ = "shift_settings"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    check_in_start  = Column(Time, nullable=False)
    check_in_end    = Column(Time, nullable=False)
    check_out_start = Column(Time, nullable=False)
    check_out_end   = Column(Time, nullable=False)
