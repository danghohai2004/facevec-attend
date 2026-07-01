from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship

from src.platform.db.base import Base


class Employee(Base):
    __tablename__ = "employees"

    emp_id   = Column(Integer, primary_key=True, autoincrement=True)
    emp_code = Column(String(50), unique=True, nullable=False)
    name     = Column(String(100), nullable=False)

    attendance_logs = relationship("AttendanceLog", back_populates="employee", cascade="all, delete-orphan")
