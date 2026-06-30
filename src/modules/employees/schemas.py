from pydantic import BaseModel, ConfigDict


class EmployeeOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    emp_id: int
    name: str
    emp_code: str


class EmployeeRegisterResponse(BaseModel):
    message: str
    employee: EmployeeOut


class EmployeeRemoveResponse(BaseModel):
    message: str
    emp_id: int | None = None
    emp_code: str | None = None


class EmployeeListResponse(BaseModel):
    items: list[EmployeeOut]
    page: int
    page_size: int
    total: int


class EmployeeDetailResponse(BaseModel):
    employee: EmployeeOut
