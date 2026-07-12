from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from src.platform.auth import require_api_key
from src.platform.db.qdrant import get_qdrant_client
from src.platform.db.session import get_db
from src.modules.employees.schemas import (
    EmployeeDetailResponse,
    EmployeeListResponse,
    EmployeeOut,
    EmployeeRegisterResponse,
    EmployeeRemoveResponse,
)
from src.modules.employees.service import (
    ERR_DUPLICATE,
    ERR_MISSING_ID,
    ERR_NOT_FOUND,
    get_employee,
    list_employees,
    register_employee,
    remove_employee,
    search_employees_by_name,
)

router = APIRouter(prefix="/api/employees", tags=["Employees"])


@router.post(
    "",
    response_model=EmployeeRegisterResponse,
    dependencies=[Depends(require_api_key)],
)
async def api_register(
    name: str = Form(...),
    emp_code: str = Form(...),
    files: list[UploadFile] = File(...),
    db: AsyncSession = Depends(get_db),
):
    # ponytail: local import avoids circular import — recognition module not yet wired
    from src.modules.recognition.extractor import extract_embeddings_from_bytes

    MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # ponytail: chặn ảnh khổng lồ/decompression bomb
    # Multi-frame enrollment: store one embedding per usable burst frame so a
    # single pose/lighting doesn't decide recognition. Skip frames with 0 or 2+
    # faces (missed/ambiguous) instead of failing the whole enrollment over one
    # bad frame; require at least one usable face overall.
    embeddings: list[list[float]] = []
    for upload in files:
        contents = await upload.read()
        if len(contents) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "Ảnh quá lớn (tối đa 5MB).")
        face_embeddings = await extract_embeddings_from_bytes(contents)
        if len(face_embeddings) == 1:
            embeddings.append(face_embeddings[0])

    if not embeddings:
        raise HTTPException(400, "Không nhận được khuôn mặt hợp lệ nào. Vui lòng thử lại.")

    qdrant = get_qdrant_client()
    employee, err = await register_employee(db, qdrant, name, emp_code, embeddings)
    if err:
        if err == ERR_DUPLICATE:
            raise HTTPException(409, "Mã nhân viên đã tồn tại.")
        raise HTTPException(500, err)
    return EmployeeRegisterResponse(
        message=f"Registered {employee.name} ({employee.emp_code})",
        employee=EmployeeOut.model_validate(employee),
    )


@router.get("", response_model=EmployeeListResponse)
async def api_list(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    employees, total, err = await list_employees(db, page, page_size)
    if err:
        raise HTTPException(500, err)
    return EmployeeListResponse(
        items=[EmployeeOut.model_validate(e) for e in employees],
        page=page,
        page_size=page_size,
        total=total,
    )


@router.get("/{identifier}", response_model=EmployeeDetailResponse | EmployeeListResponse)
async def api_get(identifier: str, db: AsyncSession = Depends(get_db)):
    if identifier.isdigit():
        employee, err = await get_employee(db, int(identifier))
        if err:
            raise HTTPException(404 if err == ERR_NOT_FOUND else 500, err)
        return EmployeeDetailResponse(employee=EmployeeOut.model_validate(employee))
    employees, err = await search_employees_by_name(db, identifier)
    if err:
        raise HTTPException(500, err)
    return EmployeeListResponse(
        items=[EmployeeOut.model_validate(e) for e in employees],
        page=1,
        page_size=len(employees),
        total=len(employees),
    )


@router.delete(
    "",
    response_model=EmployeeRemoveResponse,
    dependencies=[Depends(require_api_key)],
)
async def api_remove(
    emp_id: int | None = Query(None),
    emp_code: str | None = Query(None),
    db: AsyncSession = Depends(get_db),
):
    qdrant = get_qdrant_client()
    employee, err = await remove_employee(db, qdrant, emp_id=emp_id, emp_code=emp_code)
    if err:
        if err == ERR_MISSING_ID:
            raise HTTPException(400, "Cần emp_id hoặc emp_code.")
        if err == ERR_NOT_FOUND:
            raise HTTPException(404, "Không tìm thấy nhân viên.")
        raise HTTPException(500, err)
    return EmployeeRemoveResponse(
        message=f"Removed {employee.emp_id}",
        emp_id=employee.emp_id,
        emp_code=employee.emp_code,
    )
