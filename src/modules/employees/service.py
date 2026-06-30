import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct, FilterSelector

from src.modules.employees.models import Employee
from src.platform.db.qdrant import COLLECTION_NAME

ERR_NOT_FOUND = "EMPLOYEE_NOT_FOUND"
ERR_MISSING_ID = "MISSING_IDENTIFIER"


async def register_employee(
    db: AsyncSession,
    qdrant: AsyncQdrantClient,
    name: str,
    emp_code: str,
    embeddings: list[list[float]],
) -> tuple[Employee, str | None]:
    try:
        result = await db.execute(select(Employee).filter(Employee.emp_code == emp_code))
        employee = result.scalar_one_or_none()

        if not employee:
            employee = Employee(name=name, emp_code=emp_code)
            db.add(employee)
            await db.flush()
        else:
            employee.name = name

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=emb,
                payload={"emp_id": employee.emp_id, "emp_code": emp_code, "name": name},
            )
            for emb in embeddings
        ]

        # ponytail: commit DB first — emp_id is permanent before Qdrant is touched,
        # so a DB rollback never leaves orphaned vectors pointing at a ghost emp_id
        await db.commit()
        await db.refresh(employee)

        # Delete stale vectors then upsert; order matters: delete-then-upsert after commit
        # means worst case is missing vectors (re-register fixes it), not ghost vectors
        if employee.emp_id:
            await qdrant.delete(
                collection_name=COLLECTION_NAME,
                points_selector=FilterSelector(
                    filter=Filter(must=[FieldCondition(key="emp_id", match=MatchValue(value=employee.emp_id))])
                ),
            )
        await qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

        return employee, None
    except Exception as e:
        await db.rollback()
        return None, f"[ERROR REGISTER]: {e}"


async def remove_employee(
    db: AsyncSession,
    qdrant: AsyncQdrantClient,
    emp_id: int | None = None,
    emp_code: str | None = None,
) -> tuple[Employee, str | None]:
    if emp_id is None and emp_code is None:
        return None, ERR_MISSING_ID

    try:
        if emp_id is not None:
            stmt = select(Employee).filter(Employee.emp_id == emp_id)
        else:
            stmt = select(Employee).filter(Employee.emp_code == emp_code)

        result = await db.execute(stmt)
        employee = result.scalar_one_or_none()
        if not employee:
            return None, ERR_NOT_FOUND

        await qdrant.delete(
            collection_name=COLLECTION_NAME,
            points_selector=FilterSelector(
                filter=Filter(must=[FieldCondition(key="emp_id", match=MatchValue(value=employee.emp_id))])
            ),
        )

        await db.delete(employee)
        await db.commit()
        return employee, None
    except Exception as e:
        await db.rollback()
        return None, f"[ERROR REMOVE]: {e}"


async def list_employees(
    db: AsyncSession, page: int, page_size: int
) -> tuple[list[Employee], int, str | None]:
    try:
        total = (await db.execute(select(func.count()).select_from(Employee))).scalar_one()
        result = await db.execute(
            select(Employee).order_by(Employee.emp_id)
            .offset((page - 1) * page_size).limit(page_size)
        )
        return result.scalars().all(), total, None
    except Exception as e:
        return [], 0, f"[ERROR LIST]: {e}"


async def get_employee(db: AsyncSession, emp_id: int) -> tuple[Employee, str | None]:
    try:
        result = await db.execute(select(Employee).filter(Employee.emp_id == emp_id))
        employee = result.scalar_one_or_none()
        return (employee, None) if employee else (None, ERR_NOT_FOUND)
    except Exception as e:
        return None, f"[ERROR GET]: {e}"


async def search_employees_by_name(db: AsyncSession, name: str) -> tuple[list[Employee], str | None]:
    try:
        result = await db.execute(
            select(Employee).filter(Employee.name.ilike(f"%{name}%")).order_by(Employee.emp_id)
        )
        return result.scalars().all(), None
    except Exception as e:
        return [], f"[ERROR SEARCH]: {e}"
