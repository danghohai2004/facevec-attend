import asyncio
import pathlib
import sys

# ponytail: repo root on path so `python scripts/reconcile_vectors.py` works,
# not just `python -m scripts.reconcile_vectors` — operators reach for the former.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from qdrant_client import AsyncQdrantClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.modules.employees.models import Employee
from src.platform.db.qdrant import COLLECTION_NAME, get_qdrant_client
from src.platform.db.session import AsyncSessionLocal

_SCROLL_LIMIT = 256
_DRIFT_EXIT_CODE = 1
_FAILURE_EXIT_CODE = 2


async def reconcile(db: AsyncSession, qdrant: AsyncQdrantClient) -> int:
    result = await db.execute(select(Employee))
    employees = {employee.emp_id: employee for employee in result.scalars().all()}

    qdrant_ids: set[int] = set()
    offset = None
    while True:
        points, next_offset = await qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=_SCROLL_LIMIT,
            offset=offset,
            with_payload=["emp_id"],
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            emp_id = payload.get("emp_id")
            if not isinstance(emp_id, int):
                raise ValueError(f"Qdrant point {point.id} has no valid emp_id payload")
            qdrant_ids.add(emp_id)

        if next_offset is None:
            break
        offset = next_offset

    missing_ids = sorted(set(employees) - qdrant_ids)
    orphan_ids = sorted(qdrant_ids - set(employees))

    for emp_id in missing_ids:
        employee = employees[emp_id]
        print(
            f"MISSING VECTOR: emp_id={emp_id} "
            f"emp_code={employee.emp_code} name={employee.name}"
        )
    if missing_ids:
        print(
            "ACTION: re-register each employee listed under MISSING VECTOR; "
            "vectors cannot be rebuilt automatically."
        )

    for emp_id in orphan_ids:
        print(f"ORPHAN VECTOR: emp_id={emp_id}")
    if orphan_ids:
        print(
            "ACTION: manually verify and prune ORPHAN VECTOR entries later; "
            "this script does not delete data."
        )

    return _DRIFT_EXIT_CODE if missing_ids or orphan_ids else 0


async def main() -> int:
    try:
        async with AsyncSessionLocal() as db:
            return await reconcile(db, get_qdrant_client())
    except Exception as exc:
        print(f"RECONCILIATION FAILED: {exc}", file=sys.stderr)
        return _FAILURE_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
