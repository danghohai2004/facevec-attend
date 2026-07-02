from unittest.mock import AsyncMock, MagicMock

import pytest
from qdrant_client.models import Record

import src.modules.attendance.models  # noqa: F401 — register AttendanceLog relationship
from scripts import reconcile_vectors as module
from src.modules.employees.models import Employee


class AsyncSessionContext:
    def __init__(self, db):
        self.db = db

    async def __aenter__(self):
        return self.db

    async def __aexit__(self, exc_type, exc, tb):
        return False


def make_db(employees):
    result = MagicMock()
    result.scalars.return_value.all.return_value = employees

    db = MagicMock()
    db.execute = AsyncMock(return_value=result)
    db.add = MagicMock()
    db.delete = AsyncMock()
    db.commit = AsyncMock()
    return db


def make_qdrant(pages):
    qdrant = MagicMock()
    qdrant.scroll = AsyncMock(side_effect=pages)
    qdrant.upsert = AsyncMock()
    qdrant.delete = AsyncMock()
    return qdrant


def point(point_id, emp_id):
    return Record(id=point_id, payload={"emp_id": emp_id})


def configure_dependencies(monkeypatch, module, db, qdrant):
    monkeypatch.setattr(module, "AsyncSessionLocal", lambda: AsyncSessionContext(db))
    monkeypatch.setattr(module, "get_qdrant_client", lambda: qdrant)


@pytest.mark.asyncio
async def test_matching_employee_and_vector_ids_exit_zero(monkeypatch, capsys):
    db = make_db(
        [
            Employee(emp_id=1, emp_code="EMP-1", name="Alice"),
            Employee(emp_id=2, emp_code="EMP-2", name="Bob"),
        ]
    )
    qdrant = make_qdrant(
        [
            ([point("point-1", 1)], "next-page"),
            ([point("point-2", 2)], None),
        ]
    )
    configure_dependencies(monkeypatch, module, db, qdrant)

    exit_code = await module.main()

    assert exit_code == 0
    assert capsys.readouterr() == ("", "")
    assert qdrant.scroll.await_args_list[0].kwargs["offset"] is None
    assert qdrant.scroll.await_args_list[1].kwargs["offset"] == "next-page"


@pytest.mark.asyncio
async def test_missing_vector_is_reported_with_reregister_guidance(monkeypatch, capsys):
    db = make_db(
        [
            Employee(emp_id=1, emp_code="EMP-1", name="Alice"),
            Employee(emp_id=2, emp_code="EMP-2", name="Bob"),
        ]
    )
    qdrant = make_qdrant([([point("point-1", 1)], None)])
    configure_dependencies(monkeypatch, module, db, qdrant)

    exit_code = await module.main()
    output = capsys.readouterr()

    assert exit_code != 0
    assert "MISSING VECTOR: emp_id=2 emp_code=EMP-2 name=Bob" in output.out
    assert "re-register" in output.out
    assert output.err == ""


@pytest.mark.asyncio
async def test_orphan_vector_is_reported_with_manual_prune_guidance(monkeypatch, capsys):
    db = make_db([Employee(emp_id=1, emp_code="EMP-1", name="Alice")])
    qdrant = make_qdrant(
        [([point("point-1", 1), point("orphan-point", 9)], None)]
    )
    configure_dependencies(monkeypatch, module, db, qdrant)

    exit_code = await module.main()
    output = capsys.readouterr()

    assert exit_code != 0
    assert "ORPHAN VECTOR: emp_id=9" in output.out
    assert "manually verify and prune" in output.out
    assert output.err == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_source", ["database", "qdrant"])
async def test_reconciliation_error_is_clear_and_nonzero(
    monkeypatch, capsys, failure_source
):
    db = make_db([Employee(emp_id=1, emp_code="EMP-1", name="Alice")])
    qdrant = make_qdrant([([], None)])
    if failure_source == "database":
        db.execute.side_effect = RuntimeError("database unavailable")
    else:
        qdrant.scroll.side_effect = RuntimeError("qdrant unavailable")
    configure_dependencies(monkeypatch, module, db, qdrant)

    exit_code = await module.main()
    output = capsys.readouterr()

    assert exit_code != 0
    assert "RECONCILIATION FAILED:" in output.err
    assert f"{failure_source} unavailable" in output.err
    assert "MISSING VECTOR" not in output.out
    assert "ORPHAN VECTOR" not in output.out


@pytest.mark.asyncio
async def test_reconciliation_is_read_only(monkeypatch):
    db = make_db([Employee(emp_id=1, emp_code="EMP-1", name="Alice")])
    qdrant = make_qdrant([([point("point-1", 1)], None)])
    configure_dependencies(monkeypatch, module, db, qdrant)

    exit_code = await module.main()

    assert exit_code == 0
    db.add.assert_not_called()
    db.delete.assert_not_awaited()
    db.commit.assert_not_awaited()
    qdrant.upsert.assert_not_awaited()
    qdrant.delete.assert_not_awaited()
