import asyncio
from contextlib import suppress
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.modules.recognition import pipeline
from src.platform.queue import FrameItem, FrameQueue


@pytest.mark.asyncio
async def test_run_pipeline_limits_pending_processing_tasks(monkeypatch):
    queue = FrameQueue(maxsize=20)
    for index in range(20):
        await queue.put(
            FrameItem(
                client_id="cam1",
                frame=f"frame-{index}".encode(),
                captured_at=float(index),
            )
        )

    release_processing = asyncio.Event()
    started = 0

    async def blocked_process(*_args):
        nonlocal started
        started += 1
        await release_processing.wait()

    monkeypatch.setattr(pipeline, "_process", blocked_process)
    monkeypatch.setattr(pipeline, "_pending_tasks", set())
    monkeypatch.setattr(pipeline, "_sem", asyncio.Semaphore(4), raising=False)

    runner = asyncio.create_task(
        pipeline.run_pipeline(
            queue=queue,
            qdrant=object(),
            db_factory=object(),
            manager=object(),
            checker=object(),
        )
    )

    try:
        for _ in range(20):
            await asyncio.sleep(0)
            if started >= 4:
                break
        await asyncio.sleep(0)

        assert started == 4
        assert len(pipeline._pending_tasks) == 4
    finally:
        runner.cancel()
        release_processing.set()
        with suppress(asyncio.CancelledError):
            await runner
        await asyncio.gather(
            *tuple(pipeline._pending_tasks),
            return_exceptions=True,
        )


@pytest.mark.asyncio
async def test_process_hides_internal_error_from_websocket_payload(caplog):
    item = FrameItem(client_id="cam1", frame=b"jpeg", captured_at=0.0)
    loop = MagicMock()
    loop.run_in_executor = AsyncMock(
        side_effect=RuntimeError("qdrant://admin:secret@vector/internal")
    )
    manager = MagicMock()
    manager.send = AsyncMock()

    with caplog.at_level("ERROR", logger="src.modules.recognition.pipeline"):
        await pipeline._process(
            item=item,
            qdrant=object(),
            db_factory=object(),
            manager=manager,
            checker=object(),
            threshold=0.6,
            loop=loop,
        )

    manager.send.assert_awaited_once_with(
        "cam1",
        {"status": "error", "detail": "Lỗi hệ thống"},
    )
    assert any(record.exc_info is not None for record in caplog.records)


@pytest.mark.asyncio
async def test_run_pipeline_shutdown_drains_children_and_restores_permits(
    monkeypatch,
):
    queue = FrameQueue(maxsize=1)
    await queue.put(
        FrameItem(client_id="cam1", frame=b"jpeg", captured_at=0.0)
    )

    processing_started = asyncio.Event()
    processing_cancelled = asyncio.Event()
    keep_processing = asyncio.Event()

    async def blocked_process(*_args):
        processing_started.set()
        try:
            await keep_processing.wait()
        except asyncio.CancelledError:
            processing_cancelled.set()
            raise

    pending_tasks = set()
    semaphore = asyncio.Semaphore(4)
    monkeypatch.setattr(pipeline, "_process", blocked_process)
    monkeypatch.setattr(pipeline, "_pending_tasks", pending_tasks)
    monkeypatch.setattr(pipeline, "_sem", semaphore)

    runner = asyncio.create_task(
        pipeline.run_pipeline(
            queue=queue,
            qdrant=object(),
            db_factory=object(),
            manager=object(),
            checker=object(),
        )
    )

    try:
        await processing_started.wait()
        runner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await runner

        assert processing_cancelled.is_set()
        assert pending_tasks == set()

        for _ in range(4):
            await asyncio.wait_for(semaphore.acquire(), timeout=0.1)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(semaphore.acquire(), timeout=0.01)
    finally:
        for task in tuple(pending_tasks):
            task.cancel()
        await asyncio.gather(*tuple(pending_tasks), return_exceptions=True)
        for _ in range(4):
            semaphore.release()
