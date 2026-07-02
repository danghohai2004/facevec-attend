import asyncio
from contextlib import suppress

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
