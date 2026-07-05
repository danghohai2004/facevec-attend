import asyncio
import logging
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from src.platform.queue import FrameQueue
from src.platform.realtime.manager import ConnectionManager
from src.modules.antispoofing.service import LivenessChecker
from src.modules.recognition.extractor import extract_largest_face
from src.modules.recognition.identifier import identify_face

logger = logging.getLogger(__name__)

INTERNAL_ERROR = "Lỗi hệ thống"
_executor = ThreadPoolExecutor(max_workers=4)
# ponytail: cap in-flight _process tasks to match the executor's 4 workers —
# more concurrency would just queue on _executor anyway. Raise both together.
_sem = asyncio.Semaphore(4)
# ponytail: strong refs prevent GC from silently cancelling in-flight tasks mid-await
_pending_tasks: set[asyncio.Task] = set()


async def run_pipeline(
    queue: FrameQueue,
    qdrant,
    db_factory,
    manager: ConnectionManager,
    checker: LivenessChecker,
    threshold: float = 0.6,
) -> None:
    loop = asyncio.get_running_loop()

    def _task_done(task: asyncio.Task) -> None:
        _pending_tasks.discard(task)
        _sem.release()

    try:
        while True:
            await _sem.acquire()
            task = None
            try:
                item = await queue.get()
                task = asyncio.create_task(
                    _process(
                        item,
                        qdrant,
                        db_factory,
                        manager,
                        checker,
                        threshold,
                        loop,
                    )
                )
            finally:
                if task is None:
                    _sem.release()
            _pending_tasks.add(task)
            task.add_done_callback(_task_done)
    finally:
        pending_tasks = tuple(_pending_tasks)
        for task in pending_tasks:
            task.cancel()
        await asyncio.gather(*pending_tasks, return_exceptions=True)


async def _process(item, qdrant, db_factory, manager, checker, threshold, loop):
    from src.modules.attendance.service import log_attendance

    def _cpu_work():
        nparr = np.frombuffer(item.frame, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None, None, "no_face"
        result = extract_largest_face(img)
        if result is None:
            return None, None, "no_face"
        emb, bbox = result
        # liveness needs the face bbox, so it runs after detection
        if not checker.check(img, bbox):
            return None, None, "spoof"
        return emb, bbox, None

    try:
        embedding, bbox, early_status = await loop.run_in_executor(_executor, _cpu_work)
        ts = datetime.now(timezone.utc).isoformat()

        if early_status:
            await manager.send(item.client_id, {"status": early_status, "timestamp": ts})
            return

        # bbox is present here (a face was detected) — send it so the kiosk can
        # draw the box on the face even before recognition resolves.
        person = await identify_face(qdrant, embedding, threshold)
        if person is None:
            await manager.send(
                item.client_id,
                {"status": "unknown", "bbox": bbox, "timestamp": ts},
            )
            return

        async with db_factory() as db:
            attendance_result = await log_attendance(db, person["emp_id"])

        await manager.send(item.client_id, {
            "status": "recognized",
            "emp_id": person["emp_id"],
            "name": person["name"],
            "attendance": attendance_result,
            "bbox": bbox,
            "timestamp": ts,  # reuse ts captured before CPU work
        })
    except Exception:
        logger.exception("Recognition pipeline processing failed")
        await manager.send(
            item.client_id,
            {"status": "error", "detail": INTERNAL_ERROR},
        )
