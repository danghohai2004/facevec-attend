import asyncio
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from src.platform.queue import FrameQueue
from src.platform.realtime.manager import ConnectionManager
from src.modules.antispoofing.service import LivenessChecker
from src.modules.recognition.extractor import extract_embedding_from_frame
from src.modules.recognition.identifier import identify_face

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

    async def _process_with_release(item):
        try:
            await _process(
                item,
                qdrant,
                db_factory,
                manager,
                checker,
                threshold,
                loop,
            )
        finally:
            _sem.release()

    while True:
        await _sem.acquire()
        task = None
        try:
            item = await queue.get()
            task = asyncio.create_task(_process_with_release(item))
        finally:
            if task is None:
                _sem.release()
        _pending_tasks.add(task)
        task.add_done_callback(_pending_tasks.discard)


async def _process(item, qdrant, db_factory, manager, checker, threshold, loop):
    from src.modules.attendance.service import log_attendance

    def _cpu_work():
        nparr = np.frombuffer(item.frame, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None, "no_face"
        if not checker.check(item.frame):  # checker expects bytes
            return None, "spoof"
        emb = extract_embedding_from_frame(img)
        if emb is None:
            return None, "no_face"
        return emb, None

    try:
        embedding, early_status = await loop.run_in_executor(_executor, _cpu_work)
        ts = datetime.now(timezone.utc).isoformat()

        if early_status:
            await manager.send(item.client_id, {"status": early_status, "timestamp": ts})
            return

        person = await identify_face(qdrant, embedding, threshold)
        if person is None:
            await manager.send(item.client_id, {"status": "unknown", "timestamp": ts})
            return

        async with db_factory() as db:
            attendance_result = await log_attendance(db, person["emp_id"])

        await manager.send(item.client_id, {
            "status": "recognized",
            "emp_id": person["emp_id"],
            "name": person["name"],
            "attendance": attendance_result,
            "timestamp": ts,  # reuse ts captured before CPU work
        })
    except Exception as e:
        await manager.send(item.client_id, {"status": "error", "detail": str(e)})
