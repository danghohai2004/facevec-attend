import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.platform.db.qdrant import ensure_collection, get_qdrant_client
from src.platform.db.session import AsyncSessionLocal
from src.platform.config import THRESHOLD
from src.platform.queue import FrameQueue
from src.platform.realtime.manager import ConnectionManager
from src.modules.antispoofing.service import PassThroughChecker, get_liveness_checker
from src.modules.recognition.pipeline import run_pipeline
from src.modules.recognition.ws_ingress import make_ws_router
from src.modules.employees.api import router as employees_router
from src.modules.attendance.api import router as attendance_router


def create_app() -> FastAPI:
    queue = FrameQueue(maxsize=50)
    manager = ConnectionManager()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        checker = get_liveness_checker()
        if os.getenv("ENV", "").lower() == "production" and isinstance(
            checker,
            PassThroughChecker,
        ):
            raise RuntimeError("PassThroughChecker cannot be used in production.")

        await ensure_collection()
        qdrant = get_qdrant_client()
        app.state.pipeline_task = asyncio.create_task(
            run_pipeline(
                queue=queue,
                qdrant=qdrant,
                db_factory=AsyncSessionLocal,
                manager=manager,
                checker=checker,
                threshold=THRESHOLD,
            )
        )
        yield
        app.state.pipeline_task.cancel()

    app = FastAPI(
        title="Face Recognition Attendance System",
        description="API for managing employees and tracking attendance using facial recognition.",
        version="0.2.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?$",
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    try:
        app.mount("/static", StaticFiles(directory="static"), name="static")
    except RuntimeError:
        pass  # ponytail: static dir absent in tests/CI

    app.include_router(employees_router)
    app.include_router(attendance_router)
    app.include_router(make_ws_router(queue, manager))

    return app
