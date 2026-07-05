import asyncio
from unittest.mock import AsyncMock

import pytest

import src.app as app_module
from src.modules.antispoofing.service import LivenessChecker, PassThroughChecker


class RealLivenessChecker(LivenessChecker):
    def check(self, img, bbox) -> bool:
        return False


def stub_startup_dependencies(monkeypatch, checker):
    ensure_collection = AsyncMock()
    run_pipeline = AsyncMock()
    qdrant = AsyncMock()
    engine = AsyncMock()
    monkeypatch.setattr(app_module, "ensure_collection", ensure_collection)
    monkeypatch.setattr(app_module, "get_qdrant_client", lambda: qdrant)
    monkeypatch.setattr(app_module, "get_liveness_checker", lambda: checker)
    monkeypatch.setattr(app_module, "run_pipeline", run_pipeline)
    monkeypatch.setattr(app_module, "engine", engine)
    return ensure_collection, run_pipeline


@pytest.mark.asyncio
@pytest.mark.parametrize("environment", ["production", "PRODUCTION"])
async def test_production_rejects_pass_through_liveness_before_startup(
    monkeypatch,
    environment,
):
    monkeypatch.setenv("ENV", environment)
    ensure_collection, run_pipeline = stub_startup_dependencies(
        monkeypatch,
        PassThroughChecker(),
    )
    app = app_module.create_app()

    with pytest.raises(RuntimeError, match="PassThroughChecker"):
        async with app.router.lifespan_context(app):
            pass

    ensure_collection.assert_not_awaited()
    run_pipeline.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("environment", "checker"),
    [
        ("development", PassThroughChecker()),
        ("production", RealLivenessChecker()),
    ],
)
async def test_allowed_liveness_configuration_starts_normally(
    monkeypatch,
    environment,
    checker,
):
    monkeypatch.setenv("ENV", environment)
    ensure_collection, run_pipeline = stub_startup_dependencies(
        monkeypatch,
        checker,
    )
    app = app_module.create_app()

    async with app.router.lifespan_context(app):
        pass

    ensure_collection.assert_awaited_once()
    run_pipeline.assert_called_once()


@pytest.mark.asyncio
async def test_shutdown_awaits_pipeline_before_closing_resources(monkeypatch):
    events = []
    pipeline_started = asyncio.Event()
    keep_pipeline_running = asyncio.Event()

    async def run_pipeline(**_kwargs):
        pipeline_started.set()
        try:
            await keep_pipeline_running.wait()
        finally:
            await asyncio.sleep(0)
            events.append("pipeline_stopped")

    async def close_qdrant():
        events.append("qdrant_closed")

    async def dispose_engine():
        events.append("engine_disposed")

    qdrant = AsyncMock()
    qdrant.close = AsyncMock(side_effect=close_qdrant)
    engine = AsyncMock()
    engine.dispose = AsyncMock(side_effect=dispose_engine)

    monkeypatch.setenv("ENV", "development")
    monkeypatch.setattr(app_module, "ensure_collection", AsyncMock())
    monkeypatch.setattr(app_module, "get_qdrant_client", lambda: qdrant)
    monkeypatch.setattr(
        app_module,
        "get_liveness_checker",
        lambda: PassThroughChecker(),
    )
    monkeypatch.setattr(app_module, "run_pipeline", run_pipeline)
    monkeypatch.setattr(app_module, "engine", engine, raising=False)

    app = app_module.create_app()
    async with app.router.lifespan_context(app):
        await pipeline_started.wait()
    await asyncio.sleep(0)

    assert events == [
        "pipeline_stopped",
        "qdrant_closed",
        "engine_disposed",
    ]
