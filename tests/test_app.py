from unittest.mock import AsyncMock

import pytest

import src.app as app_module
from src.modules.antispoofing.service import LivenessChecker, PassThroughChecker


class RealLivenessChecker(LivenessChecker):
    def check(self, frame: bytes) -> bool:
        return False


def stub_startup_dependencies(monkeypatch, checker):
    ensure_collection = AsyncMock()
    run_pipeline = AsyncMock()
    monkeypatch.setattr(app_module, "ensure_collection", ensure_collection)
    monkeypatch.setattr(app_module, "get_qdrant_client", lambda: object())
    monkeypatch.setattr(app_module, "get_liveness_checker", lambda: checker)
    monkeypatch.setattr(app_module, "run_pipeline", run_pipeline)
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
