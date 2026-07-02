import pytest

from src.platform.db import qdrant as qdrant_db


def _capture_client_kwargs(monkeypatch):
    configured = {}

    class FakeAsyncQdrantClient:
        def __init__(self, **kwargs):
            configured.update(kwargs)

    monkeypatch.setenv("QDRANT_HOST", "qdrant.internal")
    monkeypatch.setenv("QDRANT_PORT", "7000")
    monkeypatch.setenv("QDRANT_API_KEY", "test-qdrant-key")
    monkeypatch.setattr(qdrant_db, "AsyncQdrantClient", FakeAsyncQdrantClient)
    monkeypatch.setattr(qdrant_db, "_client", None)
    return configured, FakeAsyncQdrantClient


def test_qdrant_client_uses_configured_api_key_and_plaintext_by_default(monkeypatch):
    configured, FakeAsyncQdrantClient = _capture_client_kwargs(monkeypatch)

    client = qdrant_db.get_qdrant_client()

    assert isinstance(client, FakeAsyncQdrantClient)
    # https defaults to False — local Qdrant serves plaintext HTTP even with an
    # api_key; sending https here caused an SSL WRONG_VERSION_NUMBER at startup.
    assert configured == {
        "host": "qdrant.internal",
        "port": 7000,
        "api_key": "test-qdrant-key",
        "https": False,
    }


@pytest.mark.parametrize(
    ("value", "expected"),
    [("true", True), ("TRUE", True), ("false", False), ("", False)],
)
def test_qdrant_https_knob(monkeypatch, value, expected):
    configured, _ = _capture_client_kwargs(monkeypatch)
    monkeypatch.setenv("QDRANT_HTTPS", value)

    qdrant_db.get_qdrant_client()

    assert configured["https"] is expected
