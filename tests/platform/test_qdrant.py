from src.platform.db import qdrant as qdrant_db


def test_qdrant_client_uses_configured_api_key(monkeypatch):
    configured = {}

    class FakeAsyncQdrantClient:
        def __init__(self, **kwargs):
            configured.update(kwargs)

    monkeypatch.setenv("QDRANT_HOST", "qdrant.internal")
    monkeypatch.setenv("QDRANT_PORT", "7000")
    monkeypatch.setenv("QDRANT_API_KEY", "test-qdrant-key")
    monkeypatch.setattr(qdrant_db, "AsyncQdrantClient", FakeAsyncQdrantClient)
    monkeypatch.setattr(qdrant_db, "_client", None)

    client = qdrant_db.get_qdrant_client()

    assert isinstance(client, FakeAsyncQdrantClient)
    assert configured == {
        "host": "qdrant.internal",
        "port": 7000,
        "api_key": "test-qdrant-key",
    }
