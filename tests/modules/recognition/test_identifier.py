import pytest
from unittest.mock import AsyncMock, MagicMock


def _query_response(points):
    """Mimic AsyncQdrantClient.query_points() → QueryResponse with .points."""
    response = MagicMock()
    response.points = points
    return response


@pytest.mark.asyncio
async def test_identify_returns_none_below_threshold():
    from src.modules.recognition.identifier import identify_face
    qdrant = AsyncMock()
    hit = MagicMock()
    hit.score = 0.3  # below 1 - 0.6 = 0.4
    hit.payload = {"emp_id": 1, "name": "A", "emp_code": "NV001"}
    qdrant.query_points = AsyncMock(return_value=_query_response([hit]))

    result = await identify_face(qdrant, [0.1] * 512, threshold=0.6)
    assert result is None


@pytest.mark.asyncio
async def test_identify_returns_payload_above_threshold():
    from src.modules.recognition.identifier import identify_face
    qdrant = AsyncMock()
    hit = MagicMock()
    hit.score = 0.8  # above 1 - 0.6 = 0.4
    hit.payload = {"emp_id": 1, "name": "Nguyen Van A", "emp_code": "NV001"}
    qdrant.query_points = AsyncMock(return_value=_query_response([hit]))

    result = await identify_face(qdrant, [0.1] * 512, threshold=0.6)
    assert result == {"emp_id": 1, "name": "Nguyen Van A", "emp_code": "NV001", "score": 0.8}


@pytest.mark.asyncio
async def test_identify_returns_none_when_no_results():
    from src.modules.recognition.identifier import identify_face
    qdrant = AsyncMock()
    qdrant.query_points = AsyncMock(return_value=_query_response([]))

    result = await identify_face(qdrant, [0.1] * 512, threshold=0.6)
    assert result is None
