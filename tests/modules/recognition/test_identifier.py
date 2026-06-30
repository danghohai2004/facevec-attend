import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.mark.asyncio
async def test_identify_returns_none_below_threshold():
    from src.modules.recognition.identifier import identify
    qdrant = AsyncMock()
    hit = MagicMock()
    hit.score = 0.3  # below 1 - 0.6 = 0.4
    hit.payload = {"emp_id": 1, "name": "A", "emp_code": "NV001"}
    qdrant.search = AsyncMock(return_value=[hit])

    result = await identify(qdrant, [0.1] * 512, threshold=0.6)
    assert result is None


@pytest.mark.asyncio
async def test_identify_returns_payload_above_threshold():
    from src.modules.recognition.identifier import identify
    qdrant = AsyncMock()
    hit = MagicMock()
    hit.score = 0.8  # above 1 - 0.6 = 0.4
    hit.payload = {"emp_id": 1, "name": "Nguyen Van A", "emp_code": "NV001"}
    qdrant.search = AsyncMock(return_value=[hit])

    result = await identify(qdrant, [0.1] * 512, threshold=0.6)
    assert result == {"emp_id": 1, "name": "Nguyen Van A", "emp_code": "NV001"}


@pytest.mark.asyncio
async def test_identify_returns_none_when_no_results():
    from src.modules.recognition.identifier import identify
    qdrant = AsyncMock()
    qdrant.search = AsyncMock(return_value=[])

    result = await identify(qdrant, [0.1] * 512, threshold=0.6)
    assert result is None
