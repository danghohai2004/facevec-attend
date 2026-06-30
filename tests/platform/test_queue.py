import asyncio
import pytest
from src.platform.queue import FrameItem, FrameQueue


def make_item(client_id: str = "cam1") -> FrameItem:
    return FrameItem(client_id=client_id, frame=b"jpeg", captured_at=0.0)


@pytest.mark.asyncio
async def test_put_and_get():
    q = FrameQueue(maxsize=2)
    item = make_item()
    await q.put(item)
    result = await q.get()
    assert result.client_id == "cam1"


@pytest.mark.asyncio
async def test_drop_oldest_when_full():
    q = FrameQueue(maxsize=2)
    old = FrameItem(client_id="cam1", frame=b"old", captured_at=1.0)
    new1 = FrameItem(client_id="cam1", frame=b"new1", captured_at=2.0)
    new2 = FrameItem(client_id="cam1", frame=b"new2", captured_at=3.0)
    await q.put(old)
    await q.put(new1)
    await q.put(new2)  # queue full → old dropped
    first = await q.get()
    assert first.frame == b"new1"  # old was dropped
