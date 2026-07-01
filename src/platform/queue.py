import asyncio
from dataclasses import dataclass


@dataclass(frozen=True)
class FrameItem:
    client_id: str
    frame: bytes
    captured_at: float


class FrameQueue:
    def __init__(self, maxsize: int = 50) -> None:
        self._queue: asyncio.Queue[FrameItem] = asyncio.Queue(maxsize=maxsize)

    async def put(self, item: FrameItem) -> None:
        if self._queue.full():
            try:
                self._queue.get_nowait()  # ponytail: drop oldest, realtime prefers newest frame
            except asyncio.QueueEmpty:
                pass
        await self._queue.put(item)

    async def get(self) -> FrameItem:
        return await self._queue.get()
