import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.platform.queue import FrameItem, FrameQueue
from src.platform.realtime.manager import ConnectionManager

router = APIRouter(tags=["Recognition"])


def make_ws_router(queue: FrameQueue, manager: ConnectionManager) -> APIRouter:
    @router.websocket("/ws/recognition/{client_id}")
    async def ws_endpoint(websocket: WebSocket, client_id: str):
        await manager.connect(client_id, websocket)
        try:
            while True:
                frame = await websocket.receive_bytes()
                await queue.put(FrameItem(
                    client_id=client_id,
                    frame=frame,
                    captured_at=time.time(),
                ))
        except WebSocketDisconnect:
            manager.disconnect(client_id)

    return router
