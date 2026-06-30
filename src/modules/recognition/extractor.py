import asyncio
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from src.platform.ml.face_app import setup_face_app

# ponytail: module-level executor, shared across all calls
_executor = ThreadPoolExecutor(max_workers=4)


async def extract_embeddings_from_bytes(frame: bytes) -> list[list[float]]:
    """Decode image bytes and return embeddings for all detected faces."""
    nparr = np.frombuffer(frame, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return []
    app = setup_face_app()
    loop = asyncio.get_running_loop()
    faces = await loop.run_in_executor(_executor, app.get, img)
    return [face.normed_embedding.tolist() for face in faces]


def extract_embedding_from_frame(img: np.ndarray) -> list[float] | None:
    """Sync helper for use inside threadpool workers. Returns largest-face embedding."""
    app = setup_face_app()
    faces = app.get(img)
    if not faces:
        return None
    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
    return faces[0].embedding.tolist()
