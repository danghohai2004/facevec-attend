"""Face analysis singleton for insightface model."""

import threading

from insightface.app import FaceAnalysis

from src.platform.config import MODEL

_face_app: FaceAnalysis | None = None
_lock = threading.Lock()


def setup_face_app() -> FaceAnalysis:
    """Initialize and return FaceAnalysis singleton (thread-safe)."""
    global _face_app
    if _face_app is None:
        with _lock:
            if _face_app is None:  # double-checked: avoid lock on hot path
                _face_app = FaceAnalysis(name=MODEL["name"], providers=MODEL["providers"])
                _face_app.prepare(ctx_id=MODEL["ctx_id"], det_size=MODEL["det_size"])
    return _face_app
