"""Face analysis singleton for insightface model."""

from insightface.app import FaceAnalysis

from src.platform.config import MODEL

_face_app: FaceAnalysis | None = None


def setup_face_app() -> FaceAnalysis:
    """Initialize and return FaceAnalysis singleton.

    Returns:
        FaceAnalysis: Initialized face analysis model instance.
    """
    global _face_app
    if _face_app is None:
        _face_app = FaceAnalysis(name=MODEL["name"], providers=MODEL["providers"])
        _face_app.prepare(ctx_id=MODEL["ctx_id"], det_size=MODEL["det_size"])
    return _face_app
