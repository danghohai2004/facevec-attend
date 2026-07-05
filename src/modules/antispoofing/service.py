import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("models/antispoofing/AntiSpoofing_bin_1.5_128.onnx")


class LivenessChecker(ABC):
    @abstractmethod
    def check(self, img: np.ndarray, bbox: list[float]) -> bool:
        """Return True if the face is real, False if spoofing detected.

        img: BGR frame; bbox: normalized [x1, y1, x2, y2] in 0..1.
        """


class PassThroughChecker(LivenessChecker):
    # ponytail: dev-only fallback — app.py refuses it in production
    def check(self, img: np.ndarray, bbox: list[float]) -> bool:
        return True


def crop_face(img: np.ndarray, bbox: list[float], bbox_inc: float = 1.5) -> np.ndarray:
    """Square crop of side max(w, h) * bbox_inc centered on the face,
    zero-padded where it falls outside the frame. Matches the reference
    inference code the model was trained against (hairymax/Face-AntiSpoofing)."""
    real_h, real_w = img.shape[:2]
    x1 = bbox[0] * real_w
    y1 = bbox[1] * real_h
    w = bbox[2] * real_w - x1
    h = bbox[3] * real_h - y1
    side = max(w, h)

    xc, yc = x1 + w / 2, y1 + h / 2
    x, y = int(xc - side * bbox_inc / 2), int(yc - side * bbox_inc / 2)
    cx1, cy1 = max(x, 0), max(y, 0)
    cx2 = min(x + int(side * bbox_inc), real_w)
    cy2 = min(y + int(side * bbox_inc), real_h)

    crop = img[cy1:cy2, cx1:cx2, :]
    return cv2.copyMakeBorder(
        crop,
        cy1 - y, int(side * bbox_inc) - cy2 + y,
        cx1 - x, int(side * bbox_inc) - cx2 + x,
        cv2.BORDER_CONSTANT, value=[0, 0, 0],
    )


def preprocess(crop: np.ndarray, size: int = 128) -> np.ndarray:
    """Aspect-preserving resize to `size`, zero-pad, CHW float32 in 0..1."""
    old_h, old_w = crop.shape[:2]
    ratio = float(size) / max(old_h, old_w)
    new_w, new_h = int(old_w * ratio), int(old_h * ratio)
    img = cv2.resize(crop, (new_w, new_h))

    dw, dh = size - new_w, size - new_h
    img = cv2.copyMakeBorder(
        img,
        dh // 2, dh - dh // 2,
        dw // 2, dw - dw // 2,
        cv2.BORDER_CONSTANT, value=[0, 0, 0],
    )
    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)


class MiniFASNetChecker(LivenessChecker):
    """Binary anti-spoofing via MiniFASNet ONNX (class 0 = real face)."""

    def __init__(self, model_path: str | os.PathLike, threshold: float = 0.5):
        import onnxruntime as ort

        self._session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        self._threshold = threshold

    def check(self, img: np.ndarray, bbox: list[float]) -> bool:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = preprocess(crop_face(rgb, bbox))
        logits = self._session.run(None, {self._input_name: tensor})[0][0]
        probs = np.exp(logits) / np.sum(np.exp(logits))
        return bool(np.argmax(probs) == 0 and probs[0] >= self._threshold)


_checker: LivenessChecker | None = None


def get_liveness_checker() -> LivenessChecker:
    global _checker
    if _checker is None:
        model_path = Path(os.getenv("ANTISPOOFING_MODEL_PATH", DEFAULT_MODEL_PATH))
        if model_path.is_file():
            _checker = MiniFASNetChecker(
                model_path,
                threshold=float(os.getenv("ANTISPOOFING_THRESHOLD", "0.5")),
            )
            logger.info("Anti-spoofing enabled: %s", model_path)
        else:
            logger.warning(
                "Anti-spoofing model missing at %s — using PassThroughChecker",
                model_path,
            )
            _checker = PassThroughChecker()
    return _checker
