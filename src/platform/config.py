"""Platform configuration for face recognition system."""

MODEL = {
    "name": "buffalo_sc",
    "det_size": (640, 640),
    "ctx_id": 0,
    "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
}

ORIGINAL_IMG_PATH = "faces"
THRESHOLD = 0.6
MAX_EMB_FACE = 50
