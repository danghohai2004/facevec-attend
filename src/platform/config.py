"""Platform configuration for face recognition system."""

MODEL = {
    "name": "buffalo_sc",
    "det_size": (640, 640),
    "ctx_id": 0,
    "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
}

THRESHOLD = 0.6
