# ==== MODEL CONFIGURATION ====
MODEL = {
    "name": "buffalo_sc",
    "det_size": (640,640),
    "ctx_id":0,
    "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"]
}

# ==== PATH IMAGE FACE ====
ORIGINAL_IMG_PATH = "faces"

# ==== THRESHOLD ====
THRESHOLD = 0.6

# ==== SET EMBEDDING APPROXIMATION (PER FACE) ====
APPROX_EMB_FACE = 50




