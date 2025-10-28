# ==== MODEL CONFIGURATION ====
MODEL = {
    "name": "buffalo_sc",
    "det_size": (640,640),
    "ctx_id":0,
    "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"]
}

# ==== PATH IMAGE FACE ====
ORIGINAL_IMG_PATH = "user_images"

# ==== THRESHOLD ====
THRESHOLD = 0.6

# ==== DB STYLE (neon_tech/local) ====
DB_STYLE = "neon_tech"





