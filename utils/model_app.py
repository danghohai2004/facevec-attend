from insightface.app import FaceAnalysis
from config import MODEL

# Prepare InsightFace model
def setup_face_app():
    app = FaceAnalysis(name=MODEL['name'], providers=MODEL['providers'])
    app.prepare(ctx_id=MODEL['ctx_id'], det_size=MODEL['det_size'])
    return app