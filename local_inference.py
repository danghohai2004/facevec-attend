import cv2
import logging
logging.getLogger("streamlit.runtime.scriptrunner.script_runner").setLevel(logging.ERROR)
from src.core.face_identifier import identify_person_pgvector
from src.core.extract_emb import setup_face_app
from utils.conn_db import get_connection

def draw_bbox(frame, bbox, color=(0,255,255), thickness=2, corner_len=10):
    x1, y1, x2, y2 = bbox

    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thickness)

    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thickness)

    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thickness)

    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thickness)

def webcam_inference(threshold, shifts_time = None):
    conn = None
    try:
        conn = get_connection()

        app = setup_face_app()

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            faces  = app.get(frame)

            if faces:
                faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
                main_face = faces[0]
                bbox = main_face.bbox.astype(int)
                emb = main_face.normed_embedding
                _,_, _, name = identify_person_pgvector(conn, emb, threshold, shifts_time)
                draw_bbox(frame, bbox)
                cv2.putText(frame, name, (bbox[0], bbox[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.imshow('Face Attendance', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    except Exception as e:
        return f"Error: {e}"
    finally:
        if conn:
            conn.close()