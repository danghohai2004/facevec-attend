import cv2
from source.core.face_identifier import identify_person_pgvector
from source.core.extract_emb import setup_face_app
from utils.conn_db import get_connection
from datetime import time
from source.services.attendance_manager import get_current_time
from config import THRESHOLD

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

def webcam_inference(conn, THRESHOLD, shifts_time):
    app = setup_face_app()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

    within_shift, now, check_type = get_current_time(shifts_time)
    if not within_shift:
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = app.get(frame)

        if faces:
            for face in faces:
                bbox = face.bbox.astype(int)
                emb = face.normed_embedding
                _, name = identify_person_pgvector(conn, emb, THRESHOLD, shifts_time)
                draw_bbox(frame, bbox)
                cv2.putText(frame, name, (bbox[0], bbox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Face Attendance', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

conn = get_connection()

shifts_time = {
    "Check in start": time(12, 57),
    "Check in end": time(12,58),
    "Check out start": time(12,59),
    "Check out end": time(13,00),
}
threshold = 0.6
webcam_inference(conn, threshold, shifts_time)