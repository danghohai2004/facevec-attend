from source.services.attendance_manager import log_attendance

def identify_person_pgvector(conn, person_frame_emb, threshold, shifts_time):
    emb_frame = person_frame_emb.astype('float32').tolist()

    with conn.cursor() as cur:
        cur.execute("""
            SELECT e.emp_id, e.name, (f.embedding <=> %s::vector) AS distance
            FROM employees e INNER JOIN face_embeddings f ON e.emp_id = f.emp_id
            ORDER BY f.embedding <=> %s::vector
            LIMIT 1;
            """, (emb_frame, emb_frame))

        result = cur.fetchone()

        if not result:
            return None, "Unknown"

        emp_id, name, distance = result
        if distance > threshold:
            return None, "Unknown"

        log_attendance(conn, emp_id, shifts_time)

        return emp_id, name



