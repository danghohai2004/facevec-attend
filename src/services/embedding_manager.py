from src.core.extract_emb import extract_embeddings
import streamlit as st

def add_info_embeddings(conn, original_img_path, name):
    embeddings = extract_embeddings(original_img_path, name)
    try:
        with conn.cursor() as cur:

            for person_name, embeddings_list in embeddings.items():
                cur.execute("SELECT emp_id FROM employees WHERE name = %s", (person_name,))
                row = cur.fetchone()
                if row:
                    emp_id = row[0]
                else:
                    cur.execute("INSERT INTO employees (name) VALUES (%s) RETURNING emp_id", (person_name, ))
                    emp_id = cur.fetchone()[0]

                for embedding in embeddings_list:
                    cur.execute("INSERT INTO face_embeddings (emp_id, embedding) VALUES (%s, %s)", (emp_id, embedding.tolist()))

            conn.commit()
        cur.close()

        return f"Successfully added the embedding {name} to the database", None

    except Exception as e:
        conn.rollback()
        return None, f"[ERROR ADD]: {e}"

def remove_embeddings(conn, emp_id):
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT emp_id FROM employees WHERE emp_id = %s", (emp_id,))
            row = cur.fetchone()
            if row is None:
                return None

            cur.execute("DELETE FROM face_embeddings WHERE emp_id = %s", (emp_id,))
            cur.execute("DELETE FROM employees WHERE emp_id = %s", (emp_id,))
            conn.commit()

    except Exception as e:
        conn.rollback()
        return f"[WARNING]: {e}"