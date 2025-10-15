import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

# Connect with database
def get_connection():
    try:
        conn = psycopg2.connect(
            host = os.getenv("DB_HOST"),
            database = os.getenv("DB_NAME"),
            user = os.getenv("DB_USER"),
            password = os.getenv("DB_PASS"),
            port = os.getenv("DB_PORT")
        )
        return conn
    except Exception as e:
        print(f"[DB ERROR] {e}")
        return None