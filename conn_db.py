import psycopg2
import os
import streamlit as st
from dotenv import load_dotenv
from config import DB_STYLE

load_dotenv()

def get_connection():
    if DB_STYLE == "local":
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST"),
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASS"),
            port=os.getenv("DB_PORT"),
        )
        return conn

    elif DB_STYLE == "neon_tech":
        conn = psycopg2.connect(**st.secrets["postgres"])
        return conn