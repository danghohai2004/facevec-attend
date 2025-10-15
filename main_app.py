import shutil

import streamlit as st
import cv2
import os
import pandas as pd
import psycopg2
import plotly.express as px
from datetime import datetime, time as tm
from time import time
from utils.conn_db import get_connection
from utils.model_app import setup_face_app
from source.core.webcam_recognition import draw_bbox
from source.core.face_identifier import identify_person_pgvector
from source.services.attendance_manager import get_current_time
from utils.embedding_utils import add_info_embeddings, remove_embeddings
from config import ORIGINAL_IMG_PATH, THRESHOLD

def get_monthly_report(selected_year, selected_month):
    query = """
                SELECT 
                    DATE_TRUNC('month', working_date) AS month,
                    ROUND(SUM(EXTRACT(EPOCH FROM working_duration)) / 3600, 2) AS total_hours,
                    COUNT(DISTINCT working_date) AS working_days,
                    ROUND(SUM(EXTRACT(EPOCH FROM working_duration)) / 3600 / COUNT(DISTINCT working_date), 2) AS avg_hours_per_day
                FROM attendance_logs
                WHERE DATE_PART('year', working_date) = %s
                  AND DATE_PART('month', working_date) = %s
                GROUP BY month
                ORDER BY month;
            """
    conn = get_connection()
    df = pd.read_sql(query, conn, params=(selected_year, selected_month))
    conn.close()
    return df

def streamlit_app(threshold, base_path):
    st.set_page_config(page_title="Face Attendance", layout="wide")
    st.image("face_icon.png", width=80)
    st.title(" Real-time Face Attendance System")

    if "capture_faces" not in st.session_state:
        st.session_state.capture_faces = []

    if "last_capture_time" not in st.session_state:
        st.session_state.last_capture_time = 0

    if "face_reco_running" not in st.session_state:
        st.session_state.face_reco_running = False

    if "capturing" not in st.session_state:
        st.session_state.capturing = False

    if "shifts_time" not in st.session_state:
        st.session_state.shifts_time = {
            "Check in start": tm(datetime.now().hour, datetime.now().minute),
            "Check in end": tm(datetime.now().hour, datetime.now().minute),
            "Check out start": tm(datetime.now().hour, datetime.now().minute),
            "Check out end": tm(datetime.now().hour, datetime.now().minute)
        }

    if "shown_warnings" not in st.session_state:
        st.session_state.shown_warnings = {
            "not_during_shift": False
        }

    conn = get_connection()

    if not conn:
        st.error("Cannot connect to database!")
        return

    tabs = st.tabs(["⏰ SHIFT SETTINGS", "🎥 ATTENDANCE", "➕ REGISTER", "🗑️ REMOVE", "📊 STATISTICS"])

    with tabs[0]:
        st.header("⏰ Shift Settings")

        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)

        with col1:
            st.session_state.shifts_time["Check in start"] = st.time_input(
                "CHECK IN START TIME",
                value=st.session_state.shifts_time["Check in start"],
                key="Check in start",
                step=60
            )

        with col2:
            st.session_state.shifts_time["Check in end"] = st.time_input(
                "CHECK IN END TIME",
                value=st.session_state.shifts_time["Check in end"],
                key="Check in end",
                step=60
            )

        with col3:
            st.session_state.shifts_time["Check out start"] = st.time_input(
                "CHECK OUT START TIME",
                value=st.session_state.shifts_time["Check out start"],
                key="Check out start",
                step=60
            )

        with col4:
            st.session_state.shifts_time["Check out end"] = st.time_input(
                "CHECK OUT END TIME",
                value=st.session_state.shifts_time["Check out end"],
                key="Check out end",
                step=60
            )

    with tabs[1]:
        st.title("🎥 REAL-TIME ATTENDANCE")

        run_reco = st.button("Start Face Recognition")
        stop_reco = st.button("Stop Face Recognition")
        st.session_state.face_reco_running = run_reco

        warning_placeholder = st.empty()
        stframe = st.empty()

        if st.session_state.face_reco_running:
            cap = cv2.VideoCapture(0)
            app = setup_face_app()

            while st.session_state.face_reco_running:
                within_shift, now, check_type = get_current_time(st.session_state.shifts_time)
                ret, frame = cap.read()
                if not ret:
                    warning_placeholder.error("Cannot Open Camera!")
                    break

                if st.session_state.shown_warnings["not_during_shift"]:
                    warning_placeholder.empty()
                    st.session_state.shown_warnings["not_during_shift"] = False

                if check_type == "check_in" or check_type == "check_out":
                    if not within_shift:
                        if not st.session_state.shown_warnings["not_during_shift"]:
                            warning_placeholder.warning("Not during working hours!")
                            st.session_state.shown_warnings["not_during_shift"] = True
                        continue
                    faces = app.get(frame)
                    if faces:
                        for face in faces:
                            emb_person_frame = face.normed_embedding
                            _, name = identify_person_pgvector(conn, emb_person_frame, threshold, st.session_state.shifts_time)
                            bbox = face.bbox.astype(int)
                            draw_bbox(frame, bbox, color=(0,255,255), thickness=2, corner_len=10)
                            cv2.putText(frame, name, (bbox[0], bbox[1] - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width=700)

                if stop_reco:
                    st.session_state.face_reco_running = False
                    break

            cap.release()
            stframe.empty()

    with tabs[2]:
        st.title("👤 REGISTER - REALTIME")
        name_person = st.text_input("Enter your name and press the Enter Key")

        if name_person:
            start_capture = st.button("Start automatic capture")
            stframe = st.empty()

            process_bar = st.progress(0)

            if start_capture:
                st.session_state.capturing = True
                st.session_state.capture_faces = []
                st.session_state.last_capture_time = 0

                cap = cv2.VideoCapture(0)

                while st.session_state.capturing:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Cannot Open Camera!")
                        break

                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    stframe.image(frame_rgb, channels="RGB", width=700)

                    if time() - st.session_state.last_capture_time > 1.5:
                        st.session_state.capture_faces.append(frame_rgb)
                        st.session_state.last_capture_time = time()

                        count = len(st.session_state.capture_faces)
                        process_bar.progress(count / 31)

                    if len(st.session_state.capture_faces) >= 31:
                        st.session_state.capturing = False

                cap.release()
                stframe.empty()

            if st.session_state.capture_faces and len(st.session_state.capture_faces) >= 31:
                save_dir = os.path.join(base_path, name_person)
                os.makedirs(save_dir, exist_ok=True)

                for i, img in enumerate(st.session_state.capture_faces[1:], start=1):
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    save_path = os.path.join(save_dir, f"{i}.jpg")
                    cv2.imwrite(save_path, img_bgr)

                st.success(f"saved {len(st.session_state.capture_faces)} the photo to {save_dir}")
                add_info_embeddings(base_path, name_person)
                st.session_state.capture_faces = []

    with tabs[3]:
        st.title("👥 EMPLOYEES LIST")

        try:
            with conn.cursor() as cur:
                cur.execute("""
                            SELECT DISTINCT e.name
                            FROM employees e
                                     INNER JOIN face_embeddings f ON e.emp_id = f.emp_id
                            """)
                all_names = [r[0] for r in cur.fetchall()]

            if all_names:
                name_person = st.selectbox("Select a name to delete:", all_names)
                start_remove = st.button("🗑️ Proceed to delete embeddings")

                if start_remove:
                    with conn.cursor() as cur:
                        cur.execute("SELECT emp_id FROM employees WHERE name = %s", (name_person,))
                        row = cur.fetchone()
                        if row:
                            emp_id = row[0]
                            remove_embeddings(emp_id)

                            name_folder_path = os.path.join(base_path, name_person)
                            if os.path.exists(name_folder_path):
                                shutil.rmtree(name_folder_path)
                                st.success(f"Deleted folder: {name_folder_path}")
                            else:
                                st.warning("Folder not found")

                            st.rerun()

            else:
                st.info("No employees with embeddings found.")

        except Exception as e:
            st.error(f"Error: {e}")

    with tabs[4]:
        st.title("📅 Monthly Attendance Report")

        current_year = datetime.today().year
        one_year_ago = datetime.today().year - 1
        two_year_ago = datetime.today().year - 2

        col1, col2 = st.columns(2)

        with col1:
            selected_year = st.selectbox("Select year", [two_year_ago, one_year_ago, current_year], index=2)
        with col2:
            selected_month = st.selectbox("Select month", list(range(1, 13)), index=9)

        if st.button("View statistics"):
            df = get_monthly_report(selected_year, selected_month)
            if not df.empty:
                st.subheader(f"Thống kê tháng {selected_month}/{selected_year}")
                st.dataframe(df)

                # Hiển thị số liệu tổng quan
                total_hours = df["total_hours"].iloc[0]
                working_days = df["working_days"].iloc[0]
                avg_hours = df["avg_hours_per_day"].iloc[0]

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("⏱ Tổng số giờ làm", f"{total_hours:.2f} giờ")
                with col2:
                    st.metric("📅 Số ngày làm việc", working_days)
                with col3:
                    st.metric("🧮 Giờ trung bình/ngày", f"{avg_hours:.2f} giờ")

                # Biểu đồ minh họa
                fig = px.pie(
                    df,
                    values="total_hours",
                    names="month",
                    title="Tỷ lệ tổng giờ làm việc giữa các tháng",
                    hole=0.3  # tạo dạng donut
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Không có dữ liệu cho tháng này.")



streamlit_app(threshold=THRESHOLD, base_path=ORIGINAL_IMG_PATH)
