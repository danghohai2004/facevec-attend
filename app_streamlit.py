import warnings
warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")
import shutil
import streamlit as st
import cv2
import os
import pandas as pd
import plotly.express as px
from datetime import datetime, time as tm
from time import time
from utils.conn_db import get_connection
from utils.model_app import setup_face_app
from local_inference import draw_bbox
from src.core.face_identifier import identify_person_pgvector
from src.services.attendance_manager import get_current_time
from src.services.embedding_manager import add_info_embeddings, remove_embeddings
from config import ORIGINAL_IMG_PATH, THRESHOLD

@st.cache_resource
def app_st():
    app = setup_face_app()
    return app

def get_monthly_report(conn, selected_year, selected_month):
    query = """
                SELECT 
                    DATE_PART('month', working_date) AS month,
                    ROUND(SUM(EXTRACT(EPOCH FROM working_duration)) / 3600, 2) AS total_hours,
                    COUNT(DISTINCT working_date) AS working_days,
                    ROUND(SUM(EXTRACT(EPOCH FROM working_duration)) / 3600 / COUNT(DISTINCT working_date), 2) AS avg_hours_per_day
                FROM attendance_logs
                WHERE DATE_PART('year', working_date) = %s AND DATE_PART('month', working_date) = %s
                GROUP BY month
                ORDER BY month;
            """
    df = pd.read_sql_query(query, conn, params=(selected_year, selected_month))
    return df

def streamlit_app(threshold, base_path):
    st.set_page_config(page_title="Face Attendance", layout="wide")
    st.image("template/face_icon.png", width=80)
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

    if "warned_checkout" not in st.session_state:
        st.session_state.warned_mess_logs = False

    if "warned_pgvector" not in st.session_state:
        st.session_state.warned_pgvector = False

    if "conn" not in st.session_state:
        st.session_state.conn = None

    tabs = st.tabs(["🗄️ CONNECT DATABASE", "⏰ SHIFT SETTINGS", "🎥 ATTENDANCE", "➕ REGISTER", "🗑️ REMOVE", "📊 STATISTICS"])

    try:
        with tabs[0]:
            st.header("🗄️ Connect Database")

            if st.button("Connect Database"):
                if st.session_state.conn is None:
                    st.session_state.conn = get_connection()
                if not st.session_state.conn:
                    st.error("Cannot connect to database!")
                    st.stop()
                else:
                    st.success("Connected to DB!")

            if st.button("Close Database") and st.session_state.conn is not None:
                st.session_state.conn.close()
                st.session_state.conn = None
                st.success("Database closed")

        with tabs[1]:
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

        with tabs[2]:
            st.title("🎥 REAL-TIME ATTENDANCE")

            if st.session_state.conn is None:
                st.error("Not connected to the database")
            elif st.session_state.conn is not None:
                run_reco = st.button("Start Face Recognition")
                stop_reco = st.button("Stop Face Recognition")
                st.session_state.face_reco_running = run_reco

                warning_placeholder = st.empty()
                stframe = st.empty()

                if st.session_state.face_reco_running:
                    cap = cv2.VideoCapture(0)

                    app = app_st()

                    while st.session_state.face_reco_running:
                        within_shift, now, check_type = get_current_time(st.session_state.shifts_time)
                        ret, frame = cap.read()
                        if not ret:
                            warning_placeholder.error("Cannot Open Camera!")
                            break

                        if st.session_state.shown_warnings["not_during_shift"]:
                            warning_placeholder.empty()
                            st.session_state.shown_warnings["not_during_shift"] = False

                        if not within_shift:
                            if not st.session_state.shown_warnings["not_during_shift"]:
                                warning_placeholder.warning("Not during working hours!")
                                st.session_state.shown_warnings["not_during_shift"] = True

                        if check_type == "check_in" or check_type == "check_out":
                            faces = app.get(frame)
                            if faces:
                                faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
                                main_face = faces[0]
                                emb_person_frame = main_face.normed_embedding
                                error_identify, mess_logs, emp_id, name = identify_person_pgvector(st.session_state.conn, emb_person_frame, threshold, st.session_state.shifts_time)

                                if mess_logs is not None:
                                    if not st.session_state.warned_mess_logs:
                                        st.warning(name + " - " + mess_logs)
                                        st.session_state.warned_mess_logs = True
                                elif mess_logs is None:
                                    st.session_state.warned_mess_logs = False

                                if error_identify is not None:
                                    if not st.session_state.warned_pgvector:
                                        st.warning(error_identify)
                                        st.session_state.warned_pgvector = True
                                elif error_identify is None:
                                    st.session_state.warned_pgvector = False

                                bbox = main_face.bbox.astype(int)
                                draw_bbox(frame, bbox, color=(0,255,255), thickness=2, corner_len=10)
                                cv2.putText(frame, name, (bbox[0], bbox[1] - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width=700)

                        if stop_reco:
                            st.session_state.face_reco_running = False
                            break

                    cap.release()
                    stframe.empty()

        with tabs[3]:
            st.title("👤 REGISTER - REALTIME")

            if st.session_state.conn is None:
                st.error("Not connected to the database")
            elif st.session_state.conn is not None:
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

                    if st.session_state.capture_faces and len(st.session_state.capture_faces) >= 1:
                        save_dir = os.path.join(base_path, name_person)
                        os.makedirs(save_dir, exist_ok=True)

                        for i, img in enumerate(st.session_state.capture_faces[1:], start=1):
                            img_bgr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            save_path = os.path.join(save_dir, f"{i}.jpg")
                            cv2.imwrite(save_path, img_bgr)

                        st.success(f"saved {len(st.session_state.capture_faces)} the photo to {save_dir}")
                        success_emb_add_db, error_emb_add_db = add_info_embeddings(st.session_state.conn, base_path, name_person)

                        if success_emb_add_db is not None:
                            st.success(success_emb_add_db)
                        if error_emb_add_db is not None:
                            st.success(error_emb_add_db)

                        st.session_state.capture_faces = []

        with tabs[4]:
            st.title("👥 EMPLOYEES LIST")

            if st.session_state.conn is None:
                st.error("Not connected to the database")
            elif st.session_state.conn is not None:
                try:
                    with st.session_state.conn.cursor() as cur:
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
                            with st.session_state.conn.cursor() as cur:
                                cur.execute("SELECT emp_id FROM employees WHERE name = %s", (name_person,))
                                row = cur.fetchone()
                                if row:
                                    emp_id = row[0]
                                    mess_remove = remove_embeddings(st.session_state.conn, emp_id)

                                    if mess_remove is not None:
                                        st.info(mess_remove)

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

        with tabs[5]:
            st.title("📅 Monthly Attendance Report")

            if st.session_state.conn is None:
                st.error("Not connected to the database")
            elif st.session_state.conn is not None:
                current_year = datetime.today().year
                one_year_ago = datetime.today().year - 1
                two_year_ago = datetime.today().year - 2

                col1, col2 = st.columns(2)

                with col1:
                    selected_year = st.selectbox("Select year", [two_year_ago, one_year_ago, current_year], index=2)
                with col2:
                    selected_month = st.selectbox("Select month", list(range(1, 13)), index=9)

                if st.button("View statistics"):
                    month_df = get_monthly_report(st.session_state.conn, selected_year, selected_month)

                    if not month_df.empty:
                        st.subheader(f"Statistics for {selected_month}/{selected_year}")

                        total_hours = month_df["total_hours"].iloc[0] or 0
                        working_days = month_df["working_days"].iloc[0] or 0
                        avg_hours = month_df["avg_hours_per_day"].iloc[0] or 0

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("⏱ Total working hours", f"{total_hours:.2f} giờ")
                        with col2:
                            st.metric("📅 Number of working days", working_days)
                        with col3:
                            st.metric("🧮 Average hours/day", f"{avg_hours:.2f} giờ")

                        monthly_data = []
                        for month in range(1, 13):
                            month_df = get_monthly_report(st.session_state.conn, selected_year, month)

                            if not month_df.empty:
                                monthly_data.append({
                                    "Month": int(round(month)),
                                    "Total hours": month_df["total_hours"].iloc[0] or 0,
                                    "Number of days": month_df["working_days"].iloc[0] or 0,
                                    "AVG hours/day": month_df["avg_hours_per_day"].iloc[0] or 0
                                })

                        if monthly_data:
                            comparison_df = pd.DataFrame(monthly_data)
                        col1, col2 = st.columns(2)
                        with col1:
                            fig = px.bar(
                                comparison_df,
                                x='Month',
                                y='AVG hours/day',
                                text='AVG hours/day',
                                title='Average hours/day of the months',
                                color_discrete_sequence=['#e74c3c']
                            )
                            fig.update_xaxes(dtick=1)
                            fig.update_traces(texttemplate='%{text:.1f}h', textposition='outside')
                            fig.update_layout(height=500)
                            st.plotly_chart(fig, use_container_width=True)

                        with col2:
                            fig2 = px.line(
                                comparison_df,
                                x='Month',
                                y='Total hours',
                                title='Working hour trends',
                                markers=True
                            )
                            fig2.update_traces(line_color='#2ecc71', line_width=3, marker_size=10)
                            fig2.update_layout(height=500)
                            st.plotly_chart(fig2, use_container_width=True)
                    else:
                        st.warning("NO DATA!")

    except Exception as e:
        st.error(f"[ERROR APP]: {e}")

streamlit_app(threshold=THRESHOLD, base_path=ORIGINAL_IMG_PATH)
