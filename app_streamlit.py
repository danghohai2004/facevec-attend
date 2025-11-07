import warnings
warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")

import shutil
import streamlit as st
import cv2
import os
import pandas as pd
import tempfile
import plotly.express as px
from datetime import datetime, time as tm
from time import time
from utils.conn_db import get_connection
from utils.model_app import setup_face_app
from inference.local_inference import draw_bbox
from src.core.face_identifier import identify_person_pgvector
from src.services.attendance_manager import get_current_time
from src.services.embedding_manager import add_info_embeddings, remove_embeddings
from config import ORIGINAL_IMG_PATH, THRESHOLD


@st.cache_resource
def app_st():
    return setup_face_app()

def get_monthly_report(conn, selected_year, selected_month):
    query = """
            SELECT DATE_PART('month', working_date) AS month,
            ROUND(SUM(EXTRACT(EPOCH FROM working_duration)) / 3600, 2) AS total_hours,
            COUNT(DISTINCT working_date) AS working_days,
            ROUND(SUM(EXTRACT(EPOCH FROM working_duration)) / 3600 / COUNT(DISTINCT working_date), 2) AS avg_hours_per_day
            FROM attendance_logs
            WHERE DATE_PART('year' \
                , working_date) = %s \
              AND DATE_PART('month' \
                , working_date) = %s
            GROUP BY month
            ORDER BY month; \
            """
    return pd.read_sql_query(query, conn, params=(selected_year, selected_month))

def initialize_session_state():
    defaults = {
        "capture_faces": [],
        "last_capture_time": 0,
        "face_reco_running": False,
        "capturing": False,
        "shifts_time": {
            "Check in start": tm(datetime.now().hour, datetime.now().minute),
            "Check in end": tm(datetime.now().hour, datetime.now().minute),
            "Check out start": tm(datetime.now().hour, datetime.now().minute),
            "Check out end": tm(datetime.now().hour, datetime.now().minute)
        },
        "shown_warnings": {"not_during_shift": False},
        "warned_mess_logs": False,
        "warned_pgvector": False,
        "conn": None,
        "different_name": False
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def save_captured_faces(valid_faces, base_path, name_person, conn):
    save_dir = os.path.join(base_path, name_person)

    if os.path.exists(save_dir):
        existing_files = [f for f in os.listdir(save_dir) if f.endswith(".jpg")]
        start_index = len(existing_files) + 1
    else:
        os.makedirs(save_dir, exist_ok=True)
        start_index = 1

    saved_count = 0
    for i, img in enumerate(valid_faces, start=start_index):
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        save_path = os.path.join(save_dir, f"{i}.jpg")
        if cv2.imwrite(save_path, img_bgr):
            saved_count += 1

    st.success(f"✅ Saved {saved_count} photos to {save_dir}")

    success_emb, error_emb = add_info_embeddings(conn, base_path, name_person)
    if success_emb:
        st.success(success_emb)
    if error_emb:
        st.error(error_emb)

    return saved_count

def handle_register_capture(base_path, conn, mode="REAL TIME"):
    key_suffix = "realtime" if mode == "REAL TIME" else "video"
    name_person = st.text_input(f"Enter your name - {mode}", key=f"name_{key_suffix}")

    if not name_person:
        return

    video_cap = None
    if mode == "VIDEO":
        video_cap = st.file_uploader("📁 Select Video", type=["mp4"], key=f"register_video_{key_suffix}")
        if not video_cap:
            return

    start_capture = st.button("🎬 Start automatic capture", key=f"start_{key_suffix}")
    stframe = st.empty()
    process_bar = st.progress(0)

    if start_capture:
        st.session_state.capturing = True
        st.session_state.capture_faces = []
        st.session_state.last_capture_time = 0

        if mode == "REAL TIME":
            cap = cv2.VideoCapture(0)
        else:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(video_cap.read())
            cap = cv2.VideoCapture(tfile.name)

        try:
            while st.session_state.capturing and len(st.session_state.capture_faces) < 31:
                ret, frame = cap.read()
                if not ret:
                    st.error("Cannot read frame!" if mode == "REAL TIME" else "Video ended!")
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame_rgb, channels="RGB", width=700)

                current_time = time()
                if current_time - st.session_state.last_capture_time > 1.5:
                    st.session_state.capture_faces.append(frame_rgb)
                    st.session_state.last_capture_time = current_time

                    count = len(st.session_state.capture_faces)
                    process_bar.progress(min(count / 31, 1.0))
        finally:
            cap.release()
            stframe.empty()
            st.session_state.capturing = False

    valid_faces = [f for f in st.session_state.capture_faces if f is not None]

    if not valid_faces:
        return

    save_dir = os.path.join(base_path, name_person)

    if os.path.exists(save_dir):
        st.warning(f"⚠️ User '{name_person}' already exists. Choose an option:")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("➕ Add images", key=f"add_{key_suffix}"):
                save_captured_faces(valid_faces, base_path, name_person, conn)
                st.session_state.capture_faces = []
                st.rerun()

        with col2:
            if st.button("🔄 Different name", key=f"diff_{key_suffix}"):
                st.session_state.different_name = True

        if st.session_state.different_name:
            diff_name = st.text_input("Enter a different name:", key=f"diff_name_{key_suffix}")

            if diff_name:
                if diff_name == name_person:
                    st.error("❌ Please choose a different name!")
                elif os.path.exists(os.path.join(base_path, diff_name)):
                    st.error(f"❌ User '{diff_name}' already exists!")
                else:
                    save_captured_faces(valid_faces, base_path, diff_name, conn)
                    st.session_state.capture_faces = []
                    st.session_state.different_name = False
                    st.rerun()

    else:
        save_captured_faces(valid_faces, base_path, name_person, conn)
        st.session_state.capture_faces = []
        st.rerun()

def handle_attendance_recognition(app, conn, shifts_time, mode="REAL TIME", video_file=None, video_file_read=None):
    start_btn = st.button("▶ Start Attendance", key=f"start_att_{mode}")
    stop_btn = st.button("⏹ Stop Attendance", key=f"stop_att_{mode}")

    warning_placeholder = st.empty()
    mess_logs_placeholder = st.empty()
    error_placeholder = st.empty()

    if start_btn:
        st.session_state.face_reco_running = True
    if stop_btn:
        st.session_state.face_reco_running = False

    if not st.session_state.face_reco_running:
        return

    if mode == "REAL TIME":
        cap = cv2.VideoCapture(0)
    else:
        if not video_file:
            return
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(video_file_read)
        cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()

    try:
        while st.session_state.face_reco_running:
            ret, frame = cap.read()
            if not ret:
                warning_placeholder.error("Cannot read frame!")
                break

            within_shift, now, check_type = get_current_time(shifts_time)

            if within_shift and st.session_state.shown_warnings["not_during_shift"]:
                warning_placeholder.empty()
                st.session_state.shown_warnings["not_during_shift"] = False
            elif not within_shift and not st.session_state.shown_warnings["not_during_shift"]:
                warning_placeholder.warning("⚠️ Not during working hours!")
                st.session_state.shown_warnings["not_during_shift"] = True

            if check_type in ["check_in", "check_out"]:
                faces = app.get(frame)
                if faces:
                    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
                    main_face = faces[0]
                    emb_person_frame = main_face.normed_embedding

                    error_identify, mess_logs, emp_id, name = identify_person_pgvector(
                        conn, emb_person_frame, THRESHOLD, shifts_time
                    )

                    if mess_logs and not st.session_state.warned_mess_logs:
                        mess_logs_placeholder.warning(f"{name} - {mess_logs}")
                        st.session_state.warned_mess_logs = True
                    elif not mess_logs and st.session_state.warned_mess_logs:
                        mess_logs_placeholder.empty()
                        st.session_state.warned_mess_logs = False

                    if error_identify and not st.session_state.warned_pgvector:
                        error_placeholder.warning(error_identify)
                        st.session_state.warned_pgvector = True
                    elif not error_identify and st.session_state.warned_pgvector:
                        error_placeholder.empty()
                        st.session_state.warned_pgvector = False

                    bbox = main_face.bbox.astype(int)
                    draw_bbox(frame, bbox, color=(0, 255, 255), thickness=2, corner_len=10)
                    cv2.putText(frame, name, (bbox[0], bbox[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width=700)

    finally:
        cap.release()
        stframe.empty()

def streamlit_app(base_path):
    st.set_page_config(page_title="Face Attendance", layout="wide")
    st.image("template/face_icon.png", width=80)
    st.title("Face Attendance System")

    initialize_session_state()

    tabs = st.tabs([
        "🗄️ DATABASE",
        "⏰ SHIFT",
        "🎥 ATTENDANCE",
        "➕ REGISTER",
        "🗑️ REMOVE",
        "📊 STATISTICS"
    ])

    try:
        with tabs[0]:
            st.header("🗄️ Database Connection")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔌 Connect", use_container_width=True):
                    if st.session_state.conn is None:
                        st.session_state.conn, mess_log = get_connection()
                        if not st.session_state.conn and mess_log:
                            st.error(f"❌ {mess_log}")
                        else:
                            st.success("✅ Connected to database!")

            with col2:
                if st.button("🔌 Disconnect", use_container_width=True):
                    if st.session_state.conn:
                        st.session_state.conn.close()
                        st.session_state.conn = None
                        st.success("✅ Disconnected!")

        with tabs[1]:
            st.header("⏰ Shift Settings")

            col1, col2 = st.columns(2)
            with col1:
                st.session_state.shifts_time["Check in start"] = st.time_input(
                    "CHECK IN START",
                    value=st.session_state.shifts_time["Check in start"],
                    step=60
                )
                st.session_state.shifts_time["Check out start"] = st.time_input(
                    "CHECK OUT START",
                    value=st.session_state.shifts_time["Check out start"],
                    step=60
                )

            with col2:
                st.session_state.shifts_time["Check in end"] = st.time_input(
                    "CHECK IN END",
                    value=st.session_state.shifts_time["Check in end"],
                    step=60
                )
                st.session_state.shifts_time["Check out end"] = st.time_input(
                    "CHECK OUT END",
                    value=st.session_state.shifts_time["Check out end"],
                    step=60
                )

        with tabs[2]:
            st.header("🎥 Attendance")

            if not st.session_state.conn:
                st.error("❌ Not connected to database!")
            else:
                app = app_st()
                mode = st.selectbox("SELECT MODE", ["REAL TIME", "VIDEO"])

                if mode == "VIDEO":
                    video_file = st.file_uploader("📁 Select Video", type=["mp4"])

                    if video_file:
                        bytes_data = video_file.read()
                        col1, col2 = st.columns(2)
                        with col1:
                            st.video(bytes_data)
                        with col2:
                            handle_attendance_recognition(
                                app, st.session_state.conn,
                                st.session_state.shifts_time,
                                mode, video_file, bytes_data
                            )
                else:
                    handle_attendance_recognition(
                        app, st.session_state.conn,
                        st.session_state.shifts_time,
                        mode
                    )

        with tabs[3]:
            st.header("➕ Register New Face")

            if not st.session_state.conn:
                st.error("❌ Not connected to database!")
            else:
                mode = st.selectbox("SELECT MODE", ["REAL TIME", "VIDEO"], key="register_mode")
                handle_register_capture(base_path, st.session_state.conn, mode)

        with tabs[4]:
            st.header("🗑️ Remove Employee")

            if not st.session_state.conn:
                st.error("❌ Not connected to database!")
            else:
                try:
                    with st.session_state.conn.cursor() as cur:
                        cur.execute("""
                                    SELECT DISTINCT e.name
                                    FROM employees e
                                             INNER JOIN face_embeddings f ON e.emp_id = f.emp_id
                                    """)
                        all_names = [r[0] for r in cur.fetchall()]

                    if all_names:
                        name_person = st.selectbox("Select employee to delete:", all_names)

                        if st.button("🗑️ Delete", type="primary"):
                            with st.session_state.conn.cursor() as cur:
                                cur.execute("SELECT emp_id FROM employees WHERE name = %s", (name_person,))
                                row = cur.fetchone()

                                if row:
                                    emp_id = row[0]
                                    mess_remove = remove_embeddings(st.session_state.conn, emp_id)

                                    if mess_remove:
                                        st.info(mess_remove)

                                    name_folder_path = os.path.join(base_path, name_person)
                                    if os.path.exists(name_folder_path):
                                        shutil.rmtree(name_folder_path)
                                        st.success(f"✅ Deleted folder: {name_folder_path}")

                                    st.rerun()
                    else:
                        st.info("ℹ️ No employees found.")

                except Exception as e:
                    st.error(f"❌ Error: {e}")

        with tabs[5]:
            st.header("📊 Monthly Statistics")

            if not st.session_state.conn:
                st.error("❌ Not connected to database!")
            else:
                current_year = datetime.today().year
                current_month = datetime.today().month

                col1, col2 = st.columns(2)
                with col1:
                    selected_year = st.selectbox(
                        "Select year",
                        [current_year - 2, current_year - 1, current_year],
                        index=2
                    )
                with col2:
                    selected_month = st.selectbox(
                        "Select month",
                        list(range(1, 13)),
                        index=current_month - 1
                    )

                if st.button("📈 View Statistics"):
                    month_df = get_monthly_report(st.session_state.conn, selected_year, selected_month)

                    if not month_df.empty:
                        st.subheader(f"Statistics for {selected_month}/{selected_year}")

                        total_hours = month_df["total_hours"].iloc[0] or 0
                        working_days = month_df["working_days"].iloc[0] or 0
                        avg_hours = month_df["avg_hours_per_day"].iloc[0] or 0

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("⏱ Total Hours", f"{total_hours:.2f}h")
                        with col2:
                            st.metric("📅 Working Days", int(working_days))
                        with col3:
                            st.metric("🧮 Avg Hours/Day", f"{avg_hours:.2f}h")

                        monthly_data = []
                        for month in range(1, 13):
                            month_df = get_monthly_report(st.session_state.conn, selected_year, month)
                            if not month_df.empty:
                                monthly_data.append({
                                    "Month": month,
                                    "Total hours": month_df["total_hours"].iloc[0] or 0,
                                    "Working days": month_df["working_days"].iloc[0] or 0,
                                    "Avg hours/day": month_df["avg_hours_per_day"].iloc[0] or 0
                                })

                        if monthly_data:
                            df = pd.DataFrame(monthly_data)

                            col1, col2 = st.columns(2)
                            with col1:
                                fig = px.bar(df, x='Month', y='Avg hours/day',
                                             title='Average Hours/Day',
                                             color_discrete_sequence=['#e74c3c'])
                                fig.update_xaxes(dtick=1)
                                st.plotly_chart(fig, use_container_width=True)

                            with col2:
                                fig2 = px.line(df, x='Month', y='Total hours',
                                               title='Total Hours Trend', markers=True)
                                fig2.update_traces(line_color='#2ecc71', line_width=3)
                                st.plotly_chart(fig2, use_container_width=True)
                    else:
                        st.warning("⚠️ No data available!")

    except Exception as e:
        st.error(f"❌ Error: {e}")


if __name__ == "__main__":
    streamlit_app(base_path=ORIGINAL_IMG_PATH)
