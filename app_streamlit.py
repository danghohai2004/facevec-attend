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
from utils.draw_bbox import draw_bbox
from src.core.face_identifier import identify_person_pgvector
from src.services.attendance_manager import get_current_time
from src.services.embedding_manager import add_info_embeddings, remove_embeddings
from config import ORIGINAL_IMG_PATH, THRESHOLD, MAX_EMB_FACE


@st.cache_resource
def app_st():
    app = setup_face_app()
    return app

def get_monthly_report(conn, selected_year, selected_month):
    query = """
                SELECT
                    EXTRACT(MONTH FROM working_date) AS month,
                    ROUND(SUM(EXTRACT(EPOCH FROM working_duration)) / 3600, 2) AS total_hours,
                    COUNT(DISTINCT working_date) AS working_days,
                    ROUND(
                        SUM(EXTRACT(EPOCH FROM working_duration)) / 3600
                        / COUNT(DISTINCT working_date),
                        2
                    ) AS avg_hours_per_day
                FROM attendance_logs
                WHERE
                    EXTRACT(YEAR FROM working_date) = %s
                    AND EXTRACT(MONTH FROM working_date) = %s
                GROUP BY month;
            """
    df = pd.read_sql_query(query, conn, params=(selected_year, selected_month))
    return df

def streamlit_app(threshold, tota_emb_face, base_path):
    st.set_page_config(page_title="Face Attendance", layout="wide")
    st.image("template/face_icon.png", width=80)
    st.title("Face Attendance System")

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

    if "different_name" not in st.session_state:
        st.session_state.different_name = False

    tabs = st.tabs(["🗄️ CONNECT DATABASE", "⏰ SHIFT SETTINGS", "🎥 ATTENDANCE", "➕ REGISTER", "🗑️ REMOVE", "📊 STATISTICS"])

    try:
        # TAB 0: Database Connection
        with tabs[0]:
            st.header("🗄️ Connect Database")

            if st.button("Connect Database"):
                if st.session_state.conn is None:
                    st.session_state.conn, mess_log = get_connection()

                if not st.session_state.conn and mess_log is not None:
                    st.error(f"{mess_log} -> Cannot connect to database!")
                else:
                    st.success("Connected to DB!")

            if st.button("Close Database") and st.session_state.conn is not None:
                st.session_state.conn.close()
                st.session_state.conn = None
                st.success("Database closed")

        # TAB 1: Shift Setting
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

        # TAB 2: Attendance
        with tabs[2]:
            st.header("🛂 ATTENDANCE")
            if st.session_state.conn is None:
                st.error("Not connected to the database")
            elif st.session_state.conn is not None:
                app = app_st()
                select_mode = st.selectbox("SELECT MODE FOR ATTENDANCE", ["REAL TIME", "VIDEO"])

                if select_mode:
                    if select_mode == "REAL TIME":
                        st.header("🎥 REAL TIME ")
                        start_attendance_realtime = st.button("▶ Start Attendance")
                        stop_attendance_realtime = st.button("⏹ Stop Attendance")

                        warning_placeholder_realtime = st.empty()

                        if start_attendance_realtime:
                            st.session_state.face_reco_running = True
                        if stop_attendance_realtime:
                            st.session_state.face_reco_running = False
                        if st.session_state.face_reco_running:
                            cap_realtime = cv2.VideoCapture(0)
                            stframe = st.empty()

                            if not cap_realtime.isOpened():
                                warning_placeholder_realtime.error("Cannot Open Camera!")
                            else:
                                frame_count = 0
                                try:
                                    while st.session_state.face_reco_running:
                                        ret, frame = cap_realtime.read()
                                        if not ret:
                                            warning_placeholder_realtime.error("Cannot read from camera!")
                                            break
                                        
                                        frame_count += 1

                                        if frame_count % 5 == 0:                                            
                                            faces = app.get(frame)

                                            if faces:
                                                faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
                                                main_face = faces[0]
                                                emb_person_frame = main_face.normed_embedding

                                                error_identify, mess_logs, emp_id, name = identify_person_pgvector(
                                                    st.session_state.conn, emb_person_frame, threshold,
                                                    st.session_state.shifts_time)
                                                    
                                                if mess_logs is not None:
                                                    if not st.session_state.warned_mess_logs:
                                                        st.warning(name + " - " + mess_logs)
                                                        st.session_state.warned_mess_logs = True
                                                else:
                                                    st.session_state.warned_mess_logs = False

                                                if error_identify is not None:
                                                    if not st.session_state.warned_pgvector:
                                                        st.warning(error_identify)
                                                        st.session_state.warned_pgvector = True
                                                else:
                                                    st.session_state.warned_pgvector = False

                                                bbox = main_face.bbox.astype(int)
                                                draw_bbox(frame, bbox, color=(0, 255, 255), thickness=2, corner_len=10)
                                                cv2.putText(frame, name, (bbox[0], bbox[1] - 10),
                                                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                                        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width=700)

                                finally:
                                    cap_realtime.release()
                                    stframe.empty()

                    if select_mode == "VIDEO": 
                        st.header("🎬 VIDEO")

                        video_file = st.file_uploader("📁 Select Video", type=["mp4"], key="attendance_video")

                        if video_file:
                            start_attendance_video = st.button("▶ Start Attendance")
                            stop_attendance_video = st.button("⏹ Stop Attendance")

                            warning_placeholder_video = st.empty()
                            mess_logs_placeholder = st.empty()
                            error_placeholder = st.empty()

                            col1, col2 = st.columns(2)
                            bytes_data = video_file.read()

                            with col1:
                                st.video(bytes_data, width=700)

                            with col2:
                                if start_attendance_video:
                                    try: 
                                        tfile = tempfile.NamedTemporaryFile(delete=False)
                                        tfile.write(bytes_data)

                                        cap_video = cv2.VideoCapture(tfile.name)
                                        stframe_video = st.empty()

                                        while cap_video.isOpened():
                                            ret, frame = cap_video.read()
                                            if not ret or stop_attendance_video:
                                                break

                                            within_shift, now, check_type = get_current_time(st.session_state.shifts_time)

                                            if st.session_state.shown_warnings["not_during_shift"]:
                                                warning_placeholder_video.empty()
                                                st.session_state.shown_warnings["not_during_shift"] = False

                                            if not within_shift:
                                                if not st.session_state.shown_warnings["not_during_shift"]:
                                                    warning_placeholder_video.warning("Not during working hours!")
                                                    st.session_state.shown_warnings["not_during_shift"] = True

                                            if within_shift:
                                                if not st.session_state.shown_warnings["not_during_shift"]:
                                                    warning_placeholder_video.info("PROCESSING...")
                                                    st.session_state.shown_warnings["not_during_shift"] = True

                                            if check_type in ["check_in", "check_out"]:
                                                faces = app.get(frame)

                                                if faces:
                                                    faces.sort(
                                                        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
                                                    main_face = faces[0]
                                                    emb_person_frame = main_face.normed_embedding

                                                    error_identify, mess_logs, emp_id, name = identify_person_pgvector(
                                                        st.session_state.conn,
                                                        emb_person_frame,
                                                        threshold,
                                                        st.session_state.shifts_time
                                                    )

                                                    if mess_logs is not None:
                                                        if not st.session_state.warned_mess_logs:
                                                            mess_logs_placeholder.warning(name + " - " + mess_logs)
                                                            st.session_state.warned_mess_logs = True
                                                    else:
                                                        if st.session_state.warned_mess_logs:
                                                            mess_logs_placeholder.empty()
                                                            st.session_state.warned_mess_logs = False

                                                    if error_identify is not None:
                                                        if not st.session_state.warned_pgvector:
                                                            error_placeholder.warning(error_identify)
                                                            st.session_state.warned_pgvector = True
                                                    else:
                                                        if st.session_state.warned_pgvector:
                                                            error_placeholder.empty()
                                                            st.session_state.warned_pgvector = False

                                                    bbox = main_face.bbox.astype(int)
                                                    draw_bbox(frame, bbox, color=(0, 255, 255), thickness=2, corner_len=10)
                                                    cv2.putText(frame, name, (bbox[0], bbox[1] - 10),
                                                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                                            stframe_video.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width=700)

                                    finally:
                                        cap_video.release()
                                        stframe_video.empty()

        # TAB 3: Register
        with tabs[3]:
            st.header("👤 REGISTER")
           
            if st.session_state.conn is None:
                st.error("Not connected to the database")
            elif st.session_state.conn is not None:
                select_mode = st.selectbox("SELECT MODE for REGISTER", ["REAL TIME", "VIDEO"])
           
                if select_mode == "REAL TIME":
                    name_person = st.text_input("Enter your name and press the Enter Key - Real Time", key="name_realtime")

                    if name_person:
                        start_capture = st.button("Start automatic capture")
                        if start_capture:
                            st.session_state.capturing = True
                            st.session_state.capture_faces = []
                            st.session_state.last_capture_time = 0

                        stop_capture = st.button("Stop automatic capture")
                        if stop_capture:
                            st.session_state.capturing = False
                            st.session_state.capture_faces = []
                            st.session_state.last_capture_time = 0

                        if st.session_state.capturing == True:
                            stframe = st.empty()
                            process_bar = st.progress(0)

                            cap_realtime = cv2.VideoCapture(0)

                            if not cap_realtime.isOpened():
                                st.error("Cannot Open Camera!")
                                
                            else:
                                try:
                                    while st.session_state.capturing:
                                        ret, frame = cap_realtime.read()
                                        if not ret:
                                            st.error("Cannot read from camera!")
                                            break

                                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                        stframe.image(frame_rgb, channels="RGB", width=700)

                                        if time() - st.session_state.last_capture_time > 1.5:
                                            st.session_state.capture_faces.append(frame_rgb)
                                            st.session_state.last_capture_time = time()

                                            count = len(st.session_state.capture_faces)
                                            process_bar.progress(count / tota_emb_face)

                                        if len(st.session_state.capture_faces) >= tota_emb_face:
                                            st.session_state.capturing = False
                                finally:
                                    cap_realtime.release()
                                    stframe.empty()

                            valid_faces = [f for f in st.session_state.capture_faces if f is not None]

                            if valid_faces:
                                save_dir = os.path.join(base_path, name_person)

                                is_exists = os.path.exists(save_dir)

                                if is_exists:
                                    st.info(f"User {name_person} already exists. Please choose a suitable option:")

                                    if st.button("Add images to this user"):
                                        existing_files = [f for f in os.listdir(save_dir) if f.endswith(".jpg")]
                                        start_index = len(existing_files) + 1

                                        for i, img in enumerate(valid_faces[1:], start=start_index):
                                            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                                            save_path = os.path.join(save_dir, f"{i}.jpg")
                                            cv2.imwrite(save_path, img_bgr)

                                        st.success(f"saved {len(valid_faces)} the photo to {save_dir}")

                                        success_emb_add_db, error_emb_add_db = add_info_embeddings(
                                            st.session_state.conn, base_path, name_person)

                                        if success_emb_add_db is not None:
                                            st.success(success_emb_add_db)
                                        if error_emb_add_db is not None:
                                            st.success(error_emb_add_db)

                                        st.session_state.capture_faces = []

                                    if st.button("Use a different name"):
                                        st.session_state.different_name = True

                                    if st.session_state.different_name:
                                        diff_name = st.text_input(
                                            "Please Enter a different name and press the Enter Key")

                                        if diff_name:
                                            if diff_name == name_person:
                                                st.error("Please choose a different name!")
                                            elif os.path.exists(os.path.join(base_path, diff_name)):
                                                st.error(f"User '{diff_name}' already exists!")
                                            else:
                                                save_dir_different = os.path.join(base_path, diff_name)
                                                os.makedirs(save_dir_different, exist_ok=True)

                                                for i, img in enumerate(valid_faces[1:], start=1):
                                                    img_bgr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                                    save_path_different = os.path.join(save_dir_different, f"{i}.jpg")
                                                    cv2.imwrite(save_path_different, img_bgr)

                                                st.success(f"saved {len(valid_faces)} the photo to {save_dir_different}")

                                                success_emb_add_db, error_emb_add_db = add_info_embeddings(
                                                    st.session_state.conn, base_path, diff_name)

                                                if success_emb_add_db is not None:
                                                    st.success(success_emb_add_db)
                                                if error_emb_add_db is not None:
                                                    st.success(error_emb_add_db)

                                                st.session_state.capture_faces = []
                                                st.session_state.different_name = False

                                else:
                                    os.makedirs(save_dir, exist_ok=True)

                                    for i, img in enumerate(valid_faces[1:], start=1):
                                        img_bgr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                        save_path = os.path.join(save_dir, f"{i}.jpg")
                                        cv2.imwrite(save_path, img_bgr)

                                    st.success(f"saved {len(valid_faces)} the photo to {save_dir}")
                                    success_emb_add_db, error_emb_add_db = add_info_embeddings(st.session_state.conn, base_path, name_person)

                                    if success_emb_add_db is not None:
                                        st.success(success_emb_add_db)
                                    if error_emb_add_db is not None:
                                        st.success(error_emb_add_db)

                                    st.session_state.capture_faces = []

                if select_mode == "VIDEO":
                    name_person = st.text_input("Enter your name and press the Enter Key - Video", key="name_video")

                    if name_person:
                        video_cap = st.file_uploader("📁 Select Video", type=["mp4"], key="register_video")
                       
                        if video_cap is not None:
                            start_capture = st.button("Start automatic capture")
                            if start_capture:
                                st.session_state.capturing = True
                                st.session_state.capture_faces = []
                                st.session_state.last_capture_time = 0

                            stop_capture = st.button("Stop automatic capture")
                            if stop_capture:
                                st.session_state.capturing = False
                                st.session_state.capture_faces = []
                                st.session_state.last_capture_time = 0

                            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                            tfile.write(video_cap.read())

                            if start_capture:
                                process_bar = st.progress(0)

                                cap = cv2.VideoCapture(tfile.name)
                                stframe = st.empty()

                                while st.session_state.capturing:
                                    ret, frame = cap.read()
                                    if not ret:
                                        st.error("Video Timed Out!")
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

                            valid_faces = [f for f in st.session_state.capture_faces if f is not None]

                            if valid_faces:
                                save_dir = os.path.join(base_path, name_person)

                                is_exists = os.path.exists(save_dir)

                                if is_exists:
                                    st.info(f"User {name_person} already exists. Please choose a suitable option:")

                                    if st.button("Add images to this user"):
                                        existing_files = [f for f in os.listdir(save_dir) if f.endswith(".jpg")]
                                        start_index = len(existing_files) + 1

                                        for i, img in enumerate(valid_faces[1:], start=start_index):
                                            img_bgr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                            save_path = os.path.join(save_dir, f"{i}.jpg")
                                            cv2.imwrite(save_path, img_bgr)

                                        st.success(f"saved {len(valid_faces)} the photo to {save_dir}")

                                        success_emb_add_db, error_emb_add_db = add_info_embeddings(
                                            st.session_state.conn, base_path, name_person)

                                        if success_emb_add_db is not None:
                                            st.success(success_emb_add_db)
                                        if error_emb_add_db is not None:
                                            st.success(error_emb_add_db)

                                        st.session_state.capture_faces = []

                                    if st.button("Use a different name"):
                                        st.session_state.different_name = True

                                    if st.session_state.different_name:
                                        diff_name = st.text_input(
                                            "Please Enter a different name and press the Enter Key")

                                        if diff_name:
                                            if diff_name == name_person:
                                                st.error("Please choose a different name!")
                                            elif os.path.exists(os.path.join(base_path, diff_name)):
                                                st.error(f"User '{diff_name}' already exists!")
                                            else:
                                                save_dir_different = os.path.join(base_path, diff_name)
                                                os.makedirs(save_dir_different, exist_ok=True)

                                                for i, img in enumerate(valid_faces[1:], start=1):
                                                    img_bgr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                                    save_path_different = os.path.join(save_dir_different, f"{i}.jpg")
                                                    cv2.imwrite(save_path_different, img_bgr)

                                                st.success(
                                                    f"saved {len(valid_faces)} the photo to {save_dir_different}")

                                                success_emb_add_db, error_emb_add_db = add_info_embeddings(
                                                    st.session_state.conn, base_path, diff_name)

                                                if success_emb_add_db is not None:
                                                    st.success(success_emb_add_db)
                                                if error_emb_add_db is not None:
                                                    st.success(error_emb_add_db)

                                                st.session_state.capture_faces = []
                                                st.session_state.different_name = False

                                else:
                                    os.makedirs(save_dir, exist_ok=True)

                                    for i, img in enumerate(valid_faces[1:], start=1):
                                        img_bgr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                        save_path = os.path.join(save_dir, f"{i}.jpg")
                                        cv2.imwrite(save_path, img_bgr)

                                    st.success(f"saved {len(valid_faces)} the photo to {save_dir}")
                                    success_emb_add_db, error_emb_add_db = add_info_embeddings(st.session_state.conn,
                                                                                               base_path, name_person)

                                    if success_emb_add_db is not None:
                                        st.success(success_emb_add_db)
                                    if error_emb_add_db is not None:
                                        st.success(error_emb_add_db)

                                    st.session_state.capture_faces = []

        # TAB 4: Remove
        with tabs[4]:
            st.header("👥 EMPLOYEES LIST")

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

        # TAB 5: Statistics
        with tabs[5]:
            st.header("📅 Monthly Attendance Report")

            if st.session_state.conn is None:
                st.error("Not connected to the database")
            elif st.session_state.conn is not None:
                current_year = datetime.today().year
                one_year_ago = datetime.today().year - 1
                two_year_ago = datetime.today().year - 2

                current_month = datetime.today().month

                col1, col2 = st.columns(2)

                with col1:
                    selected_year = st.selectbox("Select year", [two_year_ago, one_year_ago, current_year], index = 2)
                with col2:
                    selected_month = st.selectbox("Select month", list(range(1, 13)), index = current_month - 1)

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

if __name__ == "__main__":
    streamlit_app(threshold=THRESHOLD, tota_emb_face = MAX_EMB_FACE, base_path=ORIGINAL_IMG_PATH)
