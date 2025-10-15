from datetime import datetime
import streamlit as st

def get_current_time(shifts_time):
    now = datetime.now()
    current_time = now.time()

    if shifts_time["Check in start"] <= current_time <= shifts_time["Check in end"]:
        return True, now, "check_in"
    elif shifts_time["Check out start"] <= current_time <= shifts_time["Check out end"]:
        return True, now, "check_out"

    return False, now, None

def log_attendance(conn, emp_id, shifts_time):
    within_shift, now, check_type = get_current_time(shifts_time)
    log_time = now.strftime("%Y-%m-%d %H:%M:%S")

    if not within_shift:
        return

    try:
        with conn.cursor() as cur:
            working_date = now.date()
            cur.execute("""
                SELECT checkin_time, checkout_time
                FROM attendance_logs
                WHERE emp_id = %s AND working_date = %s
            """, (emp_id, working_date))

            last_log = cur.fetchone()

            if check_type == "check_in":
                if last_log is None:
                    cur.execute("""
                        INSERT INTO attendance_logs (emp_id, working_date, checkin_time)
                        VALUES (%s, %s, %s)
                    """, (emp_id, working_date, now.time()))

            elif check_type == "check_out":
                if last_log is None:
                    st.warning("Cannot check out before check in")
                elif last_log[1] is None:
                    cur.execute("""
                        UPDATE attendance_logs
                        SET checkout_time = %s, working_duration = %s - checkin_time
                        WHERE emp_id = %s AND working_date = %s
                    """, (now.time(), now.time(), emp_id, working_date))

            conn.commit()

    except Exception as e:
        conn.rollback()
        st.warning(f"[DB ERROR] {e}")
