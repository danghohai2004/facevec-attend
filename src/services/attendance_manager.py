from datetime import datetime

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
    working_date = now.date()

    try:
        with conn.cursor() as cur:

            if check_type == "check_in":
                cur.execute("""
                    SELECT checkin_time
                    FROM attendance_logs
                    WHERE emp_id = %s AND working_date = %s
                """, (emp_id, working_date))
                last_log = cur.fetchone()

                if last_log is None:
                    cur.execute("""
                        INSERT INTO attendance_logs (emp_id, working_date, checkin_time)
                        VALUES (%s, %s, %s)
                    """, (emp_id, working_date, now.time()))
                    conn.commit()
                    return "Check in successful"
                else:
                    return "Checked in today"

            elif check_type == "check_out":
                cur.execute("""
                            SELECT working_date, checkin_time
                            FROM attendance_logs
                            WHERE emp_id = %s AND checkout_time IS NULL
                            ORDER BY working_date DESC 
                            LIMIT 1
                            """, (emp_id,))
                last_log = cur.fetchone()

                if last_log is None:
                    return "Check in not found to check out"

                checkin_date, checkin_time = last_log

                cur.execute("""
                            UPDATE attendance_logs
                            SET checkout_time = %s, working_duration = CASE
                                                                            WHEN (working_date + checkin_time) <= (working_date + %s)
                                                                                THEN (working_date + %s) - (working_date + checkin_time)
                                                                            ELSE (working_date + %s + INTERVAL '1 day') - (working_date + checkin_time)
                                                                        END
                            WHERE emp_id = %s AND working_date = %s
                            """,(now.time(), now.time(), now.time(), now.time(), emp_id, checkin_date))
                conn.commit()
                return "Check out successful"

    except Exception as e:
        conn.rollback()
        return f"[ERROR LOG] {e}"
