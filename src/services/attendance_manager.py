from datetime import datetime, time, timedelta

def get_current_time(shifts_time):
    now = datetime.now()
    current_time = now.time()

    check_in_start = shifts_time["Check in start"]
    check_in_end = shifts_time["Check in end"]
    check_out_start = shifts_time["Check out start"]
    check_out_end = shifts_time["Check out end"]

    def is_time_in_range(current, start, end):
        if start <= end:
            return start <= current <= end
        else:
            return current >= start or current <= end

    if is_time_in_range(current_time, check_in_start, check_in_end):
        return True, now, "check_in"

    if is_time_in_range(current_time, check_out_start, check_out_end):
        return True, now, "check_out"

    return False, now, None

def log_attendance(conn, emp_id, shifts_time):
    within_shift, now, check_type = get_current_time(shifts_time)

    if not within_shift:
        return "Not during working hours"

    try:
        with conn.cursor() as cur:

            if check_type == "check_in":
                working_date = now.date()

                check_in_start = shifts_time["Check in start"]
                check_in_end = shifts_time["Check in end"]

                if check_in_start > check_in_end and now.time() < time(12,0):
                    working_date = (now - timedelta(days=1)).date()

                cur.execute("""
                    SELECT checkin_time
                    FROM attendance_logs
                    WHERE emp_id = %s AND working_date = %s
                """, (emp_id, working_date))
                existing_log = cur.fetchone()

                if existing_log is None:
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
                            ORDER BY working_date DESC, checkin_time DESC
                            LIMIT 1
                            """, (emp_id,))
                last_log = cur.fetchone()

                if last_log is None:
                    return "Check in not found to check out"

                checkin_date, checkin_time = last_log

                checkin_date_time = datetime.combine(checkin_date, checkin_time)
                checkout_date_time = now

                if checkout_date_time < checkin_date_time:
                    checkout_date_time = checkout_date_time + timedelta(days=1)

                working_duration = checkout_date_time - checkin_date_time

                cur.execute("""
                            UPDATE attendance_logs
                            SET checkout_time = %s, working_duration = %s
                            WHERE emp_id = %s AND working_date = %s AND checkout_time IS NULL
                            """,(now.time(), working_duration, emp_id, checkin_date))
                conn.commit()
                return "Check out successful"

    except Exception as e:
        conn.rollback()
        return f"[ERROR LOG] {e}"
