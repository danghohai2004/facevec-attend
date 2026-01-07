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

                if check_in_start > check_in_end and now.time() < time(12, 0):
                    working_date = (now - timedelta(days=1)).date()

                cur.execute("""
                    SELECT 1
                    FROM attendance_logs
                    WHERE emp_id = %s AND checkout_time IS NULL
                """, (emp_id,))

                if cur.fetchone():
                    return "Already checked in"

                cur.execute("""
                    INSERT INTO attendance_logs (emp_id, working_date, checkin_time)
                    VALUES (%s, %s, %s)
                """, (emp_id, working_date, now)) 

                conn.commit()
                return "Check in successful"

            elif check_type == "check_out":

                cur.execute("""
                    SELECT log_id
                    FROM attendance_logs
                    WHERE emp_id = %s AND checkout_time IS NULL
                    ORDER BY checkin_time DESC
                    LIMIT 1
                """, (emp_id,))

                row = cur.fetchone()
                if row is None:
                    return "Check in not found to check out"

                log_id = row[0]

                cur.execute("""
                    UPDATE attendance_logs
                    SET checkout_time = %s
                    WHERE log_id = %s
                """, (now, log_id)) 

                conn.commit()
                return "Check out successful"

    except Exception as e:
        conn.rollback()
        return f"[ERROR LOG] {e}"
