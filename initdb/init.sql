CREATE TABLE employees (
    emp_id   SERIAL PRIMARY KEY,
    emp_code VARCHAR(50) UNIQUE NOT NULL,
    name     VARCHAR(100) NOT NULL
);

CREATE TABLE attendance_logs (
    log_id           SERIAL PRIMARY KEY,
    emp_id           INT NOT NULL,
    working_date     DATE NOT NULL,
    checkin_time     TIMESTAMP NOT NULL,
    checkout_time    TIMESTAMP,
    working_duration INTERVAL GENERATED ALWAYS AS (checkout_time - checkin_time) STORED,
    FOREIGN KEY (emp_id) REFERENCES employees(emp_id) ON DELETE CASCADE,
    CONSTRAINT valid_attendance_time
        CHECK (checkout_time IS NULL OR checkout_time > checkin_time)
);

CREATE INDEX ON attendance_logs (emp_id, working_date);
CREATE UNIQUE INDEX ON attendance_logs (emp_id, working_date) WHERE checkout_time IS NULL;

CREATE TABLE shift_settings (
    id              SERIAL PRIMARY KEY,
    check_in_start  TIME NOT NULL,
    check_in_end    TIME NOT NULL,
    check_out_start TIME NOT NULL,
    check_out_end   TIME NOT NULL
);

INSERT INTO shift_settings (check_in_start, check_in_end, check_out_start, check_out_end)
VALUES ('08:00', '10:00', '17:00', '19:00');
