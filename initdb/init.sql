CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE employees (
    emp_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL
);

CREATE TABLE face_embeddings (
    embedding_id SERIAL PRIMARY KEY,
    emp_id INT NOT NULL,
    embedding vector(512),
    FOREIGN KEY (emp_id) REFERENCES employees(emp_id) ON DELETE CASCADE
);

CREATE TABLE attendance_logs (
    log_id SERIAL PRIMARY KEY,
    emp_id INT NOT NULL,
    working_date DATE NOT NULL,
    checkin_time TIMESTAMP NOT NULL,
    checkout_time TIMESTAMP,
    working_duration INTERVAL GENERATED ALWAYS AS
        (checkout_time - checkin_time) STORED,
    FOREIGN KEY (emp_id)
        REFERENCES employees(emp_id)
        ON DELETE CASCADE,
    CONSTRAINT valid_attendance_time
        CHECK (checkout_time IS NULL OR checkout_time > checkin_time)
);