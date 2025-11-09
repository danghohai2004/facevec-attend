CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE employees (
    emp_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
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
    checkin_time TIME,               
    checkout_time TIME,                
    working_duration INTERVAL,
	FOREIGN KEY (emp_id) REFERENCES employees(emp_id) ON DELETE CASCADE
);