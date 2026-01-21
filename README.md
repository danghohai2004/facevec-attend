# Facial Recognition Attendance System Using Embeddings

In many enterprises, schools, and organizations, attendance tracking still relies heavily on traditional methods such as magnetic cards, fingerprint scanners, or manual data entry. These approaches suffer from several limitations:
- Attendance fraud
- Time-consuming operation and management
- Poor scalability as the number of users increases
- Incompatibility with modern automated systems

The goal of this project is to build an automated face-recognition-based attendance system that operates in real time, delivers high accuracy, and is scalable for real-world deployment.

## Project Overview

The attendance system is built on a face embedding–based approach, rather than direct image matching or training a separate classification model for each individual.

Faces are processed using  [InsightFace](https://github.com/deepinsight/insightface/tree/master/python-package) to extract embedding vectors, which are stored in PostgreSQL as vector data via [ankane/pgvector](https://hub.docker.com/r/ankane/pgvector). During recognition, the system leverages pgvector’s cosine distance (<=>) to compare incoming embeddings with stored data, identify the user, and automatically record attendance.

This embedding-based approach provides several key advantages:
- Fast recognition suitable for real-time applications
- Strong scalability with large numbers of users
- No need to retrain the model when new users are added
- Reduced application-layer overhead by performing vector distance computation directly within PostgreSQL

### Click here to watch the demo video: [Watch the video](video_demo.mp4)

## Recognition Model & Database Design

### Model Comparison & Benchmarks

Multiple [InsightFace](https://github.com/deepinsight/insightface/tree/master/python-package) models were benchmarked in a CPU-only environment to evaluate performance under real-world deployment conditions. The recognition model can be configured via `config.py`.

Test environment:
- CPU: AMD Ryzen 7 5800H
- RAM: 16GB
- inference: CPU-only
- Evaluation strategy: process every 5 frames, average over 300 runs

| Model | End-to-End Inference Time  |
|-------|----|
| `buffalo_sc` | **~19.9 ms**|
| `buffalo_s` | ~100 ms   |
| `buffalo_m` | ~175 ms   |


**Conclusion:** `buffalo_sc` achieves the lowest latency for both embedding extraction and database cosine distance queries, making it the most suitable choice for real-time, CPU-based attendance systems.

### Database Schema & Distance Search

PostgreSQL is used as the central storage and distance search engine, including:
- User profile information
- Face embedding vectors (stored using pgvector)
- Cosine distance values between embeddings
- Attendance records: name, date, check-in time, check-out time, total working hours

Distance thresholds and related parameters can be configured via `config.py`

## Main Directories & Files

`faces/` – Stores user images after registration

`initdb/` – Scripts to initialize database tables

`src/` – Core processing source code

`utils/` – Utility functions and helpers

`app_streamlit` – User interface application

Configuration files: `.env.example`, `config.py`, `compose.yaml`, `requirements.txt`

**Note:** Update configurations in `config.py` and `.env.example`.

## 🔧 Installation & Usage
### Requirements:
- Python 3.12
- Docker

### 1) Clone the Repository
    git clone https://github.com/danghohai2004/facevec-attend.git
    cd facevec-attend

### 2) Set up Virtual Environment and Install Dependencies

```powershell
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# 3. Install required packages
pip install -r requirements.txt
```

### 3) Configure Environment Variables
```powershell
mv .env.example .env
# Please review and update the configuration in config.py and .env.
```
### 4) Run with Docker Compose
    docker-compose up --build -d

### 5) Access UI:
    python -m streamlit run app_streamlit.py

## 🗄️ Querying PostgreSQL
### 1) Access PostgreSQL container:
```powershell
docker exec -it <container_name_db> bash
# <container_name_db> -> see .env -> CONTAINER_NAME_DB
```

### 2) Connect via psql:
```powershell
psql -U <database_user>
# <database_user> -> see .env -> DB_USER
```
### 3) Select database:
```powershell
\c <database_name>
# <database_name> -> see .env -> DB_NAME
```

### 4) Query Database Tables
```powershell
# 1. List all tables
\dt
# 2. Execute a query on a specific table
\pset pager off    
SELECT * FROM <table_name>;
```
### 5) Stop and Remove All Containers, Networks, and Volumes
    docker compose down -v
   
## Future Enhancements
The system captures faces from a camera or video stream, generates embedding vectors, and stores them in PostgreSQL along with user information.

## 📜 License
The system may be updated in the future to include face anti-spoofing / liveness detection, which will help prevent fraudulent attempts using photos, videos, or masks, ensuring that only real, live faces are logged for attendance.

## 📬 Contact

- 📧 **Email** — [contact me](mailto:dhhaics2004@gmail.com?subject=Question%20about%20the%20Face%20Mask%20Detection%20Project)
- 🌐 **GitHub** — [@danghohai2004](https://github.com/danghohai2004)

---

# Thank you for your interest in this project!