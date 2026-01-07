# Facial Recognition Attendance System Using Embeddings
A smart face-recognition attendance system powered by InsightFace, providing fast and accurate face embeddings.

It detects and identifies faces in real time, logs attendance automatically, and stores all user + attendance data securely in PostgreSQL.
## Project Overview
The system captures faces from a camera or video stream, generates embedding vectors, and stores them in PostgreSQL along with user information.

During recognition, cosine similarity is computed between new and stored embeddings to identify the user, after which attendance is logged automatically.
## Recognition Model & Database
### Recognition Model
I tested several InsightFace models (`buffalo_sc`, `buffalo_s`, `buffalo_m`) on my CPU. You can try other models by changing the name in `config.py`:

| Model        | Summary                                                               |
|--------------| --------------------------------------------------------------------- |
| `buffalo_sc` | Lowest latency → best for real-time                                   |
| `buffalo_s`  | Adds 2D/3D alignment; slightly slower; small improvement              |
| `buffalo_m`  | Large detector & embeddings → slowest; higher accuracy; not ideal CPU |

**Conclusion:** `buffalo_sc` is the best fit for my setup.

### Database

PostgreSQL stores:

- User information

- Embeddings

- Cosine distance result between embeddings

- Attendance logs: name, date, check-in, check-out, total hours

Threshold parameters can be configuration via `config.py`

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

### 2) Configure Environment Variables
    mv .env.example .env
    # Please review and update the configuration in config.py and .env.

### 3) Run with Docker Compose
    docker-compose up --build

#### 4) Access UI:
    python -m streamlit run app_streamlit.py

## 🗄️ Querying PostgreSQL
### 1) Access PostgreSQL container:

    docker exec -it <container_name_db> bash

#### `container_name_db` -> see`.env` -> `CONTAINER_NAME_DB`

### 2) Connect via psql:

    psql -U <database_user>

#### `database_user` -> see`.env` -> `DB_USER`

### 3) Select database:

    \c <database_name>
    
#### `database_name` -> see`.env` -> `DB_NAME`

### 4) View tables:
    
    \dt

#### Execute a data query on the table:
    \pset pager off    
    SELECT * FROM <table_name>;

### ! Use `docker compose down -v` to remove all.

## Future Enhancements
The system captures faces from a camera or video stream, generates embedding vectors, and stores them in PostgreSQL along with user information.


## 📜 License
The system may be updated in the future to include face anti-spoofing / liveness detection, which will help prevent fraudulent attempts using photos, videos, or masks, ensuring that only real, live faces are logged for attendance.

## 📬 Contact

- 📧 **Email** — [contact me](mailto:dhhaics2004@gmail.com?subject=Question%20about%20the%20Face%20Mask%20Detection%20Project)
- 🌐 **GitHub** — [@danghohai2004](https://github.com/danghohai2004)

---

# Thank you for your interest in this project!