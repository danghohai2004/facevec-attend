# Data Schema

Hệ thống dùng **2 database tách biệt theo mục đích**:

| Database | Lưu gì |
|---|---|
| **PostgreSQL** | employees, attendance_logs, shift_settings (dữ liệu quan hệ) |
| **Qdrant** | face_embeddings (vector search) |

---

## Tổng quan Storage

```
┌─────────────────────────────────┐   ┌──────────────────────────────┐
│         PostgreSQL              │   │           Qdrant             │
│                                 │   │                              │
│  employees                      │   │  collection: face_embeddings │
│    emp_id, emp_code, name       │   │                              │
│         │                       │   │  point:                      │
│         │ emp_id dùng làm       │   │    id: UUID                  │
│         │ payload trong Qdrant  │   │    vector: float[512]        │
│         │                       │   │    payload: {                │
│  attendance_logs                │   │      emp_id: int,            │
│    log_id, emp_id,              │   │      emp_code: string,       │
│    working_date,                │   │      name: string            │
│    checkin/checkout_time        │   │    }                         │
│                                 │   │                              │
│  shift_settings                 │   │  index: HNSW + Cosine        │
│    check_in/out windows         │   │                              │
└─────────────────────────────────┘   └──────────────────────────────┘
```

**Tại sao Qdrant thay pgvector?**
- Vector search là first-class citizen, không phải extension bolt-on
- HNSW index tốt hơn, query nhanh hơn khi số lượng embedding lớn
- Có sẵn filtering theo payload (filter theo emp_id khi tìm kiếm)
- Dashboard UI tại `localhost:6333/dashboard` để debug collection

---

## PostgreSQL Tables

### `employees`

| Column | Type | Constraints | Mô tả |
|---|---|---|---|
| `emp_id` | SERIAL | PRIMARY KEY | ID tự tăng |
| `emp_code` | VARCHAR(50) | UNIQUE NOT NULL | Mã nhân viên (NV001...) |
| `name` | VARCHAR(100) | NOT NULL | Tên đầy đủ |

```sql
CREATE TABLE employees (
    emp_id   SERIAL PRIMARY KEY,
    emp_code VARCHAR(50) UNIQUE NOT NULL,
    name     VARCHAR(100) NOT NULL
);
```

---

### `attendance_logs`

Log chấm công theo ngày.

| Column | Type | Constraints | Mô tả |
|---|---|---|---|
| `log_id` | SERIAL | PRIMARY KEY | ID tự tăng |
| `emp_id` | INT | FK → employees CASCADE | Nhân viên |
| `working_date` | DATE | NOT NULL | Ngày làm việc |
| `checkin_time` | TIMESTAMP | NOT NULL | Giờ check-in |
| `checkout_time` | TIMESTAMP | nullable | Giờ check-out (null nếu chưa) |
| `working_duration` | INTERVAL | GENERATED ALWAYS | `checkout_time - checkin_time` |

```sql
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
```

**Trạng thái:**
- `checkout_time IS NULL` → đã check-in, chưa check-out
- `checkout_time IS NOT NULL` → hoàn thành ca

`working_duration` là computed column — PostgreSQL tự tính, app không cần set.

---

### `shift_settings`

Cấu hình khung giờ ca làm việc. Chỉ có 1 row (upsert).

| Column | Type | Mô tả |
|---|---|---|
| `id` | SERIAL PK | |
| `check_in_start` | TIME | Bắt đầu giờ check-in |
| `check_in_end` | TIME | Kết thúc giờ check-in |
| `check_out_start` | TIME | Bắt đầu giờ check-out |
| `check_out_end` | TIME | Kết thúc giờ check-out |

```sql
CREATE TABLE shift_settings (
    id              SERIAL PRIMARY KEY,
    check_in_start  TIME NOT NULL,
    check_in_end    TIME NOT NULL,
    check_out_start TIME NOT NULL,
    check_out_end   TIME NOT NULL
);

INSERT INTO shift_settings VALUES (DEFAULT, '08:00', '10:00', '17:00', '19:00');
```

**Logic:**
```
08:00──[check-in window]──10:00       17:00──[check-out window]──19:00
Ngoài 2 cửa sổ → "Not during working hours"
```

---

### Indexes PostgreSQL

```sql
-- Query attendance theo nhân viên + ngày
CREATE INDEX ON attendance_logs (emp_id, working_date);

-- Tìm log đang mở (chưa checkout) — dùng trong check-in guard
CREATE INDEX ON attendance_logs (emp_id, working_date) WHERE checkout_time IS NULL;
```

---

## Qdrant Collection

### `face_embeddings`

Mỗi point là 1 embedding khuôn mặt. Payload denormalize tên + mã nhân viên để tránh join về PostgreSQL sau khi search.

**Collection config:**
```python
from qdrant_client.models import Distance, VectorParams

client.create_collection(
    collection_name="face_embeddings",
    vectors_config=VectorParams(size=512, distance=Distance.COSINE),
)
```

**Point structure:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "vector": [0.12, -0.34, ..., 0.08],
  "payload": {
    "emp_id": 1,
    "emp_code": "NV001",
    "name": "Nguyen Van A"
  }
}
```

**Truy vấn nhận diện:**
```python
results = client.search(
    collection_name="face_embeddings",
    query_vector=embedding,   # float[512] từ insightface
    limit=1,
    score_threshold=1 - THRESHOLD,  # Qdrant dùng cosine similarity (cao = tốt)
)

if results:
    hit = results[0]
    emp_id   = hit.payload["emp_id"]
    name     = hit.payload["name"]
    distance = 1 - hit.score    # convert similarity → distance
```

> **Note:** pgvector dùng cosine **distance** (thấp = tốt, 0–2).  
> Qdrant dùng cosine **similarity** (cao = tốt, -1–1).  
> Convert: `distance = 1 - similarity`. `THRESHOLD = 0.6` tương đương `score_threshold = 0.4`.

**Xóa embeddings của 1 nhân viên:**
```python
client.delete(
    collection_name="face_embeddings",
    points_selector=Filter(
        must=[FieldCondition(key="emp_id", match=MatchValue(value=emp_id))]
    ),
)
```

---

## ORM Models (SQLAlchemy — PostgreSQL only)

`FaceEmbedding` model **bị xóa** — không còn trong PostgreSQL.

```python
class Employee(Base):
    __tablename__ = "employees"
    emp_id   = Column(Integer, primary_key=True)
    emp_code = Column(String(50), unique=True, nullable=False)
    name     = Column(String(100), nullable=False)
    attendance_logs = relationship("AttendanceLog", cascade="all, delete-orphan")
    # Không có embeddings relationship — embeddings lưu trong Qdrant

class AttendanceLog(Base):
    __tablename__ = "attendance_logs"
    log_id           = Column(Integer, primary_key=True)
    emp_id           = Column(Integer, ForeignKey("employees.emp_id", ondelete="CASCADE"))
    working_date     = Column(Date, nullable=False)
    checkin_time     = Column(DateTime, nullable=False)
    checkout_time    = Column(DateTime)
    working_duration = Column(Interval)  # read-only, GENERATED bởi DB

class ShiftSettings(Base):
    __tablename__ = "shift_settings"
    id              = Column(Integer, primary_key=True)
    check_in_start  = Column(Time, nullable=False)
    check_in_end    = Column(Time, nullable=False)
    check_out_start = Column(Time, nullable=False)
    check_out_end   = Column(Time, nullable=False)
```

---

## init.sql (PostgreSQL only)

```sql
-- Không cần CREATE EXTENSION vector nữa

CREATE TABLE employees ( ... );
CREATE TABLE attendance_logs ( ... );
CREATE TABLE shift_settings ( ... );

INSERT INTO shift_settings VALUES (DEFAULT, '08:00', '10:00', '17:00', '19:00');
```

Qdrant collection khởi tạo trong code khi app startup (không phải SQL).
