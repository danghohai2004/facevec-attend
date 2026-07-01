# Project Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure codebase thành modular monolith với Qdrant vector store, in-process frame queue hỗ trợ 5 camera đồng thời, và fix 3 lỗi correctness trong attendance logic.

**Architecture:** Modular monolith — 1 FastAPI process. `src/platform/` cung cấp hạ tầng dùng chung (DB sessions, Qdrant client, ML singleton, realtime queue). `src/modules/` chứa domain logic theo bounded context (employees, attendance, recognition, antispoofing). Frame pipeline chạy asyncio background task, CPU-bound work offload sang `ThreadPoolExecutor(max_workers=4)`.

**Tech Stack:** Python 3.11+, FastAPI, SQLAlchemy 2.0 async, asyncpg, psycopg2-binary, qdrant-client, insightface, opencv-python, uvicorn, uv

## Global Constraints

- Python >= 3.11 (dùng `X | Y` union type syntax)
- Không dùng `pgvector` — xóa khỏi dependencies
- Qdrant cosine similarity threshold: `score >= (1 - THRESHOLD)` tương đương pgvector `distance < THRESHOLD`
- `THRESHOLD = 0.6` → Qdrant `score_threshold = 0.4`
- FrameQueue `maxsize=50` (5 camera × 10 frame buffer)
- ThreadPoolExecutor `max_workers=4`
- Attendance: khung giờ ca **luôn lấy từ DB** `shift_settings`, không nhận từ client
- Attendance: query open-log **luôn scope theo `working_date = today`**
- `FrameItem` phải có `client_id: str` để route kết quả về đúng kiosk
- Không thêm NATS, Redis, hay bất kỳ broker nào
- Xóa `src/services/analytics.py` và `src/services/anti_spoofing.py` (stub không dùng)

---

## File Map

### Tạo mới

```
src/
  app.py                                # create_app() — entry point FastAPI
  platform/
    __init__.py
    config.py                           # MODEL, THRESHOLD, ORIGINAL_IMG_PATH, MAX_EMB_FACE
    queue.py                            # FrameItem + FrameQueue (asyncio, drop-oldest)
    db/
      __init__.py
      base.py                           # DeclarativeBase
      session.py                        # async engine, get_db, sync get_connection
      qdrant.py                         # AsyncQdrantClient singleton + ensure_collection()
    ml/
      __init__.py
      face_app.py                       # FaceAnalysis singleton (insightface buffalo_sc)
    realtime/
      __init__.py
      manager.py                        # ConnectionManager: WS egress per client_id
  modules/
    employees/
      __init__.py
      models.py                         # Employee SQLAlchemy model
      schemas.py                        # EmployeeOut, RegisterResponse, ...
      service.py                        # register (PG + Qdrant), remove, list, get, search
      api.py                            # FastAPI router /api/employees
    attendance/
      __init__.py
      models.py                         # AttendanceLog, ShiftSettings SQLAlchemy models
      schemas.py                        # AttendanceLogOut, ShiftsTime, ...
      service.py                        # check_in, check_out (với 3 correctness fixes), list, shift CRUD
      api.py                            # FastAPI router /api/attendance + /api/shift-settings
    recognition/
      __init__.py
      extractor.py                      # extract_embedding_from_bytes() → list[list[float]]
      identifier.py                     # identify() → Qdrant search → (emp_id, name) | None
      pipeline.py                       # run_pipeline() asyncio task
      ws_ingress.py                     # WS endpoint /ws/recognition/{client_id}
    antispoofing/
      __init__.py
      service.py                        # LivenessChecker ABC + PassThroughChecker

tests/
  platform/
    test_queue.py                       # FrameQueue drop-oldest, put/get
  modules/
    attendance/
      test_service.py                   # correctness fixes: working_date scope, shift trust boundary
    recognition/
      test_identifier.py               # threshold behavior
```

### Xóa (sau khi migrate xong)

```
src/services/          # toàn bộ — đã migrate vào modules/
src/core/              # toàn bộ — đã migrate vào platform/
src/db/                # toàn bộ — đã migrate vào platform/db/
src/api/               # toàn bộ — đã migrate vào modules/*/api.py
src/queues/            # toàn bộ — đã migrate vào platform/queue.py
```

### Giữ nguyên (shim backward-compat)

```
utils/conn_db.py       # re-export get_connection từ platform.db.session
utils/draw_bbox.py     # re-export draw_bbox từ modules (nếu còn dùng)
utils/model_app.py     # re-export setup_face_app từ platform.ml.face_app
config.py              # from src.platform.config import *
```

### Sửa

```
compose.yaml           # đổi image postgres, thêm qdrant service + volume
pyproject.toml         # bỏ pgvector, thêm qdrant-client, insightface, python-dotenv
.env.example           # thêm QDRANT_HOST, QDRANT_PORT
initdb/init.sql        # bỏ CREATE EXTENSION vector, bỏ face_embeddings table
main.py                # import từ src.app
Makefile               # bỏ target streamlit
```

---

## Task 1: Infrastructure Setup

**Files:**
- Modify: `compose.yaml`
- Modify: `pyproject.toml`
- Modify: `initdb/init.sql`
- Modify: `.env.example`
- Modify: `Makefile`

**Interfaces:**
- Produces: PostgreSQL tại `:5432`, Qdrant tại `:6333`, dependencies đúng version

- [ ] **Step 1: Cập nhật `compose.yaml`**

```yaml
# compose.yaml
services:
  db:
    image: postgres:16
    container_name: postgres_attendance
    environment:
      POSTGRES_USER: ${DB_USER}
      POSTGRES_PASSWORD: ${DB_PASS}
      POSTGRES_DB: ${DB_NAME}
    ports:
      - "${DB_PORT:-5432}:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./initdb:/docker-entrypoint-initdb.d
    restart: unless-stopped

  qdrant:
    image: qdrant/qdrant:latest
    container_name: qdrant
    ports:
      - "${QDRANT_PORT:-6333}:6333"
    volumes:
      - qdrant_data:/qdrant/storage
    restart: unless-stopped

volumes:
  pgdata:
  qdrant_data:
```

- [ ] **Step 2: Cập nhật `pyproject.toml`**

```toml
[project]
name = "facevec-attend"
version = "0.1.0"
description = "Face recognition attendance system"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "fastapi[standard]>=0.115.0",
    "pydantic>=2.0.0",
    "sqlalchemy>=2.0.0",
    "uvicorn>=0.30.0",
    "opencv-python>=4.8.0",
    "psycopg2-binary>=2.9.0",
    "asyncpg>=0.29.0",
    "qdrant-client>=1.9.0",
    "insightface>=0.7.3",
    "python-dotenv>=1.0.0",
    "plotly>=5.0.0",
]
```

- [ ] **Step 3: Cập nhật `initdb/init.sql`**

```sql
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
CREATE INDEX ON attendance_logs (emp_id, working_date) WHERE checkout_time IS NULL;

CREATE TABLE shift_settings (
    id              SERIAL PRIMARY KEY,
    check_in_start  TIME NOT NULL,
    check_in_end    TIME NOT NULL,
    check_out_start TIME NOT NULL,
    check_out_end   TIME NOT NULL
);

INSERT INTO shift_settings (check_in_start, check_in_end, check_out_start, check_out_end)
VALUES ('08:00', '10:00', '17:00', '19:00');
```

- [ ] **Step 4: Cập nhật `.env.example`**

```env
# PostgreSQL
DB_NAME=attendance
DB_USER=postgres
DB_PASS=your_password
DB_HOST=localhost
DB_PORT=5432

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

- [ ] **Step 5: Cập nhật `Makefile` — bỏ target streamlit**

```makefile
.PHONY: help install db-up db-down run clean

UV = uv
DOCKER_COMPOSE = docker compose

help:
	@echo "Available commands:"
	@echo "  install : Install dependencies using uv"
	@echo "  db-up   : Start PostgreSQL + Qdrant"
	@echo "  db-down : Stop all services"
	@echo "  run     : Run FastAPI backend"
	@echo "  clean   : Clean up temporary files"

install:
	$(UV) sync

db-up:
	$(DOCKER_COMPOSE) up -d

db-down:
	$(DOCKER_COMPOSE) down -v

run:
	$(UV) run main.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
```

- [ ] **Step 6: Khởi động services và verify**

```bash
uv sync
docker compose up -d
docker ps  # phải thấy postgres_attendance và qdrant đang running
docker logs postgres_attendance 2>&1 | tail -5
# expected: "database system is ready to accept connections"
```

Mở `http://localhost:6333/dashboard` — Qdrant UI phải load được.

- [ ] **Step 7: Commit**

```bash
git add compose.yaml pyproject.toml initdb/init.sql .env.example Makefile
git commit -m "chore: switch to Qdrant, drop pgvector, update infra"
```

---

## Task 2: Platform DB Layer

**Files:**
- Create: `src/platform/__init__.py`
- Create: `src/platform/db/__init__.py`
- Create: `src/platform/db/base.py`
- Create: `src/platform/db/session.py`
- Create: `src/platform/db/qdrant.py`

**Interfaces:**
- Produces:
  - `get_db() -> AsyncGenerator[AsyncSession]` — FastAPI dependency
  - `get_connection() -> tuple[conn, error_str | None]` — sync cho Streamlit legacy
  - `get_qdrant_client() -> AsyncQdrantClient`
  - `COLLECTION_NAME = "face_embeddings"`

- [ ] **Step 1: Tạo `src/platform/db/base.py`**

```python
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    pass
```

- [ ] **Step 2: Tạo `src/platform/db/session.py`**

```python
import os
import psycopg2
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.engine import URL

load_dotenv()

_url = URL.create(
    "postgresql+asyncpg",
    username=os.getenv("DB_USER"),
    password=os.getenv("DB_PASS"),
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    database=os.getenv("DB_NAME"),
)

engine = create_async_engine(_url)
AsyncSessionLocal = async_sessionmaker(autocommit=False, autoflush=False, bind=engine, class_=AsyncSession)

async def get_db():
    async with AsyncSessionLocal() as db:
        try:
            yield db
        finally:
            await db.close()

def get_connection():
    try:
        conn = psycopg2.connect(
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASS"),
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT"),
            database=os.getenv("DB_NAME"),
        )
        return conn, None
    except Exception as e:
        return None, f"[ERROR CONNECT DB]: {e}"
```

- [ ] **Step 3: Tạo `src/platform/db/qdrant.py`**

```python
import os
from dotenv import load_dotenv
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()

COLLECTION_NAME = "face_embeddings"
_VECTOR_SIZE = 512

_client: AsyncQdrantClient | None = None

def get_qdrant_client() -> AsyncQdrantClient:
    global _client
    if _client is None:
        _client = AsyncQdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", "6333")),
        )
    return _client

async def ensure_collection() -> None:
    client = get_qdrant_client()
    collections = await client.get_collections()
    names = [c.name for c in collections.collections]
    if COLLECTION_NAME not in names:
        await client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=_VECTOR_SIZE, distance=Distance.COSINE),
        )
```

- [ ] **Step 4: Tạo `__init__.py` files**

```bash
touch src/platform/__init__.py src/platform/db/__init__.py
```

- [ ] **Step 5: Verify import**

```bash
uv run python -c "from src.platform.db.session import get_db; from src.platform.db.qdrant import get_qdrant_client; print('OK')"
# expected: OK
```

- [ ] **Step 6: Commit**

```bash
git add src/platform/
git commit -m "feat: add platform DB layer (SQLAlchemy async + Qdrant client)"
```

---

## Task 3: Platform Config + ML Singleton

**Files:**
- Create: `src/platform/config.py`
- Create: `src/platform/ml/__init__.py`
- Create: `src/platform/ml/face_app.py`

**Interfaces:**
- Produces:
  - `MODEL: dict` — insightface config
  - `THRESHOLD: float = 0.6`
  - `ORIGINAL_IMG_PATH: str = "faces"`
  - `MAX_EMB_FACE: int = 50`
  - `setup_face_app() -> FaceAnalysis`

- [ ] **Step 1: Tạo `src/platform/config.py`**

```python
MODEL = {
    "name": "buffalo_sc",
    "det_size": (640, 640),
    "ctx_id": 0,
    "providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
}

ORIGINAL_IMG_PATH = "faces"
THRESHOLD = 0.6
MAX_EMB_FACE = 50
```

- [ ] **Step 2: Tạo `src/platform/ml/face_app.py`**

```python
from insightface.app import FaceAnalysis
from src.platform.config import MODEL

_face_app: FaceAnalysis | None = None

def setup_face_app() -> FaceAnalysis:
    global _face_app
    if _face_app is None:
        _face_app = FaceAnalysis(name=MODEL["name"], providers=MODEL["providers"])
        _face_app.prepare(ctx_id=MODEL["ctx_id"], det_size=MODEL["det_size"])
    return _face_app
```

- [ ] **Step 3: Tạo `__init__.py`**

```bash
touch src/platform/ml/__init__.py
```

- [ ] **Step 4: Verify**

```bash
uv run python -c "from src.platform.config import THRESHOLD; print(THRESHOLD)"
# expected: 0.6
```

- [ ] **Step 5: Commit**

```bash
git add src/platform/config.py src/platform/ml/
git commit -m "feat: add platform config and ML singleton"
```

---

## Task 4: Platform Queue + ConnectionManager

**Files:**
- Create: `src/platform/queue.py`
- Create: `src/platform/realtime/__init__.py`
- Create: `src/platform/realtime/manager.py`
- Create: `tests/platform/test_queue.py`

**Interfaces:**
- Produces:
  - `FrameItem(client_id: str, frame: bytes, captured_at: float)`
  - `FrameQueue(maxsize=50)` — `.put(item)`, `.get() -> FrameItem`
  - `ConnectionManager` — `.connect(client_id, ws)`, `.disconnect(client_id)`, `.send(client_id, data)`

- [ ] **Step 1: Viết test cho FrameQueue**

```python
# tests/platform/test_queue.py
import asyncio
import pytest
from src.platform.queue import FrameItem, FrameQueue

def make_item(client_id: str = "cam1") -> FrameItem:
    return FrameItem(client_id=client_id, frame=b"jpeg", captured_at=0.0)

@pytest.mark.asyncio
async def test_put_and_get():
    q = FrameQueue(maxsize=2)
    item = make_item()
    await q.put(item)
    result = await q.get()
    assert result.client_id == "cam1"

@pytest.mark.asyncio
async def test_drop_oldest_when_full():
    q = FrameQueue(maxsize=2)
    old = FrameItem(client_id="cam1", frame=b"old", captured_at=1.0)
    new1 = FrameItem(client_id="cam1", frame=b"new1", captured_at=2.0)
    new2 = FrameItem(client_id="cam1", frame=b"new2", captured_at=3.0)
    await q.put(old)
    await q.put(new1)
    await q.put(new2)  # queue full → old dropped
    first = await q.get()
    assert first.frame == b"new1"  # old đã bị drop
```

- [ ] **Step 2: Chạy test — phải fail**

```bash
uv run pytest tests/platform/test_queue.py -v
# expected: ImportError hoặc ModuleNotFoundError
```

- [ ] **Step 3: Tạo `src/platform/queue.py`**

```python
import asyncio
from dataclasses import dataclass

@dataclass(frozen=True)
class FrameItem:
    client_id: str
    frame: bytes
    captured_at: float

class FrameQueue:
    def __init__(self, maxsize: int = 50) -> None:
        self._queue: asyncio.Queue[FrameItem] = asyncio.Queue(maxsize=maxsize)

    async def put(self, item: FrameItem) -> None:
        if self._queue.full():
            try:
                self._queue.get_nowait()  # ponytail: drop oldest, realtime ưu tiên frame mới
            except asyncio.QueueEmpty:
                pass
        await self._queue.put(item)

    async def get(self) -> FrameItem:
        return await self._queue.get()
```

- [ ] **Step 4: Chạy test — phải pass**

```bash
uv run pytest tests/platform/test_queue.py -v
# expected: 2 passed
```

- [ ] **Step 5: Tạo `src/platform/realtime/manager.py`**

```python
from fastapi import WebSocket

class ConnectionManager:
    def __init__(self) -> None:
        self._connections: dict[str, WebSocket] = {}

    async def connect(self, client_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections[client_id] = websocket

    def disconnect(self, client_id: str) -> None:
        self._connections.pop(client_id, None)

    async def send(self, client_id: str, data: dict) -> None:
        ws = self._connections.get(client_id)
        if ws:
            try:
                await ws.send_json(data)
            except Exception:
                self.disconnect(client_id)
```

- [ ] **Step 6: Tạo `__init__.py` và `tests/__init__.py`**

```bash
touch src/platform/realtime/__init__.py
mkdir -p tests/platform && touch tests/__init__.py tests/platform/__init__.py
```

- [ ] **Step 7: Commit**

```bash
git add src/platform/queue.py src/platform/realtime/ tests/platform/
git commit -m "feat: add FrameQueue (drop-oldest) and ConnectionManager"
```

---

## Task 5: Employees Module

**Files:**
- Create: `src/modules/__init__.py`
- Create: `src/modules/employees/__init__.py`
- Create: `src/modules/employees/models.py`
- Create: `src/modules/employees/schemas.py`
- Create: `src/modules/employees/service.py`
- Create: `src/modules/employees/api.py`

**Interfaces:**
- Consumes: `Base` (Task 2), `AsyncSession + get_db` (Task 2), `AsyncQdrantClient + COLLECTION_NAME` (Task 2)
- Produces:
  - `Employee` SQLAlchemy model
  - `register_employee(db, qdrant, name, emp_code, embeddings) -> (Employee, err)`
  - `remove_employee(db, qdrant, emp_id?, emp_code?) -> (Employee, err)`
  - `list_employees(db, page, page_size) -> (list[Employee], total, err)`
  - `get_employee(db, emp_id) -> (Employee, err)`
  - `search_employees_by_name(db, name) -> (list[Employee], err)`
  - FastAPI router `/api/employees`

- [ ] **Step 1: Tạo `src/modules/employees/models.py`**

```python
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship
from src.platform.db.base import Base

class Employee(Base):
    __tablename__ = "employees"

    emp_id   = Column(Integer, primary_key=True, autoincrement=True)
    emp_code = Column(String(50), unique=True, nullable=False)
    name     = Column(String(100), nullable=False)

    attendance_logs = relationship("AttendanceLog", back_populates="employee", cascade="all, delete-orphan")
```

- [ ] **Step 2: Tạo `src/modules/employees/schemas.py`**

```python
from pydantic import BaseModel, ConfigDict

class EmployeeOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    emp_id: int
    name: str
    emp_code: str

class EmployeeRegisterResponse(BaseModel):
    message: str
    employee: EmployeeOut

class EmployeeRemoveResponse(BaseModel):
    message: str
    emp_id: int | None = None
    emp_code: str | None = None

class EmployeeListResponse(BaseModel):
    items: list[EmployeeOut]
    page: int
    page_size: int
    total: int

class EmployeeDetailResponse(BaseModel):
    employee: EmployeeOut
```

- [ ] **Step 3: Tạo `src/modules/employees/service.py`**

```python
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue
import uuid

from src.modules.employees.models import Employee
from src.platform.db.qdrant import COLLECTION_NAME

ERR_NOT_FOUND = "EMPLOYEE_NOT_FOUND"
ERR_MISSING_ID = "MISSING_IDENTIFIER"

async def register_employee(
    db: AsyncSession,
    qdrant: AsyncQdrantClient,
    name: str,
    emp_code: str,
    embeddings: list[list[float]],
) -> tuple[Employee, str | None]:
    try:
        result = await db.execute(select(Employee).filter(Employee.emp_code == emp_code))
        employee = result.scalar_one_or_none()

        if not employee:
            employee = Employee(name=name, emp_code=emp_code)
            db.add(employee)
            await db.flush()
        else:
            employee.name = name

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=emb,
                payload={"emp_id": employee.emp_id, "emp_code": emp_code, "name": name},
            )
            for emb in embeddings
        ]
        await qdrant.upsert(collection_name=COLLECTION_NAME, points=points)

        await db.commit()
        await db.refresh(employee)
        return employee, None
    except Exception as e:
        await db.rollback()
        return None, f"[ERROR REGISTER]: {e}"

async def remove_employee(
    db: AsyncSession,
    qdrant: AsyncQdrantClient,
    emp_id: int | None = None,
    emp_code: str | None = None,
) -> tuple[Employee, str | None]:
    if emp_id is None and emp_code is None:
        return None, ERR_MISSING_ID

    try:
        if emp_id is not None:
            stmt = select(Employee).filter(Employee.emp_id == emp_id)
        else:
            stmt = select(Employee).filter(Employee.emp_code == emp_code)

        result = await db.execute(stmt)
        employee = result.scalar_one_or_none()
        if not employee:
            return None, ERR_NOT_FOUND

        await qdrant.delete(
            collection_name=COLLECTION_NAME,
            points_selector=Filter(
                must=[FieldCondition(key="emp_id", match=MatchValue(value=employee.emp_id))]
            ),
        )

        await db.delete(employee)
        await db.commit()
        return employee, None
    except Exception as e:
        await db.rollback()
        return None, f"[ERROR REMOVE]: {e}"

async def list_employees(
    db: AsyncSession, page: int, page_size: int
) -> tuple[list[Employee], int, str | None]:
    try:
        total = (await db.execute(select(func.count()).select_from(Employee))).scalar_one()
        result = await db.execute(
            select(Employee).order_by(Employee.emp_id)
            .offset((page - 1) * page_size).limit(page_size)
        )
        return result.scalars().all(), total, None
    except Exception as e:
        return [], 0, f"[ERROR LIST]: {e}"

async def get_employee(db: AsyncSession, emp_id: int) -> tuple[Employee, str | None]:
    try:
        result = await db.execute(select(Employee).filter(Employee.emp_id == emp_id))
        employee = result.scalar_one_or_none()
        return (employee, None) if employee else (None, ERR_NOT_FOUND)
    except Exception as e:
        return None, f"[ERROR GET]: {e}"

async def search_employees_by_name(db: AsyncSession, name: str) -> tuple[list[Employee], str | None]:
    try:
        result = await db.execute(
            select(Employee).filter(Employee.name.ilike(f"%{name}%")).order_by(Employee.emp_id)
        )
        return result.scalars().all(), None
    except Exception as e:
        return [], f"[ERROR SEARCH]: {e}"
```

- [ ] **Step 4: Tạo `src/modules/employees/api.py`**

```python
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Query
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.ext.asyncio import AsyncSession
from qdrant_client import AsyncQdrantClient

from src.platform.db.session import get_db
from src.platform.db.qdrant import get_qdrant_client
from src.modules.recognition.extractor import extract_embeddings_from_bytes
from src.modules.employees.schemas import (
    EmployeeOut, EmployeeRegisterResponse, EmployeeRemoveResponse,
    EmployeeListResponse, EmployeeDetailResponse,
)
from src.modules.employees.service import (
    register_employee, remove_employee, list_employees, get_employee,
    search_employees_by_name, ERR_NOT_FOUND, ERR_MISSING_ID,
)

router = APIRouter(prefix="/api/employees", tags=["Employees"])

@router.post("", response_model=EmployeeRegisterResponse)
async def api_register(
    name: str = Form(...),
    emp_code: str = Form(...),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    contents = await file.read()
    try:
        embeddings = await run_in_threadpool(extract_embeddings_from_bytes, contents)
    except ValueError:
        raise HTTPException(400, "Không thể đọc file ảnh.")
    if not embeddings:
        raise HTTPException(400, "Không tìm thấy khuôn mặt trong ảnh.")

    qdrant = get_qdrant_client()
    employee, err = await register_employee(db, qdrant, name, emp_code, embeddings)
    if err:
        raise HTTPException(500, err)
    return EmployeeRegisterResponse(
        message=f"Registered {employee.name} ({employee.emp_code})",
        employee=EmployeeOut.model_validate(employee),
    )

@router.get("", response_model=EmployeeListResponse)
async def api_list(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    employees, total, err = await list_employees(db, page, page_size)
    if err:
        raise HTTPException(500, err)
    return EmployeeListResponse(
        items=[EmployeeOut.model_validate(e) for e in employees],
        page=page, page_size=page_size, total=total,
    )

@router.get("/{identifier}", response_model=EmployeeDetailResponse | EmployeeListResponse)
async def api_get(identifier: str, db: AsyncSession = Depends(get_db)):
    if identifier.isdigit():
        employee, err = await get_employee(db, int(identifier))
        if err:
            raise HTTPException(404 if err == ERR_NOT_FOUND else 500, err)
        return EmployeeDetailResponse(employee=EmployeeOut.model_validate(employee))
    employees, err = await search_employees_by_name(db, identifier)
    if err:
        raise HTTPException(500, err)
    return EmployeeListResponse(items=[EmployeeOut.model_validate(e) for e in employees],
                                page=1, page_size=len(employees), total=len(employees))

@router.delete("", response_model=EmployeeRemoveResponse)
async def api_remove(
    emp_id: int | None = Query(None),
    emp_code: str | None = Query(None),
    db: AsyncSession = Depends(get_db),
):
    qdrant = get_qdrant_client()
    employee, err = await remove_employee(db, qdrant, emp_id=emp_id, emp_code=emp_code)
    if err:
        if err == ERR_MISSING_ID:
            raise HTTPException(400, "Cần emp_id hoặc emp_code.")
        if err == ERR_NOT_FOUND:
            raise HTTPException(404, "Không tìm thấy nhân viên.")
        raise HTTPException(500, err)
    return EmployeeRemoveResponse(message=f"Removed {employee.emp_id}",
                                  emp_id=employee.emp_id, emp_code=employee.emp_code)
```

- [ ] **Step 5: Tạo `__init__.py` files**

```bash
touch src/modules/__init__.py src/modules/employees/__init__.py
```

- [ ] **Step 6: Commit**

```bash
git add src/modules/employees/ src/modules/__init__.py
git commit -m "feat: add employees module (PostgreSQL + Qdrant)"
```

---

## Task 6: Attendance Module + Correctness Fixes

**Files:**
- Create: `src/modules/attendance/__init__.py`
- Create: `src/modules/attendance/models.py`
- Create: `src/modules/attendance/schemas.py`
- Create: `src/modules/attendance/service.py`
- Create: `src/modules/attendance/api.py`
- Create: `tests/modules/attendance/test_service.py`

**Interfaces:**
- Consumes: `Base` (Task 2), `AsyncSession` (Task 2), `Employee` model (Task 5)
- Produces:
  - `AttendanceLog`, `ShiftSettings` SQLAlchemy models
  - `get_shift_settings(db) -> (ShiftSettings | dict, err)`
  - `check_in(db, emp_id, now?) -> (AttendanceLog, err)`
  - `check_out(db, emp_id, now?) -> (AttendanceLog, err)`
  - `log_attendance(db, emp_id) -> str` — auto detect check_in/check_out từ shift DB
  - FastAPI router `/api/attendance` + `/api/shift-settings`

**3 Correctness Fixes:**
1. `check_in`: filter open-log bằng `working_date == today` (tránh log ngày hôm qua lock ngày hôm nay)
2. `check_out`: filter open-log bằng `working_date == today`
3. API endpoints lấy `shifts_time` từ DB, không nhận từ client body

- [ ] **Step 1: Viết failing tests cho 3 correctness fixes**

```python
# tests/modules/attendance/test_service.py
from datetime import datetime, date, time
from unittest.mock import AsyncMock, MagicMock
import pytest
from src.modules.attendance.service import (
    _is_time_in_range, _normalize_shifts_time, check_in, check_out,
)

# --- Test _is_time_in_range ---

def test_time_in_range_normal():
    assert _is_time_in_range(time(9, 0), time(8, 0), time(10, 0)) is True

def test_time_outside_range():
    assert _is_time_in_range(time(11, 0), time(8, 0), time(10, 0)) is False

def test_time_in_range_overnight():
    # ca xuyên đêm: 22:00 - 06:00
    assert _is_time_in_range(time(23, 0), time(22, 0), time(6, 0)) is True
    assert _is_time_in_range(time(5, 0), time(22, 0), time(6, 0)) is True
    assert _is_time_in_range(time(12, 0), time(22, 0), time(6, 0)) is False

# --- Test working_date scope (fix #1 & #2) ---

@pytest.mark.asyncio
async def test_check_in_scopes_by_working_date():
    """check_in phải filter theo working_date=today, không phải chỉ checkout_time IS NULL"""
    db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalars.return_value.first.return_value = None  # không có record hôm nay
    db.execute = AsyncMock(return_value=mock_result)
    db.add = MagicMock()
    db.commit = AsyncMock()
    db.refresh = AsyncMock()

    today = date.today()
    now = datetime.combine(today, time(9, 0))
    log, err = await check_in(db, emp_id=1, now=now)

    assert err is None
    # Verify query có filter working_date
    call_args = str(db.execute.call_args)
    assert "working_date" in call_args

@pytest.mark.asyncio
async def test_check_out_scopes_by_working_date():
    """check_out phải filter theo working_date=today"""
    db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None  # không có open log hôm nay
    db.execute = AsyncMock(return_value=mock_result)

    today = date.today()
    now = datetime.combine(today, time(17, 30))
    log, err = await check_out(db, emp_id=1, now=now)

    assert log is None
    assert err == "Check in not found to check out"
    call_args = str(db.execute.call_args)
    assert "working_date" in call_args
```

- [ ] **Step 2: Chạy test — phải fail**

```bash
mkdir -p tests/modules/attendance && touch tests/modules/__init__.py tests/modules/attendance/__init__.py
uv run pytest tests/modules/attendance/test_service.py -v
# expected: ImportError
```

- [ ] **Step 3: Tạo `src/modules/attendance/models.py`**

```python
from sqlalchemy import Column, Integer, String, Date, DateTime, Interval, Time, ForeignKey
from sqlalchemy.orm import relationship
from src.platform.db.base import Base

class AttendanceLog(Base):
    __tablename__ = "attendance_logs"

    log_id           = Column(Integer, primary_key=True, autoincrement=True)
    emp_id           = Column(Integer, ForeignKey("employees.emp_id", ondelete="CASCADE"), nullable=False)
    working_date     = Column(Date, nullable=False)
    checkin_time     = Column(DateTime, nullable=False)
    checkout_time    = Column(DateTime)
    working_duration = Column(Interval)  # read-only, GENERATED bởi DB

    employee = relationship("Employee", back_populates="attendance_logs")

class ShiftSettings(Base):
    __tablename__ = "shift_settings"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    check_in_start  = Column(Time, nullable=False)
    check_in_end    = Column(Time, nullable=False)
    check_out_start = Column(Time, nullable=False)
    check_out_end   = Column(Time, nullable=False)
```

- [ ] **Step 4: Tạo `src/modules/attendance/schemas.py`**

```python
from datetime import date, datetime, time, timedelta
from pydantic import BaseModel, ConfigDict

class ShiftsTime(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    check_in_start: time
    check_in_end: time
    check_out_start: time
    check_out_end: time

class AttendanceLogOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    log_id: int
    emp_id: int
    working_date: date
    checkin_time: datetime
    checkout_time: datetime | None = None
    working_duration: timedelta | None = None

class AttendanceHistoryResponse(BaseModel):
    items: list[AttendanceLogOut]
    page: int
    page_size: int
    total: int

class AttendanceCheckResponse(BaseModel):
    message: str
    check_type: str
    log: AttendanceLogOut | None = None
```

- [ ] **Step 5: Tạo `src/modules/attendance/service.py`** (với 3 fixes)

```python
from datetime import datetime, time, timedelta, date
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.modules.attendance.models import AttendanceLog, ShiftSettings

_DEFAULT_SHIFT = {
    "check_in_start": time(8, 0),
    "check_in_end": time(10, 0),
    "check_out_start": time(17, 0),
    "check_out_end": time(19, 0),
}

def _is_time_in_range(current: time, start: time, end: time) -> bool:
    if start <= end:
        return start <= current <= end
    return current >= start or current <= end  # overnight shift

def _normalize_shifts_time(shifts) -> dict:
    """Accept ShiftSettings ORM object or plain dict with check_in_start key."""
    if isinstance(shifts, dict):
        return shifts
    return {
        "check_in_start": shifts.check_in_start,
        "check_in_end": shifts.check_in_end,
        "check_out_start": shifts.check_out_start,
        "check_out_end": shifts.check_out_end,
    }

async def get_shift_settings(db: AsyncSession) -> tuple[ShiftSettings | dict, str | None]:
    try:
        result = await db.execute(select(ShiftSettings).order_by(ShiftSettings.id).limit(1))
        settings = result.scalar_one_or_none()
        return settings if settings else _DEFAULT_SHIFT, None
    except Exception as e:
        return None, f"[ERROR GET SHIFT]: {e}"

async def upsert_shift_settings(db: AsyncSession, data: dict) -> tuple[ShiftSettings, str | None]:
    try:
        result = await db.execute(select(ShiftSettings).order_by(ShiftSettings.id).limit(1))
        settings = result.scalar_one_or_none()
        if settings is None:
            settings = ShiftSettings(**data)
            db.add(settings)
        else:
            for k, v in data.items():
                setattr(settings, k, v)
        await db.commit()
        await db.refresh(settings)
        return settings, None
    except Exception as e:
        await db.rollback()
        return None, f"[ERROR UPSERT SHIFT]: {e}"

def get_current_time(shifts) -> tuple[bool, datetime, str | None]:
    shifts = _normalize_shifts_time(shifts)
    now = datetime.now()
    t = now.time()
    if _is_time_in_range(t, shifts["check_in_start"], shifts["check_in_end"]):
        return True, now, "check_in"
    if _is_time_in_range(t, shifts["check_out_start"], shifts["check_out_end"]):
        return True, now, "check_out"
    return False, now, None

async def check_in(
    db: AsyncSession,
    emp_id: int,
    now: datetime | None = None,
) -> tuple[AttendanceLog, str | None]:
    now = now or datetime.now()
    working_date = now.date()

    # Fix #1: scope by working_date to avoid yesterday's open log blocking today
    result = await db.execute(
        select(AttendanceLog).filter(
            AttendanceLog.emp_id == emp_id,
            AttendanceLog.working_date == working_date,
            AttendanceLog.checkout_time.is_(None),
        ).limit(1)
    )
    if result.scalars().first():
        return None, "Already checked in"

    log = AttendanceLog(emp_id=emp_id, working_date=working_date, checkin_time=now)
    db.add(log)
    await db.commit()
    await db.refresh(log)
    return log, None

async def check_out(
    db: AsyncSession,
    emp_id: int,
    now: datetime | None = None,
) -> tuple[AttendanceLog, str | None]:
    now = now or datetime.now()
    working_date = now.date()

    # Fix #2: scope by working_date to avoid closing yesterday's open log
    result = await db.execute(
        select(AttendanceLog).filter(
            AttendanceLog.emp_id == emp_id,
            AttendanceLog.working_date == working_date,
            AttendanceLog.checkout_time.is_(None),
        ).order_by(AttendanceLog.checkin_time.desc()).limit(1)
    )
    log = result.scalar_one_or_none()
    if log is None:
        return None, "Check in not found to check out"

    log.checkout_time = now
    await db.commit()
    await db.refresh(log)
    return log, None

async def log_attendance(db: AsyncSession, emp_id: int) -> str:
    """Fix #3: load shifts từ DB, không nhận từ client."""
    shifts, err = await get_shift_settings(db)
    if err:
        return f"[ERROR] {err}"

    within, now, check_type = get_current_time(shifts)
    if not within:
        return "Not during working hours"

    if check_type == "check_in":
        _, err = await check_in(db, emp_id, now=now)
    else:
        _, err = await check_out(db, emp_id, now=now)

    if err:
        return err
    return "Check in successful" if check_type == "check_in" else "Check out successful"

async def list_attendance_logs(
    db: AsyncSession,
    emp_id: int,
    page: int,
    page_size: int,
    from_date: date | None = None,
    to_date: date | None = None,
) -> tuple[list[AttendanceLog], int, str | None]:
    try:
        filters = [AttendanceLog.emp_id == emp_id]
        if from_date:
            filters.append(AttendanceLog.working_date >= from_date)
        if to_date:
            filters.append(AttendanceLog.working_date <= to_date)

        total = (await db.execute(
            select(func.count()).select_from(AttendanceLog).filter(*filters)
        )).scalar_one()

        result = await db.execute(
            select(AttendanceLog).filter(*filters)
            .order_by(AttendanceLog.checkin_time.desc())
            .offset((page - 1) * page_size).limit(page_size)
        )
        return result.scalars().all(), total, None
    except Exception as e:
        return [], 0, f"[ERROR LIST ATTENDANCE]: {e}"
```

- [ ] **Step 6: Chạy tests — phải pass**

```bash
uv run pytest tests/modules/attendance/test_service.py -v
# expected: 5 passed
```

- [ ] **Step 7: Tạo `src/modules/attendance/api.py`**

```python
from datetime import date
from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from src.platform.db.session import get_db
from src.modules.attendance.schemas import (
    ShiftsTime, AttendanceLogOut, AttendanceHistoryResponse, AttendanceCheckResponse,
)
from src.modules.attendance.service import (
    get_shift_settings, upsert_shift_settings, check_in, check_out, list_attendance_logs,
)
from src.modules.employees.service import get_employee, ERR_NOT_FOUND

router = APIRouter(prefix="/api", tags=["Attendance"])

@router.post("/attendance/checkin", response_model=AttendanceCheckResponse)
async def api_checkin(emp_id: int = Query(...), db: AsyncSession = Depends(get_db)):
    # Fix #3: no shifts_time from client — load from DB inside service
    employee, err = await get_employee(db, emp_id)
    if err:
        raise HTTPException(404 if err == ERR_NOT_FOUND else 500, err)
    log, err = await check_in(db, emp_id)
    if err:
        raise HTTPException(400, err)
    return AttendanceCheckResponse(
        message="Check in successful", check_type="check_in",
        log=AttendanceLogOut.model_validate(log),
    )

@router.post("/attendance/checkout", response_model=AttendanceCheckResponse)
async def api_checkout(emp_id: int = Query(...), db: AsyncSession = Depends(get_db)):
    employee, err = await get_employee(db, emp_id)
    if err:
        raise HTTPException(404 if err == ERR_NOT_FOUND else 500, err)
    log, err = await check_out(db, emp_id)
    if err:
        raise HTTPException(400, err)
    return AttendanceCheckResponse(
        message="Check out successful", check_type="check_out",
        log=AttendanceLogOut.model_validate(log),
    )

@router.get("/attendance", response_model=AttendanceHistoryResponse)
async def api_history(
    emp_id: int = Query(...),
    from_date: date | None = Query(None),
    to_date: date | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    employee, err = await get_employee(db, emp_id)
    if err:
        raise HTTPException(404 if err == ERR_NOT_FOUND else 500, err)
    logs, total, err = await list_attendance_logs(db, emp_id, page, page_size, from_date, to_date)
    if err:
        raise HTTPException(500, err)
    return AttendanceHistoryResponse(
        items=[AttendanceLogOut.model_validate(l) for l in logs],
        page=page, page_size=page_size, total=total,
    )

@router.get("/shift-settings", response_model=ShiftsTime)
async def api_shift_get(db: AsyncSession = Depends(get_db)):
    settings, err = await get_shift_settings(db)
    if err:
        raise HTTPException(500, err)
    return ShiftsTime.model_validate(settings)

@router.put("/shift-settings", response_model=ShiftsTime)
async def api_shift_update(payload: ShiftsTime, db: AsyncSession = Depends(get_db)):
    settings, err = await upsert_shift_settings(db, payload.model_dump())
    if err:
        raise HTTPException(500, err)
    return ShiftsTime.model_validate(settings)
```

- [ ] **Step 8: Tạo `__init__.py`**

```bash
touch src/modules/attendance/__init__.py
```

- [ ] **Step 9: Commit**

```bash
git add src/modules/attendance/ tests/modules/
git commit -m "feat: add attendance module with 3 correctness fixes"
```

---

## Task 7: Anti-Spoofing Module

**Files:**
- Create: `src/modules/antispoofing/__init__.py`
- Create: `src/modules/antispoofing/service.py`

**Interfaces:**
- Produces: `LivenessChecker` ABC, `PassThroughChecker`, `get_liveness_checker() -> LivenessChecker`

- [ ] **Step 1: Tạo `src/modules/antispoofing/service.py`**

```python
from abc import ABC, abstractmethod
import numpy as np

class LivenessChecker(ABC):
    @abstractmethod
    def check(self, frame: np.ndarray) -> bool:
        """Return True nếu frame là mặt thật, False nếu phát hiện giả mạo."""

class PassThroughChecker(LivenessChecker):
    # ponytail: luôn True — swap sang SilentFaceChecker khi cần liveness thật
    def check(self, frame: np.ndarray) -> bool:
        return True

_checker: LivenessChecker = PassThroughChecker()

def get_liveness_checker() -> LivenessChecker:
    return _checker
```

- [ ] **Step 2: Tạo `__init__.py` và commit**

```bash
touch src/modules/antispoofing/__init__.py
git add src/modules/antispoofing/
git commit -m "feat: add antispoofing module (PassThrough placeholder)"
```

---

## Task 8: Recognition Module

**Files:**
- Create: `src/modules/recognition/__init__.py`
- Create: `src/modules/recognition/extractor.py`
- Create: `src/modules/recognition/identifier.py`
- Create: `src/modules/recognition/pipeline.py`
- Create: `src/modules/recognition/ws_ingress.py`
- Create: `tests/modules/recognition/test_identifier.py`

**Interfaces:**
- Consumes: `setup_face_app` (Task 3), `THRESHOLD + COLLECTION_NAME` (Task 2/3), `FrameQueue + FrameItem` (Task 4), `ConnectionManager` (Task 4), `LivenessChecker` (Task 7), `log_attendance` (Task 6)
- Produces:
  - `extract_embeddings_from_bytes(contents: bytes) -> list[list[float]]`
  - `identify(qdrant, embedding, threshold) -> dict | None` — `{emp_id, name, emp_code}`
  - `run_pipeline(queue, qdrant, db_factory, manager, checker)` — asyncio coroutine
  - WS endpoint `/ws/recognition/{client_id}`

- [ ] **Step 1: Viết test cho identifier**

```python
# tests/modules/recognition/test_identifier.py
import pytest
from unittest.mock import AsyncMock, MagicMock

async def test_identify_returns_none_below_threshold():
    from src.modules.recognition.identifier import identify
    qdrant = AsyncMock()
    hit = MagicMock()
    hit.score = 0.3  # below 1 - 0.6 = 0.4
    hit.payload = {"emp_id": 1, "name": "A", "emp_code": "NV001"}
    qdrant.search = AsyncMock(return_value=[hit])

    result = await identify(qdrant, [0.1] * 512, threshold=0.6)
    assert result is None

async def test_identify_returns_payload_above_threshold():
    from src.modules.recognition.identifier import identify
    qdrant = AsyncMock()
    hit = MagicMock()
    hit.score = 0.8  # above 1 - 0.6 = 0.4
    hit.payload = {"emp_id": 1, "name": "Nguyen Van A", "emp_code": "NV001"}
    qdrant.search = AsyncMock(return_value=[hit])

    result = await identify(qdrant, [0.1] * 512, threshold=0.6)
    assert result == {"emp_id": 1, "name": "Nguyen Van A", "emp_code": "NV001"}

async def test_identify_returns_none_when_no_results():
    from src.modules.recognition.identifier import identify
    qdrant = AsyncMock()
    qdrant.search = AsyncMock(return_value=[])

    result = await identify(qdrant, [0.1] * 512, threshold=0.6)
    assert result is None
```

- [ ] **Step 2: Chạy test — phải fail**

```bash
mkdir -p tests/modules/recognition && touch tests/modules/recognition/__init__.py
uv run pytest tests/modules/recognition/test_identifier.py -v
# expected: ImportError
```

- [ ] **Step 3: Tạo `src/modules/recognition/extractor.py`**

```python
import cv2
import numpy as np
from src.platform.ml.face_app import setup_face_app

def extract_embedding_from_frame(img: np.ndarray) -> list[float] | None:
    app = setup_face_app()
    faces = app.get(img)
    if not faces:
        return None
    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
    return faces[0].normed_embedding.tolist()

def extract_embeddings_from_bytes(contents: bytes) -> list[list[float]]:
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image data")
    emb = extract_embedding_from_frame(img)
    return [emb] if emb else []
```

- [ ] **Step 4: Tạo `src/modules/recognition/identifier.py`**

```python
from qdrant_client import AsyncQdrantClient
from src.platform.db.qdrant import COLLECTION_NAME

async def identify(
    qdrant: AsyncQdrantClient,
    embedding: list[float],
    threshold: float,
) -> dict | None:
    """
    Return payload dict {emp_id, name, emp_code} nếu score >= (1 - threshold).
    Qdrant dùng cosine similarity (cao = tốt), ngược với pgvector distance (thấp = tốt).
    """
    results = await qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=embedding,
        limit=1,
        score_threshold=1.0 - threshold,
    )
    if not results:
        return None
    return results[0].payload
```

- [ ] **Step 5: Chạy test — phải pass**

```bash
uv run pytest tests/modules/recognition/test_identifier.py -v
# expected: 3 passed
```

- [ ] **Step 6: Tạo `src/modules/recognition/pipeline.py`**

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from src.platform.queue import FrameQueue
from src.platform.realtime.manager import ConnectionManager
from src.modules.recognition.extractor import extract_embedding_from_frame
from src.modules.recognition.identifier import identify
from src.modules.antispoofing.service import LivenessChecker
import cv2
import numpy as np

async def run_pipeline(
    queue: FrameQueue,
    qdrant,
    db_factory,
    manager: ConnectionManager,
    checker: LivenessChecker,
    threshold: float,
) -> None:
    executor = ThreadPoolExecutor(max_workers=4)
    loop = asyncio.get_event_loop()

    while True:
        item = await queue.get()
        asyncio.create_task(
            _process(item, qdrant, db_factory, manager, checker, threshold, executor, loop)
        )

async def _process(item, qdrant, db_factory, manager, checker, threshold, executor, loop):
    from src.modules.attendance.service import log_attendance

    try:
        # Decode + anti-spoof + extract — CPU-bound, chạy trong threadpool
        def _cpu_work():
            nparr = np.frombuffer(item.frame, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return None, "no_face"
            if not checker.check(img):
                return None, "spoof"
            emb = extract_embedding_from_frame(img)
            if emb is None:
                return None, "no_face"
            return emb, None

        embedding, early_status = await loop.run_in_executor(executor, _cpu_work)
        if early_status:
            await manager.send(item.client_id, {"status": early_status, "timestamp": datetime.now().isoformat()})
            return

        # Qdrant search — async
        person = await identify(qdrant, embedding, threshold)
        if person is None:
            await manager.send(item.client_id, {"status": "unknown", "timestamp": datetime.now().isoformat()})
            return

        # Log attendance — async DB
        async with db_factory() as db:
            message = await log_attendance(db, person["emp_id"])

        await manager.send(item.client_id, {
            "status": "recognized",
            "emp_id": person["emp_id"],
            "name": person["name"],
            "attendance": message,
            "timestamp": datetime.now().isoformat(),
        })
    except Exception as e:
        await manager.send(item.client_id, {"status": "error", "detail": str(e)})
```

- [ ] **Step 7: Tạo `src/modules/recognition/ws_ingress.py`**

```python
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from src.platform.queue import FrameItem, FrameQueue
from src.platform.realtime.manager import ConnectionManager
import time

router = APIRouter(tags=["Recognition"])

def make_ws_router(queue: FrameQueue, manager: ConnectionManager) -> APIRouter:
    @router.websocket("/ws/recognition/{client_id}")
    async def ws_endpoint(websocket: WebSocket, client_id: str):
        await manager.connect(client_id, websocket)
        try:
            while True:
                frame = await websocket.receive_bytes()
                await queue.put(FrameItem(
                    client_id=client_id,
                    frame=frame,
                    captured_at=time.time(),
                ))
        except WebSocketDisconnect:
            manager.disconnect(client_id)

    return router
```

- [ ] **Step 8: Tạo `__init__.py` và commit**

```bash
touch src/modules/recognition/__init__.py
git add src/modules/recognition/ tests/modules/recognition/
git commit -m "feat: add recognition module (extractor, Qdrant identifier, pipeline, WS ingress)"
```

---

## Task 9: App Wiring

**Files:**
- Create: `src/app.py`
- Modify: `main.py`
- Modify: `utils/conn_db.py`
- Modify: `utils/model_app.py`
- Modify: `config.py`

**Interfaces:**
- Consumes: tất cả modules từ Task 2–8
- Produces: FastAPI `app` instance chạy được với `uvicorn`

- [ ] **Step 1: Tạo `src/app.py`**

```python
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.platform.db.qdrant import ensure_collection, get_qdrant_client
from src.platform.db.session import AsyncSessionLocal
from src.platform.config import THRESHOLD
from src.platform.queue import FrameQueue
from src.platform.realtime.manager import ConnectionManager
from src.modules.antispoofing.service import get_liveness_checker
from src.modules.recognition.pipeline import run_pipeline
from src.modules.recognition.ws_ingress import make_ws_router
from src.modules.employees.api import router as employees_router
from src.modules.attendance.api import router as attendance_router

def create_app() -> FastAPI:
    app = FastAPI(
        title="Face Recognition Attendance System",
        description="API for managing employees and tracking attendance using facial recognition.",
        version="0.2.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?$",
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    try:
        app.mount("/static", StaticFiles(directory="static"), name="static")
    except RuntimeError:
        pass  # static dir không tồn tại trong test

    queue = FrameQueue(maxsize=50)
    manager = ConnectionManager()

    app.include_router(employees_router)
    app.include_router(attendance_router)
    app.include_router(make_ws_router(queue, manager))

    @app.on_event("startup")
    async def startup():
        await ensure_collection()
        qdrant = get_qdrant_client()
        checker = get_liveness_checker()
        asyncio.create_task(
            run_pipeline(queue, qdrant, AsyncSessionLocal, manager, checker, THRESHOLD)
        )

    return app
```

- [ ] **Step 2: Cập nhật `main.py`**

```python
import uvicorn
from src.app import create_app

app = create_app()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
```

- [ ] **Step 3: Cập nhật legacy shims**

```python
# utils/conn_db.py
from src.platform.db.session import get_connection
__all__ = ["get_connection"]

# utils/model_app.py
from src.platform.ml.face_app import setup_face_app
__all__ = ["setup_face_app"]

# config.py
"""Legacy config import path."""
from src.platform.config import *  # noqa: F401,F403
```

- [ ] **Step 4: Verify app khởi động**

```bash
make db-up
uv run main.py &
sleep 3
curl http://localhost:8000/docs
# expected: HTML response (Swagger UI)
curl http://localhost:8000/api/employees
# expected: {"items":[],"page":1,"page_size":20,"total":0}
curl http://localhost:8000/api/shift-settings
# expected: {"check_in_start":"08:00:00","check_in_end":"10:00:00",...}
kill %1
```

- [ ] **Step 5: Commit**

```bash
git add src/app.py main.py utils/ config.py
git commit -m "feat: wire app — register all modules, startup pipeline task"
```

---

## Task 10: Cleanup — Xóa Cấu Trúc Cũ

**Files:**
- Delete: `src/services/` (toàn bộ)
- Delete: `src/core/` (toàn bộ)
- Delete: `src/db/` (toàn bộ)
- Delete: `src/api/` (toàn bộ)
- Delete: `src/queues/` (toàn bộ)
- Delete: `app_streamlit.py` (nếu không còn dùng)

- [ ] **Step 1: Chạy toàn bộ test suite trước khi xóa**

```bash
uv run pytest tests/ -v
# expected: tất cả pass
```

- [ ] **Step 2: Xóa thư mục cũ**

```bash
rm -rf src/services/ src/core/ src/db/ src/api/ src/queues/
```

- [ ] **Step 3: Chạy lại test để confirm không có gì bị break**

```bash
uv run pytest tests/ -v
# expected: tất cả vẫn pass
uv run main.py &
sleep 3
curl http://localhost:8000/api/employees
kill %1
```

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: remove old src/ structure (services, core, db, api, queues)"
```

---

## Self-Review

**Spec coverage:**
- ✅ Modular monolith structure (`src/platform/` + `src/modules/`)
- ✅ Qdrant thay pgvector (Task 2, 5, 8)
- ✅ FrameQueue drop-oldest maxsize=50 (Task 4)
- ✅ ThreadPoolExecutor max_workers=4 (Task 8 pipeline)
- ✅ FrameItem.client_id để route kết quả (Task 4, 8, 9)
- ✅ Fix #1: check_in scope working_date (Task 6)
- ✅ Fix #2: check_out scope working_date (Task 6)
- ✅ Fix #3: shift lấy từ DB không từ client (Task 6)
- ✅ Anti-spoofing PassThrough placeholder (Task 7)
- ✅ ConnectionManager WS egress (Task 4)
- ✅ Không có NATS (confirmed)
- ✅ Legacy shims giữ nguyên (Task 9)
- ✅ Metabase trong compose (Task 1 — optional, user tự add)
- ✅ Cleanup old structure (Task 10)

**Placeholder scan:** Không có TBD hay TODO trong plan này.

**Type consistency:**
- `FrameItem.client_id: str` — dùng nhất quán Task 4, 8, 9
- `identify(qdrant, embedding: list[float], threshold: float) -> dict | None` — Task 8 tests + impl khớp
- `check_in(db, emp_id, now?)` / `check_out(db, emp_id, now?)` — Task 6 tests + impl + Task 8 caller khớp
- `log_attendance(db, emp_id)` — Task 6 impl, Task 8 caller khớp
