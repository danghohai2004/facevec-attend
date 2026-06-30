# Getting Started

## Yêu cầu

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (package manager)
- Docker & Docker Compose
- Node.js 18+ (cho frontend kiosk)

---

## 1. Clone & cài dependencies

```bash
git clone <repo-url>
cd facevec-attend

# Backend Python
uv sync

# Frontend Next.js
cd frontend && npm install && cd ..
```

---

## 2. Cấu hình môi trường

```bash
cp .env.example .env
```

Sửa `.env`:

```env
# PostgreSQL
DB_NAME=attendance
DB_USER=postgres
DB_PASS=your_password
DB_HOST=localhost
DB_PORT=5432
CONTAINER_NAME_DB=postgres_attendance

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

---

## 3. Khởi động database

```bash
make db-up
# hoặc: docker compose up -d
```

Khởi động services:
- **PostgreSQL** — schema tự tạo từ `initdb/init.sql` (3 bảng + seed shift_settings)
- **Qdrant** — collection `face_embeddings` tự tạo khi app startup lần đầu

`compose.yaml` cần có Qdrant:
```yaml
qdrant:
  image: qdrant/qdrant:latest
  container_name: qdrant
  ports:
    - "${QDRANT_PORT:-6333}:6333"
  volumes:
    - qdrant_data:/qdrant/storage
  restart: unless-stopped
```

Kiểm tra:
```bash
docker logs postgres_attendance  # PostgreSQL ready
# Qdrant UI: http://localhost:6333/dashboard
```

---

## 4. Chạy backend

```bash
make run
# hoặc: uv run main.py
```

API docs tự động tại: `http://localhost:8000/docs`

---

## 5. Chạy frontend (kiosk)

```bash
cd frontend
npm run dev
```

Mở browser: `http://localhost:3000/kiosk`

**Chế độ kiosk thật (màn hình cửa):**

```bash
google-chrome --kiosk http://localhost:3000/kiosk
# hoặc Chromium:
chromium-browser --kiosk http://localhost:3000/kiosk
```

---

## 6. Chạy Metabase (dashboard HR)

Thêm vào `compose.yaml`:

```yaml
metabase:
  image: metabase/metabase:latest
  container_name: metabase
  ports:
    - "3001:3000"
  environment:
    MB_DB_TYPE: postgres
    MB_DB_DBNAME: ${DB_NAME}
    MB_DB_PORT: 5432
    MB_DB_USER: ${DB_USER}
    MB_DB_PASS: ${DB_PASS}
    MB_DB_HOST: db
  depends_on:
    - db
  restart: unless-stopped
```

```bash
docker compose up -d metabase
```

Mở `http://localhost:3001` → setup admin account → kết nối vào database chấm công.

---

## 7. Đăng ký nhân viên đầu tiên

```bash
curl -X POST http://localhost:8000/api/employees \
  -F "name=Nguyen Van A" \
  -F "emp_code=NV001" \
  -F "file=@/path/to/face_photo.jpg"
```

Hoặc dùng Swagger UI tại `http://localhost:8000/docs`.

---

## Makefile commands

```bash
make help       # xem tất cả lệnh
make install    # cài dependencies (uv sync)
make db-up      # khởi động PostgreSQL + Qdrant
make db-down    # dừng tất cả (xóa volume)
make run        # chạy FastAPI backend
make clean      # xóa __pycache__, .pyc
```

---

## Cấu trúc thư mục

```
facevec-attend/
  src/              # backend source code
    platform/       # hạ tầng (db, config, ml model, realtime)
    modules/        # domain logic (employees, attendance, recognition, antispoofing)
  frontend/         # Next.js kiosk web app
  initdb/           # SQL schema khởi tạo database
  docs/             # tài liệu
  compose.yaml      # Docker Compose
  pyproject.toml    # Python dependencies (uv)
  main.py           # FastAPI entry point
```

---

## Troubleshooting

**Database không kết nối được:**
```bash
docker ps  # kiểm tra container đang chạy
docker logs pgvector_face  # xem logs
```

**insightface model chưa download:**

Lần đầu chạy, `buffalo_sc` model (~300MB) sẽ tự download vào `~/.insightface/`. Cần internet.

**Qdrant collection chưa tồn tại:**

App tự tạo collection `face_embeddings` khi startup. Nếu lỗi, kiểm tra Qdrant đang chạy tại `http://localhost:6333`.

**Camera không bật trên kiosk:**

Browser yêu cầu HTTPS hoặc `localhost` để dùng `getUserMedia()`. Dev trên `localhost` là OK. Production cần HTTPS.
