# Getting Started

## Yêu cầu

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (package manager)
- Docker & Docker Compose
- Node.js 20+ (cho frontend dashboard)

---

## 1. Clone & cài dependencies

```bash
git clone https://github.com/danghohai2004/facevec-attend.git
cd facevec-attend

# Backend Python
uv sync

# Frontend Next.js
cd frontend && npm install && cd ..
```

---

## 2. Cấu hình môi trường (backend)

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

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_API_KEY=<chuỗi-random-đủ-dài>   # bắt buộc: compose sẽ fail nếu để trống

# Backend write API authentication
API_KEY=<chuỗi-random-đủ-dài>          # bảo vệ các endpoint ghi (X-API-Key)

# Deployment environment
ENV=development                        # "production" → từ chối boot nếu liveness còn là PassThrough
```

> `QDRANT_API_KEY` và `API_KEY` nên là chuỗi random ≥ 32 bytes, **không commit**. `QDRANT_API_KEY` dùng ở **cả** compose (bật auth cho Qdrant) và backend (client gửi kèm). `API_KEY` phải **khớp** giữa backend và env server của frontend (bước 5).

---

## 3. Khởi động database

```bash
make db-up
# hoặc: docker compose up -d
```

`compose.yaml` khởi động:
- **PostgreSQL** — schema tự tạo từ `initdb/init.sql` (3 bảng + seed `shift_settings`).
- **Qdrant** — collection `face_embeddings` tự tạo khi backend startup lần đầu.

Cả hai bind cổng vào `127.0.0.1` (chỉ host truy cập được), Qdrant image pin phiên bản và bật API key:

```yaml
qdrant:
  image: qdrant/qdrant:v1.18.2
  container_name: qdrant
  environment:
    QDRANT__SERVICE__API_KEY: ${QDRANT_API_KEY:?QDRANT_API_KEY is required}
  ports:
    - "127.0.0.1:${QDRANT_PORT:-6333}:6333"
  volumes:
    - qdrant_data:/qdrant/storage
  restart: unless-stopped
```

Kiểm tra:
```bash
docker logs postgres_attendance     # PostgreSQL ready
# Qdrant dashboard: http://localhost:6333/dashboard  (cần header api-key)
```

---

## 4. Chạy backend

```bash
make run
# hoặc: uv run main.py
```

API docs tự động tại: `http://localhost:8000/docs`

---

## 5. Chạy frontend (dashboard)

Frontend là **dashboard quản trị** (không phải kiosk). Để các thao tác **ghi** hoạt động, tạo `frontend/.env.local` với env **server-side** cho BFF proxy:

```env
# đọc: browser gọi thẳng backend
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000

# ghi: chỉ đọc phía server, KHÔNG đặt tiền tố NEXT_PUBLIC_
API_KEY=<đúng bằng API_KEY của backend>
BACKEND_INTERNAL_URL=http://localhost:8000
```

> `API_KEY` ở đây **phải trùng** với `API_KEY` của backend. Nếu lệch → mọi thao tác ghi trả 401. Không bao giờ đặt tên `NEXT_PUBLIC_API_KEY` (sẽ lộ key ra browser).

```bash
cd frontend
npm run dev
```

Mở browser: `http://localhost:3000` → tự redirect tới `/dashboard`. Các trang:
`/dashboard` (tổng quan — hiện dùng dữ liệu mẫu), `/employees` (CRUD + đăng ký khuôn mặt), `/attendance` (chấm công thủ công theo `emp_id`), `/shifts` (cấu hình ca).

---

## 6. Đăng ký nhân viên đầu tiên

Qua UI: `/employees` → form đăng ký (tên, mã, ảnh).

Hoặc gọi API trực tiếp (nhớ `X-API-Key`):
```bash
curl -X POST http://localhost:8000/api/employees \
  -H "X-API-Key: <API_KEY>" \
  -F "name=Nguyen Van A" \
  -F "emp_code=NV001" \
  -F "file=@/path/to/face_photo.jpg"
```

Ảnh phải rõ, có **đúng 1** khuôn mặt, ≤ 5MB.

---

## 7. Kiểm tra đồng bộ PG ↔ Qdrant (tùy chọn)

```bash
uv run python -m scripts.reconcile_vectors   # hoặc: uv run python scripts/reconcile_vectors.py
```

Báo cáo (read-only) nhân viên thiếu vector (MISSING) hoặc vector thừa (ORPHAN). Exit `0` khớp, `1` lệch, `2` lỗi kết nối.

---

## Makefile commands

```bash
make help       # xem tất cả lệnh
make install    # cài dependencies (uv sync)
make db-up      # khởi động PostgreSQL + Qdrant
make db-down    # dừng containers (GIỮ data)
make db-reset   # dừng + XÓA volume (hỏi xác nhận)
make run        # chạy FastAPI backend
make clean      # xóa __pycache__, .pyc, .pytest_cache
```

---

## Chạy tests

```bash
uv run pytest -q                 # backend
cd frontend && npm run lint && npx tsc --noEmit && npm run build   # frontend
```

---

## Cấu trúc thư mục

```
facevec-attend/
  main.py           # FastAPI entry point (uvicorn)
  src/              # backend source
    platform/       # hạ tầng (config, auth, queue, db, ml, realtime)
    modules/        # domain (employees, attendance, recognition, antispoofing)
  scripts/          # reconcile_vectors.py
  frontend/         # Next.js dashboard + BFF write proxy
  initdb/           # SQL schema khởi tạo PostgreSQL
  tests/            # pytest suite
  docs/             # tài liệu
  compose.yaml      # PostgreSQL + Qdrant
  pyproject.toml    # Python deps (uv)
```

---

## Troubleshooting

**Database không kết nối được:**
```bash
docker ps                        # container đang chạy?
docker logs postgres_attendance  # xem log PostgreSQL
docker logs qdrant               # xem log Qdrant
```

**Compose báo `QDRANT_API_KEY is required`:** chưa set `QDRANT_API_KEY` trong `.env`.

**insightface model chưa download:** lần đầu chạy, model `buffalo_sc` (~300MB) tự tải vào `~/.insightface/`. Cần internet.

**Qdrant collection chưa tồn tại:** backend tự tạo `face_embeddings` khi startup. Nếu lỗi, kiểm tra Qdrant tại `http://localhost:6333` và `QDRANT_API_KEY` đúng.

**Thao tác ghi trên dashboard trả 401/503:** kiểm tra `API_KEY` ở `frontend/.env.local` **khớp** backend (401), và backend đã set `API_KEY` (503).

**Backend từ chối khởi động ở production:** khi `ENV=production` mà liveness vẫn là `PassThroughChecker`, app cố tình raise — phải cung cấp `LivenessChecker` thật trước khi chạy prod.
