# Architecture Overview

## Tóm tắt

Hệ thống chấm công bằng khuôn mặt realtime. Camera gửi frame → backend nhận diện → tự động ghi check-in/check-out. Ngoài luồng nhận diện, có một **dashboard quản trị** (Next.js) để quản lý nhân viên, cấu hình ca làm việc, và chấm công thủ công theo mã nhân viên.

> **Trạng thái hiện tại:** luồng nhận diện qua WebSocket đã hoạt động ở **backend**. Frontend hiện là **dashboard quản trị** — client camera/kiosk (mở webcam, gửi frame qua WebSocket) **chưa được wire**, xếp vào Phase 3. Xem [`../security/follow-ups.md`](../security/follow-ups.md).

---

## Deployment Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                            │
│  ┌─────────────────┐   GET (đọc, trực tiếp)   ┌──────────────────────┐    │
│  │  Admin Browser  │ ───────────────────────► │   FastAPI Server     │    │
│  │  Next.js :3000  │   ghi (POST/PUT/DELETE)  │   :8000              │    │
│  │  /dashboard     │ ──► BFF proxy ─(X-API-Key)──►                   │    │
│  │  /employees     │     /api/write/*          │  ┌──────────────┐   │    │
│  │  /attendance    │ ◄─────────────────────────│  │ Frame Queue  │   │    │
│  │  /shifts        │                           │  │ (asyncio)    │   │    │
│  └─────────────────┘                           │  └──────┬───────┘   │    │
│                                                │         │            │    │
│  ┌─────────────────┐   WebSocket (JPEG frames) │  ┌──────▼───────┐   │    │
│  │  Camera client  │ ─────────────────────────►│  │   Pipeline   │   │    │
│  │  (Phase 3 —      │ ◄──────────────────────── │  │ liveness →   │   │    │
│  │   chưa wire)     │   kết quả nhận diện       │  │ extract →    │   │    │
│  └─────────────────┘                           │  │ identify →   │   │    │
│                                                │  │ log          │   │    │
│                                                │  └──────┬───────┘   │    │
│                                                └─────────┼───────────┘    │
│                          ┌───────────────────────────────┤                │
│              ┌───────────▼──────────┐          ┌──────────▼───────────┐   │
│              │  PostgreSQL :5432     │          │   Qdrant :6333        │   │
│              │  employees            │          │   face_embeddings     │   │
│              │  attendance_logs      │          │   (512-dim, cosine,   │   │
│              │  shift_settings       │          │    HNSW)              │   │
│              └───────────────────────┘          └──────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
```

Cổng PostgreSQL và Qdrant được bind `127.0.0.1` (chỉ truy cập từ host), Qdrant bật API key. Xem [`../setup/getting-started.md`](../setup/getting-started.md).

---

## Components

### 1. Frontend — Admin Dashboard (Next.js, App Router)

Chạy như **Node server** (`next start`, không phải static export) để có Route Handlers phía server. Các trang:

| Route | Nội dung | Dữ liệu |
|---|---|---|
| `/` | redirect → `/dashboard` | — |
| `/dashboard` | tổng quan / KPI | **mock** (placeholder, chưa nối API thật) |
| `/employees` | CRUD nhân viên + đăng ký khuôn mặt | API thật |
| `/attendance` | chấm công thủ công theo `emp_id` + trạng thái | API thật (`checkin`/`checkout`) |
| `/shifts` | cấu hình khung giờ ca | API thật (`shift-settings`) |
| `/settings` | (disabled) | — |

**Đọc vs ghi:**
- **Đọc (GET):** gọi thẳng backend qua `NEXT_PUBLIC_API_BASE_URL` (mặc định `http://localhost:8000`).
- **Ghi (POST/PUT/DELETE):** đi qua **BFF proxy** cùng origin tại `/api/write/[...path]` (xem mục Security). Browser không bao giờ thấy `X-API-Key`.

### 2. Backend API (FastAPI) — Modular Monolith

1 process, tách rõ ranh giới domain:

```
src/
  app.py                      # app factory: lifespan (prod guard → ensure_collection →
                              #   pipeline task → shutdown drain), CORS, mount routers
  platform/                   # hạ tầng dùng chung, KHÔNG chứa business logic
    config.py                 #   MODEL (buffalo_sc, det_size, providers) + THRESHOLD = 0.6
    auth.py                   #   require_api_key: fail-closed, so sánh hằng-thời-gian X-API-Key
    queue.py                  #   FrameQueue: bounded asyncio.Queue, drop-oldest
    db/
      base.py                 #   DeclarativeBase
      session.py              #   async engine + AsyncSessionLocal + get_db()  (asyncpg)
      qdrant.py               #   AsyncQdrantClient singleton + ensure_collection()
    ml/
      face_app.py             #   singleton InsightFace FaceAnalysis
    realtime/
      manager.py              #   ConnectionManager: push kết quả về đúng WS client_id
  modules/
    employees/                # CRUD nhân viên + đăng ký embedding
      api.py  service.py  schemas.py  models.py
    attendance/               # check-in/out (enforce shift-window), ca làm việc
      api.py  service.py  schemas.py  models.py
    recognition/
      ws_ingress.py           # WS /ws/recognition/{client_id}: nhận frame → FrameQueue
      pipeline.py             # asyncio task: queue → liveness → extract → identify → log → push
      identifier.py           # Qdrant cosine search (score_threshold = 1 - THRESHOLD)
      extractor.py            # insightface embedding extraction
    antispoofing/
      service.py              # LivenessChecker (ABC) + PassThroughChecker (placeholder)
```

Entry point: `main.py` → `create_app()` → `uvicorn` (:8000).

### 3. Storage

**PostgreSQL** — dữ liệu quan hệ (employees, attendance_logs, shift_settings), truy cập async qua SQLAlchemy/asyncpg.

**Qdrant** — vector search (collection `face_embeddings`, 512-dim, Cosine, HNSW).

```
PostgreSQL :5432    ← SQLAlchemy async (asyncpg)
Qdrant     :6333    ← qdrant-client (AsyncQdrantClient, có API key)
```

Embedding **không** lưu trong PostgreSQL. Qdrant lưu vector + payload `{emp_id, emp_code, name}` để tránh join sau khi search. Ảnh khuôn mặt **không** được lưu ở đâu cả — chỉ tồn tại trong RAM lúc đăng ký đủ để trích embedding.

Xem chi tiết: [`../data/schema.md`](../data/schema.md)

---

## Frame Processing Pipeline

```
Camera client (Phase 3)
  │  JPEG bytes qua WebSocket  /ws/recognition/{client_id}
  ▼
ws_ingress.py
  │  FrameQueue.put(FrameItem{client_id, frame, captured_at})
  │  └─ drop-oldest nếu queue đầy (ưu tiên frame mới nhất)
  ▼
pipeline.py  ← asyncio background task (run_pipeline)
  │  Semaphore(4): tối đa 4 frame xử lý đồng thời (khớp ThreadPoolExecutor 4 worker)
  │
  ├─► LivenessChecker.check(frame)         [chạy trong threadpool cùng bước decode]
  │     PassThroughChecker → True (placeholder). status "spoof" nếu bị từ chối.
  │
  ├─► extract_embedding_from_frame(img)    [insightface, threadpool, không block loop]
  │     → normed_embedding 512-dim của khuôn mặt lớn nhất; None → status "no_face"
  │
  ├─► identify_face(qdrant, embedding, THRESHOLD)
  │     Qdrant search cosine, limit=1, score_threshold = 1 - THRESHOLD
  │     không đạt → status "unknown"; đạt → payload {emp_id, name, emp_code}
  │
  └─► log_attendance(db, emp_id)
        Khung giờ lấy từ DB shift_settings (không tin client); tự chọn check_in/check_out
        theo thời điểm hiện tại (timezone Asia/Ho_Chi_Minh). Ngoài giờ → "Not during working hours".
        ▼
      ConnectionManager.send(client_id, result)   → đẩy JSON về đúng client
```

**Concurrency & backpressure:**
- `FrameQueue` giới hạn 50 frame, drop-oldest → luôn ưu tiên frame mới nhất.
- `run_pipeline` acquire `asyncio.Semaphore(4)` **trước khi** lấy frame, nên khi đủ tải nó nhường chỗ cho frame mới nhất thay vì xử lý frame cũ.
- CPU-bound (decode + insightface) chạy trong `ThreadPoolExecutor(max_workers=4)`, không block event loop.

**Graceful shutdown:** khi tắt app, lifespan cancel pipeline task, **drain** các frame đang xử lý (cancel + gather), rồi đóng Qdrant client và dispose DB engine — không bỏ sót resource.

**FrameItem** mang `client_id` để route kết quả về đúng client:
```python
@dataclass(frozen=True)
class FrameItem:
    client_id: str   # định danh client gửi frame — dùng để route kết quả về
    frame: bytes
    captured_at: float
```

---

## Security Model (write protection)

Chi tiết & threat model: [`../api/api_spec.md`](../api/api_spec.md) và [`../security/follow-ups.md`](../security/follow-ups.md).

```
Browser ──POST/PUT/DELETE──► Next BFF /api/write/[...path] ──+ X-API-Key──► FastAPI (require_api_key)
 (không có key)               (server-side, key ở env server)                (fail-closed)
```

- **Backend fail-closed** (`platform/auth.py`): các endpoint **ghi** phụ thuộc `require_api_key`. Nếu server chưa cấu hình `API_KEY` → `503`; thiếu/sai `X-API-Key` → `401`. So sánh hằng-thời-gian (`hmac.compare_digest`).
- **BFF proxy** (`frontend/src/app/api/write/[...path]/route.ts`): chỉ cho phép các route trong `ALLOWED_TARGETS`, chèn `X-API-Key` từ env **server** (`API_KEY`), forward tới `BACKEND_INTERNAL_URL`. Có **CSRF guard**: chặn request có `Origin` khác host (403). Key không bao giờ lộ ra browser (không dùng `NEXT_PUBLIC_`).
- **Endpoint được bảo vệ:** `POST/DELETE /api/employees`, `PUT /api/shift-settings`, `POST /api/attendance/checkin|checkout`. **Công khai:** các GET + WebSocket ingress.

> Đây **không** phải user-auth: bất kỳ ai mở được frontend vẫn gọi được write (proxy tự chèn key). Login/RBAC + audit là Phase 3.

**Production boot guard:** nếu `ENV=production` mà liveness vẫn là `PassThroughChecker`, app **từ chối khởi động** (RuntimeError) — tránh chạy prod với anti-spoofing giả.

---

## Anti-Spoofing

Phát hiện người dùng giơ ảnh/video/mask thay vì mặt thật. Interface cố định, pipeline không đổi khi thay implementation:

```
frame(bytes) → LivenessChecker.check() → bool
```

- **Hiện tại:** `PassThroughChecker` — luôn trả `True` (placeholder).
- **Production:** implement `LivenessChecker` thật (vd Silent-Face-Anti-Spoofing) và trả về từ `get_liveness_checker()`. Guard ở trên chặn boot prod nếu quên.

---

## Data Consistency (PG ↔ Qdrant)

Đăng ký commit PostgreSQL **trước**, upsert Qdrant **sau**; xoá thì xoá Qdrant trước rồi PostgreSQL. Nếu bước Qdrant fail sau khi PG đã commit, hai kho có thể lệch. PostgreSQL **không** lưu ảnh/embedding nên **không thể** tự dựng lại vector.

Công cụ: `scripts/reconcile_vectors.py` (read-only) — so sánh `emp_id` giữa PG và Qdrant, báo:
- **MISSING VECTOR** (có ở PG, thiếu vector) → operator đăng ký lại.
- **ORPHAN VECTOR** (có vector, không còn ở PG) → operator prune thủ công.

Exit `0` khớp, `1` lệch, `2` lỗi kết nối. Không tự sửa/không mutate.

---

## Scalability Path (khi thật sự cần)

| Nhu cầu | Thay đổi |
|---|---|
| Nhiều camera → 1 server | Tăng `FrameQueue` maxsize + `Semaphore`/`ThreadPoolExecutor` workers |
| Worker nhận diện sang máy GPU riêng | Thay `asyncio.Queue` bằng message broker (1 adapter) |
| Nhiều văn phòng | Multi-tenant schema hoặc tách DB per-tenant |
| Liveness thật | Implement `LivenessChecker` (Silent-Face…) |
| Phân quyền theo người | User login/RBAC + audit (Phase 3) |

**Không có ngay bây giờ:** NATS/Redis/message broker, separate worker process, login/RBAC.
