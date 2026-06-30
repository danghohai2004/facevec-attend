# Architecture Overview

## Tóm tắt

Hệ thống chấm công khuôn mặt realtime. Nhân viên nhìn vào camera kiosk đặt trước cửa → nhận diện khuôn mặt → tự động ghi check-in/check-out. HR xem báo cáo qua Metabase.

---

## Deployment Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                         PRODUCTION                               │
│                                                                  │
│  ┌─────────────────┐   WebSocket    ┌──────────────────────┐    │
│  │  Kiosk Browser  │ ─────────────► │   FastAPI Server     │    │
│  │  (Chrome kiosk) │ ◄───────────── │   :8000              │    │
│  │  /kiosk         │   kết quả      │                      │    │
│  └─────────────────┘                │   ┌──────────────┐   │    │
│                                     │   │ Frame Queue  │   │    │
│                                     │   │ (asyncio)    │   │    │
│                                     │   └──────┬───────┘   │    │
│                                     │          │            │    │
│                                     │   ┌──────▼───────┐   │    │
│                                     │   │   Pipeline   │   │    │
│                                     │   │ anti-spoof   │   │    │
│                                     │   │ + extract    │   │    │
│                                     │   │ + identify   │   │    │
│                                     │   └──────┬───────┘   │    │
│                                     └──────────┼───────────┘    │
│                                                │                 │
│  ┌─────────────────┐                ┌──────────▼───────────┐    │
│  │  HR Browser     │                │   PostgreSQL :5432    │    │
│  │  Metabase :3001 │ ─────────────► │   employees          │    │
│  │  (báo cáo)      │                │   attendance_logs    │    │
│  └─────────────────┘                │   shift_settings     │    │
│                                     └──────────────────────┘    │
│                                     ┌──────────────────────┐    │
│                                     │   Qdrant :6333        │    │
│                                     │   face_embeddings    │    │
│                                     │   (vector search)    │    │
│                                     └──────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

---

## Components

### 1. Kiosk Frontend (Next.js)

**Là web app** chạy trên Chrome ở chế độ kiosk — fullscreen, ẩn thanh địa chỉ:

```bash
google-chrome --kiosk http://localhost:3000/kiosk
```

- Dùng `getUserMedia()` để mở camera, không cần app native
- Chụp frame định kỳ → gửi qua WebSocket lên backend
- Nhận kết quả → hiển thị "Xin chào [tên] — Check in 08:02"
- Deploy chỉ cần update server, không cài lại trên từng màn hình

### 2. Backend API (FastAPI)

**Modular monolith** — 1 process, tách rõ ranh giới domain:

```
src/
  app.py                      # entry point, đăng ký router + startup pipeline task
  platform/                   # hạ tầng dùng chung, không chứa business logic
    config.py                 # MODEL, THRESHOLD, ORIGINAL_IMG_PATH, MAX_EMB_FACE
    db/
      base.py                 # DeclarativeBase
      session.py              # async engine + get_db + sync get_connection
    ml/
      face_app.py             # singleton FaceAnalysis (insightface buffalo_sc)
    realtime/
      manager.py              # ConnectionManager: push kết quả về đúng WS client
      sse.py                  # SSE stream (fallback)
    queue.py                  # FrameQueue: bounded asyncio.Queue, drop-oldest
  modules/
    employees/                # CRUD nhân viên + đăng ký embedding
      api.py  service.py  schemas.py  models.py
    attendance/               # check-in/check-out, ca làm việc
      api.py  service.py  schemas.py  models.py
    recognition/
      ws_ingress.py           # WS endpoint: nhận frame → FrameQueue
      pipeline.py             # asyncio task: queue → anti-spoof → extract → identify → push
      identifier.py           # Qdrant cosine similarity search
      extractor.py            # insightface embedding extraction
    antispoofing/
      service.py              # LivenessChecker interface + PassThroughChecker default
```

### 3. Storage

**PostgreSQL** — dữ liệu quan hệ (employees, attendance_logs, shift_settings)

**Qdrant** — vector search (face_embeddings collection, HNSW + Cosine)

```
PostgreSQL :5432    ← SQLAlchemy async
Qdrant     :6333    ← qdrant-client
```

Embedding không lưu trong PostgreSQL nữa. Qdrant lưu vector + payload `{emp_id, name, emp_code}` để tránh join sau khi search.

Xem chi tiết: [`../data/schema.md`](../data/schema.md)

### 4. Dashboard (Metabase)

**Web tool có sẵn** — không viết code, chạy qua Docker:

- Kết nối trực tiếp vào PostgreSQL
- HR tự tạo báo cáo: bảng công, thống kê muộn/vắng, export Excel
- Tương tự Apache Superset / Grafana, chọn Metabase vì nhẹ và dễ self-host nhất

---

## Frame Processing Pipeline

```
Kiosk Browser
  │  JPEG bytes qua WebSocket
  ▼
ws_ingress.py
  │  FrameQueue.put(FrameItem)
  │  └─ drop-oldest nếu queue đầy (ưu tiên frame mới nhất)
  ▼
pipeline.py  ← asyncio background task
  │
  ├─► AntiSpoofing.check(frame)
  │     PassThroughChecker: True  (swap Silent-Face model khi cần)
  │     Từ chối nếu score ≤ threshold
  │
  ├─► Extractor.extract(frame)
  │     insightface → normed_embedding vector 512 chiều
  │     Trả [] nếu không tìm thấy khuôn mặt
  │
  ├─► Identifier.identify(embedding)
  │     Qdrant search: cosine similarity, limit=1, score_threshold=1-THRESHOLD
  │     score < threshold → Unknown
  │     payload trả về: {emp_id, name} — không cần join PostgreSQL
  │
  └─► Attendance.log(emp_id, shift_from_db)
        Khung giờ lấy từ DB shift_settings, không tin client
        check_in / check_out tùy thời điểm hiện tại
        ▼
      ConnectionManager.send(client_id, result)
        ▼
      Kiosk hiển thị kết quả realtime
```

**Backpressure:** FrameQueue giới hạn 50 frames (5 camera × 10 frame buffer mỗi cái). Pipeline chạy CPU-bound trong `ThreadPoolExecutor(max_workers=4)`, không block asyncio event loop. Đủ xử lý 5 camera trên 1 máy có GPU.

**FrameItem** mang `client_id` để `ConnectionManager` push kết quả về đúng kiosk:
```python
@dataclass
class FrameItem:
    client_id: str   # ID của kiosk gửi frame — dùng để route kết quả về
    frame: bytes
    captured_at: float
```

---

## Anti-Spoofing

Phát hiện người dùng giơ ảnh/video/mask trước camera thay vì mặt thật.

**Kỹ thuật áp dụng:** Passive Liveness Detection — phân tích texture frame, không yêu cầu hành động từ người dùng.

```
frame → LivenessChecker.check() → bool
```

- **Hiện tại:** `PassThroughChecker` — luôn trả `True` (placeholder)
- **Production:** swap sang `SilentFaceChecker` (Silent-Face-Anti-Spoofing model)

Interface cố định, pipeline không đổi khi thay implementation.

---

## Tất cả là Web

| Thành phần | Loại | Ai viết |
|---|---|---|
| `/kiosk` | Web (Next.js) + Chrome kiosk mode | Dev |
| Metabase | Web tool Docker, HR tự dùng | Không viết |
| FastAPI | Backend API | Dev |
| PostgreSQL | Database | Không viết |

---

## Scalability Path (khi thật sự cần)

| Nhu cầu | Thay đổi |
|---|---|
| Nhiều camera → 1 server | Tăng FrameQueue maxsize + ThreadPoolExecutor workers |
| Worker nhận diện sang máy GPU riêng | Thay asyncio.Queue bằng NATS (1 file adapter) |
| Nhiều văn phòng | Multi-tenant schema hoặc tách DB per-tenant |
| Liveness detection thật | Implement `SilentFaceChecker(LivenessChecker)` |

**Không có ngay bây giờ:** NATS, Redis, message broker, separate worker process.
