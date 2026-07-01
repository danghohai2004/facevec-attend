# System Architecture — Face Recognition Attendance System

**Date:** 2026-06-30  
**Status:** Approved

---

## 1. Overview

Hệ thống chấm công khuôn mặt realtime cho văn phòng. Nhân viên nhìn vào camera kiosk đặt trước cửa → hệ thống nhận diện → tự động ghi chấm công. HR xem báo cáo qua Metabase.

```
┌─────────────────────────────────────────────────────────────────┐
│                        DEPLOYMENT                               │
│                                                                 │
│  [Kiosk browser]  ──WS──►  [FastAPI server]  ◄──►  [PostgreSQL]│
│  /kiosk (Next.js)                │                  + pgvector  │
│                                  │                              │
│  [HR browser]  ─────────►  [Metabase]  ──────────►  [PostgreSQL]│
│  (báo cáo)                                                      │
│                                          [Qdrant :6333]         │
│                                          face_embeddings        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Components

### 2.1 Kiosk Frontend (`frontend/` — Next.js)

**Là web app** chạy trên Chrome ở chế độ kiosk (fullscreen, ẩn thanh địa chỉ, ẩn tab):

```bash
google-chrome --kiosk http://localhost:3000/kiosk
```

Trông như app native nhưng thực ra là trang web — deploy chỉ cần update server, không cần cài lại trên từng màn hình.

- Route `/kiosk`: fullscreen camera, chụp frame định kỳ, gửi qua WebSocket
- Nhận kết quả từ server qua WebSocket: hiển thị "Xin chào [tên] — Check in 08:02"
- Không cần login, chạy trên browser kiosk cố định tại cửa
- Camera access qua `getUserMedia()` API của browser — không cần app native

### 2.2 Backend API (`src/` — FastAPI)

Modular monolith: 1 process, tách rõ ranh giới domain.

```
src/
  app.py                      # create_app() + đăng ký router + startup worker task
  platform/                   # hạ tầng, không chứa business logic
    config.py                 # biến cấu hình (model, threshold, paths)
    db/
      base.py                 # DeclarativeBase dùng chung
      session.py              # async engine (SQLAlchemy) + get_db + sync get_connection
    ml/
      face_app.py             # singleton FaceAnalysis (insightface)
    realtime/
      manager.py              # ConnectionManager: gửi kết quả về đúng WS client
      sse.py                  # SSE stream (fallback cho client không support WS)
    queue.py                  # FrameQueue: bounded asyncio.Queue, drop-oldest khi đầy
  modules/
    employees/                # CRUD nhân viên + đăng ký embedding khuôn mặt
      api.py  service.py  schemas.py  models.py
    attendance/               # logic chấm công (check-in/check-out, ca làm việc)
      api.py  service.py  schemas.py  models.py
    recognition/
      ws_ingress.py           # WS endpoint: nhận frame → FrameQueue.put()
      pipeline.py             # asyncio task: get frame → anti-spoof → extract → identify → push
      identifier.py           # truy vấn pgvector tìm nhân viên gần nhất
      extractor.py            # extract embedding từ frame (insightface)
    antispoofing/
      service.py              # interface LivenessChecker + PassThroughChecker (default)
```

### 2.3 Database (PostgreSQL + pgvector)

```sql
employees          -- nhân viên (emp_id, emp_code, name)
face_embeddings    -- vector 512 chiều mỗi khuôn mặt (nhiều/người)
attendance_logs    -- log chấm công (checkin_time, checkout_time, working_date)
shift_settings     -- khung giờ ca làm việc (check_in_start..check_out_end)
```

### 2.4 Dashboard (Metabase)

**Là web tool có sẵn** — không viết code, chạy qua Docker, HR mở browser vào `localhost:3001`:

```
┌─────────────────────────────────────────────────┐
│  Metabase                          [HR login]   │
├─────────────────────────────────────────────────┤
│  Báo cáo tháng 6                               │
│                                                 │
│  Nhân viên    Ngày công   Đi muộn   Vắng       │
│  Nguyễn Văn A   22/22      0         0          │
│  Trần Thị B     18/22      3         4          │
│                                                 │
│  [Kéo thả tạo chart, filter, export Excel]     │
└─────────────────────────────────────────────────┘
```

- Kết nối trực tiếp vào PostgreSQL, tự đọc schema bảng `attendance_logs`
- HR tự tạo báo cáo: bảng công, thống kê muộn/vắng, export Excel — không cần dev
- Tương tự Apache Superset, Grafana, Power BI — chọn Metabase vì nhẹ và dễ self-host nhất

---

## 3. Frame Processing Pipeline

```
Browser (kiosk)
  │  binary frame (JPEG bytes) qua WebSocket
  ▼
ws_ingress.py
  │  FrameQueue.put(FrameItem)   ← drop-oldest nếu queue đầy (ưu tiên frame mới)
  ▼
pipeline.py  (asyncio background task, CPU-bound chạy trong threadpool)
  │
  ├─► antispoofing.check(frame)
  │     PassThroughChecker → True (placeholder, cắm model liveness sau)
  │
  ├─► extractor.extract(frame)
  │     insightface → embedding vector 512 chiều
  │
  ├─► identifier.identify(embedding, threshold)
  │     pgvector cosine distance → emp_id, name, distance
  │
  └─► attendance.log(emp_id, shift)
        check-in hoặc check-out tùy khung giờ
        ▼
      ConnectionManager.send(client_id, result)
        ▼
      Kiosk hiển thị kết quả
```

**Backpressure:** FrameQueue bounded (mặc định 30 frames). Khi đầy → drop frame cũ nhất. Pipeline xử lý 1 frame/lần trong threadpool, không block event loop.

---

## 4. Correctness Fixes (đưa vào khi restructure)

Ba lỗi từ code review cần fix khi chuyển sang cấu trúc mới:

1. **Trust boundary:** `/attendance/checkin` và `/checkout` không nhận `shifts_time` từ client nữa — load từ DB `shift_settings`. Client không tự override được khung giờ.

2. **Open-log scope:** Query "already checked in" và "find log to check out" thêm filter `working_date = today`. Tránh log ngày hôm qua khóa chéo ngày hôm nay.

3. **Cross-day checkout:** Nhân viên quên check-out hôm qua, hôm nay check-out sẽ không bị ghi nhầm vào log cũ.

---

## 5. Infrastructure

```yaml
# docker-compose.yaml
services:
  db:        # PostgreSQL + pgvector (đã có)
  api:       # FastAPI (uvicorn)
  metabase:  # Metabase dashboard
  frontend:  # Next.js (kiosk)
```

**Không có:** NATS, Redis, message broker, separate worker process. Tất cả chạy in-process. Tách ra khi có nhu cầu scale thật.

---

## 6. Scalability Path (khi cần, không phải bây giờ)

| Nhu cầu | Thay đổi |
|---|---|
| Nhiều camera → 1 server | Tăng `maxsize` FrameQueue + ThreadPoolExecutor workers |
| Camera → GPU server riêng | Thay `asyncio.Queue` bằng NATS adapter (1 file) |
| Nhiều văn phòng | Multi-tenant schema hoặc tách DB per-tenant |
| Liveness detection thật | Implement `LivenessChecker` interface, cắm vào pipeline |

---

## 7. Tất cả đều là Web

| Thành phần | Loại | Ai viết |
|---|---|---|
| `/kiosk` | Web (Next.js) chạy fullscreen trên Chrome kiosk mode | Bạn viết |
| Metabase | Web tool có sẵn, self-host qua Docker | Không viết |
| FastAPI | Backend API | Bạn viết |
| PostgreSQL | Database | Không viết |

---

## 8. Out of Scope

- Authentication kiosk (kiosk không cần login theo thiết kế)
- Mobile app (kiosk cố định, không cần)
- NATS / message broker (in-process queue đủ cho usecase hiện tại)
- Custom dashboard frontend (Metabase đảm nhiệm)
