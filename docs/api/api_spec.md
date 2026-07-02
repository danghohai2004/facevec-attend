# API Specification

Base URL: `http://localhost:8000/api`  
Interactive docs: `http://localhost:8000/docs`

---

## Authentication (write endpoints)

Read endpoints (GET) and the WebSocket ingress are **public**. Write endpoints are **protected** and require the header `X-API-Key: <API_KEY>`:

| Method + Path | Protected |
|---|---|
| `POST /api/employees` | ✓ |
| `DELETE /api/employees` | ✓ |
| `POST /api/attendance/checkin` | ✓ |
| `POST /api/attendance/checkout` | ✓ |
| `PUT /api/shift-settings` | ✓ |
| all GET endpoints, `WS /ws/recognition/{client_id}` | public |

Auth is **fail-closed** (`src/platform/auth.py`):

- Server has no `API_KEY` configured → **503** `{"detail": "Server auth not configured"}`
- Missing or wrong `X-API-Key` → **401** `{"detail": "Invalid or missing API key"}`
- Correct key → request proceeds

**Frontend never sends the key from the browser.** The Next.js dashboard calls a same-origin **BFF proxy** at `POST/PUT/DELETE /api/write/[...path]`, which injects `X-API-Key` from a server-only env var and forwards to the backend. The proxy also rejects cross-origin writes (403) as a CSRF guard, returns 503 if unconfigured, and 502 if the backend is unreachable. See [`../architecture/overview.md`](../architecture/overview.md#security-model-write-protection).

> **Internal errors are not leaked.** On an unexpected server/DB error, write and read endpoints return a generic `500 {"detail": "Lỗi hệ thống"}`; the real cause is logged server-side only.

---

## Employees

### POST `/api/employees` — Đăng ký nhân viên 🔒

Đăng ký nhân viên mới và lưu embedding khuôn mặt từ **1 ảnh** (đúng 1 khuôn mặt).

**Request:** `multipart/form-data`

| Field | Type | Required | Mô tả |
|---|---|---|---|
| `name` | string | ✓ | Tên nhân viên |
| `emp_code` | string | ✓ | Mã nhân viên (unique) |
| `file` | file | ✓ | Ảnh rõ khuôn mặt (JPEG/PNG), tối đa 5MB, đúng 1 mặt |

**Response 200:**
```json
{
  "message": "Registered Nguyen Van A (NV001)",
  "employee": { "emp_id": 1, "name": "Nguyen Van A", "emp_code": "NV001" }
}
```

**Errors:**
- `400` — Không đọc được ảnh, hoặc ảnh không có **đúng 1** khuôn mặt
- `409` — `emp_code` đã tồn tại (`{"detail": "Employee code already exists."}`)
- `413` — Ảnh quá lớn (> 5MB)
- `401` / `503` — Auth (xem trên)

---

### GET `/api/employees` — Danh sách nhân viên

**Query params:**

| Param | Type | Default | Mô tả |
|---|---|---|---|
| `page` | int | 1 | Trang hiện tại (≥1) |
| `page_size` | int | 20 | Số lượng/trang (1–100) |

**Response 200:**
```json
{
  "items": [ { "emp_id": 1, "name": "Nguyen Van A", "emp_code": "NV001" } ],
  "page": 1, "page_size": 20, "total": 1
}
```

---

### GET `/api/employees/{identifier}` — Tìm nhân viên

- Nếu `identifier` là số → tìm theo `emp_id` (trả 1 nhân viên).
- Nếu là chuỗi → tìm theo tên (case-insensitive, partial) → trả danh sách.

**Response 200 (theo ID):**
```json
{ "employee": { "emp_id": 1, "name": "Nguyen Van A", "emp_code": "NV001" } }
```

**Response 200 (theo tên):** cùng schema với `GET /api/employees` (`items`, `page`, `page_size`, `total`).

**Errors:** `404` — Không tìm thấy (chỉ khi tìm theo ID).

---

### DELETE `/api/employees` — Xóa nhân viên 🔒

Xóa nhân viên khỏi PostgreSQL (cascade xóa `attendance_logs`) và xóa embeddings khỏi Qdrant.

**Query params** (cần ít nhất 1): `emp_id` (int) hoặc `emp_code` (string).

**Response 200:**
```json
{ "message": "Removed 1", "emp_id": 1, "emp_code": "NV001" }
```

**Errors:** `400` — thiếu cả `emp_id` và `emp_code`; `404` — không tìm thấy; `401`/`503` — auth.

---

## Attendance

> **Chấm công thủ công tuân theo khung giờ ca.** `emp_id` truyền qua **query param** (không có request body). Khung giờ lấy từ DB `shift_settings` — server **không** tin client. Timestamp theo timezone **Asia/Ho_Chi_Minh**.

### POST `/api/attendance/checkin` — Check in 🔒

**Query params:** `emp_id` (int, required).

**Response 200:**
```json
{
  "message": "Check in successful",
  "check_type": "check_in",
  "log": {
    "log_id": 42, "emp_id": 1, "working_date": "2026-07-02",
    "checkin_time": "2026-07-02T08:05:00", "checkout_time": null, "working_duration": null
  }
}
```

**Errors:**
- `400` — Ngoài khung giờ check-in (`"Ngoài khung giờ check-in."`) hoặc đã check-in rồi (`"Already checked in"`)
- `404` — Không tìm thấy nhân viên
- `401` / `503` — Auth

---

### POST `/api/attendance/checkout` — Check out 🔒

**Query params:** `emp_id` (int, required).

**Response 200:**
```json
{
  "message": "Check out successful",
  "check_type": "check_out",
  "log": {
    "log_id": 42, "emp_id": 1, "working_date": "2026-07-02",
    "checkin_time": "2026-07-02T08:05:00", "checkout_time": "2026-07-02T17:30:00",
    "working_duration": "PT9H25M"
  }
}
```

**Errors:**
- `400` — Ngoài khung giờ check-out (`"Ngoài khung giờ check-out."`) hoặc chưa check-in (`"Check in not found to check out"`)
- `404` — Không tìm thấy nhân viên
- `401` / `503` — Auth

---

### GET `/api/attendance` — Lịch sử chấm công

**Query params:**

| Param | Type | Required | Mô tả |
|---|---|---|---|
| `emp_id` | int | ✓ | ID nhân viên |
| `from_date` | date | | Từ ngày (YYYY-MM-DD) |
| `to_date` | date | | Đến ngày (YYYY-MM-DD) |
| `page` | int | | Trang (default 1) |
| `page_size` | int | | Số lượng/trang (default 20, max 100) |

**Response 200:**
```json
{
  "items": [
    {
      "log_id": 42, "emp_id": 1, "working_date": "2026-07-02",
      "checkin_time": "2026-07-02T08:05:00", "checkout_time": "2026-07-02T17:30:00",
      "working_duration": "PT9H25M"
    }
  ],
  "page": 1, "page_size": 20, "total": 22
}
```

**Errors:** `400` — `from_date` > `to_date`; `404` — không tìm thấy nhân viên.

---

## Shift Settings

### GET `/api/shift-settings` — Lấy ca làm việc

Trả về ca hiện tại; nếu chưa cấu hình, trả về mặc định (`08:00–10:00` / `17:00–19:00`).

**Response 200:**
```json
{
  "check_in_start": "08:00:00", "check_in_end": "10:00:00",
  "check_out_start": "17:00:00", "check_out_end": "19:00:00"
}
```

---

### PUT `/api/shift-settings` — Cập nhật ca làm việc 🔒

**Request body:**
```json
{
  "check_in_start": "07:30:00", "check_in_end": "09:30:00",
  "check_out_start": "16:30:00", "check_out_end": "18:30:00"
}
```

**Response 200:** settings đã lưu (cùng schema). **Errors:** `401`/`503` — auth.

---

## WebSocket — Nhận diện khuôn mặt realtime

### WS `/ws/recognition/{client_id}` (public)

Client kết nối WebSocket, gửi frame liên tục, nhận kết quả. `client_id` là định danh duy nhất của từng client — server dùng nó để route kết quả về đúng nơi (hỗ trợ nhiều camera đồng thời).

> **Lưu ý:** endpoint này đã sẵn sàng ở backend. Client camera trong browser (mở webcam, gửi frame) **chưa được implement** ở frontend — xem [`../security/follow-ups.md`](../security/follow-ups.md) (Phase 3).

**Gửi (binary):** JPEG bytes của frame camera.

**Nhận (JSON):**

```json
// Nhận diện thành công (kèm kết quả chấm công)
{ "status": "recognized", "emp_id": 1, "name": "Nguyen Van A",
  "attendance": "Check in successful", "timestamp": "2026-07-02T08:05:00+00:00" }

// Không nhận ra
{ "status": "unknown", "timestamp": "..." }

// Phát hiện giả mạo
{ "status": "spoof", "timestamp": "..." }

// Không có khuôn mặt trong frame
{ "status": "no_face", "timestamp": "..." }

// Lỗi xử lý (chi tiết đã được ẩn)
{ "status": "error", "detail": "Lỗi hệ thống" }
```

**Flow:**
1. Client mở WebSocket tới `/ws/recognition/{client_id}`.
2. Gửi frame JPEG định kỳ (khuyến nghị ~1 frame/giây).
3. Server đẩy frame vào `FrameQueue` → pipeline xử lý (liveness → extract → identify → log).
4. Kết quả push về ngay client qua cùng connection.

---

## Error format chung

```json
{ "detail": "Mô tả lỗi" }
```

| HTTP | Ý nghĩa |
|---|---|
| 400 | Bad request (thiếu field, ngoài giờ làm, đã check-in, khoảng ngày sai, ảnh sai...) |
| 401 | Thiếu/sai `X-API-Key` (write endpoints) |
| 403 | BFF proxy chặn write cross-origin (CSRF guard) |
| 404 | Không tìm thấy resource |
| 409 | Trùng `emp_code` khi đăng ký |
| 413 | Ảnh upload quá lớn (> 5MB) |
| 500 | Lỗi server/DB — luôn trả `"Lỗi hệ thống"` (chi tiết log server-side) |
| 502 | BFF proxy không gọi được backend |
| 503 | Server chưa cấu hình `API_KEY` (backend) hoặc proxy chưa cấu hình |
