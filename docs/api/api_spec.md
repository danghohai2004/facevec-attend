# API Specification

Base URL: `http://localhost:8000/api`  
Interactive docs: `http://localhost:8000/docs`

---

## Employees

### POST `/api/employees` — Đăng ký nhân viên

Đăng ký nhân viên mới và lưu embedding khuôn mặt từ ảnh.

**Request:** `multipart/form-data`

| Field | Type | Required | Mô tả |
|---|---|---|---|
| `name` | string | ✓ | Tên nhân viên |
| `emp_code` | string | ✓ | Mã nhân viên (unique) |
| `file` | file | ✓ | Ảnh chụp rõ khuôn mặt (JPEG/PNG) |

**Response 200:**
```json
{
  "message": "Successfully registered employee Nguyen Van A (NV001)",
  "employee": {
    "emp_id": 1,
    "name": "Nguyen Van A",
    "emp_code": "NV001"
  }
}
```

**Errors:**
- `400` — Không đọc được ảnh hoặc không tìm thấy khuôn mặt
- `500` — Lỗi database

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
  "items": [
    { "emp_id": 1, "name": "Nguyen Van A", "emp_code": "NV001" }
  ],
  "page": 1,
  "page_size": 20,
  "total": 1
}
```

---

### GET `/api/employees/{name}` — Tìm nhân viên

- Nếu `name` là số nguyên → tìm theo `emp_id`
- Nếu là chuỗi → tìm theo tên (case-insensitive, partial match)

**Response 200 (tìm theo ID):**
```json
{
  "employee": { "emp_id": 1, "name": "Nguyen Van A", "emp_code": "NV001" }
}
```

**Response 200 (tìm theo tên):**
```json
{
  "items": [...],
  "page": 1,
  "page_size": 2,
  "total": 2
}
```

**Errors:**
- `404` — Không tìm thấy nhân viên (chỉ khi tìm theo ID)

---

### DELETE `/api/employees` — Xóa nhân viên

Xóa nhân viên khỏi PostgreSQL (cascade xóa `attendance_logs`) và đồng thời xóa embeddings khỏi Qdrant collection.

**Query params** (cần ít nhất 1):

| Param | Type | Mô tả |
|---|---|---|
| `emp_id` | int | ID nhân viên |
| `emp_code` | string | Mã nhân viên |

**Response 200:**
```json
{
  "message": "Successfully removed employee 1",
  "emp_id": 1,
  "emp_code": "NV001"
}
```

**Errors:**
- `400` — Thiếu cả `emp_id` và `emp_code`
- `404` — Không tìm thấy nhân viên

---

## Attendance

### POST `/api/attendance/checkin` — Check in thủ công

**Request body:**
```json
{
  "emp_id": 1,
  "shifts_time": {
    "check_in_start": "08:00:00",
    "check_in_end": "10:00:00",
    "check_out_start": "17:00:00",
    "check_out_end": "19:00:00"
  }
}
```

> ⚠️ **Note:** Endpoint này dùng `shifts_time` từ client — sẽ được fix trong restructure để load từ DB thay vì tin client.

**Response 200:**
```json
{
  "message": "Check in successful",
  "check_type": "check_in",
  "log": {
    "log_id": 42,
    "emp_id": 1,
    "working_date": "2026-06-30",
    "checkin_time": "2026-06-30T08:05:00",
    "checkout_time": null,
    "working_duration": null
  }
}
```

**Errors:**
- `400` — Ngoài giờ check-in hoặc đã check-in rồi
- `404` — Không tìm thấy nhân viên

---

### POST `/api/attendance/checkout` — Check out thủ công

Request body giống `/checkin`.

**Response 200:**
```json
{
  "message": "Check out successful",
  "check_type": "check_out",
  "log": {
    "log_id": 42,
    "emp_id": 1,
    "working_date": "2026-06-30",
    "checkin_time": "2026-06-30T08:05:00",
    "checkout_time": "2026-06-30T17:30:00",
    "working_duration": "PT9H25M"
  }
}
```

**Errors:**
- `400` — Ngoài giờ check-out hoặc chưa check-in
- `404` — Không tìm thấy nhân viên

---

### GET `/api/attendance` — Lịch sử chấm công

**Query params:**

| Param | Type | Required | Mô tả |
|---|---|---|---|
| `emp_id` | int | ✓ | ID nhân viên |
| `from_date` | date | | Từ ngày (YYYY-MM-DD) |
| `to_date` | date | | Đến ngày (YYYY-MM-DD) |
| `page` | int | | Trang (default: 1) |
| `page_size` | int | | Số lượng/trang (default: 20, max: 100) |

**Response 200:**
```json
{
  "items": [
    {
      "log_id": 42,
      "emp_id": 1,
      "working_date": "2026-06-30",
      "checkin_time": "2026-06-30T08:05:00",
      "checkout_time": "2026-06-30T17:30:00",
      "working_duration": "PT9H25M"
    }
  ],
  "page": 1,
  "page_size": 20,
  "total": 22
}
```

---

## Shift Settings

### GET `/api/shift-settings` — Lấy ca làm việc

**Response 200:**
```json
{
  "check_in_start": "08:00:00",
  "check_in_end": "10:00:00",
  "check_out_start": "17:00:00",
  "check_out_end": "19:00:00"
}
```

---

### PUT `/api/shift-settings` — Cập nhật ca làm việc

**Request body:**
```json
{
  "check_in_start": "07:30:00",
  "check_in_end": "09:30:00",
  "check_out_start": "16:30:00",
  "check_out_end": "18:30:00"
}
```

**Response 200:** trả về settings đã lưu (cùng schema).

---

## WebSocket — Nhận diện khuôn mặt realtime

### WS `/ws/recognition/{client_id}`

Kiosk kết nối WebSocket, gửi frame liên tục, nhận kết quả nhận diện. `client_id` là định danh duy nhất của từng kiosk — server dùng nó để route kết quả về đúng màn hình (hỗ trợ 5+ camera đồng thời).

**Gửi (binary):**

```
JPEG bytes của frame camera
```

**Nhận (JSON):**

```json
// Nhận diện thành công
{
  "status": "recognized",
  "emp_id": 1,
  "name": "Nguyen Van A",
  "attendance": "Check in successful",
  "timestamp": "2026-06-30T08:05:00"
}

// Không nhận ra
{
  "status": "unknown",
  "timestamp": "2026-06-30T08:05:00"
}

// Phát hiện giả mạo
{
  "status": "spoof",
  "timestamp": "2026-06-30T08:05:00"
}

// Không có khuôn mặt trong frame
{
  "status": "no_face",
  "timestamp": "2026-06-30T08:05:00"
}
```

**Flow:**
1. Kiosk mở WebSocket tới `/ws/recognition/{client_id}`
2. Gửi frame JPEG định kỳ (khuyến nghị: 1 frame/giây)
3. Server đẩy frame vào FrameQueue → pipeline xử lý
4. Kết quả push về ngay client qua cùng connection

---

## Error format chung

```json
{
  "detail": "Mô tả lỗi"
}
```

| HTTP Code | Ý nghĩa |
|---|---|
| 400 | Bad request (thiếu field, ngoài giờ làm, đã check-in...) |
| 404 | Không tìm thấy resource |
| 500 | Lỗi server / database |
