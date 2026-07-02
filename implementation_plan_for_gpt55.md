# Implementation Plan — facevec-attend

> Nguồn: đã second-pass review report `review_sourcecode_gpt5.5medium.md`, verify trực tiếp trên code.
> Nguyên tắc: **fix nhỏ, đúng root cause, không over-engineer.** Không làm outbox/saga, không RBAC đầy đủ, không Alembic ở giai đoạn này.
> Mỗi task độc lập, có thể commit riêng. Làm theo thứ tự Phase.

## Quy ước
- Backend: `uv run pytest` phải xanh sau mỗi task backend.
- Frontend: `npm run lint` + `npx tsc --noEmit` phải xanh sau mỗi task frontend.
- Không sửa những thứ đã đúng: overnight-shift logic, partial unique index, CORS config, `.env` (đã gitignore, không leak).

---

## PHASE 1 — Quick wins (rủi ro thấp, giá trị cao). Làm trước.

### Task 1.1 — `make db-down` không được xoá data
- **File:** `Makefile`
- **Vấn đề:** `db-down` chạy `docker compose down -v` → xoá volume `pgdata` + `qdrant_data` khi chỉ định "stop".
- **Sửa:**
  - `db-down` → `$(DOCKER_COMPOSE) down` (bỏ `-v`).
  - Thêm target `db-reset` chạy `$(DOCKER_COMPOSE) down -v` có xác nhận: `@read -p "Xoá toàn bộ data? [y/N] " c && [ "$$c" = "y" ]`.
  - Thêm `db-reset` vào `.PHONY` và block `help`.
- **Verify:** `make -n db-down` không có `-v`; `make -n db-reset` có `-v`.
- **Acceptance:** Stop service không mất data.
- **Rollback risk:** Không.

### Task 1.2 — Sửa hợp đồng API attendance ở frontend (đang 422)
- **File:** `frontend/src/lib/api.ts`
- **Vấn đề:** `checkIn`/`checkOut` POST body JSON `{emp_id, shifts_time}`, backend yêu cầu `emp_id: int = Query(...)`, không nhận body → **422 mọi lần**. Backend cũng bỏ qua `shifts_time` (load shift từ DB).
- **Sửa:**
  ```ts
  export async function checkIn(payload: { empId: string }) {
    const empId = Number(payload.empId);
    if (Number.isNaN(empId)) throw new Error("Employee ID phải là số.");
    const response = await api.post("/attendance/checkin", null, { params: { emp_id: empId } });
    return response.data;
  }
  ```
  Tương tự `checkOut`. Bỏ tham số `shiftSettings` khỏi 2 hàm này.
- **Cập nhật caller:** `frontend/src/components/attendance/attendance-client.tsx` — `mutate({ empId: targetEmpId })` (bỏ `shiftSettings`).
- **Verify:** backend up, submit emp_id số hợp lệ → 200; `npm run lint` + `tsc`.
- **Acceptance:** Check-in thật tạo được row `attendance_logs`.
- **Note:** `manualEmpId` là free-text (placeholder "EMP-1024") — validate số ở trên đã chặn; giữ thông báo lỗi rõ ràng.
- **Rollback risk:** Thấp.

### Task 1.3 — Bỏ/khoá detection giả + dọn lãng phí camera
- **File:** `frontend/src/components/attendance/attendance-client.tsx`, `frontend/src/components/employees/employee-registration.tsx`
- **Vấn đề:** `attendance-client` chọn random `demoEmployees` mỗi 2s làm "detection"; tạo `snapshot` JPEG không dùng. Registration tạo object URL không revoke.
- **Sửa:**
  - Xoá block random-detection (dòng `demoEmployees[Math.floor(...)]` + `setDetection`). Nếu muốn giữ demo: bọc sau `process.env.NEXT_PUBLIC_DEMO === "1"`.
  - Xoá state `snapshot` và phần `getScreenshot()` không dùng.
  - Registration: `URL.revokeObjectURL(url)` trong cleanup của `useEffect`.
- **Verify:** `npm run lint` + `tsc`; không còn danh tính giả hiện lên.
- **Acceptance:** Không có employee bịa; không leak object URL.
- **Rollback risk:** Thấp.

### Task 1.4 — Router factory tự tạo router riêng
- **File:** `src/modules/recognition/ws_ingress.py`
- **Vấn đề:** `router` là module-global; `make_ws_router` decorate lại nó mỗi lần gọi → route trùng nếu `create_app()` chạy 2 lần (test/reload).
- **Sửa:** Chuyển `router = APIRouter(tags=["Recognition"])` vào **trong** `make_ws_router`, rồi `return router`.
- **Verify:** `pytest`; gọi `create_app()` 2 lần → mỗi app 1 WS route.
- **Acceptance:** Số route ổn định qua 2 lần `create_app()`.
- **Rollback risk:** Không.

### Task 1.5 — Giới hạn upload enrollment: đúng 1 mặt + cap size
- **File:** `src/modules/employees/api.py`
- **Vấn đề:** `file.read()` không giới hạn; `extract_embeddings_from_bytes` trả **tất cả** mặt → mọi mặt trong ảnh bị enroll chung 1 nhân viên (poisoning + DoS).
- **Sửa:** Trong `api_register`, sau `contents = await file.read()`:
  ```python
  MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # ponytail: chặn ảnh khổng lồ/decompression bomb
  if len(contents) > MAX_UPLOAD_BYTES:
      raise HTTPException(413, "Ảnh quá lớn (tối đa 5MB).")
  ```
  Sau khi có `embeddings`:
  ```python
  if len(embeddings) != 1:
      raise HTTPException(400, "Ảnh phải có đúng 1 khuôn mặt.")
  ```
- **Verify:** test: ảnh 2 mặt → 400; ảnh >5MB → 413; 1 mặt → 200.
- **Acceptance:** Chỉ ảnh 1 mặt, đúng size mới enroll.
- **Rollback risk:** Thấp (cố ý reject ảnh nhiều mặt).

### Task 1.6 — Dọn dead code / deps / docs
- **File:** `src/platform/db/session.py`, `src/platform/config.py`, `pyproject.toml`, `frontend/src/lib/{api.ts,format.ts,mock-data.ts}`, `frontend/src/components/ui/textarea.tsx`, `src/modules/recognition/identifier.py`, `tests/modules/recognition/test_identifier.py`, `docs/setup/getting-started.md`
- **Sửa (đã grep xác nhận 0 caller production):**
  - Xoá `get_connection()` + import `psycopg2`; gỡ dep `psycopg2-binary` khỏi `pyproject.toml`.
  - Xoá `MAX_EMB_FACE`, `ORIGINAL_IMG_PATH` trong `config.py`.
  - Xoá `listAttendanceHistory` (`api.ts`), `formatPercent` (`format.ts`), `recentAttendance` (`mock-data.ts`), `textarea.tsx`.
  - Gỡ dep `plotly` (không import), `next-themes` (đang dùng `theme-provider.tsx` custom).
  - Xoá alias `identify = identify_face` (`identifier.py:28`); đổi test sang `identify_face`.
  - `getting-started.md:8`: "Node.js 18+" → "Node.js 20+" (Next 16.2.6 cần >=20).
- **Verify:** `pytest`, `uv sync`, `npm run lint`/`tsc`/`npm run build`.
- **Acceptance:** Build + test xanh sau khi xoá.
- **Rollback risk:** Thấp.

---

## PHASE 2 — Correctness & maintainability

## Phase 2, Task 2.1 — Secure write endpoint protection

> **REVISED 2026-07-01.** Bản cũ (`if x_api_key != os.getenv("API_KEY")` + `NEXT_PUBLIC_API_KEY`) đã bị **bác bỏ**: nó (a) **fail-open** khi `API_KEY` chưa set và request không có header (`None != None` → `False` → cho qua), và (b) lộ secret trong bundle browser nếu dùng `NEXT_PUBLIC_*`. Không dùng lại 2 pattern đó.

### Decision
- **Chosen option: C — Server-side frontend proxy (BFF).**
- **Reason:** Frontend chạy như **Node server** (`next.config.ts` không có `output: 'export'`; script `next start`), nên route handler server-side là khả thi **không** đổi mô hình deploy. Browser hiện đang thực hiện write thật (đăng ký/xoá nhân viên, sửa shift, checkin/checkout thủ công), nên **Option A** sẽ làm hỏng các nút đó cho tới khi có proxy/auth. **Option B** (`NEXT_PUBLIC_API_KEY`) không phải authentication — bác bỏ. **Option D** (hoãn) không cần thiết vì C đủ nhỏ (5 write endpoints, Next đã là server). C giữ secret **chỉ ở server**, backend **fail-closed**, browser không bao giờ giữ key.

### Threat model (trung thực)
- **Bảo vệ được:** backend write endpoints khỏi client mạng tuỳ ý (vd `curl` thẳng vào `:8000`). API secret **không bao giờ** xuất hiện trong browser bundle/network. Backend chỉ tin request đến từ Next server (nơi giữ key).
- **KHÔNG bảo vệ:** người dùng — bất kỳ ai mở được frontend Next đều có thể kích hoạt write (proxy sẽ inject key). Đây **không phải** user authentication. Muốn chặn theo người dùng cần **login/session/RBAC** → task riêng ở Phase 3, không nằm trong 2.1.
- **Cũng không bảo vệ:** nếu host chạy Next server bị chiếm, hoặc `API_KEY` bị lộ.

### Backend scope
- **Protected write endpoints** (bắt buộc `X-API-Key`, fail-closed):
  - `POST /api/employees`
  - `DELETE /api/employees`
  - `PUT /api/shift-settings`
  - `POST /api/attendance/checkin`
  - `POST /api/attendance/checkout`
- **Public endpoints** (KHÔNG đụng, không yêu cầu key):
  - GET: `GET /api/employees`, `GET /api/employees/{identifier}`, `GET /api/attendance`, `GET /api/shift-settings`
  - WebSocket: `/ws/recognition/{client_id}`
- **Behavior (không được fail-open):**
  - **`API_KEY` unset/empty** → **HTTP 503** `"Server auth not configured"` trên **mọi protected write** (fail-closed; tuyệt đối không coi unset là "cho qua"). GET/WS không ảnh hưởng.
  - **Missing `X-API-Key`** → **HTTP 401** `"Invalid or missing API key"`.
  - **Wrong `X-API-Key`** → **HTTP 401** (cùng message, không tiết lộ đúng/sai để tránh oracle).
  - **Correct `X-API-Key`** → cho qua, xử lý bình thường.
- **So sánh key: constant-time.** Dùng `hmac.compare_digest`. Dependency mẫu:
  ```python
  # src/platform/auth.py
  import hmac
  import os
  from fastapi import Header, HTTPException

  def require_api_key(x_api_key: str | None = Header(default=None)):
      api_key = os.getenv("API_KEY")
      if not api_key:                       # unset/empty → fail CLOSED
          raise HTTPException(503, "Server auth not configured")
      if not x_api_key or not hmac.compare_digest(x_api_key, api_key):
          raise HTTPException(401, "Invalid or missing API key")
  ```
  Gắn `dependencies=[Depends(require_api_key)]` vào đúng 5 route trên (thêm ở decorator route hoặc `APIRouter(dependencies=...)` con — **không** gắn lên router chứa GET).

### Frontend scope
- **Browser behavior:** browser **KHÔNG** gửi bất kỳ API key nào. Các hàm write trong `api.ts` gọi **same-origin** Next route handler (đường dẫn tương đối, không qua `NEXT_PUBLIC_API_BASE_URL`).
- **Server-side proxy behavior (bắt buộc):** route handler chạy **chỉ server-side** (App Router `route.ts` — mặc định server). Nó:
  1. Đọc `API_KEY` (server-only) + `BACKEND_INTERNAL_URL` (server-only).
  2. Forward request tới backend endpoint tương ứng, **thêm header `X-API-Key`**, giữ nguyên method + body (kể cả multipart cho đăng ký ảnh — dùng `await request.arrayBuffer()`/stream và forward `content-type`).
  3. Trả nguyên status + body của backend về browser.
  - **Allowlist bắt buộc:** proxy chỉ được forward tới đúng 5 (method, path) trên. **Không** forward path tuỳ ý (chống SSRF/path-injection). Có thể làm 5 route handler riêng, hoặc 1 handler với allowlist tường minh — **allowlist là bắt buộc dù chọn cách nào**.
  - GET **không** proxy: giữ browser gọi thẳng backend qua `NEXT_PUBLIC_API_BASE_URL` như hiện tại.
- **Environment variables:**
  - **Server-only (KHÔNG có tiền tố `NEXT_PUBLIC_`):** `API_KEY`, `BACKEND_INTERNAL_URL` (vd `http://localhost:8000`). Chỉ đọc trong `route.ts`.
  - **Public (đã có, giữ nguyên):** `NEXT_PUBLIC_API_BASE_URL` cho GET từ browser.
  - **TUYỆT ĐỐI KHÔNG** tạo `NEXT_PUBLIC_API_KEY`.

### Files allowed to change
Backend:
- Mới: `src/platform/auth.py`
- Sửa (chỉ thêm dependency vào 5 route write): `src/modules/employees/api.py`, `src/modules/attendance/api.py`
- Mới/sửa test: `tests/platform/test_auth.py`; cập nhật `tests/modules/employees/test_api.py` (3 test Phase 1 giờ cần key — xem dưới)
- `.env.example` (thêm `API_KEY=`)

Frontend:
- Mới: route handler(s) dưới `frontend/src/app/api/write/...` (proxy write)
- Sửa: `frontend/src/lib/api.ts` (trỏ `createEmployee`, `deleteEmployee`, `updateShiftSettings`, `checkIn`, `checkOut` sang same-origin proxy)
- Mới: `frontend/.env.example` (thêm `API_KEY=`, `BACKEND_INTERNAL_URL=`, ghi rõ server-only)

### Files not allowed to change
- Bất kỳ endpoint GET nào và logic của chúng; `src/modules/recognition/ws_ingress.py`; pipeline; models; `compose.yaml` (network là Task 2.8).
- Toàn bộ file Phase 1 đã accept (trừ việc cập nhật `tests/modules/employees/test_api.py` để gửi header).
- Không refactor các hàm GET trong `api.ts`, không đụng `normalizeEmployee`/`normalizeShiftSettings`.
- Không thêm `NEXT_PUBLIC_API_KEY` ở bất cứ đâu.

### Tests to add/update
Backend (`uv run pytest`):
- `test_auth.py`: `require_api_key` — `API_KEY` unset → 503; missing header → 401; wrong header → 401; correct header → pass. (Test qua một app nhỏ gắn dependency, hoặc gọi trực tiếp dependency bằng cách bắt `HTTPException`.)
- Cập nhật `tests/modules/employees/test_api.py`: 3 test hiện tại POST `/api/employees` → giờ phải `monkeypatch.setenv("API_KEY", "test-key")` và gửi `headers={"X-API-Key": "test-key"}`, **hoặc** override dependency `app.dependency_overrides[require_api_key] = lambda: None`. Không được để test đi vòng qua auth một cách vô tình.
- Thêm 1 test: protected write **không** header → 401 (regression guard cho fail-open).

Frontend:
- `npm run lint`, `npx tsc --noEmit`, `npm run build` phải xanh.
- Route handler chỉ server-side: xác nhận **không** import từ client component, **không** dùng `"use client"`, và `API_KEY` không lọt vào bundle (grep bundle nếu cần).

### Verification commands
```bash
# Backend
uv run pytest

# Backend manual (chạy backend với/không API_KEY)
# unset API_KEY → protected write trả 503:
curl -s -o /dev/null -w "%{http_code}\n" -X POST "http://localhost:8000/api/attendance/checkin?emp_id=1"   # 503
# set API_KEY=secret rồi:
curl -s -o /dev/null -w "%{http_code}\n" -X POST "http://localhost:8000/api/attendance/checkin?emp_id=1"   # 401 (no header)
curl -s -o /dev/null -w "%{http_code}\n" -H "X-API-Key: wrong" -X POST ".../checkin?emp_id=1"              # 401
curl -s -o /dev/null -w "%{http_code}\n" -H "X-API-Key: secret" -X POST ".../checkin?emp_id=1"            # xử lý (200/400/404 tuỳ nghiệp vụ, KHÔNG 401/503)
# GET vẫn public:
curl -s -o /dev/null -w "%{http_code}\n" "http://localhost:8000/api/employees"                            # KHÔNG 401/503

# Frontend
cd frontend && npm run lint && npx tsc --noEmit && npm run build
# Xác nhận không có NEXT_PUBLIC_API_KEY:
grep -rn "NEXT_PUBLIC_API_KEY" frontend/src   # phải rỗng
```

### Non-goals
- **Không** làm full login/session/RBAC (đó là task Phase 3 nếu cần chặn theo người dùng).
- **Không** đưa `API_KEY` vào browser bundle; **không** `NEXT_PUBLIC_API_KEY`.
- **Không** bảo vệ GET/WebSocket (giữ public theo thiết kế kiosk).
- **Không** proxy GET; **không** refactor client/api code không liên quan.
- **Không** để backend fail-open trong bất kỳ nhánh nào.

### GPT 5.5 implementation instructions
> Implement Task 2.1 theo **Option C**. Bước:
> 1. Tạo `src/platform/auth.py` với `require_api_key` **fail-closed** đúng như snippet (unset→503, missing/wrong→401, constant-time `hmac.compare_digest`).
> 2. Gắn `Depends(require_api_key)` vào đúng 5 write route (`POST/DELETE /api/employees`, `PUT /api/shift-settings`, `POST /api/attendance/checkin|checkout`). Không đụng GET/WS.
> 3. Thêm `API_KEY=` vào `.env.example` (backend).
> 4. Tạo Next route handler(s) server-side dưới `frontend/src/app/api/write/...` với **allowlist** đúng 5 (method,path); đọc `API_KEY` + `BACKEND_INTERNAL_URL` (server-only), forward kèm `X-API-Key`, giữ nguyên body/multipart, trả nguyên status/body.
> 5. Sửa `frontend/src/lib/api.ts`: `createEmployee`, `deleteEmployee`, `updateShiftSettings`, `checkIn`, `checkOut` gọi same-origin proxy (path tương đối). GET giữ nguyên.
> 6. Tạo `frontend/.env.example` với `API_KEY=` và `BACKEND_INTERNAL_URL=` + chú thích **server-only, không bao giờ NEXT_PUBLIC**.
> 7. Cập nhật `tests/modules/employees/test_api.py` để gửi header/override, thêm `tests/platform/test_auth.py` (4 nhánh) + 1 regression test no-header→401.
> 8. Chạy toàn bộ Verification commands; tất cả phải khớp mã trạng thái ghi trên.
> Ràng buộc cứng: backend không fail-open; secret không rời server; chỉ chạm các file trong "Files allowed to change".

- **Rollback risk:** Trung bình — thêm proxy layer + đổi call-site write ở frontend. GET/WS/pipeline không đổi. Nếu proxy lỗi, chỉ ảnh hưởng các nút write, revert được độc lập.

### Task 2.2 — Chặn boot production khi liveness là PassThrough
- **File:** `src/app.py` (hoặc `src/modules/antispoofing/service.py`)
- **Vấn đề:** `PassThroughChecker` luôn `True` (placeholder cố ý). Rủi ro nếu deploy prod.
- **Sửa:** Trong lifespan startup: nếu `os.getenv("ENV") == "production"` và checker là `PassThroughChecker` → raise RuntimeError chặn boot.
- **Verify:** set `ENV=production` → app từ chối start.
- **Acceptance:** Prod không boot với liveness giả.
- **Rollback risk:** Thấp (chỉ thêm guard).

### Task 2.3 — `POST /employees` trả 409 khi trùng emp_code
- **File:** `src/modules/employees/service.py`, `src/modules/employees/api.py`
- **Vấn đề:** Đang upsert theo `emp_code` → ghi đè danh tính (thay tên + toàn bộ vector).
- **Sửa:** Thêm sentinel `ERR_DUPLICATE`; trong `register_employee`, nếu tìm thấy employee theo `emp_code` → return `(None, ERR_DUPLICATE)`. Trong `api_register` map `ERR_DUPLICATE` → HTTP 409.
- **Verify:** test: đăng ký lại `emp_code` đã tồn tại → 409.
- **Acceptance:** Không ghi đè danh tính qua create. (Re-enroll để riêng, sau này.)
- **Rollback risk:** Thấp (frontend không dựa vào upsert).

### Task 2.4 — Quyết định hợp đồng multi-file enrollment
- **File:** `frontend/src/lib/api.ts` (khuyến nghị) HOẶC `src/modules/employees/api.py`
- **Vấn đề:** Frontend append nhiều `file`, backend nhận 1 `UploadFile` → các frame dư bị bỏ âm thầm.
- **Sửa (khuyến nghị, khớp Task 1.5):** frontend chỉ gửi **1 frame** đã chọn. `createEmployee` append đúng 1 `file`; UI reg chọn 1 ảnh đại diện.
- **Verify:** `npm run lint`/`tsc`; enroll 1 ảnh → 200.
- **Acceptance:** UI không báo enroll nhiều frame trong khi backend chỉ nhận 1.
- **Rollback risk:** Thấp.

### Task 2.5 — Giới hạn concurrency pipeline recognition
- **File:** `src/modules/recognition/pipeline.py`
- **Vấn đề:** `run_pipeline` `create_task` mỗi item không giới hạn → task/frame bytes tích luỹ vô hạn dưới tải cao (queue đã bounded 50 + drop-oldest, nhưng pending tasks thì không).
- **Sửa:** Thêm `_sem = asyncio.Semaphore(4)`; acquire trước khi tạo task (hoặc bọc thân `_process`), release trong `finally`.
- **Verify:** flood frame → số pending task không tăng vô hạn.
- **Acceptance:** Task count có trần.
- **Rollback risk:** Thấp. (Per-camera fairness: defer.)

### Task 2.6 — Ngừng leak chuỗi exception ra client
- **File:** `src/modules/employees/service.py`, `src/modules/attendance/service.py`, `src/modules/recognition/pipeline.py`
- **Vấn đề:** Trả `f"[ERROR ...]: {e}"` → lộ chi tiết DB/Qdrant qua HTTP/WS.
- **Sửa:** `logging.exception(...)` phía server; trả message generic ổn định (vd "Lỗi hệ thống"). Giữ sentinel nghiệp vụ (`ERR_NOT_FOUND`, ...) như cũ.
- **Verify:** gây lỗi DB → response không chứa chi tiết internal; log server có stacktrace.
- **Acceptance:** Không lộ chi tiết nội bộ.
- **Rollback risk:** Thấp.

### Task 2.7 — Lifecycle cleanup + timezone
- **File:** `src/app.py`, `src/modules/attendance/service.py`
- **Vấn đề:** Shutdown `cancel()` không `await`, không đóng Qdrant/engine. Pipeline emit UTC iso trong khi attendance lưu `datetime.now()` naive (local host) → lệch tz.
- **Sửa:**
  - Lifespan: sau `cancel()`, `await task` trong `try/except CancelledError`; đóng qdrant client + `await engine.dispose()`.
  - Thống nhất 1 timezone nghiệp vụ: dùng tz-aware (vd `ZoneInfo("Asia/Ho_Chi_Minh")`) cho `datetime.now(...)` ở attendance service, hoặc UTC đồng bộ với pipeline. Chọn 1, ghi rõ.
  - Thêm validate `from_date <= to_date` trong `api_history` (hoặc schema).
- **Verify:** `pytest`; reload không leak; timestamp nhất quán.
- **Acceptance:** Timezone nhất quán; shutdown sạch.
- **Rollback risk:** Trung bình (đổi tz ảnh hưởng so sánh shift-window — test kỹ `get_current_time`).

### Task 2.8 — Compose: pin Qdrant + hạn chế expose
- **File:** `compose.yaml`
- **Vấn đề:** Qdrant `latest`, không auth; Postgres + Qdrant publish port ra host.
- **Sửa:** `qdrant/qdrant:latest` → pin phiên bản cụ thể (vd `v1.12.4`). Nếu backend chạy trong compose network: bỏ `ports` publish (chỉ expose nội bộ) hoặc bind `127.0.0.1:...`. **Xác minh topology deploy trước** (nếu backend chạy trên host thì vẫn cần port).
- **Verify:** `docker compose config` hợp lệ; backend vẫn kết nối được.
- **Acceptance:** Image pinned; port không expose công khai không cần thiết.
- **Rollback risk:** Trung bình — phụ thuộc cách deploy.

## Phase 2, Task 2.9 — Qdrant vector reconciliation report (report-only)

> **Đã sửa lại (2026-07-02):** bản gốc của Task 2.9 giả định *sai* rằng có thể **khôi phục** vector Qdrant bị thiếu từ Postgres. Giả định đó không đúng với thiết kế hiện tại (xem bên dưới). Task 2.9 chuyển sang **detection / report-only**. Không đụng Phase 1 và Task 2.1–2.8 (đã accept).

### Decision
- **Chọn: Option A — Detection / report-only.**
- **Lý do:** Postgres **không** lưu ảnh hay embedding, nên **không thể** rebuild vector tự động (đã kiểm chứng trên code, mục dưới). Option B (persist ảnh/embedding) kéo theo schema mới, quyết định privacy/security, migration, retention policy, và có thể đổi API → **quá lớn cho một fix nhỏ**; nếu cần, tách thành task riêng ở **Phase 3** (đã ghi ở Non-goals). Tuân theo "minimal safe policy".

### Corrected data model assumption (vì sao giả định gốc sai)
- `src/modules/employees/models.py`: `Employee` chỉ có `emp_id`, `emp_code`, `name`. **Không** có cột ảnh/embedding/`LargeBinary` (đã grep toàn repo — không có nguồn nào khác).
- Ảnh upload chỉ tồn tại **trong RAM** lúc register: `src/modules/employees/api.py` đọc `file.read()` → `extract_embeddings_from_bytes()` → truyền `embeddings` cho `register_employee` → `qdrant.upsert(...)`, rồi bytes bị bỏ. Không lưu đâu cả.
- `register_employee` commit Postgres **trước**, upsert Qdrant **sau**. Nếu upsert fail sau commit → Postgres đã có employee nhưng **không** còn dữ liệu nào để tái tạo vector.
- ⇒ **Qdrant là nơi DUY NHẤT lưu vector.** Employee có trong PG nhưng thiếu vector Qdrant là **không khôi phục tự động được** → chỉ có thể **báo cáo** và yêu cầu operator **đăng ký lại**.
- Chiều ngược lại phát hiện được read-only: payload Qdrant lưu `{emp_id, emp_code, name}`, nên **orphan vector** (emp_id không còn trong PG — ví dụ `remove_employee` xoá PG xong nhưng `qdrant.delete` fail) có thể dò bằng set-diff. Orphan **nguy hiểm về nghiệp vụ**: recognition có thể match một `emp_id` đã bị xoá → `log_attendance` ghi cho nhân viên không tồn tại. Vì vậy report **cả hai chiều**.

### Scope
**Implement:**
- Script chạy tay `scripts/reconcile_vectors.py` (đọc PG + Qdrant, so sánh theo `emp_id`).
- `pg_ids` = tập `emp_id` trong bảng `employees`; `qdrant_ids` = tập `emp_id` lấy từ payload mọi point trong collection (scroll/paginate qua `COLLECTION_NAME`).
- **MISSING VECTOR** = `pg_ids - qdrant_ids`: in rõ `emp_id`, `emp_code`, `name`; guidance: **operator phải re-register** các nhân viên này (không thể tự phục hồi).
- **ORPHAN VECTOR** = `qdrant_ids - pg_ids`: in rõ `emp_id`; guidance: **operator prune thủ công** (script chỉ report).
- Exit code **≠ 0** nếu phát hiện bất kỳ inconsistency nào (để dùng được trong cron/CI sau này); exit `0` khi khớp hoàn toàn.
- Nếu **không kết nối được / lỗi Qdrant hoặc Postgres**: fail rõ ràng — in lỗi (đã theo phong cách Task 2.6, không leak chi tiết nhạy cảm ra ngoài là không bắt buộc ở đây vì đây là CLI cho operator, nhưng phải **không** trả exit 0 giả thành công), exit ≠ 0.
- **Read-only tuyệt đối:** không `upsert`/`delete`/`commit`/`add` gì cả.

**Do NOT implement:**
- Tự re-embed / tái tạo vector.
- Lưu ảnh khuôn mặt.
- Lưu embedding trong Postgres.
- Schema migration.
- Thay đổi API hoặc frontend.
- Tự động xoá orphan (đó là mutation → chỉ report; nếu sau này muốn, thêm flag `--prune-orphans` ở task khác).

### Files allowed to change
- **Tạo mới:** `scripts/reconcile_vectors.py`
- **Tạo mới:** `tests/scripts/test_reconcile_vectors.py` (thêm `tests/scripts/__init__.py` nếu bộ test cần package).
- **Được cập nhật (housekeeping):** `docs/security/follow-ups.md` (ghi nhận công cụ + policy re-register).
- Tái sử dụng (import, **không sửa**): `get_qdrant_client`, `COLLECTION_NAME` từ `src/platform/db/qdrant.py`; `AsyncSessionLocal` từ `src/platform/db/session.py`; `Employee` từ `src/modules/employees/models.py`.

### Files NOT allowed to change
- `src/modules/employees/models.py`, `src/modules/attendance/models.py` (không đổi schema).
- `src/modules/employees/service.py`, `src/modules/employees/api.py` (không thêm persistence/không đổi API — logging Qdrant-fail đã có từ Task 2.6).
- `src/platform/db/qdrant.py` (cấu hình client là Task 2.8; script chỉ import, dùng read-only).
- `compose.yaml`, `frontend/**`, và mọi file đã accept ở Phase 1 / Task 2.1–2.8.
- `README.md` (đang có sửa đổi ngoài phạm vi — đừng đụng).

### Tests
Mock-based (theo phong cách hiện có: `AsyncMock`/`MagicMock`, **không** cần Postgres/Qdrant thật):
- **Khớp hoàn toàn:** mọi employee đều có vector → không báo gì, exit `0`.
- **Thiếu vector:** một employee có trong PG, không có trong Qdrant → được report kèm `emp_id`/`emp_code`, exit ≠ 0.
- **Orphan:** `emp_id` có trong Qdrant, không có trong PG → được report, exit ≠ 0.
- **Qdrant lỗi/không kết nối:** raise khi đọc Qdrant → script fail rõ ràng (exit ≠ 0), **không** báo thành công giả.
- **Read-only:** assert **không** gọi `upsert`/`delete`/`commit`/`add`/`delete` trên mock db & qdrant.

### Verification commands
- `uv run pytest tests/scripts/test_reconcile_vectors.py -q`
- `uv run pytest -q`  *(toàn bộ suite vẫn xanh — không hồi quy Task 2.1–2.8)*
- `git diff --check`
- *(tùy chọn, cần service thật):* `uv run python -m scripts.reconcile_vectors` (hoặc `uv run python scripts/reconcile_vectors.py`); kiểm tra report + `echo $?` (exit code).

### Non-goals
- Không đổi schema.
- Không lưu ảnh khuôn mặt / embedding.
- Không tự động khôi phục vector.
- Không tự động xoá orphan.
- Không đổi frontend/API.

### GPT 5.5 implementation instructions
> Implement **Task 2.9 (Option A — report-only)**. Viết `scripts/reconcile_vectors.py`: dùng `asyncio.run(main())`; mở `AsyncSessionLocal()` đọc toàn bộ `emp_id` từ `Employee`; dùng `get_qdrant_client()` scroll toàn bộ point trong `COLLECTION_NAME` (phân trang tới hết) và gom `emp_id` từ payload. So sánh set: in **MISSING VECTOR** (`pg_ids - qdrant_ids`, kèm `emp_code`/`name`, ghi rõ "operator phải re-register") và **ORPHAN VECTOR** (`qdrant_ids - pg_ids`, ghi rõ "prune thủ công"). Exit `0` nếu khớp hoàn toàn, ngược lại exit ≠ 0. Lỗi kết nối/đọc PG hoặc Qdrant → in lỗi + exit ≠ 0 (không giả thành công). **Read-only:** tuyệt đối không upsert/delete/commit. Thêm `tests/scripts/test_reconcile_vectors.py` (mock `Employee` query + Qdrant scroll) cho 5 case ở mục Tests, khẳng định không có mutation. Chạy `uv run pytest -q` và `git diff --check`. **Không** đụng schema, ảnh/embedding persistence, API, frontend, hay bất kỳ file đã accept nào.

## Phase 2, Task 2.10 — Enforce shift-window attendance behavior

> **Đã chốt (2026-07-02):** bỏ ambiguity "chọn 1". `api_checkin`/`api_checkout` hiện gọi thẳng `check_in`/`check_out` → **bỏ qua shift-window** (chỉ `log_attendance` mới enforce). Chốt **Option B: coi đây là chấm công thường, enforce shift-window**. Không đụng Phase 1 và Task 2.1–2.9.

### Decision
- **Chọn: Option B — enforce shift-window cho manual check-in/out.**
- **Lý do:** (1) Không thể tạo admin-override nếu không có admin auth/RBAC thật + audit log — mà project **chưa có login/RBAC** (Task 2.1 BFF/API-key chỉ chặn truy cập backend trực tiếp, **không** authorize người gọi frontend: bất kỳ ai mở được frontend đều gọi được BFF route). Tạo override "bỏ qua giờ" mà không có admin thật = lỗ hổng chính sách. (2) UI đã hiển thị khung giờ shift → kỳ vọng nghiệp vụ là *theo giờ*. Admin-override thật (Option A) **dời sang Phase 3**, chỗ đó bắt buộc có login/RBAC + audit.

### Corrected product policy
Hành động check-in/check-out qua API/frontend hiện tại là **chấm công thường (normal attendance)**, **không** phải admin override. Nó phải tuân **cùng** ràng buộc khung giờ như luồng recognition (`log_attendance`). Field "manual" trên UI chỉ là *nhập tay emp_id* (thay cho face detection), **không** phải "override luật".

### Public API contract (phải giữ nguyên)
- Response shape **không đổi**: `POST /api/attendance/checkin` và `/checkout` vẫn trả `AttendanceCheckResponse { message, check_type, log }`.
- Khi thành công, `AttendanceCheckResponse.log` **vẫn là object `AttendanceLogOut`** (không được để `None`).
- Business error giữ nguyên cơ chế: trả `(None, "<thông báo>")` từ service → API `raise HTTPException(400, err)`. Thêm sentinel mới **ngoài giờ** cũng đi qua path 400 này.
- **KHÔNG** đổi `AttendanceCheckResponse`, `AttendanceLogOut`, hay contract string của `log_attendance` (pipeline recognition đang phụ thuộc — nó vẫn trả `str`).

### Backend scope
**Cách làm (tối giản, enforce trong business service — KHÔNG route qua `log_attendance`):**
Thêm 2 wrapper mỏng trong `src/modules/attendance/service.py`, giữ `check_in`/`check_out` nguyên vẹn (để không phá test đã accept ở 2.7):
```
async def manual_check_in(db, emp_id) -> tuple[AttendanceLog | None, str | None]:
    shifts, err = await get_shift_settings(db)
    if err:
        return None, err
    within, now, check_type = get_current_time(shifts)
    if not within or check_type != "check_in":
        return None, "Ngoài khung giờ check-in."
    return await check_in(db, emp_id, now=now)   # dùng lại logic + timestamp đã tính

async def manual_check_out(db, emp_id) -> tuple[AttendanceLog | None, str | None]:
    ... # tương tự, check_type != "check_out" → "Ngoài khung giờ check-out."
    return await check_out(db, emp_id, now=now)
```
- API (`api.py`): đổi `check_in`→`manual_check_in`, `check_out`→`manual_check_out`. Phần dựng `AttendanceCheckResponse(..., log=AttendanceLogOut.model_validate(log))` **giữ nguyên** → response shape bảo toàn. Import thêm `manual_check_in`, `manual_check_out` (bỏ import trực tiếp `check_in`/`check_out` khỏi api nếu không còn dùng).
- **Tại sao truyền `now` từ `get_current_time` vào `check_in`:** đồng nhất timestamp dùng để quyết định cửa sổ và timestamp ghi log → tránh race khi thời điểm rơi đúng ranh giới khung giờ.
- **Ghi chú status code (chấp nhận đơn giản hoá):** `get_shift_settings` lỗi (hiếm — DB đã sống vì `get_employee` chạy trước) sẽ đi qua path 400 như business error. Đây là edge cực hiếm; không cần tách 500 riêng cho Task 2.10.

**Do NOT implement:** admin override; RBAC/login; audit-log table; đổi tên route (`/checkin`,`/checkout` giữ nguyên → **allowlist proxy KHÔNG đổi**); schema migration; sửa `check_in`/`check_out`/`log_attendance` core.

### Frontend scope
- Chỉ sửa **copy gây hiểu nhầm**: `frontend/src/components/attendance/attendance-client.tsx:141` `Label` `"Employee ID (manual override)"` → `"Employee ID (manual entry)"` (hoặc `"Employee ID"`). Vì hành động không còn là "override".
- **Không** đổi `api.ts` (route + response shape không đổi). `redirect: "manual"` trong `route.ts` là fetch-mode, **không** liên quan — đừng đụng.

### Files allowed to change
- `src/modules/attendance/service.py` (thêm 2 wrapper).
- `src/modules/attendance/api.py` (đổi lời gọi sang wrapper; import).
- `frontend/src/components/attendance/attendance-client.tsx` (chỉ đổi copy Label).
- `tests/modules/attendance/test_service.py` và/hoặc mới `tests/modules/attendance/test_api.py` (thêm test — file `test_api.py` đã tồn tại từ Task 2.7).

### Files NOT allowed to change
- `src/modules/attendance/schemas.py` (giữ `AttendanceCheckResponse`/`AttendanceLogOut`).
- `check_in`, `check_out`, `log_attendance` core logic trong `service.py` (chỉ *thêm* wrapper, không sửa).
- `src/platform/auth.py`, `require_api_key` dependency (lớp auth 2.1 giữ nguyên).
- `frontend/src/app/api/write/[...path]/route.ts` (allowlist), `frontend/src/lib/api.ts`.
- `README.md`, và mọi file đã accept ở Phase 1 / Task 2.1–2.9.

### Tests
Mock-based (theo phong cách hiện có; monkeypatch `datetime`/`get_current_time` hoặc mock `get_shift_settings` để cố định khung giờ):
- **Check-in trong khung giờ** → `manual_check_in` gọi `check_in`, trả `(log, None)`.
- **Check-in ngoài khung giờ** → trả `(None, "Ngoài khung giờ check-in.")`, **không** gọi `check_in` (assert `check_in`/`db.add` không được gọi).
- **Check-out trong khung giờ** → gọi `check_out`, trả `(log, None)`.
- **Check-out ngoài khung giờ** → trả `(None, "Ngoài khung giờ check-out.")`, không gọi `check_out`.
- **Response giữ `log`:** test API (TestClient, override `manual_check_in` trả `(fake_log, None)`, override `require_api_key`) → 200 và body có `log` là object `AttendanceLogOut` (không `None`).
- **Auth 2.1 còn nguyên:** test API `POST /api/attendance/checkin` thiếu `X-API-Key` → 401/503 (giữ test `test_auth.py` xanh; không cần viết lại).
- **Không hồi quy:** các test `check_in`/`check_out` trực tiếp ở `test_service.py` (2.7) vẫn xanh (vì core không đổi).

### Verification commands
- `uv run pytest tests/modules/attendance/ -q`
- `uv run pytest -q`  *(toàn bộ suite xanh — không hồi quy 2.1–2.9)*
- `cd frontend && npx tsc --noEmit && npm run build`  *(copy Label đổi không phá build/type)*
- `git diff --check`

### Non-goals
- Không admin override.
- Không RBAC/login.
- Không audit logging.
- Không schema migration.
- Không đổi route / allowlist proxy.
- Không đụng follow-up bảo mật của Task 2.1 (trừ khi trực tiếp cần — ở đây không cần).

### GPT 5.5 implementation instructions
> Implement **Task 2.10 (Option B — enforce shift-window)**. Trong `src/modules/attendance/service.py` thêm `manual_check_in(db, emp_id)` và `manual_check_out(db, emp_id)`: load `get_shift_settings`, gọi `get_current_time(shifts)`; nếu `not within` hoặc `check_type` không khớp hành động → trả `(None, "Ngoài khung giờ check-in.")` / `"...check-out."`; ngược lại `return await check_in(db, emp_id, now=now)` / `check_out(...)`. **Không** sửa `check_in`/`check_out`/`log_attendance`. Trong `api.py` đổi `api_checkin`/`api_checkout` gọi wrapper mới; **giữ nguyên** phần `AttendanceCheckResponse(..., log=AttendanceLogOut.model_validate(log))` để bảo toàn response shape. Sửa copy `attendance-client.tsx:141` `"manual override"` → `"manual entry"`. Thêm test (in-window / out-window cho cả in & out, response-có-`log`, auth-401). Chạy `uv run pytest -q`, frontend `tsc --noEmit && npm run build`, `git diff --check`. **Không** đổi schema, route, allowlist, auth, hay bất kỳ file đã accept nào.

### Task 2.11 — Chạy lại npm audit
- **File:** `frontend/package-lock.json`
- **Sửa:** `npm audit --omit=dev`; nếu advisory nằm ở build-tool (`shadcn`→`hono`, `axios`→`form-data`) và không reachable ở runtime → ghi chú, cân nhắc update lockfile. Không ép update phá build.
- **Acceptance:** Có đánh giá reachability rõ ràng.
- **Rollback risk:** Thấp.

---

## PHASE 3 — Lớn/rủi ro. CHỈ làm khi thực sự cần.
1. Wire WebSocket recognition thật ở frontend (`/ws/recognition/{client_id}`, gửi frame binary, tiêu thụ kết quả) — đây mới là sản phẩm thật.
2. Alembic migrations (thay `init.sql`-only) + CI (test/lint/typecheck/audit).
3. Liveness detection thật + calibrate threshold.
4. Employee soft-delete + attendance retention policy (thay cascade delete).

---

## KHÔNG làm ở giai đoạn này (defer / bác bỏ)
- **Outbox/saga PG↔Qdrant** → over-engineered; dùng Task 2.9 reconcile thay thế.
- **Consolidate 2 executor vì "GPU oversubscription"** → chưa có bằng chứng lỗi; chỉ gộp nếu đo được vấn đề.
- **Thay `theme-provider` bằng `next-themes` (giữ code)** → lưu ý: nếu Task 1.6 gỡ dep `next-themes` thì giữ `theme-provider` custom. ROI thấp, rủi ro hydration — không refactor theming.
- **RBAC/audit đầy đủ** → API key (Task 2.1) là đủ đến khi thật sự cần role.
- **Per-camera queue fairness** → sau khi có bounded worker và chỉ khi multi-camera là thật.

---

## Thứ tự khuyến nghị
1. Phase 1 toàn bộ (1.1 → 1.6) — nhỏ, high-confidence, làm sản phẩm chạy được.
2. Phase 2: 2.1 (+2.2), 2.3, 2.4, 2.5, 2.6 trước; 2.7–2.11 sau.
3. Phase 3 chỉ khi có yêu cầu rõ.
