# FaceVec Attend Project Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tạo bộ 10 file Markdown tiếng Việt mô tả toàn diện FaceVec Attend cho cả báo cáo học thuật và tra cứu kỹ thuật, bám sát code hiện tại.

**Architecture:** Tài liệu được chia thành một file điều hướng và chín chương độc lập trong `docs/project-report/`. Mỗi chương có một trách nhiệm rõ ràng, liên kết chéo bằng đường dẫn tương đối và dùng Mermaid cho các quan hệ hoặc chuỗi hoạt động khó diễn đạt tuyến tính.

**Tech Stack:** Markdown, Mermaid, FastAPI, SQLAlchemy async, PostgreSQL, Qdrant, InsightFace, MiniFASNet ONNX, Piper TTS, Next.js App Router, React Query, MediaPipe, Docker Compose.

## Global Constraints

- Code và cấu hình trên nhánh hiện tại là nguồn sự thật chính; tài liệu cũ chỉ là nguồn tham khảo.
- Phân biệt rõ **đã triển khai**, **giới hạn hiện tại** và **đề xuất tương lai**.
- Mô tả backend là modular monolith, không gọi kiến trúc hiện tại là microservice.
- Không đưa secret thật, `.env` cá nhân, dữ liệu khuôn mặt hoặc dữ liệu cá nhân vào tài liệu.
- Không tuyên bố độ chính xác hoặc hiệu năng production nếu repository không có số đo chứng minh.
- Ảnh khuôn mặt chỉ tồn tại tạm thời trong bộ nhớ; embedding được lưu trong Qdrant.
- Mọi sơ đồ Mermaid phải có đoạn giải thích bằng văn bản ngay sau sơ đồ.
- Không thay đổi code ứng dụng, API, schema hoặc hành vi runtime.

## File Map

| File | Trách nhiệm |
|---|---|
| `docs/project-report/README.md` | Mục lục, phạm vi, quy ước và hướng dẫn sử dụng với Word |
| `docs/project-report/01-tong-quan-du-an.md` | Bối cảnh, mục tiêu, phạm vi, yêu cầu và công nghệ |
| `docs/project-report/02-kien-truc-he-thong.md` | Kiến trúc tổng thể, module, deployment và phân tích microservice |
| `docs/project-report/03-luong-hoat-dong.md` | Tất cả luồng nghiệp vụ và vòng đời hệ thống |
| `docs/project-report/04-xu-ly-ai-va-nhan-dien.md` | Pipeline AI, embedding, Qdrant, anti-spoofing và MediaPipe |
| `docs/project-report/05-du-lieu-va-api.md` | Schema, ERD, REST, WebSocket và tính nhất quán dữ liệu |
| `docs/project-report/06-frontend-dashboard-va-kiosk.md` | Dashboard, kiosk, state machine, BFF và client data flow |
| `docs/project-report/07-ha-tang-bao-mat-hieu-nang-kiem-thu.md` | Triển khai, concurrency, bảo mật, kiểm thử và đo tải |
| `docs/project-report/08-danh-gia-va-huong-phat-trien.md` | Đánh giá, giới hạn, rủi ro và lộ trình kỹ thuật |
| `docs/project-report/09-phu-luc-tra-cuu.md` | Cây thư mục, endpoint, env, thuật ngữ và source map |

---

### Task 1: Khung tài liệu, tổng quan và kiến trúc

**Files:**
- Create: `docs/project-report/README.md`
- Create: `docs/project-report/01-tong-quan-du-an.md`
- Create: `docs/project-report/02-kien-truc-he-thong.md`

**Sources:**
- `README.md`
- `pyproject.toml`
- `frontend/package.json`
- `compose.yaml`
- `src/app.py`
- Toàn bộ cây `src/modules/` và `src/platform/`

**Produces:** Bộ khái niệm và thuật ngữ kiến trúc chuẩn để các chương sau dùng nhất quán: modular monolith, admin dashboard, kiosk, BFF, recognition pipeline, PostgreSQL relational store và Qdrant vector store.

- [ ] **Step 1: Tạo `README.md` làm cổng vào bộ tài liệu**

  Viết các mục: mục đích, đối tượng đọc, phạm vi khảo sát ngày 2026-07-20, quy ước ba trạng thái, mục lục đủ chín chương, lộ trình đọc cho báo cáo học thuật, lộ trình đọc cho kỹ thuật, cách render Mermaid và cách đưa sơ đồ vào Word.

- [ ] **Step 2: Viết chương tổng quan dự án**

  Trình bày bài toán chấm công, hạn chế của thẻ/vân tay/thủ công, giải pháp embedding-based, mục tiêu, phạm vi, actor, danh sách chức năng hiện có, yêu cầu chức năng, yêu cầu phi chức năng và bảng công nghệ. Ghi rõ dashboard thống kê đã gọi API thật và kiosk đã được wire trên nhánh hiện tại.

- [ ] **Step 3: Viết chương kiến trúc với ba sơ đồ Mermaid**

  Thêm:

  - Flowchart system context: quản trị viên, nhân viên, browser, Next.js, FastAPI, PostgreSQL và Qdrant.
  - Flowchart deployment/component: các cổng `3000`, `8000`, `5432`, `6333`, REST, WebSocket và BFF.
  - Flowchart module backend: `platform` và năm domain employees, attendance, recognition, antispoofing, TTS.

  Giải thích lý do hệ thống là modular monolith; so sánh ngắn với microservice; mô tả ranh giới có thể tách trong tương lai nhưng không trình bày đó là hiện trạng.

- [ ] **Step 4: Kiểm tra nhóm file thứ nhất**

  Run:

  ```bash
  test -s docs/project-report/README.md
  test -s docs/project-report/01-tong-quan-du-an.md
  test -s docs/project-report/02-kien-truc-he-thong.md
  rg -n "modular monolith|microservice|```mermaid|kiosk" docs/project-report/{README,01-tong-quan-du-an,02-kien-truc-he-thong}.md
  ```

  Expected: ba file tồn tại, có nội dung; kiến trúc và kiosk được mô tả; chương 02 có ít nhất ba Mermaid block.

- [ ] **Step 5: Commit nhóm tổng quan và kiến trúc**

  ```bash
  git add docs/project-report/README.md docs/project-report/01-tong-quan-du-an.md docs/project-report/02-kien-truc-he-thong.md
  git commit -m "docs: add project overview and architecture report"
  ```

### Task 2: Luồng hoạt động và xử lý AI

**Files:**
- Create: `docs/project-report/03-luong-hoat-dong.md`
- Create: `docs/project-report/04-xu-ly-ai-va-nhan-dien.md`

**Sources:**
- `src/app.py`
- `src/platform/queue.py`
- `src/platform/realtime/manager.py`
- `src/platform/ml/face_app.py`
- `src/modules/employees/api.py`
- `src/modules/employees/service.py`
- `src/modules/recognition/`
- `src/modules/attendance/service.py`
- `src/modules/antispoofing/service.py`
- `src/modules/tts/`
- `frontend/src/components/kiosk/`
- `frontend/src/lib/kiosk.ts`

**Consumes:** Thuật ngữ và ranh giới thành phần được định nghĩa trong Task 1.

**Produces:** Mô tả tuần tự chuẩn về enrollment, recognition, attendance, TTS và các trạng thái lỗi để chương dữ liệu và frontend tham chiếu.

- [ ] **Step 1: Viết vòng đời khởi động và dừng**

  Mô tả production liveness guard, tạo Qdrant collection, tạo pipeline task, nhận request, cancel task, drain in-flight work, đóng Qdrant và dispose SQLAlchemy engine. Thêm sequence diagram startup/shutdown.

- [ ] **Step 2: Viết sequence đăng ký nhân viên nhiều frame**

  Bám đúng flow: admin nhập tên/mã → deep-link `/kiosk?mode=register` → MediaPipe kiểm tra khoảng cách → countdown 3 giây → tối đa 5 frame cách 400 ms → BFF thêm API key → backend bỏ frame không có đúng một mặt → commit PostgreSQL → upsert nhiều point Qdrant → invalidate cache và quay lại danh sách.

- [ ] **Step 3: Viết sequence nhận diện và chấm công realtime**

  Bám đúng flow: camera 1280x720 → proximity gate → JPEG chất lượng 0.7 mỗi 1000 ms → WebSocket → bounded queue → largest face → liveness → embedding → Qdrant top-1 → shift lookup → check-in/out → JSON response → state reducer → TTS. Trình bày các status `no_face`, `spoof`, `unknown`, `recognized`, `error`.

- [ ] **Step 4: Viết các flow quản trị còn lại**

  Bao gồm list/search/delete employee, cập nhật ca, summary/monthly/daily statistics, xuất Excel hai sheet và TTS. Nêu rõ xóa nhân viên xóa vector trước rồi mới xóa PostgreSQL và attendance cascade.

- [ ] **Step 5: Viết chương AI**

  Giải thích InsightFace `buffalo_sc`, detector `640x640`, provider CUDA rồi CPU, normed embedding 512 chiều, cosine score và code hiện tại dùng `score_threshold = 1 - THRESHOLD` với `THRESHOLD = 0.6`. Phân tích chính xác ý nghĩa của ngưỡng theo implementation, không tự gán nó là độ chính xác 60%.

  Trình bày MiniFASNet: crop vuông mở rộng 1.5, RGB, resize/pad 128, CHW float32, softmax, class 0 và threshold mặc định 0.5. Trình bày MediaPipe BlazeFace chỉ phục vụ tracking/proximity phía client, không quyết định danh tính.

- [ ] **Step 6: Thêm flowchart pipeline AI và bảng trạng thái**

  Flowchart phải có các nhánh decode lỗi, không có mặt, spoof, unknown, recognized và system error. Bảng trạng thái phải ghi rõ payload WebSocket tương ứng và tác động UI.

- [ ] **Step 7: Kiểm tra nhóm flow và AI**

  Run:

  ```bash
  test -s docs/project-report/03-luong-hoat-dong.md
  test -s docs/project-report/04-xu-ly-ai-va-nhan-dien.md
  rg -n "multi-frame|512|Qdrant|MiniFASNet|MediaPipe|drop-oldest|```mermaid" docs/project-report/{03-luong-hoat-dong,04-xu-ly-ai-va-nhan-dien}.md
  ```

  Expected: hai file có nội dung; có đủ thuật ngữ kỹ thuật và ít nhất ba sơ đồ cho startup, enrollment, recognition/AI.

- [ ] **Step 8: Commit nhóm flow và AI**

  ```bash
  git add docs/project-report/03-luong-hoat-dong.md docs/project-report/04-xu-ly-ai-va-nhan-dien.md
  git commit -m "docs: document system flows and recognition pipeline"
  ```

### Task 3: Dữ liệu, API và frontend

**Files:**
- Create: `docs/project-report/05-du-lieu-va-api.md`
- Create: `docs/project-report/06-frontend-dashboard-va-kiosk.md`

**Sources:**
- `initdb/init.sql`
- `src/modules/*/models.py`
- `src/modules/*/schemas.py`
- `src/modules/*/api.py`
- `src/modules/*/service.py`
- `src/modules/recognition/ws_ingress.py`
- `src/modules/recognition/pipeline.py`
- `frontend/src/app/`
- `frontend/src/components/`
- `frontend/src/lib/api.ts`
- `frontend/src/lib/types.ts`
- `frontend/src/lib/kiosk.ts`

**Consumes:** Flow và thuật ngữ từ Tasks 1–2.

**Produces:** Bảng tra cứu hợp đồng dữ liệu và mô hình trạng thái frontend.

- [ ] **Step 1: Viết mô hình dữ liệu và ERD**

  Mô tả `employees`, `attendance_logs`, `shift_settings`, foreign key cascade, generated `working_duration`, check constraint và partial unique index. Dùng `erDiagram` cho PostgreSQL; nối Qdrant bằng một flowchart riêng vì Qdrant không có foreign key vật lý.

- [ ] **Step 2: Viết collection Qdrant và quy tắc đồng bộ**

  Ghi collection `face_embeddings`, vector 512 chiều, Cosine, nhiều point cho một `emp_id`, payload `emp_id`, `emp_code`, `name`. Phân tích registration commit PG trước Qdrant, deletion Qdrant trước PG, các failure window và `scripts/reconcile_vectors.py` exit code 0/1/2.

- [ ] **Step 3: Lập bảng toàn bộ REST API hiện tại**

  Bao gồm employees, attendance history/check-in/check-out, summary, monthly, daily, report Excel, shift settings và TTS. Với mỗi endpoint ghi method, path, auth, input, output, lỗi chính và consumer. Sửa khác biệt với API doc cũ: enrollment dùng `files` nhiều phần, dashboard stats là API thật và BFF hiện không allowlist manual attendance writes.

- [ ] **Step 4: Ghi hợp đồng WebSocket**

  Mô tả đường dẫn `/ws/recognition/{client_id}`, binary JPEG client-to-server và năm union message server-to-client. Nêu WebSocket hiện công khai, connection registry dùng một socket cuối cùng cho mỗi `client_id` và không có acknowledgement theo sequence frame.

- [ ] **Step 5: Viết chương frontend và state machine kiosk**

  Mô tả App Router, route group dashboard, `/kiosk`, React Query/Axios, read-direct/write-BFF, cache invalidation, navigation và dashboard charts. Thêm `stateDiagram-v2` cho camera/socket/greeting, giải thích reducer priority, reconnect 2 giây, greeting 5 giây và TTS best-effort.

- [ ] **Step 6: Kiểm tra nhóm data/API/frontend**

  Run:

  ```bash
  test -s docs/project-report/05-du-lieu-va-api.md
  test -s docs/project-report/06-frontend-dashboard-va-kiosk.md
  rg -n "attendance/summary|attendance/report|files|stateDiagram-v2|face_embeddings|reconcile" docs/project-report/{05-du-lieu-va-api,06-frontend-dashboard-va-kiosk}.md
  ```

  Expected: hai file có nội dung; endpoint mới, multi-file enrollment, ERD/Qdrant và kiosk state machine đều xuất hiện.

- [ ] **Step 7: Commit nhóm dữ liệu, API và frontend**

  ```bash
  git add docs/project-report/05-du-lieu-va-api.md docs/project-report/06-frontend-dashboard-va-kiosk.md
  git commit -m "docs: add data API and frontend reference"
  ```

### Task 4: Hạ tầng, bảo mật, hiệu năng, kiểm thử và đánh giá

**Files:**
- Create: `docs/project-report/07-ha-tang-bao-mat-hieu-nang-kiem-thu.md`
- Create: `docs/project-report/08-danh-gia-va-huong-phat-trien.md`

**Sources:**
- `compose.yaml`
- `.env.example`
- `frontend/.env.example`
- `Makefile`
- `pyproject.toml`
- `frontend/package.json`
- `src/app.py`
- `src/platform/auth.py`
- `src/platform/queue.py`
- `src/modules/recognition/pipeline.py`
- `frontend/src/app/api/write/[...path]/route.ts`
- `tests/`
- `scripts/load_test.py`
- `scripts/reconcile_vectors.py`

**Consumes:** Kiến trúc, flow và hợp đồng từ Tasks 1–3.

**Produces:** Đánh giá có bằng chứng về đặc tính vận hành hiện tại và roadmap không bị nhầm với chức năng đã có.

- [ ] **Step 1: Viết triển khai và cấu hình**

  Mô tả host process cho FastAPI/Next.js, container PostgreSQL/Qdrant, volume, loopback port binding, Qdrant image pin, uv/npm và nhóm biến môi trường. Không chép giá trị secret.

- [ ] **Step 2: Viết concurrency và backpressure**

  Giải thích queue 50, drop-oldest, semaphore 4, thread pool 4, task strong reference, CPU work off event loop, one-second client capture và graceful shutdown. Không khẳng định số camera tối đa; trình bày `scripts/load_test.py` là công cụ đo, không phải bằng chứng kết quả nếu chưa chạy.

- [ ] **Step 3: Viết threat model và sơ đồ BFF**

  Thêm sơ đồ browser → same-origin BFF → FastAPI. Phân tích fail-closed API key, constant-time compare, BFF allowlist, same-origin `Origin`/`Host` check, CORS localhost, Qdrant API key và loopback binding. Ghi rõ thiếu user login, RBAC, audit, rate limit và TLS; WebSocket/TTS/read endpoints hiện public.

- [ ] **Step 4: Viết chiến lược kiểm thử**

  Phân loại unit, service, API, integration-like và frontend pure-logic tests theo file hiện có. Ghi các lệnh backend `uv run pytest -q` và frontend `npm run lint`, `npx tsc --noEmit`, `npm run build`; mô tả load test và chỉ số response rate, p50/p95/p99, CPU, RAM.

- [ ] **Step 5: Viết đánh giá và roadmap**

  Tách ba bảng: điểm mạnh đã có, giới hạn/rủi ro, ưu tiên phát triển. Đề xuất theo thứ tự: benchmark/observability, auth-RBAC-audit, TLS/rate limit, liveness validation, outbox/reconciliation, GPU worker/message broker, rồi mới cân nhắc microservice. Thêm sơ đồ kiến trúc mục tiêu có API gateway/auth, recognition worker và message broker nhưng gắn nhãn đề xuất.

- [ ] **Step 6: Kiểm tra nhóm vận hành và đánh giá**

  Run:

  ```bash
  test -s docs/project-report/07-ha-tang-bao-mat-hieu-nang-kiem-thu.md
  test -s docs/project-report/08-danh-gia-va-huong-phat-trien.md
  rg -n "drop-oldest|Semaphore|CSRF|RBAC|rate limit|đề xuất|load_test" docs/project-report/{07-ha-tang-bao-mat-hieu-nang-kiem-thu,08-danh-gia-va-huong-phat-trien}.md
  ```

  Expected: hai file có nội dung; concurrency, security debt, test strategy và roadmap được phân biệt rõ.

- [ ] **Step 7: Commit nhóm hạ tầng và đánh giá**

  ```bash
  git add docs/project-report/07-ha-tang-bao-mat-hieu-nang-kiem-thu.md docs/project-report/08-danh-gia-va-huong-phat-trien.md
  git commit -m "docs: cover operations security and project roadmap"
  ```

### Task 5: Phụ lục, liên kết chéo và kiểm chứng toàn bộ

**Files:**
- Create: `docs/project-report/09-phu-luc-tra-cuu.md`
- Modify: `docs/project-report/README.md`
- Modify when verification finds factual inconsistencies: `docs/project-report/01-tong-quan-du-an.md` through `docs/project-report/08-danh-gia-va-huong-phat-trien.md`

**Sources:** Toàn bộ repository, ưu tiên `src/`, `frontend/src/`, config, schema và tests.

**Consumes:** Toàn bộ tài liệu từ Tasks 1–4.

**Produces:** Bộ tài liệu hoàn chỉnh, điều hướng được, có source map và không chứa placeholder hoặc secret.

- [ ] **Step 1: Viết phụ lục tra cứu**

  Thêm cây thư mục chú giải, bảng endpoint cô đọng, bảng biến môi trường gồm nơi dùng/mức nhạy cảm/default an toàn, glossary Việt–Anh, bảng đối chiếu chủ đề với source file và gợi ý ánh xạ chín chương Markdown sang chương/mục Word.

- [ ] **Step 2: Hoàn thiện liên kết hai chiều**

  Mỗi chương có liên kết về mục lục và chương tiếp theo; `README.md` liên kết đủ chín chương bằng đường dẫn tương đối chính xác.

- [ ] **Step 3: Kiểm tra đủ file và liên kết local**

  Run:

  ```bash
  find docs/project-report -maxdepth 1 -type f -name '*.md' | sort
  rg -n '\]\([^)]*\.md\)' docs/project-report
  ```

  Expected: đúng 10 file Markdown và các liên kết chương dùng đường dẫn tương đối.

- [ ] **Step 4: Quét placeholder và secret**

  Run:

  ```bash
  rg -n "TODO|TBD|FIXME|your_password|BEGIN (RSA|OPENSSH|EC) PRIVATE KEY" docs/project-report || true
  ```

  Expected: không có placeholder hoặc secret. Nếu từ `TODO`/`TBD` xuất hiện trong câu hướng dẫn, đổi cách diễn đạt để phép quét sạch hoàn toàn.

- [ ] **Step 5: Kiểm chứng endpoint và thuật ngữ với source**

  Run:

  ```bash
  rg -n '@router\.(get|post|put|delete)|@router\.websocket' src/modules src/platform
  rg -n 'ALLOWED_TARGETS|CAPTURE_INTERVAL_MS|GREETING_MS|RECONNECT_MS|ENROLLMENT_BURST|THRESHOLD|FrameQueue|max_workers|Semaphore' src frontend/src
  rg -n 'attendance/(summary|monthly|daily|report)|api/tts|ws/recognition|face_embeddings' docs/project-report
  ```

  Expected: mọi endpoint và hằng số quan trọng trong source có mô tả tương ứng, không dùng giá trị từ tài liệu cũ trái với code.

- [ ] **Step 6: Kiểm tra Mermaid cơ bản và Markdown hygiene**

  Run:

  ```bash
  rg -c '^```mermaid$' docs/project-report/*.md
  git diff --check -- docs/project-report
  ```

  Expected: các chương kiến trúc/flow/data/frontend/security/roadmap có Mermaid; không có whitespace error.

- [ ] **Step 7: Chạy test repository để bảo đảm thay đổi tài liệu không ảnh hưởng project**

  Run:

  ```bash
  uv run pytest -q
  npm run lint --prefix frontend
  npx --prefix frontend tsc --noEmit
  npm run build --prefix frontend
  ```

  Expected: backend tests pass, frontend lint/typecheck/build pass. Nếu lỗi do trạng thái có sẵn hoặc môi trường, ghi rõ lệnh, lỗi và xác nhận không phát sinh từ file Markdown.

- [ ] **Step 8: Commit phụ lục và các sửa chữa kiểm chứng**

  ```bash
  git add docs/project-report
  git commit -m "docs: complete FaceVec Attend project report"
  ```

- [ ] **Step 9: Báo cáo bàn giao**

  Nêu đường dẫn `docs/project-report/README.md`, danh sách chương, số sơ đồ Mermaid, các lệnh kiểm chứng đã chạy, kết quả test và mọi giới hạn còn lại. Không tuyên bố hoàn thành nếu bất kỳ file bắt buộc nào còn thiếu.
