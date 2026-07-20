# Chương 9. Phụ lục tra cứu

## 9.1. Cây thư mục chú giải

```text
facevec-attend/
├── src/
│   ├── app.py                         # Composition root, lifespan, CORS và router
│   ├── modules/
│   │   ├── employees/                 # Hồ sơ nhân viên và enrollment nhiều frame
│   │   ├── attendance/                # Ca làm, chấm công, thống kê và Excel
│   │   ├── recognition/               # Trích embedding, định danh và pipeline realtime
│   │   ├── antispoofing/               # MiniFASNet và fallback liveness
│   │   └── tts/                        # Piper TTS trả WAV
│   └── platform/
│       ├── auth.py                     # Xác thực shared API key cho write
│       ├── config.py                   # Cấu hình model và ngưỡng khoảng cách
│       ├── db/                         # SQLAlchemy session và Qdrant client
│       ├── ml/                         # Singleton InsightFace
│       ├── queue.py                    # Queue frame hữu hạn, drop-oldest
│       └── realtime/                   # Registry WebSocket theo client_id
├── frontend/src/
│   ├── app/                            # Next.js App Router: dashboard, kiosk và BFF
│   ├── components/                     # UI nghiệp vụ và các hook kiosk
│   └── lib/                            # Axios client, type, reducer và tiện ích
├── initdb/init.sql                     # DDL PostgreSQL và ca mặc định
├── scripts/reconcile_vectors.py        # Đối soát read-only PostgreSQL–Qdrant
├── tests/                              # Backend/unit/API/integration-like tests
├── compose.yaml                        # PostgreSQL 16 và Qdrant 1.18.2
├── .env.example                        # Mẫu biến môi trường backend/datastore
└── frontend/.env.example               # Mẫu biến public và server-only của Next.js
```

`src/modules/` là các domain trong cùng một tiến trình FastAPI, không phải các service triển khai độc lập. `frontend/src/app/api/write/[...path]/route.ts` là BFF chỉ cho phép ba cặp method/đích; browser vẫn đọc trực tiếp từ FastAPI. Compose chỉ quản lý hai datastore, không chứa container backend hoặc frontend.

## 9.2. Bảng endpoint cô đọng

Tất cả đường dẫn REST dưới đây có prefix `/api`. “Bảo vệ write” nghĩa là FastAPI yêu cầu header `X-API-Key`; nó không tương đương đăng nhập hay RBAC.

| Method | Đường dẫn | Input chính | Kết quả chính | Xác thực hiện tại | BFF expose |
|---|---|---|---|---|---|
| `POST` | `/api/employees` | Multipart `name`, `emp_code`, `files[]`; tối đa 5 MiB/file | Nhân viên vừa tạo | Bảo vệ write | Có: `POST /api/write/employees` |
| `GET` | `/api/employees` | `page`, `page_size` 1..100 | Danh sách phân trang | Công khai | Không cần |
| `GET` | `/api/employees/{identifier}` | ID số hoặc chuỗi tên | Chi tiết hoặc kết quả tìm tên | Công khai | Không cần |
| `DELETE` | `/api/employees` | `emp_id` hoặc `emp_code` | Nhân viên đã xóa | Bảo vệ write | Có: `DELETE /api/write/employees` |
| `GET` | `/api/attendance/summary` | Không có | Tổng quan hôm nay và delta hôm qua | Công khai | Không cần |
| `GET` | `/api/attendance/monthly` | `year` | Thống kê từng tháng và năm khả dụng | Công khai | Không cần |
| `GET` | `/api/attendance/daily` | `year`, `month` 1..12 | Giờ trung bình theo ngày | Công khai | Không cần |
| `GET` | `/api/attendance/report` | `year`, `month` 1..12 | Workbook XLSX hai sheet | Công khai | Không cần |
| `POST` | `/api/attendance/checkin` | `emp_id` | Log check-in thủ công | Bảo vệ write | Không |
| `POST` | `/api/attendance/checkout` | `emp_id` | Log check-out thủ công | Bảo vệ write | Không |
| `GET` | `/api/attendance` | `emp_id`, khoảng ngày, phân trang | Lịch sử nhân viên | Công khai | Không cần |
| `GET` | `/api/shift-settings` | Không có | Bốn mốc thời gian ca | Công khai | Không cần |
| `PUT` | `/api/shift-settings` | Bốn mốc thời gian ca | Cấu hình đã upsert | Bảo vệ write | Có: `PUT /api/write/shift-settings` |
| `GET` | `/api/tts` | `text` dài 1..200 ký tự | Audio WAV | Công khai | Không cần |
| WebSocket | `/ws/recognition/{client_id}` | Frame JPEG dạng binary | JSON `no_face`, `spoof`, `unknown`, `recognized` hoặc `error` | Công khai | Không áp dụng |

Nguồn chuẩn cho bảng là `src/modules/*/api.py`, `src/modules/recognition/ws_ingress.py` và allowlist trong `frontend/src/app/api/write/[...path]/route.ts`. OpenAPI do FastAPI sinh là nơi phù hợp để xem schema REST chi tiết khi ứng dụng đang chạy.

## 9.3. Biến môi trường

Mức nhạy cảm trong bảng là đánh giá vận hành: **cao** cần secret manager/quyền truy cập hạn chế; **trung bình** có thể lộ topology hoặc danh tính kỹ thuật; **thấp** là cấu hình công khai. “Không có” nghĩa là code không đặt fallback an toàn và bên triển khai phải cung cấp giá trị.

| Biến | Nơi dùng | Nhạy cảm | Default/fallback an toàn theo source | Ghi chú |
|---|---|---:|---|---|
| `DB_NAME` | SQLAlchemy, Compose PostgreSQL | Thấp | Không có; mẫu dùng `attendance` | Bắt buộc để tạo URL backend và database container. |
| `DB_USER` | SQLAlchemy, Compose PostgreSQL | Trung bình | Không có; mẫu dùng `postgres` | Không phải secret nhưng có thể hỗ trợ dò credential. |
| `DB_PASS` | SQLAlchemy, Compose PostgreSQL | Cao | Không có | Phải đặt secret thật ngoài Git. |
| `DB_HOST` | SQLAlchemy | Thấp | Không có; mẫu local dùng `localhost` | Trong network container cần hostname phù hợp deployment. |
| `DB_PORT` | SQLAlchemy, port bind Compose | Thấp | Compose fallback `5432`; backend không tự đặt | Hai nơi dùng phải nhất quán. |
| `QDRANT_HOST` | Qdrant client | Thấp | `localhost` | Ứng dụng backend dùng trực tiếp. |
| `QDRANT_PORT` | Qdrant client, Compose | Thấp | `6333` | Compose chỉ bind vào loopback host. |
| `QDRANT_API_KEY` | Qdrant client, Qdrant container | Cao | Không có; Compose fail nếu thiếu | Không đưa vào client browser. |
| `QDRANT_HTTPS` | Qdrant client | Thấp | `false` | Chỉ bật khi endpoint Qdrant thực sự có TLS. |
| `API_KEY` | FastAPI auth và Next.js BFF | Cao | Không có; hai phía fail-closed | Cùng credential kỹ thuật phải được cấp server-side ở cả hai process. |
| `ENV` | FastAPI startup guard | Thấp | Chuỗi rỗng; mẫu dùng `development` | Giá trị `production` từ chối `PassThroughChecker`. |
| `ANTISPOOFING_MODEL_PATH` | Liveness factory | Thấp | `models/antispoofing/AntiSpoofing_bin_1.5_128.onnx` | Thiếu model dẫn đến fallback; production guard sẽ chặn fallback. |
| `ANTISPOOFING_THRESHOLD` | MiniFASNet checker | Thấp | `0.5` | Khác `THRESHOLD=0.6` trong nhận diện Qdrant. |
| `PIPER_VOICE_PATH` | Piper TTS service | Thấp | Model Việt trong `models/piper/` | Đường dẫn artifact cục bộ, không phải API credential. |
| `NEXT_PUBLIC_API_BASE_URL` | Axios và kiosk browser | Thấp | `http://localhost:8000` | Có mặt trong bundle browser theo chủ đích. |
| `BACKEND_INTERNAL_URL` | Next.js BFF | Trung bình | Không có; BFF trả 503 nếu thiếu/sai | Chỉ server-side; phải là URL HTTP(S). |

Không đặt prefix `NEXT_PUBLIC_` cho `API_KEY`, `DB_PASS` hoặc `QDRANT_API_KEY`. File `.env` thực và credential không thuộc nội dung báo cáo hoặc version control.

## 9.4. Hằng số và invariant tra nhanh

| Giá trị hiện hành | Ý nghĩa | Source chuẩn |
|---|---|---|
| `MODEL.name = buffalo_sc` | Gói InsightFace | `src/platform/config.py` |
| `det_size = 640 × 640` | Kích thước detector InsightFace | `src/platform/config.py` |
| Vector 512, `COSINE` | Schema collection `face_embeddings` | `src/platform/db/qdrant.py` |
| `THRESHOLD = 0.6` | Khoảng cách tối đa; code chấp nhận score `>= 1 - THRESHOLD`, tức `>= 0.4` | `src/platform/config.py`, `src/modules/recognition/identifier.py` |
| Queue `maxsize=50` | Backpressure drop-oldest cho frame | `src/app.py`, `src/platform/queue.py` |
| `Semaphore(4)` và pool 4 worker | Tối đa bốn task pipeline in-flight và bốn thread CPU | `src/modules/recognition/pipeline.py` |
| Capture `1000 ms` | Chu kỳ gửi frame kiosk | `frontend/src/components/kiosk/use-recognition.ts` |
| Greeting `5000 ms` | Thời gian giữ trạng thái chào | `frontend/src/components/kiosk/use-recognition.ts` |
| Reconnect `2000 ms` | Độ trễ thử nối lại WebSocket | `frontend/src/components/kiosk/use-recognition.ts` |
| Burst 5 frame, cách `400 ms` | Enrollment phía kiosk | `frontend/src/components/kiosk/use-enrollment.ts` |
| TTS tối đa 200 ký tự | Giới hạn query text | `src/modules/tts/service.py` |

Các giá trị concurrency và thời gian ở trên là cấu hình code, không phải benchmark, SLA hay bằng chứng số kiosk tối đa.

## 9.5. Glossary Việt–Anh

| Thuật ngữ dùng trong báo cáo | Thuật ngữ Anh | Nghĩa trong dự án |
|---|---|---|
| Chấm công | Attendance | Ghi check-in/check-out của nhân viên theo ngày và cửa sổ ca. |
| Đăng ký khuôn mặt | Face enrollment | Thu nhiều frame, lấy các embedding hợp lệ và gắn với nhân viên. |
| Nhận diện | Face recognition | So embedding frame hiện tại với vector đã lưu để tìm danh tính. |
| Phát hiện khuôn mặt | Face detection | Xác định vị trí/mặt có trong ảnh; chưa phải nhận dạng danh tính. |
| Vector đặc trưng | Face embedding | Vector 512 chiều đã chuẩn hóa biểu diễn khuôn mặt. |
| Độ tương đồng cosine | Cosine similarity | Score Qdrant dùng để xếp ứng viên gần nhất. |
| Kiểm tra người thật | Liveness / anti-spoofing | Phân biệt khuôn mặt thật với ảnh/video giả mạo theo model hiện tại. |
| Hàng đợi có áp lực ngược | Bounded queue / backpressure | Queue hữu hạn bỏ frame cũ nhất khi đầy để ưu tiên dữ liệu mới. |
| Khối nguyên khối theo module | Modular monolith | Một backend deploy unit có ranh giới package theo domain. |
| Kho quan hệ | Relational store | PostgreSQL lưu employee, shift settings và attendance log. |
| Kho vector | Vector store | Qdrant lưu/tìm embedding và payload danh tính. |
| Backend cho frontend | Backend for Frontend (BFF) | Route Next.js server-side allowlist write và gắn API key. |
| Đối soát | Reconciliation | So ID PostgreSQL với payload Qdrant để phát hiện missing/orphan. |
| Cửa sổ lỗi | Failure window | Khoảng giữa hai thao tác ghi độc lập có thể để lại lệch hai datastore. |
| Vòng đời ứng dụng | Application lifecycle | Trình tự startup, phục vụ, cancel task và đóng datastore client. |

## 9.6. Source map theo chủ đề

| Chủ đề cần kiểm chứng | Chương liên quan | Source/config/test ưu tiên |
|---|---|---|
| Composition, CORS, startup/shutdown | 2, 3, 7 | `src/app.py`, `tests/test_app.py` |
| Employee CRUD và enrollment | 1, 3, 5, 6 | `src/modules/employees/`, `src/modules/recognition/extractor.py`, `tests/modules/employees/` |
| Shift và quy tắc attendance | 1, 3, 5 | `src/modules/attendance/service.py`, `schemas.py`, `models.py`, `tests/modules/attendance/` |
| Analytics và báo cáo Excel | 3, 5, 6 | `src/modules/attendance/api.py`, `service.py`, `tests/modules/attendance/test_report.py` |
| WebSocket và queue | 2, 3, 5, 7 | `src/modules/recognition/ws_ingress.py`, `src/platform/queue.py`, `src/platform/realtime/`, các test recognition/platform |
| Detection, embedding và định danh | 4 | `src/platform/ml/face_app.py`, `src/modules/recognition/extractor.py`, `identifier.py`, test identifier |
| Liveness | 4, 7, 8 | `src/modules/antispoofing/service.py`, `tests/modules/antispoofing/` |
| PostgreSQL schema | 5 | `initdb/init.sql`, `src/modules/*/models.py`, `src/platform/db/session.py` |
| Qdrant schema và drift | 4, 5, 8 | `src/platform/db/qdrant.py`, employee service, `scripts/reconcile_vectors.py`, test script |
| Dashboard | 1, 3, 6 | `frontend/src/app/(dashboard)/`, `frontend/src/components/dashboard/`, `frontend/src/lib/api.ts` |
| Kiosk capture/state/TTS | 3, 4, 6 | `frontend/src/components/kiosk/`, `frontend/src/lib/kiosk.ts`, pure-logic tests, `src/modules/tts/` |
| BFF và secret boundary | 2, 5, 6, 7 | `frontend/src/app/api/write/[...path]/route.ts`, `src/platform/auth.py`, auth tests, hai file `.env.example` |
| Deployment và dependency | 1, 2, 7 | `compose.yaml`, `pyproject.toml`, `frontend/package.json`, lockfiles |
| Roadmap và giới hạn đo lường | 7, 8 | Code/test hiện hành và khoảng trống quan sát được; không suy diễn từ artifact ngoài Git |

`scripts/load_test.py` không có trong tree được Git track của nhánh khảo sát. Một file cùng tên chỉ tồn tại dạng untracked ở working copy gốc tại thời điểm bàn giao, nên báo cáo chỉ xem nó là gợi ý harness cần review/version hóa, không xem nội dung hoặc kết quả của nó là bằng chứng repository.

## 9.7. Ánh xạ sang báo cáo Microsoft Word

| Markdown | Chương/mục Word gợi ý | Thành phần nên chuyển |
|---|---|---|
| `README.md` | Trang đầu, lời dẫn và mục lục | Mục đích, đối tượng đọc, phạm vi khảo sát, quy ước trạng thái. |
| `01-tong-quan-du-an.md` | Chương 1 — Giới thiệu | Bối cảnh, mục tiêu, phạm vi, tác nhân, yêu cầu và công nghệ. |
| `02-kien-truc-he-thong.md` | Chương 2 — Kiến trúc | System context, deployment, module và so sánh kiến trúc. |
| `03-luong-hoat-dong.md` | Chương 3 — Phân tích luồng | Startup/shutdown, enrollment, recognition, quản trị và TTS. |
| `04-xu-ly-ai-va-nhan-dien.md` | Chương 4 — Phương pháp AI | InsightFace, embedding, cosine, liveness và nhánh kết quả. |
| `05-du-lieu-va-api.md` | Chương 5 — Thiết kế dữ liệu/API | ERD, Qdrant, nhất quán hai kho, REST/BFF/WebSocket. |
| `06-frontend-dashboard-va-kiosk.md` | Chương 6 — Thiết kế giao diện | App Router, data access, dashboard và state machine kiosk. |
| `07-ha-tang-bao-mat-hieu-nang-kiem-thu.md` | Chương 7 — Triển khai và kiểm thử | Hạ tầng, threat model, concurrency, hiệu năng và test strategy. |
| `08-danh-gia-va-huong-phat-trien.md` | Chương 8 — Đánh giá và kết luận | Điểm mạnh, giới hạn, roadmap và kiến trúc mục tiêu có nhãn đề xuất. |
| `09-phu-luc-tra-cuu.md` | Phụ lục A–F | Cây source, endpoint, môi trường, hằng số, glossary và source map. |

Khi chuyển Mermaid sang Word, xuất SVG/PNG và đánh số hình theo chương; giữ đoạn giải thích ngay sau hình. Bảng endpoint, biến môi trường và source map phù hợp đặt ở phụ lục để thân báo cáo không bị ngắt bởi chi tiết tra cứu. Đường dẫn source dùng kiểu monospace và có thể chuyển thành chú thích cuối trang hoặc hyperlink đến repository nội bộ.

## 9.8. Kết luận bộ tài liệu

Phụ lục gom các điểm tra cứu dễ thay đổi nhất nhưng luôn chỉ về source chuẩn, giúp người đọc phân biệt hợp đồng hiện hành với đề xuất. Khi code đổi endpoint, biến môi trường, model, ngưỡng, concurrency hoặc BFF allowlist, cần cập nhật bảng tương ứng và rà lại các chương được liệt kê trong source map trước khi phát hành lại báo cáo.

---

[Về mục lục](README.md) · [Trước: Chương 8 — Đánh giá và hướng phát triển](08-danh-gia-va-huong-phat-trien.md) · [Tiếp: Mục lục và lộ trình đọc](README.md)
