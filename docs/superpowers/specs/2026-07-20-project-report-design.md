# Thiết kế bộ tài liệu tổng hợp dự án FaceVec Attend

## Trạng thái

Đã được người dùng phê duyệt ngày 2026-07-20.

## 1. Mục tiêu

Tạo một bộ tài liệu Markdown tiếng Việt mô tả toàn diện dự án FaceVec Attend, phục vụ đồng thời hai nhu cầu:

- Làm nguồn nội dung có văn phong phù hợp để biên tập thành báo cáo Word, đồ án hoặc khóa luận.
- Làm tài liệu kỹ thuật để đọc, tra cứu, bảo trì và phát triển hệ thống.

Tài liệu phải phản ánh code trên nhánh hiện tại, không sao chép máy móc các tài liệu cũ đã lỗi thời. Những nội dung chưa được triển khai phải được ghi rõ là giới hạn hoặc hướng phát triển.

## 2. Đối tượng đọc

- Giảng viên, hội đồng hoặc người đọc báo cáo cần hiểu bài toán, giải pháp và kết quả.
- Lập trình viên cần hiểu kiến trúc, luồng dữ liệu, API và chi tiết triển khai.
- Người vận hành cần biết cách cấu hình, triển khai, kiểm thử và xử lý các rủi ro thường gặp.

## 3. Nguyên tắc nội dung

1. Code và cấu hình hiện tại là nguồn sự thật chính.
2. Mỗi nhận định quan trọng được phân biệt theo ba trạng thái: **đã triển khai**, **giới hạn hiện tại**, hoặc **đề xuất tương lai**.
3. Không gọi backend hiện tại là kiến trúc microservice. Hệ thống được mô tả đúng là FastAPI modular monolith, kết hợp với frontend Next.js và các dịch vụ lưu trữ triển khai độc lập.
4. Phần microservice sẽ phân tích ranh giới module hiện tại, khả năng tách dịch vụ và điều kiện cần thiết để việc tách có giá trị.
5. Ảnh khuôn mặt không được mô tả là dữ liệu lưu trữ lâu dài; hệ thống chỉ lưu embedding trong Qdrant theo code hiện tại.
6. Mermaid được dùng cho sơ đồ để người đọc có thể xem trực tiếp trong Markdown hoặc xuất thành ảnh khi đưa vào Word.
7. Không đưa secret thật, giá trị trong `.env` cá nhân hoặc dữ liệu nhạy cảm vào tài liệu.

## 4. Cách tổ chức

Bộ tài liệu được đặt tại `docs/project-report/` và chia theo chương. Cấu trúc này cân bằng khả năng đọc tuần tự như báo cáo với khả năng tra cứu kỹ thuật theo chủ đề.

### `README.md`

- Mục lục tổng thể.
- Mô tả đối tượng đọc và cách sử dụng.
- Hướng dẫn xem sơ đồ Mermaid và đưa nội dung vào Word.
- Phạm vi khảo sát, ngày khảo sát và quy ước trạng thái nội dung.

### `01-tong-quan-du-an.md`

- Bối cảnh và vấn đề cần giải quyết.
- Mục tiêu, phạm vi và đối tượng sử dụng.
- Yêu cầu chức năng và phi chức năng suy ra từ hệ thống.
- Công nghệ chính và lý do phù hợp với bài toán.
- Các chức năng đang có trên dashboard và kiosk.

### `02-kien-truc-he-thong.md`

- System context và deployment view.
- Kiến trúc Next.js, FastAPI, PostgreSQL và Qdrant.
- Ranh giới giữa `platform` và các domain module.
- Phân tích modular monolith so với microservice.
- Sơ đồ hiện trạng và phương án tách dịch vụ trong tương lai.

### `03-luong-hoat-dong.md`

- Khởi động và dừng hệ thống.
- Đăng ký nhân viên bằng nhiều frame.
- Nhận diện và chấm công realtime.
- Quản lý, tìm kiếm và xóa nhân viên.
- Cấu hình ca làm việc.
- Tổng hợp dashboard và xuất Excel.
- Phát lời chào TTS.
- Luồng lỗi, retry và phản hồi trạng thái cho kiosk.

### `04-xu-ly-ai-va-nhan-dien.md`

- Face detection và embedding bằng InsightFace `buffalo_sc`.
- Vector 512 chiều và chuẩn hóa embedding.
- Cosine similarity, ngưỡng nhận diện và truy vấn Qdrant.
- Chọn khuôn mặt lớn nhất trong frame.
- Multi-frame enrollment.
- Anti-spoofing MiniFASNet ONNX và fallback phát triển.
- MediaPipe phía trình duyệt cho tracking và proximity gating.
- Phân biệt vai trò của mô hình trình duyệt và mô hình backend.
- Các yếu tố ảnh hưởng độ chính xác và cách đánh giá thực nghiệm.

### `05-du-lieu-va-api.md`

- ERD PostgreSQL.
- Cấu trúc collection Qdrant và payload.
- Hợp đồng REST API.
- Hợp đồng WebSocket và các trạng thái phản hồi.
- Ràng buộc dữ liệu chấm công.
- Transaction, thứ tự ghi/xóa giữa PostgreSQL và Qdrant.
- Drift hai kho và script reconciliation.

### `06-frontend-dashboard-va-kiosk.md`

- Next.js App Router và cấu trúc route.
- Dashboard thống kê, quản lý nhân viên và cấu hình ca.
- React Query, Axios, chuẩn hóa dữ liệu và cache invalidation.
- BFF write proxy.
- Kiosk state machine.
- Camera lifecycle, capture interval và WebSocket reconnect.
- Face tracking, hysteresis và face-loss grace period.
- Luồng đăng ký kiosk với countdown và burst capture.
- TTS và trải nghiệm phản hồi cho người dùng.

### `07-ha-tang-bao-mat-hieu-nang-kiem-thu.md`

- Docker Compose, uv, npm và các biến môi trường.
- Vòng đời FastAPI và khởi tạo Qdrant collection.
- Queue giới hạn, drop-oldest, semaphore và thread pool.
- Backpressure, graceful shutdown và khả năng chịu nhiều camera.
- API key, BFF, CSRF guard, CORS và cô lập cổng dữ liệu.
- Các thiếu hụt: login/RBAC, rate limiting, TLS, audit và retention.
- Chiến lược test backend/frontend và load test WebSocket.
- Các chỉ số nên thu thập khi đánh giá hiệu năng.

### `08-danh-gia-va-huong-phat-trien.md`

- Ưu điểm kỹ thuật và nghiệp vụ.
- Hạn chế, rủi ro và nợ kỹ thuật.
- Khả năng mở rộng theo số nhân viên và số kiosk.
- Lộ trình liveness, auth/RBAC, observability, GPU worker và message broker.
- Tiêu chí quyết định khi nào nên tách microservice.

### `09-phu-luc-tra-cuu.md`

- Cây thư mục có chú giải.
- Bảng endpoint cô đọng.
- Bảng biến môi trường không chứa secret.
- Thuật ngữ Việt–Anh.
- Bảng đối chiếu chương tài liệu với source code liên quan.
- Gợi ý bố cục chương khi chuyển sang Word.

## 5. Danh sách sơ đồ

Tài liệu sẽ chứa tối thiểu các sơ đồ Mermaid sau:

1. System context.
2. Deployment/component architecture.
3. Kiến trúc module backend.
4. Sequence đăng ký nhân viên nhiều frame.
5. Sequence nhận diện và ghi chấm công realtime.
6. Flowchart pipeline AI.
7. ERD PostgreSQL và liên hệ logic với Qdrant.
8. State machine kiosk.
9. Luồng BFF và API key.
10. Vòng đời startup/shutdown.
11. Kiến trúc hiện tại và phương án tách microservice tương lai.

Sơ đồ phải dùng nhãn ngắn, không chứa cú pháp dễ làm Mermaid parser lỗi, và có đoạn văn giải thích ngay bên dưới.

## 6. Nguồn kỹ thuật bắt buộc đối chiếu

- `src/app.py` và `main.py` cho app lifecycle và router wiring.
- `src/platform/` cho auth, queue, DB client, ML singleton và WebSocket manager.
- `src/modules/` cho employees, attendance, recognition, anti-spoofing và TTS.
- `frontend/src/app/`, `frontend/src/components/` và `frontend/src/lib/` cho dashboard, kiosk và BFF.
- `initdb/init.sql`, `compose.yaml`, `.env.example`, `frontend/.env.example`, `pyproject.toml` và `frontend/package.json`.
- `tests/` và `scripts/load_test.py` cho kiểm thử và đánh giá tải.
- Các tài liệu cũ trong `docs/` chỉ dùng làm nguồn tham khảo, không được ưu tiên hơn code hiện tại.

## 7. Tiêu chí hoàn thành

- Có đủ 10 file theo cấu trúc đã duyệt.
- Tất cả liên kết nội bộ giữa các file hoạt động.
- Không còn `TODO`, `TBD`, placeholder hoặc khẳng định mơ hồ.
- API và schema phản ánh code hiện tại, bao gồm dashboard thống kê, báo cáo Excel, kiosk, multi-frame enrollment, anti-spoofing và TTS.
- Có giải thích chính xác về modular monolith và microservice.
- Các flow chính có cả mô tả bằng lời và sơ đồ.
- Người đọc có thể dùng riêng phần học thuật mà không cần đọc source code.
- Người phát triển có thể lần từ tài liệu tới file source tương ứng.
- Tài liệu không chứa secret hoặc dữ liệu cá nhân.

## 8. Ngoài phạm vi

- Không thay đổi code ứng dụng, API, schema hoặc hành vi runtime.
- Không tự nhận hệ thống đã đạt độ chính xác hoặc hiệu năng production nếu repository không có số đo chứng minh.
- Không tạo số liệu benchmark mới nếu không chạy kiểm thử thực nghiệm phù hợp.
- Không biến các đề xuất tương lai thành mô tả của chức năng hiện tại.
