# Chương 1. Tổng quan dự án

## 1.1. Bối cảnh và bài toán

Chấm công là quá trình xác nhận một nhân viên có mặt và ghi nhận thời điểm vào/ra để phục vụ quản lý ngày công. Các phương thức phổ biến đều có điểm yếu riêng:

- ghi tay tốn thời gian tổng hợp, dễ sai sót và khó kiểm tra tính trung thực;
- thẻ từ có thể bị quên, mất hoặc nhờ người khác quẹt hộ;
- thiết bị vân tay tạo điểm tiếp xúc chung, có thể chậm khi đông người và phụ thuộc chất lượng cảm biến/ngón tay;
- đối chiếu ảnh hoặc đào tạo một classifier riêng cho từng người làm tăng chi phí cập nhật khi số nhân viên thay đổi.

FaceVec Attend giải quyết bài toán bằng nhận diện khuôn mặt dựa trên **embedding**. Ảnh được xử lý thành vector đặc trưng 512 chiều đã chuẩn hóa; Qdrant tìm vector gần nhất theo cosine, còn PostgreSQL quản lý dữ liệu quan hệ. Khi thêm nhân viên mới, hệ thống thêm các embedding từ nhiều frame hợp lệ thay vì huấn luyện lại mô hình phân loại cho toàn bộ tập người dùng. Code ứng dụng không chủ ý ghi bền ảnh nguồn vào PostgreSQL hoặc Qdrant; tuy nhiên, multipart `UploadFile` của FastAPI/Starlette dùng tệp tạm dạng spooled và có thể chuyển dữ liệu upload lớn sang đĩa tạm của runtime trong lúc xử lý. Sau xử lý, artifact sinh trắc mà ứng dụng chủ ý lưu bền là embedding trong Qdrant.

## 1.2. Mục tiêu

Hệ thống hướng tới các mục tiêu sau:

1. Cung cấp chấm công tại kiosk bằng khuôn mặt với phản hồi gần thời gian thực.
2. Quản lý nhân viên, khung giờ vào/ra và lịch sử chấm công trên một admin dashboard.
3. Tách dữ liệu nghiệp vụ khỏi dữ liệu vector: PostgreSQL đảm nhiệm quan hệ và giao dịch, Qdrant đảm nhiệm tìm kiếm tương đồng.
4. Cho phép thêm người mới bằng enrollment nhiều frame mà không huấn luyện lại mô hình nhận diện.
5. Giữ bí mật ghi ở phía server bằng BFF thay vì đưa API key vào JavaScript của browser.
6. Tổ chức backend theo domain để dễ kiểm thử, bảo trì và tạo ranh giới tách hệ thống trong tương lai khi thực sự cần.

Các mục tiêu trên không đồng nghĩa với cam kết độ chính xác hoặc tải production. Việc xác lập các chỉ số đó cần bộ dữ liệu đánh giá, kịch bản tải và kết quả đo có thể tái lập.

## 1.3. Phạm vi

### Đã triển khai

- API và dịch vụ quản lý nhân viên: đăng ký nhiều ảnh/frame, danh sách, tìm kiếm, xem và xóa.
- Quản lý một bộ khung giờ check-in/check-out, có giá trị mặc định nếu chưa có bản ghi cấu hình.
- Chấm công vào/ra theo múi giờ `Asia/Ho_Chi_Minh` và theo cửa sổ ca làm.
- WebSocket nhận frame JPEG, hàng đợi có giới hạn, nhận diện khuôn mặt, kiểm tra liveness, tìm Qdrant và ghi nhật ký chấm công.
- Thống kê tổng quan, theo tháng, theo ngày và xuất báo cáo Excel hai sheet.
- Admin dashboard cho quản lý và analytics; các card/biểu đồ thống kê gọi API thật `/attendance/summary`, `/attendance/monthly` và `/attendance/daily`.
- Kiosk tại `/kiosk` đã được wire: xin quyền camera, gửi frame qua `/ws/recognition/{client_id}`, tự kết nối lại, hiển thị kết quả và gọi TTS tiếng Việt.
- Kiosk enrollment nhận deep-link từ luồng quản trị, chụp nhiều frame và gửi đăng ký qua BFF.
- TTS offline phía backend bằng Piper để các kiosk có giọng phản hồi nhất quán.
- API key bảo vệ các endpoint ghi được đánh dấu; BFF chỉ chuyển tiếp các đích ghi nằm trong allowlist.

### Giới hạn hiện tại

- Chưa có đăng nhập người dùng, RBAC và audit trail đầy đủ.
- Một số endpoint đọc, WebSocket recognition và TTS hiện không yêu cầu xác thực.
- Nếu không tìm thấy model anti-spoofing, môi trường phát triển dùng `PassThroughChecker`; ứng dụng từ chối khởi động với checker này khi `ENV=production`.
- PostgreSQL và Qdrant là hai kho độc lập, nên các thao tác ghi chéo kho không có transaction phân tán nguyên tử.
- Compose chỉ định nghĩa PostgreSQL và Qdrant; FastAPI và Next.js vẫn là các tiến trình ứng dụng chạy bên ngoài hai container đó trong cấu hình hiện tại.
- Chưa có bằng chứng trong phạm vi khảo sát để khẳng định số kiosk đồng thời tối đa, độ chính xác production hoặc SLA.
- Backend có API manual check-in/check-out, nhưng allowlist BFF hiện không chuyển tiếp hai thao tác này từ admin dashboard.

### Ngoài phạm vi hiện trạng

Microservice, message broker, worker nhận diện độc lập, API gateway, autoscaling theo GPU và identity provider là các hướng có thể cân nhắc sau này. Chúng không phải thành phần của hệ thống đang chạy tại mốc khảo sát.

## 1.4. Tác nhân và lợi ích

| Tác nhân | Tương tác chính | Lợi ích mong đợi |
|---|---|---|
| Quản trị viên | Dùng admin dashboard để quản lý nhân viên, ca làm, thống kê và báo cáo | Giảm thao tác tổng hợp thủ công, có dữ liệu tập trung |
| Nhân viên | Đứng trước kiosk để đăng ký hoặc chấm công | Không cần mang thẻ hay chạm cảm biến dùng chung |
| Người vận hành kỹ thuật | Cấu hình ứng dụng, PostgreSQL, Qdrant, model và biến môi trường | Có ranh giới thành phần rõ để triển khai và xử lý sự cố |
| Browser kiosk/dashboard | Gọi REST/BFF, gửi frame WebSocket và phát audio | Là client kỹ thuật kết nối người dùng với backend |

## 1.5. Chức năng hiện có theo nhóm

| Nhóm | Chức năng | Thành phần chính |
|---|---|---|
| Nhân viên | Đăng ký nhiều frame, list/search/detail, xóa | `employees`, kiosk enrollment, Qdrant |
| Chấm công | Check-in/out theo ca, lịch sử theo ngày, thao tác manual ở backend | `attendance`, PostgreSQL |
| Nhận diện | Decode JPEG, chọn mặt lớn nhất, liveness, embedding, top-1 vector search | `recognition`, `antispoofing`, InsightFace, Qdrant |
| Realtime | Nhận binary frame và trả JSON status theo `client_id` | FastAPI WebSocket, `FrameQueue`, `ConnectionManager` |
| Quản trị | Danh sách nhân viên, ca làm, dashboard analytics | Next.js, Axios, React Query, Recharts |
| Báo cáo | Summary, monthly/daily stats, file `.xlsx` | `attendance`, SQLAlchemy, OpenPyXL |
| Phản hồi kiosk | Thông báo trạng thái và lời chào tiếng Việt | Kiosk state, Piper TTS |
| Bảo vệ ghi | Kiểm tra API key và proxy ghi cùng origin | FastAPI auth, Next.js BFF |

## 1.6. Yêu cầu chức năng

| Mã | Yêu cầu | Trạng thái |
|---|---|---|
| FR-01 | Quản trị viên có thể đăng ký nhân viên bằng mã, tên và nhiều frame khuôn mặt hợp lệ | Đã triển khai |
| FR-02 | Hệ thống lưu thông tin nhân viên ở PostgreSQL và embedding ở Qdrant | Đã triển khai |
| FR-03 | Quản trị viên có thể xem, tìm kiếm và xóa nhân viên | Đã triển khai |
| FR-04 | Quản trị viên có thể xem và cập nhật cửa sổ check-in/check-out | Đã triển khai |
| FR-05 | Kiosk có thể thu camera, gửi frame và nhận trạng thái nhận diện theo thời gian thực | Đã triển khai |
| FR-06 | Hệ thống chỉ ghi check-in/check-out phù hợp với cửa sổ ca và dữ liệu trong ngày làm việc | Đã triển khai |
| FR-07 | Dashboard hiển thị summary, xu hướng tháng, trung bình theo ngày từ API thật | Đã triển khai |
| FR-08 | Hệ thống xuất báo cáo tháng gồm sheet tổng hợp và chi tiết | Đã triển khai |
| FR-09 | Kiosk phát lời chào/nhắc trạng thái bằng TTS khi nhận diện thành công | Đã triển khai |
| FR-10 | Hệ thống phân biệt các kết quả không có mặt, spoof, không xác định, nhận diện được và lỗi | Đã triển khai |
| FR-11 | Người dùng được phân quyền theo vai trò và mọi thao tác quản trị được audit | Đề xuất tương lai |

## 1.7. Yêu cầu phi chức năng

| Mã | Yêu cầu | Cách hệ thống hiện đáp ứng hoặc giới hạn |
|---|---|---|
| NFR-01 — Độ trễ realtime | Ưu tiên frame mới và không để backlog tăng vô hạn | Queue tối đa 50, đầy thì drop-oldest; pipeline giới hạn bốn tác vụ xử lý đồng thời |
| NFR-02 — Khả năng mở rộng dữ liệu | Không huấn luyện lại classifier khi thêm người | Enrollment upsert embedding mới; Qdrant thực hiện vector search cosine |
| NFR-03 — Tính nhất quán | Giữ quan hệ nhân viên/chấm công hợp lệ | PostgreSQL dùng khóa ngoại và transaction; đồng bộ PostgreSQL–Qdrant vẫn có failure window |
| NFR-04 — Bảo mật secret | Không để API key ghi xuất hiện ở browser | BFF server-side gắn `X-API-Key`; backend fail-closed nếu thiếu cấu hình key |
| NFR-05 — Riêng tư | Hạn chế dữ liệu sinh trắc không cần thiết | Ứng dụng không chủ ý persist ảnh nguồn vào PostgreSQL/Qdrant và chỉ lưu bền embedding trong Qdrant; multipart buffering vẫn có thể dùng đĩa tạm runtime, nên cần kiểm soát cả vòng đời dữ liệu tạm lẫn embedding |
| NFR-06 — Khả dụng | Client kiosk tự phục hồi khi socket mất | Kiosk tự kết nối lại; chưa có SLA, HA hoặc health orchestration đầy đủ |
| NFR-07 — Bảo trì | Tách nghiệp vụ theo domain và hạ tầng dùng chung | Backend modular monolith với năm domain và `platform`; hợp đồng nội bộ vẫn là lời gọi Python trực tiếp |
| NFR-08 — Khả chuyển | Chạy được trên stack phổ biến và có fallback CPU | InsightFace cấu hình CUDA rồi CPU; PostgreSQL/Qdrant chạy bằng Compose |
| NFR-09 — Kiểm chứng | Có thể đo và kiểm thử trước khi tuyên bố chất lượng | Repository có test/tooling liên quan, nhưng không suy ra chỉ số production nếu chưa chạy benchmark phù hợp |

## 1.8. Công nghệ

| Lớp | Công nghệ theo source hiện tại | Vai trò |
|---|---|---|
| Backend | Python 3.11+, FastAPI, Uvicorn, Pydantic 2 | REST, WebSocket, validation và app lifecycle |
| Truy cập dữ liệu | SQLAlchemy 2 async, asyncpg | ORM và kết nối PostgreSQL bất đồng bộ |
| Dữ liệu quan hệ | PostgreSQL 16 | Nhân viên, ca làm, nhật ký chấm công |
| Dữ liệu vector | Qdrant 1.18.2, `qdrant-client` | Collection embedding 512 chiều và cosine search |
| Computer vision | InsightFace, OpenCV | Phát hiện mặt và trích embedding chuẩn hóa |
| Liveness | MiniFASNet ONNX hoặc `PassThroughChecker` có guard production | Phân loại khuôn mặt thật/giả trước nhận diện |
| TTS | Piper TTS | Sinh WAV tiếng Việt offline ở backend |
| Frontend | Next.js 16.2.6, React 19.2.4, TypeScript | Admin dashboard, BFF và kiosk |
| Client data/UI | Axios, TanStack React Query, Recharts, MediaPipe Tasks Vision và Media Capture API của browser | REST cache, biểu đồ, camera và proximity tracking |
| Báo cáo | OpenPyXL | Sinh workbook Excel |
| Hạ tầng phát triển | Docker Compose, `uv`, npm | Chạy kho dữ liệu và quản lý dependency/build |

## 1.9. Kết luận chương

FaceVec Attend là một hệ thống chấm công embedding-based đã có đầy đủ đường đi từ browser camera đến WebSocket recognition pipeline, hai kho dữ liệu và dashboard quản trị. Giá trị chính của kiến trúc hiện tại là ranh giới domain rõ trong một deployment backend đơn giản. Chương tiếp theo phân tích các thành phần, kết nối và lý do gọi hệ thống là modular monolith thay vì microservice.

---

[Về mục lục](README.md) · [Tiếp: Chương 2 — Kiến trúc hệ thống](02-kien-truc-he-thong.md)
