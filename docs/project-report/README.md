# Báo cáo dự án FaceVec Attend

## Mục đích

Bộ tài liệu này mô tả FaceVec Attend theo hai góc nhìn bổ trợ nhau: một báo cáo học thuật về bài toán chấm công bằng nhận diện khuôn mặt và một tài liệu kỹ thuật có thể dùng để đọc, vận hành, kiểm chứng và tiếp tục phát triển hệ thống. Nội dung ưu tiên mã nguồn và cấu hình trên nhánh được khảo sát, không suy diễn chức năng từ ý tưởng hoặc tài liệu đã lỗi thời.

Các thuật ngữ kiến trúc được dùng thống nhất trong toàn bộ báo cáo:

- **modular monolith**: backend FastAPI chạy trong một tiến trình, nhưng mã nguồn được chia theo domain;
- **admin dashboard**: giao diện quản trị và thống kê viết bằng Next.js;
- **kiosk**: giao diện camera phục vụ đăng ký khuôn mặt và chấm công tại chỗ;
- **BFF (Backend for Frontend)**: route server-side của Next.js chuyển tiếp các thao tác ghi và gắn `X-API-Key`;
- **recognition pipeline**: chuỗi nhận frame, phát hiện/liveness, trích embedding, tìm danh tính và ghi chấm công;
- **PostgreSQL relational store**: nơi lưu nhân viên, ca làm và nhật ký chấm công;
- **Qdrant vector store**: nơi lưu và tìm kiếm embedding khuôn mặt.

## Đối tượng đọc

- Sinh viên, giảng viên và hội đồng cần đánh giá bối cảnh, phương pháp, kiến trúc và kết quả của đồ án.
- Lập trình viên cần hiểu ranh giới module, luồng dữ liệu, API, frontend và pipeline nhận diện.
- Người triển khai hoặc đánh giá an toàn cần tra cứu cấu hình, hạ tầng, giới hạn và hướng phát triển.

## Phạm vi khảo sát

Mốc khảo sát là **2026-07-20**. Trạng thái được xác định từ code và cấu hình hiện có tại mốc này. Dashboard thống kê đã gọi các API backend thật; kiosk tại `/kiosk` đã được nối với camera, WebSocket recognition pipeline, shift settings và TTS. Những mô tả cũ coi dashboard là số liệu giả hoặc kiosk chỉ thuộc giai đoạn tương lai không còn phản ánh đúng nhánh khảo sát.

Báo cáo không công bố secret, dữ liệu cá nhân, ảnh khuôn mặt hay embedding thực. Báo cáo cũng không tự khẳng định độ chính xác, công suất camera đồng thời hoặc hiệu năng production khi repository chưa cung cấp phép đo tương ứng.

## Quy ước ba trạng thái

Mỗi nhận định về chức năng hoặc kiến trúc thuộc một trong ba trạng thái sau:

| Trạng thái | Ý nghĩa | Cách nhận biết trong tài liệu |
|---|---|---|
| **Đã triển khai** | Có code hoặc cấu hình hiện hành làm bằng chứng | Mô tả ở thì hiện tại và nêu thành phần thực thi |
| **Giới hạn hiện tại** | Hành vi còn thiếu, có điều kiện hoặc chưa được chứng minh | Gắn nhãn “Giới hạn hiện tại” và không diễn đạt như chức năng hoàn chỉnh |
| **Đề xuất tương lai** | Phương án chưa phải kiến trúc đang chạy | Gắn nhãn “Đề xuất”, “có thể” hoặc “trong tương lai” |

## Mục lục chín chương

1. [Tổng quan dự án](01-tong-quan-du-an.md)
2. [Kiến trúc hệ thống](02-kien-truc-he-thong.md)
3. [Luồng hoạt động](03-luong-hoat-dong.md)
4. [Xử lý AI và nhận diện](04-xu-ly-ai-va-nhan-dien.md)
5. [Dữ liệu và API](05-du-lieu-va-api.md)
6. [Frontend dashboard và kiosk](06-frontend-dashboard-va-kiosk.md)
7. [Hạ tầng, bảo mật, hiệu năng và kiểm thử](07-ha-tang-bao-mat-hieu-nang-kiem-thu.md)
8. [Đánh giá và hướng phát triển](08-danh-gia-va-huong-phat-trien.md)
9. [Phụ lục tra cứu](09-phu-luc-tra-cuu.md)

## Lộ trình đọc

### Cho báo cáo học thuật

Đọc chương 1 để nắm bài toán, mục tiêu và phạm vi; chương 2 để hiểu lựa chọn kiến trúc; chương 3–4 để theo dõi phương pháp và pipeline nhận diện; chương 7–8 để đánh giá đặc tính vận hành, giới hạn và hướng phát triển. Chương 5–6 cung cấp bằng chứng triển khai khi cần đối chiếu sâu hơn.

### Cho kỹ thuật

Bắt đầu từ chương 2, sau đó đọc chương 5 cho hợp đồng dữ liệu/API, chương 6 cho frontend, chương 3–4 cho luồng realtime và AI, rồi chương 7 cho triển khai và kiểm thử. Dùng chương 9 như bảng tra nhanh source, endpoint, biến môi trường và thuật ngữ.

## Sử dụng sơ đồ Mermaid

Các nền tảng hỗ trợ Mermaid như GitHub/GitLab có thể render trực tiếp các khối `mermaid` trong Markdown. Với Mermaid CLI, có thể cài `@mermaid-js/mermaid-cli`, chép từng khối vào file `.mmd`, rồi xuất SVG hoặc PNG, ví dụ:

```bash
mmdc -i architecture.mmd -o architecture.svg
```

Khi đưa sơ đồ vào Microsoft Word, nên ưu tiên SVG để giữ chữ và đường nét sắc khi phóng to; nếu phiên bản Word không hỗ trợ SVG thì xuất PNG độ phân giải cao. Chèn bằng **Insert → Pictures**, đặt chú thích và số hình trong Word, đồng thời giữ phần giải thích bằng văn bản ngay dưới sơ đồ để tài liệu vẫn hiểu được khi hình không render.

---

[Bắt đầu với Chương 1](01-tong-quan-du-an.md)
