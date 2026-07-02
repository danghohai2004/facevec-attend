# Security follow-ups

Debt còn lại sau Task 2.1 (secure write endpoints, Option C BFF). Cập nhật 2026-07-01.

Task 2.1 **chỉ** bảo vệ backend write khỏi client mạng tuỳ ý (curl :8000) và giữ API secret khỏi browser. Nó **không** phải user-auth. Các mục dưới là những gì còn thiếu.

## Cao — chưa có task cụ thể

- **User authentication cho write actions.** Sau Option C, bất kỳ ai mở được frontend Next vẫn kích hoạt được write (proxy tự inject key). Chưa có login/session/RBAC → không phân biệt được người dùng, không audit theo người. Cần task riêng (Phase 3). *Đây là giới hạn cố ý đã ghi trong plan Task 2.1 threat model.*
- **API_KEY strength & rotation.** `API_KEY` phải là chuỗi random đủ dài (≥32 bytes), không commit, và cùng giá trị ở backend + Next server env. Chưa có quy trình xoay key. Rủi ro nếu leak: mở toàn bộ write.
- **Rate limiting.** Không có throttle trên write endpoints hay WebSocket ingress → brute-force key / DoS frame flood. Cân nhắc slowapi (backend) hoặc giới hạn ở reverse proxy.
- **TLS / transport encryption.** Traffic browser→proxy→backend, backend→Postgres, backend→Qdrant hiện plaintext. Prod cần HTTPS ở edge và mã hoá kênh nội bộ (hoặc mạng tin cậy).

## Đã có task trong `implementation_plan_for_gpt55.md`

- **Anti-spoofing / liveness thật** — Task 2.2 chỉ chặn boot prod khi còn PassThrough; liveness thật ở Phase 3. **Lưu ý:** guard là *opt-in* — chỉ kích hoạt khi `ENV=production` (đã chuẩn hoá hoa/thường). Nếu prod quên set `ENV`, guard im lặng và liveness giả vẫn chạy. Cân nhắc chuyển sang default-deny (chỉ cho PassThrough khi `ENV` rõ ràng là dev) nếu muốn secure-by-default.
- **Internal error leakage** — Task 2.6 (ngừng trả chuỗi exception ra client).
- ~~**Network isolation + Qdrant auth + pin image**~~ **[Đã xử lý trong Task 2.8]**: Qdrant + Postgres publish port bind `127.0.0.1` (không lộ ra network), Qdrant image pin `v1.18.2`, và bật `QDRANT__SERVICE__API_KEY` (compose fail nếu thiếu key, client gửi key). *Còn lại:* key đi qua HTTP loopback plaintext → TLS vẫn ở mục dưới; `QDRANT_API_KEY` là secret 2-copy (compose env + backend env) phải cùng giá trị.
- **npm audit advisories** — Task 2.11.
- **Attendance retention vs cascade delete** — Phase 3 (xoá employee đang xoá luôn lịch sử chấm công).

## Ghi chú nhỏ

- **Upload size check sau khi read hết body** (Task 1.5): cap 5MB kiểm tra *sau* `await file.read()` → body lớn vẫn buffer trước khi 413. Nếu cần siết, check `Content-Length`/`file.size` trước khi đọc.
- **CORS**: hiện `allow_origin_regex` cho localhost. Prod phải khai báo origin thật, không để lỏng.
- ~~**Register race → 500 thay vì 409** (`register_employee`)~~ **[Đã xử lý trong Task 2.6]**: check-then-insert không atomic; giờ bắt `IntegrityError` → `ERR_DUPLICATE` (api map sang 409), và exception khác đã ngừng leak (trả `INTERNAL_ERROR`, log server-side).
- **Proxy allowlist**: nếu đổi/thêm write endpoint, phải cập nhật cả `ALLOWED_TARGETS` trong `frontend/src/app/api/write/[...path]/route.ts` **và** backend dependency — lệch nhau sẽ tạo endpoint không được bảo vệ hoặc không gọi được.
- **`API_KEY` tồn tại ở 2 nơi** (backend env + Next server env) và phải **cùng giá trị**; lệch → toàn bộ write trả 401 khó chẩn đoán. Cân nhắc nạp từ một secret store chung.
- **CSRF trên proxy write**: đã thêm same-origin guard (chặn khi `Origin` khác host của app) trong `route.ts`. Guard chỉ chặn request có `Origin` cross-site; client không-browser (server-to-server) vẫn dựa vào API key ở backend. Sau reverse proxy phải bảo đảm header `Host` phản ánh đúng host công khai để guard không chặn nhầm.
