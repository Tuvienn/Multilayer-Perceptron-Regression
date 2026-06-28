# Project Rules

Tài liệu này dựa trên `working_rule.md`. Mọi xung đột (nếu có) sẽ ưu tiên theo `working_rule.md`.

## 1. Quy tắc chung (General Rules)
- **Clarify First:** Làm rõ yêu cầu trước khi hành động. Không tự assume context.
- **Confirm Before Update:** Xin phép và chờ xác nhận trước khi thay đổi kiến trúc hoặc logic lớn.
- **Evaluation:** Đánh giá kết quả (risk, side-effects) sau mỗi task.

## 2. Quy tắc viết code (Coding Rules)
- Tuân thủ architecture và coding style hiện hành. Không tự refactor diện rộng.
- Naming convention: Python dùng `snake_case` cho variables/functions/files, `PascalCase` cho Classes.
- Không sửa "stable code" nếu task không yêu cầu.
- Luôn kiểm tra safety & security trước khi commit thay đổi.

## 3. Quy tắc cho project Machine Learning
- **Baseline First:** Luôn bắt đầu với mô hình cơ bản (ví dụ: Linear Regression/Random Forest) trước khi dùng mô hình phức tạp.
- **Metric Before Model:** Thống nhất evaluation metric trước khi train/build mô hình.
- **Explainability:** Ưu tiên mô hình có khả năng giải thích (explainable) hoặc phải nêu rõ trade-off nếu dùng black-box model.

## 4. Quy tắc cho notebook (Notebook Rules)
- Chạy các cell theo thứ tự từ trên xuống dưới (top-to-bottom execution) để đảm bảo state reproducibility.
- Xóa output hoặc dùng format thích hợp để tránh làm file .ipynb quá lớn trước khi commit (tùy vào convention cụ thể, nếu có).
- Ghi chú rõ ràng mục đích của từng cell (Markdown cells) để giải thích quá trình EDA hoặc Training.
- Restart kernel và run all để xác minh notebook chạy không lỗi.

## 5. Quy tắc bảo mật (Security Rules)
- Tuyệt đối không hardcode credentials, API keys, password.
- Không expose thông tin nhạy cảm qua log hay API response.
- Dùng environment variables cho mọi secret.

## 6. Quy tắc phòng tránh data leakage (Data Leakage Prevention)
- Tách biệt rõ ràng tập Train, Validation và Test **trước khi** thực hiện bất kỳ bước Data Preprocessing hay Feature Engineering nào.
- Chỉ fit (transformations, scalers, imputers) trên tập Train, sau đó transform cho tập Validation/Test.
- Cẩn thận với time-series data: không dùng data tương lai để predict quá khứ, chia split theo thời gian.
- Tránh dùng Target Variable trong Feature Engineering mà không có cơ chế chống leakage (ví dụ: cross-validation target encoding).

## 7. Quy tắc cập nhật tiến độ (Progress & Context Update)
- Sau khi hoàn thành task, agent phải cập nhật `.agents/TASK_STATUS.md`.
- Agent cũng phải cập nhật `description_for_AI_working/ai_generation_log.md`.
- Agent phải ghi rõ task đã làm, file đã sửa, phần đã hoàn thành, phần còn thiếu, kết quả kiểm tra và bước tiếp theo.
- Agent phải append log, tuyệt đối không được xóa lịch sử cũ.
- Không được đánh dấu hoàn thành nếu task chưa được test hoặc chưa thực hiện xong.
