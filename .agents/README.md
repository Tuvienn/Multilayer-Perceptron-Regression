# Agent Instructions & Workflow

Project này sử dụng `working_rule.md` ở thư mục gốc làm file hướng dẫn chính cho tất cả các AI agent (Codex, Gemini/Antigravity, Cursor, v.v.).

Tất cả các agent **BẮT BUỘC** phải đọc và hiểu các file sau trước khi tiến hành chỉnh sửa project:

1. `working_rule.md` (Nằm ở root directory) - Đây là rules cao nhất.
2. `.agents/rules/project_rules.md` - Rules áp dụng cho codebase.
3. `.agents/workflows/1_understand_project.md` - Quy trình tiếp nhận task.
4. `.agents/workflows/2_modify_code.md` - Quy trình sửa code an toàn.
5. `.agents/workflows/3_final_review.md` - Quy trình kiểm tra trước khi hoàn thành task.
6. `.agents/workflows/4_update_context.md` - Quy trình cập nhật bối cảnh sau khi xong task.

Việc tuân thủ các tài liệu này là bắt buộc không có ngoại lệ.
