# Workflow 4: Cập nhật bối cảnh (Context) của Agent sau mỗi Task

Sau khi hoàn thành bất kỳ task nào, AI agent bắt buộc phải cập nhật bối cảnh của project.

## Các cập nhật bắt buộc

Agent phải cập nhật các file sau khi có liên quan:

1. `.agents/TASK_STATUS.md`
2. `description_for_AI_working/ai_generation_log.md`

## Nội dung cần cập nhật trong `.agents/TASK_STATUS.md`

Agent phải ghi lại:

- Tên task
- Ngày tháng
- Agent được sử dụng: Codex, Gemini, Antigravity, Claude, Cursor, v.v.
- Các file đã thay đổi
- Các phần đã hoàn thành
- Các phần còn thiếu/chưa làm
- Các vấn đề phát hiện được
- Bước tiếp theo được đề xuất

## Nội dung cần cập nhật trong `ai_generation_log.md`

Agent phải ghi lại:

- Prompt đã sử dụng
- Công cụ AI đã sử dụng
- Các file đã thay đổi
- Tóm tắt nội dung được tạo hoặc chỉnh sửa
- Kết quả kiểm tra (testing result)
- Các vấn đề còn tồn đọng (remaining issues)

## Các quy tắc quan trọng

- Không được ghi đè (overwrite) lên các log trước đó.
- Luôn luôn nối thêm (append) các bản cập nhật mới vào cuối file.
- Không được xóa lịch sử task trước đó.
- Nếu task chưa hoàn thành, phải đánh dấu rõ ràng là chưa hoàn thành.
- Nếu có thay đổi code, phải ghi rõ code đó đã được test hay chưa.
- Nếu có thay đổi ở các cell trong notebook, phải ghi rõ phần nào (section) đã bị thay đổi.
