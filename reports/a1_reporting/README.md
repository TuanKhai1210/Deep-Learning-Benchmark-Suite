# A1 Final Report — Tiki-taka

Bản hoàn thiện ngày 26/10/2026, giữ mẫu HCMUT do nhóm cung cấp và viết theo ba phần của handbook. Nội dung tiếng Anh gồm dữ liệu, cả năm kiến trúc, phương pháp, kết quả, phân tích lỗi, giới hạn, tái lập, đóng góp/AI và tài liệu tham khảo.

- **Tiki-taka_A1_Report.pdf**: bản đọc hoàn chỉnh.
- **main.tex**: tài liệu chính, nhập các mục từ sections/ và bảng từ tables/.
- **report.tex**: cùng nội dung, gộp toàn bộ phần chữ/bảng vào một tệp; vẫn cần class, logo và figures/ đi kèm.
- **evidence/**: cấu hình thực tế của 15 run, số liệu, nguồn timing và hồ sơ kiểm tra.
- **reproduction/**: reproduce.py và hai notebook đánh giá/timing gốc.

## Mở và biên dịch

Upload toàn bộ gói lên Overleaf và chọn main.tex làm Main document. Dùng XeLaTeX; report.tex là lựa chọn thay thế nếu muốn sửa một tệp. Cần biên dịch đủ lượt để cập nhật mục lục và tham chiếu.

Với Tectonic 0.17.0:

    tectonic --keep-logs --keep-intermediates --untrusted main.tex

PDF đã được biên dịch bằng Tectonic, kiểm tra toàn bộ 45 trang qua ảnh và đối chiếu bảng số với dữ liệu lưu. Hồ sơ chi tiết nằm trong evidence/final_validation.json. hcmut-report.cls, hcmut.png, LICENSE.txt và TEMPLATE_SOURCE.txt được giữ nguyên từ gói đầu vào.

## Cơ sở kết quả

Kết quả dựa trên năm mô hình × ba seed và các checkpoint tốt nhất được chọn bằng validation macro-F1 ở revision a8539bee778d99e2902a1dd36aa157c6ebb3a42c. Bảng phân loại dùng CPU test đã được xác minh ngày 24/10; không huấn luyện hoặc chạy lại checkpoint để viết báo cáo này.

Inference là phép đo **Windows CPU**, hai thread, FP32, batch 64, 20 warm-up và 100 batch đo cho mỗi checkpoint. Đây là profile cpu-f7208f3550fc ngày 24/10 có log gốc. Thời gian train/validation là từ T4; số timing Colab trong bản trung gian không có raw log nên không được dùng làm số liệu cuối.

## Sửa và tạo lại tài liệu

Sửa main.tex cùng các tệp trong sections/ hoặc sửa report.tex. Hai cách là bản thay thế; các thay đổi ở một cách không tự đồng bộ sang cách còn lại.

tools/consolidate.py cập nhật report.tex từ main.tex và các tệp con. tools/write_sections.py và tools/finish_sources.py là các công cụ đã dùng để tạo bản hiện tại; chạy lại sẽ ghi lại phần chữ tương ứng. Không cần chạy các công cụ này để biên dịch PDF.

tools/build_assets.py tạo bảng/hình từ artifact hiện có, không chạy mô hình. Muốn tạo lại hình cần NumPy, matplotlib, Pillow, PyTorch, torchvision cùng các nguồn bên ngoài gói này. Truyền --train-root, --evaluation-root, --timing-root, --raw-root và --code-root theo máy đang dùng; đường dẫn mặc định ghi lại môi trường tác giả. Dataset/checkpoint lớn không được nhúng vào gói LaTeX. Quy trình tái dựng mô hình và đường dẫn artifact của nhóm được mô tả trong báo cáo.

Gói nguồn có cả bảng CSV, cấu hình và mẫu timing thô để kiểm tra số liệu; không có xác nhận nộp bài hay công bố bên ngoài.
