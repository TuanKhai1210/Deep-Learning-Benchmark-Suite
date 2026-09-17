---
layout: article
title: Handbook revision - 14 September 2026
---

# Handbook v2: cập nhật ngày 14/09/2026

Nguồn: `handbook-ene-v2.pdf`, Revision 14 September 2026, đối chiếu với `handbook-ene.pdf`. Tham chiếu bên dưới là số mục trong đề. Đây là ghi nhận yêu cầu, không xác nhận nhóm đã đăng ký hoặc nộp bài. Theo yêu cầu của nhóm, lần cập nhật này không đánh giá thời gian đăng ký nhóm.

## 1. Những gì thay đổi

| Nội dung | Quy định v2 | Việc nhóm cần làm |
|---|---|---|
| Đăng ký nhóm (1.1-1.2) | Dùng Google workbook chính thức, không đăng ký trên LMS; đăng nhập bằng email `@hcmut.edu.vn` | Mỗi người tự kiểm tra dòng MSSV của mình |
| `GroupRegistration` (1.2) | Chỉ điền cột `Group Name` tại dòng MSSV có sẵn; cả nhóm dùng tên giống hệt, kể cả chữ hoa/thường | Không thêm dòng, không sửa tên/MSSV/email của người khác |
| `GroupLink` (1.2) | Một đại diện điền URL landing page của cả project | Nếu URL thay đổi, sửa cùng ô; không tạo bản ghi nhóm thứ hai |
| Final report (4.1, 7.1) | Đại diện nộp PDF lên LMS; tên file có Group Name chính xác và số assignment | Dùng mẫu tên file bên dưới |
| Draft (7.1) | Có thể liên kết từ landing page; quy định PDF LMS nói trên áp dụng cho Final | Giữ hạn Draft, kiểm tra thông báo riêng trên LMS |

[Workbook đăng ký chính thức](https://docs.google.com/spreadsheets/d/11kdDATnpZLvYbC6Sc47cecwZjxFSUGtx/edit?usp=drive_link&ouid=102981698505363655024&rtpof=true&sd=true) và [LMS nộp bài](https://lms.hcmut.edu.vn/course/view.php?id=142848) được chép từ đề; chưa kiểm tra trạng thái đăng nhập hay bản ghi của nhóm.

Landing page cần dẫn tới A1/A2/A3 và các deliverables khi có: code, report, slides, YouTube video, checkpoints hoặc hướng dẫn tái tạo, AI disclosure. Link website không thay thế việc nộp Final PDF trên LMS.

Tên Final PDF bắt buộc:

```text
<GroupName>_A1_Report.pdf
<GroupName>_A2_Report.pdf
<GroupName>_A3_Report.pdf
```

`<GroupName>` phải khớp chính xác tên trong `GroupRegistration`; không mặc định lấy repository name hoặc tự đặt tên mới. Group Name và đại diện nộp bài hiện chờ nhóm xác nhận.

## 2. Những gì không đổi

- Deadline các assignment, trọng số milestone, late policy và địa chỉ LMS nộp file không đổi.
- A1 Draft: 23/09/2026, 23:59 GMT+7, 25% A1; EDA, Dataset/DataLoader, train/validation loop, Linear và MLP chạy được. CNN optional ở Draft.
- A1 Final: 21/10/2026, 23:59 GMT+7, 75% A1; đủ Linear, MLP, custom CNN, LSTM hoặc GRU, Transformer và toàn bộ deliverables.
- Fashion-MNIST vẫn là dataset chính; MNIST chỉ development/debug, CIFAR-10 optional.
- Metrics, fairness, representation/inductive-bias analysis và yêu cầu tái lập không đổi. Phần kỹ thuật A2/A3 không có thay đổi nội dung trong đối chiếu này; một số số mục tham chiếu được cập nhật.
- Split seed `36`, run seeds `[69420, 67, 69]`, kế hoạch 15 main runs và phân công Thiên/Khoa/Khải là quyết định nội bộ, không phải giá trị do đề chỉ định. Giữ nguyên protocol draft `a1-v0`; cập nhật hành chính này không yêu cầu đổi code hoặc chạy lại thí nghiệm.

Thông báo LMS cụ thể cho milestone được ưu tiên nếu khác handbook. Không truy cập hoặc chỉnh sửa workbook/LMS trong lần cập nhật tài liệu này.

## 3. Checklist và người thực hiện

- [ ] Cả nhóm xác nhận một Group Name chính xác và một đại diện cho GroupLink/nộp Final; không tự coi Khải là đại diện vì phụ trách Pages.
- [ ] Thiên (2453194), Khoa (2452539), Khải (2452515): mỗi người tự kiểm tra Group Name tại dòng MSSV của mình bằng email trường.
- [ ] Đại diện đã chọn: kiểm tra GroupLink trỏ tới landing page công khai của project, không chỉ repo hoặc riêng trang A1.
- [ ] Khải tích hợp nội dung Pages; Thiên review: Group Name khớp đăng ký và các link deliverable hoạt động khi xuất bản.
- [ ] Trước từng Final, đại diện kiểm tra tên PDF, nộp LMS và lưu biên nhận; một thành viên khác kiểm tra lại.
