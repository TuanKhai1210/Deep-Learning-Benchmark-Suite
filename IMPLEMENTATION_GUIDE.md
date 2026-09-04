# Hướng dẫn triển khai Assignment 1

Project đang phát triển. Data pipeline, các model và thí nghiệm cần được nhóm hiện thực và kiểm chứng; chưa có kết quả benchmark. Các vị trí `TODO A/B/C` đánh dấu phần người phụ trách cần triển khai, không phải kết quả đã hoàn thành.

## 1. Thiết lập môi trường

Dùng Python 3.11 trở lên và làm theo phần [Getting started trong README](README.md#getting-started). Cài package cơ bản để kiểm tra config/CLI trước; chốt hardware và phiên bản PyTorch/torchvision trước khi cài ML dependencies. Ghi môi trường thực dùng trong `environment/`.

Config thí nghiệm dùng Python:

- `configs/a1/protocol.py`: quy tắc chung; trạng thái vẫn draft.
- `configs/a1/models/*.py`: một dictionary literal `CONFIG = {...}` với kiến trúc và thiết lập train riêng.
- `protocol_file` trong model config trỏ tới `../protocol.py`.
- Loader đọc dữ liệu bằng `ast.literal_eval`, không import hay thực thi file. Cho phép comments/docstrings; không dùng imports, lời gọi hàm, biến tham chiếu hoặc phép tính trong config.

Bốn run seed đã chốt: **36, 69420, 67, 69**. Seed mặc định để phát triển là `36`. Split seed `42` vẫn là đề xuất riêng; không chia lại dữ liệu khi đổi run seed. CLI `--seed` chỉ đổi run seed trong bản config đã load, không ghi lại file hoặc đổi split; seed ngoài bộ đã chốt bị từ chối.

## 2. Phân công theo file

Các đường dẫn bên dưới tính từ thư mục gốc repo.

| Người | File/thư mục chính | Đầu ra cần có |
|---|---|---|
| A — Nguyễn Hạo Thiên | `src/dlbench/a1/data/` | Dataset, saved split, preprocessing, DataLoader và EDA |
| A — Nguyễn Hạo Thiên | `src/dlbench/a1/models/linear.py`, `cnn.py` | Model, config, tests và method notes tương ứng |
| B — Nguyễn Anh Khoa | `src/dlbench/a1/engine.py`, `trainer.py` | Train/validation loop và điều phối full run |
| B — Nguyễn Anh Khoa | `src/dlbench/a1/metrics.py`, `checkpoint.py`, `benchmark.py` | Metrics, best-checkpoint rule, timing có định nghĩa rõ |
| B — Nguyễn Anh Khoa | `src/dlbench/a1/models/mlp.py`, `rnn.py` | MLP, một trong LSTM/GRU, configs/tests/method notes |
| C — Tạ Tuấn Khải | `src/dlbench/common/`, `src/dlbench/a1/contracts.py`, `cli.py` | Config, seeds, logging/artifacts, interface và CLI tích hợp |
| C — Tạ Tuấn Khải | `src/dlbench/a1/models/transformer.py`, `registry.py` | Transformer và chọn model từ config |
| Cả nhóm | `tests/`, `src/dlbench/a1/analysis.py`, `reports/a1/`, `docs/` | Test phần mình, phân tích có bằng chứng, nội dung báo cáo |

Thiên phụ trách nội dung EDA/error examples; Khoa kiểm tra số liệu/curves; Khải tích hợp phân tích representation và report/Pages. Khải không viết thay toàn bộ báo cáo. Review theo vòng: Khoa → Thiên, Khải → Khoa, Thiên → Khải.

## 3. Interface chung

Đọc `src/dlbench/a1/contracts.py` trước khi viết module. Giữ nguyên chữ ký hàm hiện có hoặc thống nhất thay đổi với nhóm trước khi sửa.

```text
images: float32 [B, 1, 28, 28]
labels: int64 [B], giá trị 0–9
model(images): raw logits [B, 10]
```

Linear/MLP tự flatten; RNN tự chuyển ảnh thành chuỗi; Transformer tự tạo patch/token bên trong model. Không softmax trước `CrossEntropyLoss`. Mọi model phải dùng shared trainer/evaluator, không tạo năm training loops riêng.

## 4. Thứ tự triển khai

1. **Cả nhóm:** đọc và duyệt các quyết định trong `docs/a1-experiment-contract.md`; chốt interface. Mean/std, compute budget và timing có thể được đo/xác định sau smoke test, không điền số giả để vượt validation.
2. **Thiên:** tạo split một lần, kiểm tra counts/overlap, đo mean/std từ đúng training subset; hoàn thiện DataLoader và Linear.
3. **Khoa:** xây metrics và engine trên batch nhỏ; kiểm tra gradient update, validation và save/load checkpoint; ghép MLP.
4. **Khải:** kiểm tra config/logging/run metadata và CLI; viết integration tests; chuẩn bị Transformer.
5. **Cả nhóm:** chạy Linear/MLP end-to-end cho Draft, review độc lập evaluator và split, cập nhật README bằng lệnh thực chạy được.
6. **Thiên/Khoa/Khải:** lần lượt hoàn thiện CNN/RNN/Transformer qua cùng engine, kèm shape/tiny-batch tests.
7. **Cả nhóm:** freeze protocol và model configs, xác minh strict validation, rồi chạy 20 main runs (5 model × 4 seed). Mỗi run khởi tạo lại model, dùng cùng split và giữ đủ evidence.
8. **Cả nhóm:** tổng hợp mean ± std của cả bốn seed, phân tích lỗi/representation/giới hạn và hoàn thành tài liệu nộp.

## 5. Lệnh kiểm tra phát triển

Sau khi cài package trong môi trường đang dùng:

```bash
python -m dlbench.a1.cli --help
python -m dlbench.a1.cli validate-config --config configs/a1/models/linear.py --seed 36
python -m unittest discover -s tests -v
```

`--strict` dành cho protocol đã hoàn thiện và được duyệt. Trong trạng thái draft, strict validation phải báo các quyết định còn thiếu. `prepare`, `train`, `evaluate`, `analyze` hiện chưa hoàn thiện ML; đọc `--help` để xem interface, không coi việc có tên lệnh là bằng chứng pipeline đã chạy được.

## 6. Khi nào một phần được coi là hoàn thành?

- Đúng interface và config dùng chung.
- Có tests phù hợp, không bỏ qua lỗi bằng dữ liệu/kết quả giả.
- Người khác chạy lại được bằng lệnh ghi trong PR.
- Results, nếu có, truy lại được tới config/split/seed/checkpoint/Git revision.
- Đã cập nhật tài liệu và AI usage nếu có hỗ trợ.
- Reviewer đã kiểm tra, không chỉ nhìn tên file hoặc thông báo test thành công.

Giữ data, logs và weights cục bộ ngoài Git. Commit code/config/split indices cùng các bảng/figures nhỏ đã review; dùng hướng dẫn tải hoặc tái tạo cho artifact lớn.
