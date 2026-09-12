# Assignment 1 — Phân công triển khai chi tiết

Ngày lập: **04/09/2026**. Đối chiếu code tại commit **`f2e140e`**.

Mục tiêu: mỗi thành viên biết **mở file nào, viết hàm nào, nhận gì, trả gì, kiểm tra thế nào và bàn giao cho ai**. Tài liệu này phân công công việc; không khẳng định các hàm ML đã được triển khai.

Phạm vi hiện tại chỉ là **Assignment 1**. Giữ A2/A3 ở trạng thái chuẩn bị, không chia thêm model/dataset cho các phần đó trong tài liệu này.

## Mục lục

- [1. Phân công và quy tắc làm việc](#1-ph%C3%A2n-c%C3%B4ng-v%C3%A0-quy-t%E1%BA%AFc-l%C3%A0m-vi%E1%BB%87c)
- [2. Quy ước chung phải đọc trước](#2-quy-%C6%B0%E1%BB%9Bc-chung-ph%E1%BA%A3i-%C4%91%E1%BB%8Dc-tr%C6%B0%E1%BB%9Bc)
- [3. Bản đồ thư mục và file](#3-b%E1%BA%A3n-%C4%91%E1%BB%93-th%C6%B0-m%E1%BB%A5c-v%C3%A0-file)
- [4. Thiên — Data, EDA, Linear và CNN](#4-thi%C3%AAn--data-eda-linear-v%C3%A0-cnn)
- [5. Khoa — Training, evaluation, MLP và RNN](#5-khoa--training-evaluation-mlp-v%C3%A0-rnn)
- [6. Khải — Reproducibility, tích hợp và Transformer](#6-kh%E1%BA%A3i--reproducibility-t%C3%ADch-h%E1%BB%A3p-v%C3%A0-transformer)
- [7. File dùng chung: ai được sửa phần nào?](#7-file-d%C3%B9ng-chung-ai-%C4%91%C6%B0%E1%BB%A3c-s%E1%BB%ADa-ph%E1%BA%A7n-n%C3%A0o)
- [8. Hợp đồng bàn giao dữ liệu và artifacts](#8-h%E1%BB%A3p-%C4%91%E1%BB%93ng-b%C3%A0n-giao-d%E1%BB%AF-li%E1%BB%87u-v%C3%A0-artifacts)
- [9. Phân công tests](#9-ph%C3%A2n-c%C3%B4ng-tests)
- [10. Trình tự bắt đầu và các mốc tích hợp](#10-tr%C3%ACnh-t%E1%BB%B1-b%E1%BA%AFt-%C4%91%E1%BA%A7u-v%C3%A0-c%C3%A1c-m%E1%BB%91c-t%C3%ADch-h%E1%BB%A3p)
- [11. Chạy thí nghiệm, báo cáo và nộp bài](#11-ch%E1%BA%A1y-th%C3%AD-nghi%E1%BB%87m-b%C3%A1o-c%C3%A1o-v%C3%A0-n%E1%BB%99p-b%C3%A0i)
- [12. Checklist nhận việc và hoàn thành](#12-checklist-nh%E1%BA%ADn-vi%E1%BB%87c-v%C3%A0-ho%C3%A0n-th%C3%A0nh)

## 1. Phân công và quy tắc làm việc

| Ký hiệu | Thành viên                     | Trách nhiệm chính                                                                              | Reviewer |
| ------- | ------------------------------ | ---------------------------------------------------------------------------------------------- | -------- |
| A       | **Nguyễn Hạo Thiên — 2453194** | Data, split, preprocessing, EDA, Linear, CNN, phân tích ảnh dự đoán                            | Khoa     |
| B       | **Nguyễn Anh Khoa — 2452539**  | Engine, trainer, metrics, checkpoint, timing, MLP, LSTM/GRU, tổng hợp số liệu                  | Khải     |
| C       | **Tạ Tuấn Khải — 2452515**     | Config, seed/environment, artifacts/logging, interface, CLI/CI, Transformer, tích hợp tài liệu | Thiên    |

Quy tắc:

1. **Owner là người viết và chịu trách nhiệm chính**, không phải người duy nhất được hiểu hoặc sửa phần đó.
2. Reviewer phải đọc diff và chạy kiểm tra phù hợp; không chỉ bấm approve.
3. Ai viết chức năng thì viết tests và ghi phương pháp cho chức năng đó. Khải không làm toàn bộ tests/báo cáo thay nhóm.
4. Mọi model dùng chung DataLoader, engine, evaluator, split và protocol. Không tạo ba project hoặc năm training loop riêng.
5. Giữ chữ ký hàm hiện có. Nếu cần đổi, cập nhật đồng thời nơi gọi, tests và tài liệu qua một PR được review.
6. Các mục ghi **MỚI** là công việc cần bổ sung; file/hàm đó chưa tồn tại. Những mục khác sử dụng tên đang có trong repo.
7. Ưu tiên Draft: data/EDA + Linear/MLP chạy end-to-end. CNN/RNN/Transformer và 20 main runs thuộc giai đoạn tiếp theo.

### Phần đã có: bảo trì, không viết lại từ đầu

- `src/dlbench/common/config.py`: đọc/kiểm tra cấu hình Python.
- `src/dlbench/a1/contracts.py`: các kiểu dữ liệu chung.
- `src/dlbench/a1/models/registry.py`: chọn class model từ tên config.
- `src/dlbench/a1/cli.py`: parser và điều hướng các lệnh hiện có.
- `setup.py`, cấu hình CI và các test config/CLI.

Các model, data pipeline, training/evaluation, ghi artifacts và phân tích vẫn cần triển khai. Có tên hàm/CLI không có nghĩa chức năng đã chạy được.

## 2. Quy ước chung phải đọc trước

### 2.1. Đã thống nhất và còn cần chốt

| Nội dung                       | Quy ước/trạng thái                                                           |
| ------------------------------ | ---------------------------------------------------------------------------- |
| Dataset chính                  | Fashion-MNIST; PyTorch là framework chính                                    |
| Run seeds                      | **`[36, 69420, 67, 69]`**, đã chốt; mặc định phát triển `36`                 |
| Main runs                      | 5 model × 4 seed = **20 runs**; chưa tính debug/tuning                       |
| Split seed                     | `42` đang là đề xuất riêng; đổi run seed không tạo split mới                 |
| Split đề xuất                  | 50.000 train / 10.000 validation từ official train; giữ 10.000 official test |
| Model RNN                      | Chọn **LSTM hoặc GRU**; config hiện đề xuất GRU, Khoa xác nhận với nhóm      |
| Normalization                  | Chưa đo; Thiên tính từ training partition chưa augmentation                  |
| Batch size/epoch/tuning budget | Chưa chốt; dựa trên smoke test và máy thực tế                                |
| Augmentation                   | Random crop padding 2 là đề xuất, chưa phải policy được duyệt                |
| Protocol                       | `a1-v0`, trạng thái `draft`; chưa điền approvers thay nhóm                   |

Không điền số giả vào mean/std hoặc budget để làm `--strict` chạy qua. Những con số chưa đo không cản việc viết code/unit tests, nhưng chưa được dùng để gọi một run là main benchmark đã duyệt.

### 2.2. Interface dữ liệu và model

Đọc contracts.py trước khi code.

| Kiểu               | Các trường/ý nghĩa                                                                          |
| ------------------ | ------------------------------------------------------------------------------------------- |
| `Batch`            | `images`: float32 `[B,1,28,28]`; `labels`: int64 `[B]`, nhãn 0–9; `sample_ids`: `list[str]` |
| `DataLoaders`      | `train`, `validation`, `test`; chú ý tên trường là `validation`, không tự đổi thành `val`   |
| `EpochMetrics`     | `loss`, `accuracy`, `macro_f1`, `num_samples`                                               |
| `Predictions`      | `sample_ids`, `targets`, `predicted_labels`, `probabilities` `[N,10]`                       |
| `EvaluationResult` | `metrics: EpochMetrics`, `predictions: Predictions`                                         |
| `FitResult`        | `run_dir`, `best_checkpoint`, `history_file`, đều là `Path`                                 |
| `SplitManifest`    | `dataset`, `split_seed`, `train_indices`, `validation_indices`, `test_indices`              |

Mọi model nhận `images [B,1,28,28]` và trả **raw logits `[B,10]`**. Flatten/tạo sequence/patch nằm trong model. `CrossEntropyLoss` nhận logits, **không nhận softmax output**. Chỉ tính probabilities để xuất predictions/phân tích.

### 2.3. Config là dữ liệu, không phải nơi viết pipeline

- `configs/a1/protocol.py`: một nguồn quy tắc chung.
- `configs/a1/models/*.py`: config từng model; chỉ chứa `CONFIG = {...}` bằng literal Python.
- Không import, gọi hàm, tham chiếu biến hay tính toán trong các file config; loader hiện đọc bằng `ast.literal_eval`.
- Model config không tự ghi đè data/preprocessing/metrics/checkpoint policy.
- `--seed` chỉ đổi run seed trong bộ nhớ; không sửa file config hay split.

## 3. Bản đồ thư mục và file

Các đường dẫn tính từ gốc repo; không hard-code đường dẫn ổ đĩa cá nhân vào code.

```
src/dlbench/
├── common/                         Khải
│   ├── config.py                   Bảo trì config/validator đang có
│   ├── reproducibility.py          Seeds, workers, environment
│   └── artifacts.py                Run directory, config/log/metadata
└── a1/
    ├── contracts.py                Khải điều phối; cả nhóm duyệt
    ├── cli.py                      Khải
    ├── data/                       Thiên
    │   ├── dataset.py              Dataset nguồn + wrapper + prepare
    │   ├── split.py                Tạo/lưu/đọc/kiểm tra split
    │   ├── transforms.py           Normalization, augmentation
    │   ├── loaders.py              Collate + ba DataLoader
    │   └── eda.py                  EDA
    ├── models/
    │   ├── linear.py               Thiên
    │   ├── cnn.py                  Thiên
    │   ├── mlp.py                  Khoa
    │   ├── rnn.py                  Khoa
    │   ├── transformer.py          Khải
    │   └── registry.py             Khải bảo trì
    ├── engine.py                   Khoa: một epoch
    ├── trainer.py                  Khoa: điều phối nhiều epoch
    ├── metrics.py                  Khoa
    ├── checkpoint.py               Khoa
    ├── benchmark.py                Khoa
    └── analysis.py                 Chia theo hàm; xem mục 7

configs/a1/
├── protocol.py                     Khải điều phối; Thiên/Khoa cung cấp quyết định
├── models/{linear,cnn}.py          Thiên
├── models/{mlp,rnn}.py             Khoa
├── models/transformer.py           Khải
└── splits/                         Thiên tạo; Khoa kiểm tra; Khải dùng metadata

tests/                              Ai viết chức năng, người đó viết test
environment/                        Khải tổng hợp; mọi người cung cấp môi trường
notebooks/                          Thiên EDA; logic dùng lại vẫn nằm trong src
reports/a1/                         Cả nhóm viết; Khải ghép
docs/a1.md, docs/assets/a1/          Khải tích hợp; owner cung cấp nội dung/figures
data/, runs/, checkpoints/          Artifacts cục bộ, không commit dữ liệu/weights
```

## 4. Thiên — Data, EDA, Linear và CNN

**Reviewer chính: Khoa.** Bàn giao sớm nhất là dataset/split/normalization để trainer không phải chờ toàn bộ EDA.

### T01 — Đọc dataset và tạo sample có ID

**File:** dataset.py.

| Hàm/class                                               | Công việc phải viết                                                                                                                       | Kết quả bàn giao                                     |
| ------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| `load_official_dataset(root, *, train, download=False)` | Dùng torchvision FashionMNIST, chọn đúng official train/test; không transform ngẫu nhiên tại nguồn; không download khi import module      | Dataset nguồn, đọc được ảnh/nhãn và danh sách labels |
| **MỚI:** `IndexedImageDataset`                          | Wrapper cho dataset nguồn + indices + transform + tên nguồn; tên class này là đề xuất cần nhóm thống nhất                                 | Adapter dùng được cho train/validation/test          |
| **MỚI:** `IndexedImageDataset.__len__()`                | Trả đúng số indices, không trả số mẫu của toàn bộ nguồn                                                                                   | `int`                                                |
| **MỚI:** `IndexedImageDataset.__getitem__(index)`       | Map local index về original index, áp dụng transform đúng tập, trả mapping `images`, `labels`, `sample_ids`; ở cấp sample ID là một chuỗi | Một sample để `collate_samples()` ghép batch         |

Sample ID đề xuất: `fashion_mnist:official_train:123` hoặc `fashion_mnist:official_test:123`. Hai nguồn có thể cùng số `123` nhưng không phải cùng một ảnh.

**Điều kiện xong:** đọc được cả hai nguồn; nhãn đúng; wrapper giữ original ID; train và validation không dùng chung một transform bị mutate. Model không cần biết ảnh đến từ dataset wrapper nào.

### T02 — Tạo và lưu split một lần

**File:** split.py; đầu ra nhỏ trong `configs/a1/splits/`.

| Hàm                                                    | Input → output                          | Logic cần có                                                                           |
| ------------------------------------------------------ | --------------------------------------- | -------------------------------------------------------------------------------------- |
| `create_split(labels, *, validation_size, split_seed)` | Labels official train → `SplitManifest` | Stratified split theo policy đã duyệt; official test là nguồn riêng; giữ exact counts  |
| `validate_split(manifest)`                             | Manifest → `None` hoặc `ValueError`     | Kiểm tra range, duplicate, train/val overlap, coverage đủ official train/test          |
| `save_split(manifest, path)`                           | Manifest + đường dẫn → file JSON        | Serialize rõ schema; không âm thầm ghi đè split khác                                   |
| `load_split(path)`                                     | JSON có sẵn → `SplitManifest`           | Deserialize, kiểm tra hợp lệ; không tạo split mới khi thiếu file hoặc khi đổi run seed |

**MỚI cần phối hợp Khải:** `SplitManifest` chưa chứa hash, class counts hay provenance. Thống nhất một metadata JSON đi kèm theo mục 8; không nhét thêm fields vào dataclass một phía.

**Điều kiện xong:** save/load round-trip đúng; same split seed tạo same indices; không trùng train/val; count đúng; đổi run seed không đổi manifest. Khoa kiểm tra độc lập trước khi merge.

### T03 — Normalization và transforms

**File:** transforms.py.

| Hàm                                             | Công việc phải viết                                                                                | Điều kiện xong                                                     |
| ----------------------------------------------- | -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| `compute_normalization(dataset, train_indices)` | Đưa pixels về `[0,1]`, tính mean/std theo kênh trên ảnh train chưa augmentation                    | Trả `([mean], [std])`; std > 0; không đọc val/test để fit thống kê |
| `build_transforms(preprocessing, *, training)`  | Tạo pipeline train/eval riêng; đề xuất crop → tensor → normalize khi train; eval không random crop | Một ảnh thành float32 `[1,28,28]`; eval lặp lại cho cùng kết quả   |

Thiên lưu số đo thật và nguồn đo; Khải tích hợp vào protocol sau review. Không lấy số normalization từ ví dụ trên mạng rồi ghi như số đã đo của nhóm.

**Tests:** so với dữ liệu nhỏ tính tay; đảm bảo chỉ chọn train indices; tách transform train/validation; kích thước không đổi sau augmentation.

### T04 — Batch và DataLoaders

**File:** loaders.py.

| Hàm                                         | Công việc phải viết                                                                                                      | Bàn giao                                        |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------- |
| `collate_samples(samples)`                  | Stack ảnh/nhãn; ép đúng dtype; giữ IDs theo đúng thứ tự                                                                  | `Batch` đúng contract                           |
| `build_dataloaders(config, *, smoke=False)` | Load manifest đã lưu, tạo ba wrapper/transforms riêng, batch size hợp lệ, train shuffle, val/test không shuffle/drop mẫu | `DataLoaders(train, validation, test)` cho Khoa |

Phụ thuộc: T01–T03 và `seed_worker()` từ Khải. Dùng DataLoader generator riêng theo run seed, không để RNG khởi tạo model làm thay đổi shuffle ngoài ý muốn.

**Điều kiện xong:** batch size 1 và batch cuối lẻ đều đúng; tổng số mẫu eval đủ; IDs/labels khớp ảnh; val/test deterministic. Ban đầu có thể dùng `num_workers=0` để debug Windows, sau đó kiểm tra thêm multi-worker khi cần.

### T05 — Điều phối chuẩn bị dữ liệu

**File:** dataset.py, hàm `prepare_data(config)`.

Viết chuỗi xử lý:

1. Đọc/tải official train/test bằng `load_official_dataset()` khi người dùng gọi `prepare`.
2. Nếu đã có split: đọc và kiểm tra tương thích; nếu chưa có: tạo theo policy đã thống nhất.
3. Tính/kiểm tra normalization từ đúng train indices.
4. Lưu manifest + metadata nhỏ theo schema chung, không tự freeze protocol.
5. Trả dictionary **JSON-serializable**: đường dẫn artifact, exact counts, seed, mean/std và thông tin xác minh để CLI hiển thị.

Thiên cung cấp số liệu; Khải cập nhật config đã review. Lần gọi sau không được âm thầm đổi split hoặc làm hỏng artifact lần trước.

**Bàn giao:** lệnh `prepare` hoạt động sau khi hoàn thiện hàm; có thể chạy lại an toàn. `generate_eda()` chưa tự được gọi bởi CLI; phần nối do Khải phụ trách C04.

### T06 — EDA và tài liệu dữ liệu

**File:** eda.py, hàm `generate_eda(config, output_dir)`.

- Xuất số mẫu, kích thước, dtype/range, phân bố lớp, imbalance và ảnh đại diện.
- Ghi nguồn/license, tên lớp, split policy và kiểm tra leakage.
- Lưu bản sinh tự động trong `runs/`; chọn ảnh nhỏ đã review vào `docs/assets/a1/`.
- Viết report mục **1.2–1.4** và phần data của **2.1**. Không đợi xong model mới viết EDA.

**Điều kiện xong:** bảng/biểu đồ lấy từ dữ liệu thật, có nhãn/đơn vị/caption; Khoa đối chiếu counts với manifest.

### T07 — Linear trước, CNN sau

| File/config                                       | Class và methods                                                            | Code cần viết                                                                                         |
| ------------------------------------------------- | --------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `models/linear.py`; `configs/a1/models/linear.py` | `LinearClassifier.__init__(parameters)`, `LinearClassifier.forward(images)` | Flatten 784 chiều, linear → 10 logits; không hidden layer/softmax                                     |
| `models/cnn.py`; `configs/a1/models/cnn.py`       | `CNNClassifier.__init__(parameters)`, `CNNClassifier.forward(images)`       | CNN tự thiết kế; conv/activation/pooling và head; kiểm tra feature dimensions thay vì đoán kích thước |

Đường dẫn hai model nằm trong `src/dlbench/a1/`. Không viết optimizer, data loading hay vòng lặp train trong class model; không hard-code `.cuda()`.

**Mỗi model phải có:** shape test cho B=1/B=7; finite logits; backward/update được ít nhất một parameter; small-data learning check; config và giải thích thiết kế trong report **2.2**. Chạy qua engine của Khoa, không chạy loop riêng.

### T08 — Phân tích ảnh đúng/sai

**File:** analysis.py, owner của `analyze_run(run_dir)` là **Thiên**; Khải review provenance.

- Đọc history/predictions/config đã lưu; tạo curves, confusion matrix, ví dụ đúng/sai.
- **MỚI:** helper resolve `sample_id` → ảnh nguồn, sử dụng data root/provenance trong metadata; tên/signature helper thống nhất trước khi thêm.
- Ghi quy tắc chọn ví dụ để tránh chỉ chọn ảnh có lợi; giữ sample IDs và đúng split trên caption.
- Không retrain, đổi checkpoint hay tune bằng các ảnh test vừa xem.
- Viết report **3.3**, cung cấp observations cho **3.4–3.5**.

**Phụ thuộc:** Khoa xuất predictions đúng schema; Khải lưu data/source metadata. Khoa sở hữu hai hàm khác trong cùng file — xem mục 7.

## 5. Khoa — Training, evaluation, MLP và RNN

**Reviewer chính: Khải.** Bắt đầu metrics/checkpoint comparator và engine với batch tổng hợp, không cần chờ Thiên hoàn thành EDA.

### K01 — Metrics dùng chung

**File:** metrics.py.

| Hàm                                                 | Code cần viết                                                                                                      | Điều kiện xong                                                                     |
| --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------- |
| `classification_metrics(targets, predicted_labels)` | Accuracy + macro-F1 trên toàn tập; labels 0–9, `zero_division=0`; từ chối input rỗng/lệch độ dài/nhãn không hợp lệ | Dictionary có `accuracy`, `macro_f1` trong `[0,1]`; kiểm tra dữ liệu tính tay      |
| `count_parameters(model)`                           | Đếm `numel()` của toàn bộ parameters và phần `requires_grad`                                                       | Có `total_parameters`, `trainable_parameters`; không tính bằng công thức ước lượng |

Không trung bình macro-F1 từng batch. Engine chịu trách nhiệm gom predictions và trung bình loss theo số mẫu trước khi gọi metric.

### K02 — Checkpoint selection và lưu/đọc

**File:** checkpoint.py.

| Hàm                                            | Code cần viết                                                                                                        | Điều kiện xong                                                     |
| ---------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| `is_better(candidate, incumbent)`              | So sánh `val_macro_f1` cao hơn → `val_loss` thấp hơn → epoch sớm hơn; xử lý incumbent `None` và metric không hữu hạn | Test thắng/thua/hòa; test metrics không tham gia chọn checkpoint   |
| `save_checkpoint(path, payload)`               | Lưu an toàn, metadata đầy đủ; phân biệt `best.pt` và `last.pt`                                                       | Đọc lại được; lỗi ghi file không để lại một best checkpoint hỏng   |
| `load_checkpoint(path, *, map_location="cpu")` | Chỉ load checkpoint đáng tin cậy; kiểm tra schema/keys, hỗ trợ map về CPU                                            | Load lại model cho predictions tương ứng trên cùng input/eval mode |

Payload thống nhất với Khải: weights, epoch, model/effective config, split/statistics hash, run seed, val metrics, Git revision. Nếu hỗ trợ resume, bổ sung optimizer/scheduler/RNG state và ghi rõ resume ở ranh giới epoch hay giữa epoch; không hứa exact resume khi chưa test.

### K03 — Một epoch train và evaluate

**File:** engine.py.

| Hàm                                                            | Input → output                                | Trình tự cần có                                                                                                                      |
| -------------------------------------------------------------- | --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `train_one_epoch(model, loader, optimizer, criterion, device)` | Các thành phần của một epoch → `EpochMetrics` | `train()`, chuyển images/labels lên device, zero grad, forward logits, CE, backward, optimizer step; tích lũy loss/count/predictions |
| `evaluate_epoch(model, loader, criterion, device)`             | Model + eval loader → `EvaluationResult`      | `eval()` + `inference_mode()`, không cập nhật weights; gom đủ labels/IDs/predictions/probabilities và tính metrics toàn tập          |

`sample_ids` giữ ở CPU dạng chuỗi. Không sửa Batch contract cho riêng GPU/model. Loss trung bình phải theo số mẫu, kể cả batch cuối nhỏ.

**Tests:** synthetic loader để kiểm tra cơ học; weights đổi khi train và không đổi khi evaluate; dropout/eval mode đúng; sample count đúng; probabilities/IDs không lệch thứ tự. Dữ liệu tổng hợp chỉ dùng test, không đưa số liệu đó vào report Fashion-MNIST.

### K04 — MLP và RNN

| File/config                                                | Class và methods                                                      | Code cần viết                                                                                                   |
| ---------------------------------------------------------- | --------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `src/dlbench/a1/models/mlp.py`; `configs/a1/models/mlp.py` | `MLPClassifier.__init__(parameters)`, `MLPClassifier.forward(images)` | Flatten, ít nhất một hidden layer, activation/regularization có giải thích, output 10 logits                    |
| `src/dlbench/a1/models/rnn.py`; `configs/a1/models/rnn.py` | `RNNClassifier.__init__(parameters)`, `RNNClassifier.forward(images)` | Một LSTM hoặc GRU; đề xuất rows `[B,28,28]`, `batch_first=True`; chọn hidden representation/aggregation và head |

Ưu tiên MLP cho Draft. RNN phải giải thích: timestep là gì, input size, hidden size, số layers và cách dùng output/state để phân loại. Với LSTM, xử lý đúng hidden/cell state; không áp công thức GRU một cách máy móc.

Mỗi model có shape/backward/tiny-data tests, config và nội dung report **2.2** như T07.

### K05 — Điều phối một lần training

**File:** trainer.py, hàm `fit(config, *, smoke=False)`.

Viết theo thứ tự:

1. Kiểm tra config phù hợp chế độ; xác minh split/statistics thực tế, không chỉ `--strict`.
2. Seed bằng utility của Khải; khởi tạo lại model cho run hiện tại.
3. Lấy loaders của Thiên và model từ registry.
4. Chọn device; tạo criterion, optimizer và các thành phần training đã được duyệt.
5. Tạo run directory và ghi resolved config/environment/provenance trước khi train.
6. Gọi `train_one_epoch()` và `evaluate_epoch()` trên validation theo từng epoch.
7. Ghi history, chọn/lưu best checkpoint, áp dụng early stopping nếu bật.
8. Ghi thời gian và số epoch thực chạy; trả `FitResult` với đường dẫn có thật.

**MỚI:** logic/helper chọn optimizer, kiểm tra device, early stopping và chuẩn hóa smoke budget thuộc **Khoa**. Có thể thêm helper nội bộ trong `trainer.py`; tên mới phải ghi rõ trong PR, không tạo framework tổng quát không cần thiết.

Config hiện có optimizer/LR/weight decay/patience, **chưa có scheduler/min-delta**. Nếu bổ sung, Khoa đề xuất fields/semantics, Khải cập nhật validator/tests. Không quảng cáo options chưa được code hỗ trợ.

Training không tự đánh giá official test mỗi epoch. `fit()` không thay thế logic engine, checkpoint hay artifact writer bằng bản riêng.

### K06 — Đánh giá checkpoint đã chọn

**File:** trainer.py, hàm `evaluate_checkpoint(config, checkpoint_path, *, split="validation")`.

- Tái dựng model từ config checkpoint; xác minh supplied config tương thích về model, seed, split, preprocessing và class order.
- Đọc đúng checkpoint, không tự chọn lại bằng test score.
- Chọn validation hoặc test loader; gọi `evaluate_epoch()`.
- Ghi metrics/predictions qua hàm dùng chung, giữ thông tin checkpoint/run gốc.
- Trả `EvaluationResult`. Không âm thầm ghi đè predictions của một split bằng split khác.

**Điều kiện xong:** eval trước/sau load tương ứng; mismatch config/hash bị báo lỗi; default là validation; chỉ đánh giá test khi quyết định mô hình đã cố định và người chạy xác nhận `--allow-test`.

### K07 — Timing

**File:** benchmark.py, hàm `benchmark_inference(model, sample_batch, timing_config)`.

- Dùng cùng device, batch size và precision cho so sánh timing.
- Batch sẵn trên device khi scope là `forward_only`; không tính DataLoader/chuyển dữ liệu vào số đó.
- `eval()` + không gradient; warm-up, đo lặp và đồng bộ CUDA đúng lúc.
- Trả metadata phép đo, median batch time, throughput và amortized time/image với đơn vị rõ.
- Không gọi thời gian chia cho batch lớn là latency một ảnh đơn lẻ.

Training-time được đo trong `fit()` theo ranh giới nhóm duyệt: ghi rõ có tính validation/checkpoint/logging không; kèm actual epochs. Tuning cost ghi riêng. Khải kiểm tra cùng môi trường; không xếp hạng tốc độ từ các máy không tương đương.

### K08 — Xuất predictions và bảng so sánh

**File:** analysis.py, hai hàm thuộc **Khoa**:

| Hàm                                   | Việc cần làm                                                                                              | Output                                                             |
| ------------------------------------- | --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| `save_predictions(predictions, path)` | Kiểm tra số dòng/IDs/probabilities khớp; xuất CSV ổn định                                                 | `sample_id,target,predicted_label,probability_0,...,probability_9` |
| `compare_runs(run_dirs, output_path)` | Kiểm tra tương thích protocol/split/timing trước khi gộp; phát hiện run thiếu/trùng và smoke run lẫn main | Bảng từng run và bảng tổng hợp theo model/seed                     |

Một model phải có đủ bốn seed đã chốt trước khi báo tổng hợp Final; giữ số liệu từng seed. Không ghép kết quả có augmentation/protocol khác mà không ghi riêng. Mean/std theo policy được thống nhất tại mục 11.

Khoa phụ trách report **2.3–2.4, 3.1–3.2**, cung cấp bằng chứng trade-off cho **3.5**.

## 6. Khải — Reproducibility, tích hợp và Transformer

**Reviewer chính: Thiên.** Giao seed/artifact interface sớm để Thiên/Khoa có thể ghép; không chờ hoàn thiện Transformer mới hỗ trợ pipeline.

### C01 — Config và contracts đang có

**Files:** config.py, contracts.py, `configs/a1/protocol.py`.

| Thành phần                                                                                              | Trách nhiệm                                                                         |
| ------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `ConfigError`                                                                                           | Giữ loại lỗi cấu hình rõ cho CLI/tests                                              |
| `_read_python_config(path)`                                                                             | Bảo trì đọc literal-only; không đổi sang thực thi file tùy ý                        |
| `load_config(path)`                                                                                     | Bảo vệ common sections, resolve protocol_file, giữ `_sources`, mỗi lần load độc lập |
| `validate_config(config, *, strict=False)`                                                              | Bảo trì cấu trúc/ràng buộc và readiness; mở rộng có tests khi nhóm bổ sung fields   |
| `Batch`, `DataLoaders`, `EpochMetrics`, `Predictions`, `EvaluationResult`, `FitResult`, `SplitManifest` | Điều phối thay đổi và kiểm tra mọi producer/consumer dùng cùng schema               |

Các helper `require()`, `nonnegative_int()`, `finite_number()` nằm bên trong validator cũng thuộc Khải. Không giao viết lại loader chỉ để có commit; ưu tiên phần thiếu ở C02–C03.

Thiên đề xuất data/statistics; Khoa đề xuất budget/optimizer/checkpoint/timing; Khải tích hợp vào protocol. Không tự điền tên mọi người vào `approved_by` khi họ chưa review.

### C02 — Seeds và environment

**File:** reproducibility.py.

| Hàm                                            | Code cần viết                                                                                           | Điều kiện xong                                                                                    |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `seed_everything(seed, *, deterministic=True)` | Seed Python/NumPy/PyTorch, cấu hình deterministic theo môi trường; ghi nhận giới hạn                    | Cùng seed trên môi trường test cho chuỗi ngẫu nhiên tương ứng; không hứa bitwise across platforms |
| `seed_worker(worker_id)`                       | Top-level function dùng được với Windows multiprocessing; lấy worker seed từ PyTorch, seed NumPy/random | Worker random state không bị lặp sai; dùng được với generator của Thiên                           |
| `collect_environment()`                        | Thu thập Python/dependency/CUDA/CPU/GPU/precision/determinism liên quan                                 | Dictionary serializable, không chứa secrets hoặc toàn bộ environment variables                    |

**Bàn giao sớm:** Thiên dùng `seed_worker`; Khoa gọi `seed_everything` trước xây model/loaders. Lưu hướng dẫn môi trường đã kiểm chứng trong `environment/`, không chỉ danh sách dependency không có versions.

### C03 — Run directories, logs và metadata

**File:** artifacts.py.

| Hàm                                   | Code cần viết                                                                                             | Điều kiện xong                                                      |
| ------------------------------------- | --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| `create_run_dir(output_root, run_id)` | Kiểm tra run ID an toàn; tạo mới và từ chối overwrite                                                     | Trả `Path`; ID trùng hoặc traversal như `../` bị từ chối            |
| `save_run_metadata(run_dir, config)`  | Lưu resolved config, hai file config nguồn, môi trường, Git revision/dirty state, split/statistics hashes | Sau khi config nguồn đổi vẫn truy lại được cấu hình đúng của run cũ |
| `append_history(run_dir, row)`        | Ghi CSV theo schema thống nhất, một row/epoch, không lặp header                                           | Số rows/epoch khớp, kiểu/đơn vị rõ, lỗi schema báo rõ               |
| `save_metrics(run_dir, metrics)`      | Ghi JSON có split/checkpoint/model/run/seed/units; xử lý giá trị không serializable/nonfinite             | File đọc lại được, không có điểm số giả hoặc provenance thiếu       |

Thiên tạo metadata dữ liệu; Khải sao chụp/ghi tham chiếu và hash cho mỗi run; Khoa gọi writer. Chốt cách không ghi đè metrics/predictions giữa validation/test theo mục 8.

### C04 — CLI và lệnh chưa nối

**File:** cli.py, `build_parser()` và `main(argv=None)`.

- Bảo trì `validate-config`, `prepare`, `train`, `evaluate`, `analyze`; giữ `--seed` không đổi split.
- Khi Thiên/Khoa triển khai xong hàm đích, kiểm tra lệnh gọi đúng và lỗi hiển thị rõ. Không chuyển logic training vào CLI.
- Phân biệt `train --smoke` và main run; giữ strict guard cho main/test.
- Bảo đảm `--help`/config checks vẫn không cần import torch hoặc tự download.
- **MỚI:** nối `generate_eda()` bằng lệnh `eda` **hoặc** một option rõ của `prepare`; Khải và Thiên chọn một cách trước khi thêm. Hiện chưa có lệnh `eda`.
- **MỚI:** entry point cho `compare_runs()`: Khải và Khoa chọn thêm lệnh `compare` hoặc một script riêng; hiện chưa có lệnh `compare`.
- Thêm tests và README cho các entry point mới; không ghi lệnh chưa tồn tại thành lệnh chạy được ngay.

### C05 — Transformer và model registry

**Files:** transformer.py, registry.py, `configs/a1/models/transformer.py`.

| Thành phần                                    | Code/trách nhiệm                                                                                                           |
| --------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `TransformerClassifier.__init__(parameters)`  | Patch/token projection, positional encoding, encoder và head; kiểm tra embed_dim/head divisibility                         |
| `TransformerClassifier.forward(images)`       | Từ ảnh chung tạo tokens, xử lý attention, pooling/CLS theo thiết kế, trả `[B,10]` logits                                   |
| `MODEL_REGISTRY`, `build_model(model_config)` | Mapping hiện đã có; bảo trì constructor signature và lỗi tên không hợp lệ; không eager-import mọi model khi chỉ đọc config |

Config đang đề xuất patch 4×4 → 49 patch, mỗi patch có 16 giá trị trước projection. Nếu dùng CLS token phải giải thích chiều sequence tăng thêm; chốt pooling/CLS policy, không mô tả như đã có sẵn.

**Điều kiện xong:** shape/backward/tiny-data tests; positional information thật sự được dùng; giải thích attention input/output và representation; chạy qua engine chung. Khải viết phần Transformer trong report **2.2**, tích hợp **3.4** từ notes của cả ba người.

### C06 — Package, CI và kiểm thử tích hợp

**Files:** `setup.py`, `.github/workflows/ci.yml`, `.github/pull_request_template.md`, `.gitignore`, `.gitattributes`, `.editorconfig`, `tests/`.

- Giữ `pip install -e .`, CLI và import package chạy từ clean environment.
- Dependency declarations không thay thế phiên bản môi trường đã được kiểm chứng.
- CI hiện chỉ cài package cơ bản; khi bật tests import torch, thêm môi trường/job CPU với dependency phù hợp. Không download full dataset hoặc chạy 20 main experiments trong CI.
- Giữ data/weights/logs/secrets bị ignore; split indices và tài liệu nhỏ vẫn tracked.
- Điều phối tests dùng chung; mỗi model owner vẫn viết tests riêng của mình.
- Đảm bảo người không viết module có thể chạy lại lệnh trong PR/README.

Các `__init__.py` do owner module bảo trì export nếu cần, Khải kiểm tra import không có side effects. Không bỏ logic training/download vào các file này.

### C07 — Tích hợp tài liệu và tiến độ

**Files:** `README.md`, `IMPLEMENTATION_GUIDE.md`, `CONTRIBUTING.md`, `PROJECT_PLAN.md`, tài liệu này, `docs/index.md`, `docs/a1.md`, `reports/a1/report.md`, `reports/a1/presentation-outline.md`, `AI_USAGE.md`.

- Khải ghép nội dung, kiểm tra link, đồng bộ trạng thái và hướng dẫn chạy.
- Thiên/Khoa nộp phần viết/figures/run evidence của mình; Khải không viết thay cả nhóm.
- Mỗi thành viên ghi AI usage cho phần đã dùng; Khải kiểm tra disclosure đồng nhất trên repo/Pages/report.
- Thông tin tên/MSSV/giảng viên giữ đúng; Group ID chỉ điền khi có xác nhận.
- Việc publish/submit thực hiện riêng theo thống nhất của nhóm, không xem thao tác tạo tài liệu là đã nộp bài.

## 7. File dùng chung: ai được sửa phần nào?

| File                                 | Người giữ nhịp tích hợp     | Phân chia cụ thể                                                                                                                   |
| ------------------------------------ | --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| `configs/a1/protocol.py`             | Khải                        | Thiên đề xuất data/preprocessing; Khoa đề xuất training/metrics/checkpoint/timing; cả ba duyệt                                     |
| `src/dlbench/a1/contracts.py`        | Khải                        | Người cần đổi gửi ví dụ input/output; update producer + consumer + tests trong cùng PR                                             |
| `src/dlbench/a1/analysis.py`         | Khải điều phối thứ tự merge | Khoa: `save_predictions`, `compare_runs`; Thiên: `analyze_run` và helper đọc ảnh; Khải review provenance, không viết đè phần owner |
| `tests/test_config.py`               | Khải                        | Duy trì config/CLI tests; owner chức năng thông báo khi hành vi TODO/draft thay đổi                                                |
| `tests/test_pipeline_contract.py`    | Khải điều phối              | Khoa: metrics/checkpoint tests; Thiên: split test; chỉ sửa/bỏ skip phần đã hoàn thành                                              |
| `tests/test_model_contract.py`       | Khải điều phối              | Tách/bật checks theo model; owner model viết kiểm thử tương ứng                                                                    |
| `reports/a1/report.md`, `docs/a1.md` | Khải                        | Thiên: data/errors; Khoa: experiments/results; cả ba: model notes; Khải ghép narrative/links                                       |

Với file cùng sửa: báo trước hàm/section sẽ đổi; PR nhỏ; merge lần lượt rồi cập nhật nhánh. Không thay toàn bộ file chỉ để nhận thêm một hàm. Phần chưa có kết quả vẫn ghi đang làm, không điền kết luận trước thí nghiệm.

## 8. Hợp đồng bàn giao dữ liệu và artifacts

Các tên metadata/định dạng ở mục này là **đề xuất triển khai cần duyệt ngày đầu**; không phải file đã được sinh ra.

### 8.1. Thiên → Khoa/Khải

- `SplitManifest` JSON đúng cấu trúc đã có.
- Một metadata JSON nhỏ cạnh manifest, ví dụ `fashion_mnist_seed42.metadata.json`: schema version, source/train-test namespaces, exact counts, class counts, split-file hash, normalization, cách tính và phạm vi mẫu dùng để tính.
- Wrapper/DataLoaders trả đúng contract, có sample IDs truy về nguồn.
- Mean/std đo thật; PR cập nhật protocol sau review hoặc một cơ chế đọc thống kê đã được nhóm duyệt. Không để trainer dùng `[]` hay mỗi người tự điền số khác nhau.

**Thiên tạo nội dung; Khoa kiểm tra độc lập; Khải thống nhất schema và tích hợp.** Nếu đổi tên/hash/schema phải cập nhật prepare, artifact writer, evaluator và checkpoint cùng nhau.

### 8.2. Khải → Khoa/Thiên

- Seed utilities và environment collector.
- Run directory mới, resolved config và metadata đúng schema.
- Quy tắc lịch sử epoch/metrics để Khoa ghi và Thiên đọc.
- Data-root/source identity được lưu đủ để `analyze_run()` truy lại ảnh; khi chuyển máy, phải có cách chỉ định lại data root có kiểm tra source, không hard-code đường dẫn F: của Khải.

### 8.3. Khoa → Thiên/Khải

- `FitResult` chỉ tới run/checkpoint/history có thật.
- `EvaluationResult` có đủ predictions/IDs/probabilities và metrics.
- File predictions và bảng so sánh dùng cùng thứ tự lớp và đơn vị.
- Training time/inference time có scope, device, batch, repetitions và actual epochs.

### 8.4. Cách tổ chức một run cần thống nhất

```
runs/a1/<unique-run-id>/
├── config.json                 Resolved config thực sự dùng
├── sources/                    Hai config Python gốc đã chụp lại
├── environment.json
├── metadata.json               Commit, split/stats identity, run mode, provenance
├── history.csv
├── best.pt
├── last.pt                     Chỉ nếu hỗ trợ resume
└── evaluation/
    ├── validation/<eval-id>/    metrics.json, predictions.csv, figures/
    └── test/<eval-id>/          metrics.json, predictions.csv, figures/
```

Đây là layout đề xuất, **chưa được code tạo tự động**. Cấu trúc evaluator phải giữ validation/test tách biệt và từ chối overwrite ngoài ý muốn. Khải có thể dùng `save_metrics()` với thư mục evaluation đích; Khoa/Thiên phải thống nhất đường dẫn trước khi implement.

History tối thiểu: `epoch, train_loss, val_loss, train_accuracy, val_accuracy, train_macro_f1, val_macro_f1, learning_rate, epoch_seconds`.

Giá trị accuracy/F1 trong logs dùng `[0,1]`; report có thể dùng `%` nhưng phải ghi rõ. Không ghi zero thay cho metric chưa đo. Chỉ commit code/config/split metadata nhỏ và figures đã duyệt, không commit toàn bộ run.

## 9. Phân công tests

Dùng `unittest` theo project hiện tại. Các file ghi **MỚI** dưới đây là đề xuất owner sẽ tạo trong PR triển khai, chưa được tạo bởi tài liệu này.

| Owner          | Test file                                                          | Nội dung cần kiểm tra                                                                                        |
| -------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------ |
| Thiên          | **MỚI** `tests/test_data.py`                                       | Wrapper/ID, split save/load/overlap, train-only stats, transforms và collate/dataloader                      |
| Thiên          | **MỚI** `tests/test_linear.py`, `tests/test_cnn.py`                | Shape, finite logits, backward/parameter update và input batch khác nhau                                     |
| Khoa           | **MỚI** `tests/test_engine.py`                                     | Loss theo số mẫu, whole-split metric, train/eval modes, weights/IDs đúng                                     |
| Khoa           | **MỚI** `tests/test_checkpoint.py`                                 | Tie-breaking, nonfinite metric, save/load, config/hash mismatch                                              |
| Khoa           | **MỚI** `tests/test_mlp.py`, `tests/test_rnn.py`                   | Model contract và sequence representation                                                                    |
| Khoa           | **MỚI** `tests/test_analysis.py`                                   | Prediction CSV round-trip, run trùng/thiếu seed/protocol mismatch; Thiên bổ sung tests cho helper ảnh khi có |
| Khải           | **MỚI** `tests/test_reproducibility.py`, `tests/test_artifacts.py` | Seed/worker behavior, unique run path, schema/serialization, chống overwrite                                 |
| Khải           | **MỚI** `tests/test_transformer.py`                                | Patch/token shape, positional information, backward                                                          |
| Khải           | `tests/test_config.py`                                             | Giữ literal-only/seed/protocol checks; cập nhật fixtures và CLI khi behavior thay đổi                        |
| Khải điều phối | `tests/test_model_contract.py`, `tests/test_pipeline_contract.py`  | Tích hợp checks hiện có; owner từng phần chịu trách nhiệm assertions                                         |

### Ba điểm cần sửa sớm trong tests hiện tại

1. `test_model_contract.py` đang skip toàn class và loop cả năm model. **Không chờ đủ năm model mới test Linear/MLP.** Khải tách checks theo model/file, hoặc bật từng test tương ứng; aggregate check cả năm chỉ bật khi tất cả đã sẵn sàng. Tests của model đã hoàn thành không được skip.
2. `test_pipeline_contract.py`: Thiên bật `test_split_overlap_is_rejected()` sau T02; Khoa bật `test_metrics_use_all_ten_classes()` sau K01 và `test_checkpoint_order_ignores_test_metrics()` sau comparator K02.
3. `test_config.py` hiện có `test_prepare_fails_honestly_until_implemented()` và assertions dựa vào config thật luôn draft. Khi prepare hoạt động hoặc protocol frozen, **Khải chuyển các test trạng thái sang fixture/mock phù hợp**, thêm success-path test, vẫn giữ kiểm tra draft phải bị strict từ chối. Không giữ test đòi production code mãi báo TODO; không xóa bảo vệ test-set policy để làm tests xanh.

Trong CI, tests cơ học dùng tensors/samples tổng hợp và file tạm; không tự tải Fashion-MNIST. Chạy dữ liệu thật bằng smoke/integration command riêng khi đã chuẩn bị môi trường. Ghi rõ test nào đã chạy, test nào còn pending.

## 10. Trình tự bắt đầu và các mốc tích hợp

### 10.1. Ngày đầu: những việc cần chốt trong một buổi ngắn

- [ ] Cả ba xác nhận interface và người phụ trách ở tài liệu này.
- [ ] Thiên/Khoa xác nhận split proposal; Khải ghi quyết định, không đổi split theo run seed.
- [ ] Chọn schema split/statistics/run metadata ở mục 8.
- [ ] Chốt smoke policy dùng chung. **Đề xuất để thử:** 2 epochs, batch 32, tối đa 512 train/128 validation, run seed 36, không đọc official test; ghi mode `smoke`, seed và subset identity. Điều chỉnh nếu máy không phù hợp; không dùng đây làm budget main runs.
- [ ] Khoa thực hiện smoke config trong bản copy in-memory; Thiên chọn subset từ fixed train/val; Khải đảm bảo log ghi đúng resolved values. Không thay config main bằng số debug mà không ghi nhận.
- [ ] Khải ghi máy/Python; nhóm chọn compatible ML dependencies trước khi chạy tests cần torch.
- [ ] Tạo task/branch nhỏ theo task ID, thống nhất lịch review; không cần mở một branch khổng lồ cho cả tháng.

### 10.2. Bảy ngày làm việc đầu — lịch đề xuất

`D1` là ngày nhóm thực sự bắt đầu; đây là lịch triển khai đề xuất, không thay hạn nộp chính thức.

| Ngày | Thiên                                        | Khoa                                                | Khải                                                    | Điểm kiểm tra cuối ngày                                     |
| ---- | -------------------------------------------- | --------------------------------------------------- | ------------------------------------------------------- | ----------------------------------------------------------- |
| D1   | T01 nguồn dữ liệu; T02 split                 | K01 metrics; K02 comparator                         | C01 xác nhận contracts/config; C02 seeds                | Schema thống nhất; tasks/PR đầu tiên rõ owner               |
| D2   | T02 save/load/validate; T03 stats/transforms | K03 engine với synthetic loader; K02 checkpoint I/O | C02 environment; C03 run metadata/history               | Unit tests nhỏ chạy; bàn giao interfaces thật               |
| D3   | T01 wrapper; T04 loaders; T07 Linear         | K04 MLP; K05 trainer orchestration                  | C03 hoàn thiện writer; C04 ghép smoke flow; C06 tests   | Linear/MLP đúng shape; loader và trainer nhận đúng contract |
| D4   | T05 prepare; hỗ trợ tích hợp data            | K05 smoke fit + K06 validation checkpoint           | Tách tests theo model, cập nhật test TODO; kiểm tra CLI | Một smoke run Linear và một MLP từ lệnh chung               |
| D5   | T06 EDA; viết data section                   | Xác minh checkpoint reload, metrics và history      | Kiểm tra provenance; bổ sung guide/CI phù hợp           | Reviewer chạy lại; sửa lỗi tích hợp trước tối ưu            |
| D6   | Hoàn thiện EDA/Linear; chuẩn bị CNN          | Hoàn thiện MLP/training; chạy validation phát triển | Reproduction check; chuẩn bị Transformer                | Artifacts không bị ghi đè; thiếu sót Draft được liệt kê     |
| D7   | Bàn giao data notes/figures                  | Bàn giao preliminary results/curves                 | Ghép draft report/Pages và task tuần sau                | Chốt danh sách đạt/chưa đạt, không ép ghi Done              |

Khoa không cần chờ dataset để viết metric/comparator/engine tests. Thiên không cần chờ Transformer để bàn giao data. Khải không cần đợi training xong mới viết artifact writer. Công việc độc lập thực hiện song song; phần dùng chung tích hợp hằng ngày.

### 10.3. Các gate

**Gate 1 — nối được data và engine:**

- Dataset wrapper, saved split, normalization và Batch đúng.
- Linear/MLP forward/backward hoạt động.
- Một epoch train/validation dùng chung, không có test leakage.
- Run directory/config/history/checkpoint có thật.

**Gate 2 — sẵn sàng A1 Draft:**

- EDA đạt nội dung yêu cầu; Linear/MLP chạy end-to-end.
- Có preliminary metrics/curves từ thí nghiệm thật, checkpoint reload và reproduction instructions.
- Config, split, seeds và môi trường được ghi; nội dung report/Pages đúng trạng thái.
- CNN có thể làm thêm nếu đủ tiến độ; chưa bắt buộc đủ cả năm model để qua phần triển khai Draft.

**Gate 3 — trước 20 main runs:**

- CNN/RNN/Transformer đã ghép cùng engine và pass tests liên quan.
- Mean/std, augmentation, budget/tuning/timing được duyệt; protocol và model configurations được freeze đúng thời điểm.
- Tuning chỉ dùng validation; khi chọn xong model config mới đánh giá final test.
- Cả năm model dùng cùng split/seeds/evaluator; cùng máy và cùng scope khi so sánh timing.

**Gate 4 — Final:** đủ số liệu/qualitative analysis, report, slides/video, Pages, AI disclosure, checkpoint/reconstruction và kiểm tra nộp bài theo yêu cầu.

Theo kế hoạch đang có: **A1 Draft 23/09/2026**, **A1 Final 21/10/2026**, 23:59 GMT+7; đối chiếu LMS nếu có cập nhật. Xem [PROJECT_PLAN.md](PROJECT_PLAN.md) cho timeline toàn môn, không tự dời các mốc này vì lịch D1–D7.

### 10.4. Lệnh nào chạy ngay, lệnh nào chờ code?

Sau khi cài package trong môi trường của project, các kiểm tra hiện có:

```powershell
python -m dlbench.a1.cli --help
python -m dlbench.a1.cli validate-config --config configs/a1/models/linear.py --seed 36
python -m unittest discover -s tests -v
```

Các lệnh sau là **mục tiêu bàn giao**, hiện còn phụ thuộc các hàm ML chưa triển khai:

```powershell
python -m dlbench.a1.cli prepare --config configs/a1/models/linear.py
python -m dlbench.a1.cli train --config configs/a1/models/linear.py --seed 36 --smoke
python -m dlbench.a1.cli train --config configs/a1/models/mlp.py --seed 36 --smoke
```

Muốn eval/analyze phải dùng đường dẫn checkpoint/run thực đã sinh, không sao chép một đường dẫn ví dụ không tồn tại. Chưa có lệnh `eda`/`compare` riêng; C04 sẽ quyết định và bổ sung.

## 11. Chạy thí nghiệm, báo cáo và nộp bài

### 11.1. Ai chịu trách nhiệm kết quả model nào?

| Owner | Model              | Seeds phải lưu đủ cho Final     | Số main runs |
| ----- | ------------------ | ------------------------------- | ------------ |
| Thiên | Linear, CNN        | 36, 69420, 67, 69 cho mỗi model | 8            |
| Khoa  | MLP, LSTM hoặc GRU | 36, 69420, 67, 69 cho mỗi model | 8            |
| Khải  | Transformer        | 36, 69420, 67, 69               | 4            |

Owner chịu trách nhiệm model/config và xử lý lỗi, **không nhất thiết chạy trên máy riêng**. Lập hàng đợi trên máy benchmark chung; không chạy nhiều job tranh GPU trong lúc đo timing. Khoa kiểm tra evaluator, Khải kiểm tra môi trường/provenance, reviewer xác minh run đủ dữ liệu.

Khoa lưu bảng từng seed trước khi tổng hợp. **Đề xuất báo cáo:** mean và sample standard deviation (`ddof=1`) qua bốn seed, accuracy/F1 thể hiện rõ fraction hay percent; cả nhóm duyệt cách này trước khi freeze output schema. Không coi bốn seed là bốn split hoặc là số trials tuning. Với timing, lưu phép đo lặp trong mỗi run riêng với biến thiên giữa các seed, không gộp hai loại std mà không giải thích.

### 11.2. Phân chia báo cáo theo sections hiện có

**File:** reports/a1/report.md.

| Phần                                        | Người viết chính                            | Người cung cấp/review                                    |
| ------------------------------------------- | ------------------------------------------- | -------------------------------------------------------- |
| Thông tin nhóm, course, revision/date       | Khải                                        | Cả nhóm xác nhận                                         |
| 1.1 Objective                               | Khải                                        | Thiên/Khoa                                               |
| 1.2 Dataset; 1.3 EDA; 1.4 Split/leakage     | Thiên                                       | Khoa kiểm tra số liệu                                    |
| 2.1 Shared pipeline/preprocessing           | Thiên phần data; Khoa phần engine           | Khải ghép sơ đồ/contract                                 |
| 2.2 Models                                  | Owner từng model                            | Reviewer tương ứng                                       |
| 2.3 Training/tuning; 2.4 Evaluation/timing  | Khoa                                        | Khải                                                     |
| 2.5 Reproducibility                         | Khải                                        | Thiên/Khoa cung cấp run/environment                      |
| 3.1 Main comparison; 3.2 Learning dynamics  | Khoa                                        | Owner giải thích run/model của mình                      |
| 3.3 Error analysis                          | Thiên                                       | Khoa/Khải                                                |
| 3.4 Representation/inductive bias           | Khải ghép                                   | Mỗi owner viết phân tích model, không chỉ đọc định nghĩa |
| 3.5 Trade-offs/limitations; 3.6 Conclusion  | Cả ba, Khải tích hợp                        | Đối chiếu kết luận với số liệu thật                      |
| 3.7 Contributions/AI; 3.8 Sources/artifacts | Mỗi người cập nhật phần mình, Khải kiểm tra | Cả nhóm                                                  |

Ở đây inductive bias là đặc điểm kiến trúc/biểu diễn, không chỉ là mất cân bằng lớp. Không kết luận model tốt nhất chỉ từ accuracy. Mỗi figure/table phải có caption, đơn vị, split, run/protocol reference và người kiểm tra.

### 11.3. File ngoài code và người giữ

| Folder/file                                              | Owner/đầu ra                                                                               |
| -------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `docs/assets/a1/`                                        | Thiên cung cấp EDA/error figures; Khoa cung cấp curves/tables; Khải duyệt tên/link/caption |
| `docs/a1.md`, `docs/index.md`                            | Khải tích hợp và kiểm tra link; giữ thông tin thật                                         |
| `reports/a1/presentation-outline.md`                     | Khải điều phối; cả ba nói phần mình và hiểu phần chung                                     |
| `README.md`, `IMPLEMENTATION_GUIDE.md`                   | Khải cập nhật lệnh thật; owner chạy lại lệnh phần mình                                     |
| `PROJECT_PLAN.md`, tài liệu này                          | Khải cập nhật trạng thái theo xác nhận owner/reviewer                                      |
| `AI_USAGE.md`                                            | Cả ba ghi sử dụng/kiểm chứng; Khải kiểm tra đủ và nhất quán                                |
| `environment/`                                           | Khải tổng hợp versions/hardware; mọi máy dùng cho runs phải được ghi                       |
| `data/`, `runs/`, `checkpoints/`                         | Thiên quản lý dataset; mỗi owner quản lý runs/checkpoints; Khoa/Khải kiểm tra evidence     |
| `configs/a2/`, `configs/a3/`, `docs/a2.md`, `docs/a3.md` | Chưa chia implementation trong A1; giữ đúng trạng thái theo kế hoạch toàn môn              |

## 12. Checklist nhận việc và hoàn thành

### Thiên nhận việc ngay

- [ ] T01: nguồn Fashion-MNIST + thiết kế wrapper/IDs.
- [ ] T02: split/save/load/validation + test overlap.
- [ ] T03: train-only normalization + transforms.
- [ ] T04–T05: loaders + prepare; bàn giao cho Khoa/Khải.
- [ ] T07 Linear: model + tests, ghép vào engine.
- [ ] T06: EDA và sections dữ liệu trong report.
- [ ] Sau Gate 2: CNN, T08 error analysis và đủ 8 main runs của hai model.

### Khoa nhận việc ngay

- [ ] K01: accuracy/macro-F1/parameter count + toy tests.
- [ ] K02: comparator + checkpoint I/O, bật tests từng phần.
- [ ] K03: train/evaluate epoch với synthetic batches.
- [ ] K04 MLP: model + tests.
- [ ] K05–K06: fit/smoke/validation checkpoint evaluation và predictions.
- [ ] Ghép Linear/MLP, kiểm tra data của Thiên.
- [ ] Sau Gate 2: RNN, timing/compare runs, đủ 8 main runs của hai model.

### Khải nhận việc ngay

- [ ] C01: xác nhận config/contract; thống nhất schema mới, không viết lại phần đang chạy.
- [ ] C02: seeds/workers/environment, bàn giao sớm.
- [ ] C03: run directories/metadata/history/metrics.
- [ ] C04/C06: ghép CLI, thiết kế smoke policy, chỉnh test TODO/draft và test-per-model.
- [ ] Hỗ trợ clean-run reproduction cho Linear/MLP; tích hợp tài liệu Draft.
- [ ] Sau Gate 2: Transformer + tests, đủ 4 main runs; tích hợp analysis/Final package.

### Definition of Done cho mỗi task

- [ ] Có code thật trong đúng hàm/file, không trả kết quả giả để vượt test.
- [ ] Input/output khớp contract; config và metadata đủ cho người nhận.
- [ ] Tests đúng chức năng đã bật và chạy; ghi rõ tests chưa chạy/lý do.
- [ ] PR có lệnh tái kiểm tra, kết quả thật, artifact cần xem và limitations.
- [ ] Reviewer đã chạy/đọc phần cần thiết; thay đổi file dùng chung được thống nhất.
- [ ] Owner bổ sung method notes, số liệu/figures thật và AI usage nếu có.
- [ ] Không commit dataset, credentials, logs thô hoặc weights lớn.

Suggested branch theo phần việc: `feat/a1-data-split`, `feat/a1-loaders-linear`, `feat/a1-metrics-engine`, `feat/a1-trainer-mlp`, `feat/a1-repro-artifacts`, `feat/a1-transformer`. Đây là tên gợi ý, tài liệu không tự tạo branch/PR.

Mỗi PR ghi ngắn: **Task ID — owner — reviewer — file/hàm đổi — lệnh test — output bàn giao — phần chưa xong**. Một người khác phải dùng được kết quả mà không cần hỏi lại cách đặt key/shape/đường dẫn.

Tài liệu liên quan: [Hướng dẫn triển khai](IMPLEMENTATION_GUIDE.md), [Quy trình đóng góp](CONTRIBUTING.md), Experiment contract, [Kế hoạch toàn môn](PROJECT_PLAN.md).
