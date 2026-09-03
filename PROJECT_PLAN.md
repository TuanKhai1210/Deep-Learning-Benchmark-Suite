# CO3133 - Kế hoạch dự án cho nhóm 3 người

Ngày bắt đầu: **02/09/2026**  
Nhóm: 

## 1. Các mốc chính

| Mốc | Hạn chính thức | Hạn nội bộ |
|---|---:|---:|
| Group + GitHub Pages skeleton | 09/09/2026 | 08/09/2026 |
| A1-M1 Draft | 23/09/2026 | 22/09/2026 |
| A2-M1 Dataset Proposal | 07/10/2026 | 06/10/2026 |
| A1-M2 Final | 21/10/2026 | 20/10/2026 |
| A2-M2 Draft | 28/10/2026 | 27/10/2026 |
| A2-M3 Final | 11/11/2026 | 10/11/2026 |
| A3-M1 Dataset Proposal | 18/11/2026 | 17/11/2026 |
| A3-M2 Draft | 25/11/2026 | 24/11/2026 |
| A3-M3 Final | 02/12/2026 | 01/12/2026 |

## 2. Vai trò xuyên suốt

| Thành viên | Vai trò chính | Assignment 1 | Reviewer chính |
|---|---|---|---|
| A | Data & Vision Lead | EDA, split, preprocessing, Linear, CNN, error analysis | B |
| B | Training & Evaluation Lead | Trainer/evaluator, MLP, LSTM/GRU, metrics, timing | C |
| C | Reproducibility & Sequence Lead | Config, tests, logging, Transformer, Pages/report integration | A |

Vai trò là trách nhiệm chính, không phải vùng làm việc độc quyền. Mỗi hạng mục phải có ít nhất một người khác review.

## 3. Quy tắc làm việc chung

1. Không tạo ba notebook hoặc ba training loop độc lập.
2. Tất cả model dùng chung split, DataLoader contract, trainer, evaluator và metrics.
3. Mỗi model chỉ cung cấp model module, config và representation adapter riêng.
4. Không dùng test set để tune hyperparameter hoặc chọn checkpoint.
5. Main benchmark phải chạy trên cùng máy/hardware và cùng phương pháp đo thời gian.
6. Mỗi task hoàn thành qua branch và Pull Request; không push thẳng vào `main` sau giai đoạn khởi tạo.
7. AI hỗ trợ phải được ghi vào `AI_USAGE.md` trong cùng ngày hoặc cùng Pull Request.

Review vòng tròn:

- B review công việc của A.
- C review công việc của B.
- A review công việc của C.

## 4. Experiment contract cần chốt trước khi code

Deadline: **03/09/2026**

- Dataset chính: Fashion-MNIST qua `torchvision`.
- MNIST chỉ dùng cho smoke test/debug; CIFAR-10 chỉ là extension.
- Split đề xuất: 50.000 train / 10.000 validation / 10.000 official test.
- Seed đề xuất: `42`.
- Lưu split indices hoặc split manifest trong `configs/a1/splits/`.
- Validation split có stratification; ghi exact sample counts.
- Augmentation chỉ áp dụng cho train.
- Chọn preprocessing/normalization dùng chung.
- Metrics: accuracy, macro-F1, parameter count, training time, inference time.
- Checkpoint rule đề xuất: validation macro-F1 cao nhất.
- Chốt cách đo inference time: cùng hardware, batch size, warm-up và số lần lặp.
- Lưu config, log, checkpoint path, predictions và Git commit cho mỗi main run.

Đầu ra: `docs/a1-experiment-contract.md` hoặc một section tương đương trong `docs/a1.md`.

## 5. Timeline Assignment 1

### 02/09-03/09: Chốt protocol và interface

| Người | Công việc | Đầu ra |
|---|---|---|
| A | Kiểm tra Fashion-MNIST, class labels, split strategy và EDA checklist | Data note + split proposal |
| B | Thiết kế trainer/evaluator API, metrics và checkpoint rule | Trainer/evaluator contract |
| C | Chốt config schema, seed utilities, folder structure và command interface | Config/reproducibility contract |
| Cả nhóm | Duyệt experiment contract | Contract được approve bởi 3 người |

Gate: chưa triển khai model riêng trước khi contract được thống nhất.

### 04/09-08/09: Xây nền tảng và Pages skeleton

| Người | Công việc | Đầu ra |
|---|---|---|
| A | Download bằng `torchvision`; EDA; stratified split; Dataset/DataLoader; Linear model | EDA figures, split manifest, data module, Linear smoke run |
| B | Train/validation/test loop; CrossEntropyLoss; macro-F1; timing; checkpoint; MLP | Shared engine + MLP smoke run |
| C | Config loader; deterministic seed; logging; model interface tests; README/Pages skeleton | Repro utilities, tests, site links |

**08/09 internal freeze:** repo, landing page, A1/A2/A3 links, group information và `AI_USAGE.md` hoạt động.  
**09/09 official gate:** Group + GitHub Pages skeleton.

### 09/09-14/09: Tích hợp Linear và MLP

| Người | Công việc | Đầu ra |
|---|---|---|
| A | Hoàn thiện preprocessing/EDA; chạy Linear qua shared engine | Linear config, curves, metrics |
| B | Hoàn thiện MLP và evaluator; tạo confusion matrix/prediction export | MLP config, curves, metrics |
| C | Integration tests; clean command; draft report/Pages; bắt đầu CNN nếu nền tảng ổn định | Repro smoke test + report skeleton |
| Cả nhóm | Review split, sample counts, transforms và test-set policy | Protocol freeze |

### 15/09-19/09: Chuẩn bị A1 Draft

| Người | Công việc | Đầu ra |
|---|---|---|
| A | EDA narrative và leakage analysis; correct/incorrect examples ban đầu | Draft Part 1 |
| B | Results table cho Linear/MLP; kiểm tra parameter count và timing | Preliminary results table |
| C | README commands, config documentation, Pages A1 và draft methodology | Reproducible draft package |
| Cả nhóm | Một người khác chạy lại từ clean clone | Reproduction record |

Song song nhưng giới hạn tối đa 60 phút/người: mỗi thành viên đề xuất một candidate task/dataset cho A2, kèm source, license, quy mô, metric và compute sơ bộ. Việc này không được làm chậm A1 Draft.

### 20/09-23/09: Freeze và nộp A1-M1

- **20/09:** code freeze cho Linear/MLP và data pipeline.
- **21/09:** result/figure freeze.
- **22/09:** report, Pages, README, AI disclosure và link QA.
- **23/09:** buffer và nộp trước 23:59 GMT+7.

A1-M1 Definition of Done:

- EDA bắt buộc hoàn chỉnh.
- Dataset/DataLoader và train/validation loop chạy được.
- Linear và MLP chạy end-to-end.
- Cùng frozen split và seed.
- Có preliminary metrics/curves.
- Có hướng dẫn chạy và Pages draft.
- CNN là optional ở mốc này.

### 24/09-30/09: Ba model còn lại

| Người | Model chính | Đầu ra |
|---|---|---|
| A | CNN tự thiết kế | Model, config, shape test, tiny-batch test, method note |
| B | LSTM hoặc GRU | Sequence adapter, model, config, tests, method note |
| C | Transformer | Token/patch projection, positional encoding, model, config, tests, method note |

Mỗi model phải chạy qua shared engine. Không chấp nhận training loop riêng.

Song song tối đa 15-20% effort cho A2 Proposal:

- A audit dataset source, license, size và khả năng truy cập.
- B đề xuất split/leakage controls, baseline và metrics.
- C dựng proposal skeleton, compute/risk table và dataset dự phòng.
- **30/09:** chốt task, dataset chính và một dataset dự phòng cho A2.

### 01/10-07/10: Tích hợp A1 và hoàn thành A2 Proposal

Phân bổ: **70% A1, 30% A2 Proposal**.

| Người | A1 | A2 Proposal |
|---|---|---|
| A | CNN integration/tuning trên validation | Dataset source, license, stats, split/leakage plan |
| B | LSTM/GRU integration/tuning | Metrics, baseline, pretrained model, compute estimate |
| C | Transformer integration/tuning | Tổng hợp proposal, task definition, repository/Pages record |

- **05/10:** review A2 Proposal.
- **06/10:** freeze.
- **07/10:** submit Proposal; không chạy main A2 experiments trước khi được duyệt.

### 08/10-12/10: Main A1 experiments

- Freeze model configs trước main runs.
- Chạy cả 5 model trên cùng benchmark machine.
- Lưu config, seed, checkpoint, training/inference time, predictions và Git commit.
- Chỉ tune bằng validation; official test dùng cho báo cáo cuối.
- Nếu đủ compute, lặp nhiều seed và báo mean +/- standard deviation.

Owner model chịu trách nhiệm debug; B kiểm tra evaluator/timing; reviewer xác minh artifact.

Nếu A2 Proposal đã ở trạng thái `Approved` hoặc `Approved with conditions`, không để A2 chờ tới sau A1 Final:

- A dành khoảng 25-30% thời gian cho dataset adapter và EDA A2.
- B dựng simple-baseline smoke config bằng shared engine.
- C chuẩn bị pretrained-model environment/config và cập nhật điều kiện phê duyệt.
- B/C vẫn ưu tiên A1; tổng effort A2 của cả nhóm không vượt khoảng 20% trong giai đoạn này.

Không chạy bất kỳ main A2 experiment nào trước khi proposal được phê duyệt.

### 13/10-17/10: Phân tích và result freeze

| Người | Công việc |
|---|---|
| A | Confusion matrices, correct/error examples, failure taxonomy, data limitations |
| B | Tổng hợp accuracy, macro-F1, parameters, train/inference time và curves |
| C | Representation/inductive-bias comparison, methodology integration, reproducibility table |
| Cả nhóm | Kết luận trade-off; không chọn model tốt nhất chỉ dựa trên accuracy |

- **17/10:** đóng băng toàn bộ số liệu và figure.

Trong thời gian chờ main A1 runs hoặc review, tiếp tục A2 adapter/baseline smoke work đã được phê duyệt. Mục tiêu đến **20/10**: A2 DataLoader chạy được, EDA khung đã có, simple baseline và pretrained path đều pass smoke test. Nhờ vậy A2 Draft không bắt đầu từ số 0 sau ngày 21/10.

### 18/10-21/10: Final package

- **18/10:** report và GitHub Pages content complete.
- **19/10:** slides/video recording; checkpoint/reconstruction instructions.
- **20/10:** clean-clone reproduction, link QA, AI disclosure audit, tag/release candidate.
- **21/10:** buffer và submit trước 23:59 GMT+7.

## 6. Definition of Done cho mỗi model

Một model chỉ được đánh dấu Done khi có đủ:

- Model module tuân thủ input/output contract.
- Config được version control.
- Unit/shape test.
- Tiny-batch overfit hoặc smoke test.
- Train/validation/test chạy bằng shared engine.
- Checkpoint được chọn theo rule đã chốt.
- Accuracy, macro-F1, parameter count, train/inference time.
- Curves, predictions và confusion matrix artifact.
- Method note giải thích architecture/input representation/loss.
- Reviewer chạy lại ít nhất smoke command.
- AI usage log được cập nhật nếu có AI hỗ trợ.

## 7. Timeline sau Assignment 1

### A2 - sau khi Proposal được duyệt

| Khoảng thời gian | A | B | C | Mốc |
|---|---|---|---|---|
| 22/10-27/10 | Dataset adapter + EDA | Simple baseline | Pretrained model + fine-tuning | Draft freeze 27/10 |
| 28/10 | Cả nhóm kiểm tra package | Cả nhóm | Cả nhóm | A2 Draft |
| 29/10-04/11 | Data/error analysis | Controlled experiment + compute | Fine-tuning/ablation integration | Experiment freeze |
| 05/11-09/11 | Qualitative analysis | Final result table | Report/Pages/slides/video integration | Package freeze 09/11 |
| 10/11 | Cross-check và buffer | Cross-check và buffer | Cross-check và buffer | Internal buffer |
| 11/11 | Submission | Submission | Submission | A2 Final |

Từ 05/11, cả nhóm dành một phiên ngắn để audit hai candidate dataset cho A3; chỉ chuẩn bị proposal, chưa chạy main experiment.

### A3 - sau khi Proposal được duyệt

| Khoảng thời gian | A | B | C | Mốc |
|---|---|---|---|---|
| 12/11-17/11 | Pair validity, split, modality A plan | Metrics, modality B baseline plan | Fusion/model plan + proposal editor | Proposal freeze 17/11 |
| 18/11 | Submit proposal | Submit proposal | Submit proposal | A3 Proposal |
| 19/11-24/11 | Unimodal A | Unimodal B | Simple fusion | Draft freeze 24/11 |
| 25/11 | Cả nhóm kiểm tra package | Cả nhóm | Cả nhóm | A3 Draft |
| 26/11-29/11 | Pair/error/support-conflict analysis | Compute/metrics/ablation runs | Improved fusion + integration | Result freeze 29/11 |
| 30/11-01/12 | Qualitative analysis | Result table/reproduction | Report/Pages/slides/video | Final freeze 01/12 |
| 02/12 | Buffer/submission | Buffer/submission | Buffer/submission | A3 Final |

## 8. Nhịp phối hợp hàng tuần

- Đầu tuần: 20 phút chốt task, owner, reviewer và acceptance criteria.
- Giữa tuần: 30 phút integration; mỗi người demo command chạy được.
- Cuối tuần: 45 phút review kết quả, blockers, report và AI log.
- Không để Pull Request mở quá 48 giờ mà không có review.
- Mỗi tuần phải có ít nhất một clean smoke run từ người không viết module đó.

## 9. Definition of Done cho milestone

Milestone chỉ hoàn thành khi:

- Core requirements đầy đủ; extension không thay thế core.
- Main results truy ngược được tới config, split, checkpoint, log và commit.
- README commands được chạy thử từ clean clone.
- Report, Pages, slides/video/checkpoint links đã kiểm tra ở chế độ đăng xuất.
- Tất cả thành viên hiểu và giải thích được phần không do mình viết.
- `AI_USAGE.md` đã được audit.
