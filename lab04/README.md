# LAB 04 — Text Classification

Theo W4, hạn nộp **23:59 ngày 14/10/2026**. Dataset: [IMDb, Stanford NLP trên Hugging Face](https://huggingface.co/datasets/stanfordnlp/imdb), revision `e6281661ce1c48d982bc483cf8a173c1bbeb5d31`. Application: **Option A — Sentiment classification**, negative=0, positive=1.

Notebook chỉ có code và output máy sinh. Các phần prediction, calculation, interpretation của metrics, lựa chọn nguyên nhân lỗi, reflection và individual learning check phải do người học tự làm theo mục 24. Không có câu trả lời AI cho các phần này trong notebook hoặc Markdown.

## Phân công từng mục trong đề

[Dàn ý điền trống, checklist và nguồn output cho từng câu tự viết](HUONG_DAN_VIET_TAY.md). Đây là tài liệu hướng dẫn thêm, không phải câu trả lời để nộp và không thuộc danh sách deliverables bắt buộc.

| Mục W4 | Phần code/output | Phần tự viết, không dùng AI | Có cần dùng output để viết? | File chứa PDF scan |
|---|---|---|---|---|
| 1–4 | Bối cảnh/lý thuyết, không có code riêng bắt buộc | Ôn pipeline supervised classification, không có câu nộp riêng | Không | Không có file riêng |
| 5 | Hai representation được triển khai ở mục 15 | Dự đoán representation phù hợp với IMDb và giải thích, trước khi chạy | **Không; không xem kết quả trước** | `prediction.md` |
| 6 — Bài 1 | Không code giải bài tính tay | Tính logit z, sigmoid và P(y=1\|x) từ số liệu đề | **Không** | `calculations.md` |
| 7 — Bài 2 | Không code giải bài tính tay | Dự đoán class với hai threshold và giải thích precision/recall | **Không; giải từ dữ kiện đề** | `calculations.md` |
| 8 — Bài 3 | Không code giải ma trận số liệu đề | Tính precision, recall, F1; nêu ý nghĩa FP/FN trong spam | **Không** | `calculations.md` |
| 9 — Bài 4 | Không code giải bài accuracy trap | Đánh giá model luôn dự đoán negative bằng các metric thích hợp | **Không** | `calculations.md` |
| 10 | Xác nhận prediction đã được tự viết; khởi tạo môi trường, không huấn luyện model | Tự viết Prediction 1–5, đủ Prediction/Reason/Confidence, trước khi chạy | **Không; không xem kết quả trước** | `prediction.md` |
| 11 | Samples, classes, distribution, lengths, duplicates, missing text/labels; vẽ distribution | Trả lời dataset có imbalance không | **Có**: audit và class counts/percentages | `reflection.md` |
| 12 | Clean/deduplicate, stratified train/validation split, test riêng, fit train-only, overlap audit | Vì sao fit TF-IDF trước split là data leakage? | **Không bắt buộc**; có thể dẫn audit thực tế | `reflection.md` |
| 13 | Định nghĩa các hàm cho pipeline TF-IDF → classifier → prediction/evaluation; chưa train | Hiểu công thức logit/sigmoid; phần tính nằm ở Bài 1 | Không cần câu nộp riêng ngoài Bài 1 | `calculations.md` |
| 14 | Train baseline; training time, vocabulary, accuracy, precision, recall, F1 trên validation | Diễn giải kết quả baseline; final test metrics ở mục 19 | **Có** | `reflection.md` |
| 15 | TF-IDF + LR, Average Word2Vec + LR; thêm TF-IDF + Linear SVM; so sánh trên validation | Phần giải thích representation chuyển sang mục 16 | **Có** | `reflection.md` |
| 16 | Bảng so sánh, dimension, OOV coverage | Vì sao hai representation cho kết quả khác nhau? | **Có**: bảng cùng test split và thông tin cấu hình | `reflection.md` |
| 17 | S1/S2 đúng đề, probability, cosine; thêm hoán vị từ, không viết kết luận | Representation có hiểu vai trò của `not` không? | **Có**: probe vectors và predictions | `reflection.md` |
| 18 | Unigram vs unigram+bigram: vocabulary, F1, training time trên validation | Vì sao bigram có thể giúp classification nhưng làm feature space lớn hơn? Nhận xét thí nghiệm; final test metrics ở mục 19 | **Có** cho nhận xét thí nghiệm; phần lý thuyết không phụ thuộc output | `reflection.md` |
| 19 | Accuracy, precision, recall, F1; thêm macro/weighted và per-class metrics | Interpretation của metrics | **Có** | `reflection.md` |
| 20 | Confusion matrix bắt buộc; baseline PNG và hình so sánh | Class nào nhầm nhiều nhất, loại error nào nhiều, metric nào phản ánh? | **Có** | `error_analysis.md` |
| 21 | CSV đầy đủ text/true/predicted; candidate pool đúng/sai; không có cause labels | Tự chọn ít nhất 3 đúng + 3 sai, tự phân tích từng mẫu và nguyên nhân lỗi | **Có**: nguyên văn, nhãn thật/dự đoán và evidence | `error_analysis.md` |
| 22 | Top 20 positive/negative features và weights; numeric feature contributions | Các feature có ý nghĩa với task không? | **Có** | `error_analysis.md` |
| 23 | Application A: sentiment từ IMDb, predict review mới; chọn model bằng validation | Nếu viết nhận xét application, tự viết | **Có nếu nhận xét** | `reflection.md` |
| 24 | AI assistance statement dưới đây | Người học tự kiểm tra và bổ sung phần Modified/Verified của mình | Không cần scan riêng | `README.md` |
| 25 | Không code trả lời | Tự ôn để trả lời ngẫu nhiên một câu trong 3 phút | **Không bắt buộc** | Không có file nộp riêng; danh sách câu trong `reflection.md` |
| 26–28 | Kiểm tra deliverables; số liệu/hình phục vụ rubric và peer assessment | Hoàn thành scan và tự kiểm tra các câu trả lời | Có với các mục định lượng | Bốn file `.md` tương ứng |
| 29 | Không experiment riêng bắt buộc | Ôn mạch LAB01→04 và giới hạn representation | Không bắt buộc | `reflection.md` (tùy chọn) |

## Cấu hình và cách chạy

Mỗi code cell gắn đúng một số mục, theo thứ tự **10 → 11 → 12 → 13 → 14 → 15 → 17 → 18 → 19 → 20 → 21 → 22 → 23 → 26**. Các mục chỉ yêu cầu tự viết/lý thuyết không có cell Markdown trả lời hoặc cell giả để lấp số.

Mục 13 chỉ định nghĩa implementation. Mục 14 train baseline, mục 15 train/so sánh representation, mục 18 train/so sánh n-gram. Các bảng tại ba mục này ghi rõ **validation**. Mục 19 đánh giá các cấu hình đã chốt trên **test**, xuất `results.csv` và per-class metrics. Mục 23 thực hiện application. Không có cell gộp nhiều số mục.

1. Cài dependencies: `python -m pip install -r requirements.txt`.
2. Mở terminal tại `lab04/`. Mở `classification.ipynb` và chạy tất cả cell theo thứ tự, hoặc dùng lệnh bên dưới.
3. **Trước lần chạy đầu tiên**, người học phải tự viết mục 5 và Prediction 1–5 ở mục 10. Người dùng đã xác nhận hoàn thành bước này trước lần chạy do ChatGPT hỗ trợ trong phiên làm việc này. Scan phần prediction sau đó; giữ nguyên nội dung đã viết trước experiment.
4. Hai parquet có nhãn được tải và kiểm tra SHA256 trong `data/`; không dùng unsupervised split. Có thể giữ cache để chạy lại, không cần đưa parquet lên repository nộp bài nếu giảng viên không yêu cầu.

```powershell
$env:PYTHONHASHSEED = '42'
python -X utf8 -c "import nbformat; from nbclient import NotebookClient; from pathlib import Path; p=Path('classification.ipynb'); n=nbformat.read(p,as_version=4); c=NotebookClient(n,timeout=1800,kernel_name='python3',resources={'metadata':{'path':str(Path.cwd())}}); c.execute(); nbformat.write(n,p)"
```

Seed 42. Giữ official test, loại duplicate theo cùng tokenizer giữa các split; tách validation 20% từ official train đã clean, stratify theo nhãn. Giữ negation trong tokenizer. TF-IDF fit trên train; Word2Vec train trên train; StandardScaler của average embeddings fit trên train. C chọn trong `0.1/1/10` bằng F1 positive trên validation, test không dùng chọn cấu hình hoặc model application.

TF-IDF baseline unigram có cap 20.000, `min_df=2`, sublinear TF. Uni+bigram cap 40.000. Word2Vec Skip-gram: dimension 100, window 5, min_count 2, negative 5, epochs 10, workers 1. Các model dùng đúng cùng train/validation/test. Notebook giữ bốn model: unigram LR, Average Word2Vec LR, unigram Linear SVM và unigram+bigram LR. Các thí nghiệm vocabulary/stopwords và threshold bổ sung trước đây được lưu cùng bản notebook cũ ngoài thư mục nộp, tại `tmp/w4/before_section_order_fix/` ở thư mục project.

Các cột precision/recall/F1 chính là binary, positive label=1. Macro/weighted F1 và per-class metrics được lưu riêng. Training time = thời gian fit representation trên train (bao gồm average+scaler với Word2Vec) + thời gian fit classifier được chọn. Thời gian thử cả C grid lưu riêng; transform validation/test không tính vào training time. Các model dùng chung TF-IDF chia sẻ representation fit nên không cộng training time giữa các dòng để tính thời gian toàn bộ run.

Đây là so sánh pipeline đã ghi cấu hình, không phải bằng chứng mọi cấu hình TF-IDF/Word2Vec đều có thứ hạng tương tự. W2V train riêng trên tập train IMDb; không dùng pretrained embeddings. Synthetic probes ở mục 17 và application không phải benchmark có nhãn. Dữ liệu chỉ được loại trùng theo token, chưa thực hiện phát hiện near-duplicate.

## Output cần mở khi tự viết

| Mục | Output trong notebook / CSV |
|---|---|
| 11–12 | `outputs/dataset_audit_raw.csv`, `class_distribution_raw.csv`, `document_lengths_raw.csv`, `cleaning_audit.csv`, `split_distribution.csv`, `split_overlap_audit.csv` |
| 14 | `outputs/baseline_validation.csv`; final test metrics trong `results.csv` ở mục 19 |
| 15 | `outputs/representation_validation.csv`; final test metrics trong `results.csv` ở mục 19 |
| 18 | `outputs/ngram_validation.csv`; final test metrics trong `results.csv` ở mục 19 |
| 16, 19 | `results.csv`, `outputs/per_class_metrics.csv`, `locked_model_configs.csv`, `validation_hyperparameters.csv`, `word2vec_coverage_*.csv` |
| 17 | `outputs/negation_word_order_predictions.csv`, `word_order_vector_comparison.csv` |
| 20 | `confusion_matrix.png`, `outputs/confusion_matrices_comparison.png`, TN/FP/FN/TP trong `results.csv` |
| 21 | `outputs/error_analysis_candidates.csv`, `six_baseline_examples.csv`, `predictions_<model>.csv`; CSV giữ nguyên văn đầy đủ |
| 22 | `outputs/top_model_weights.csv`, `candidate_feature_contributions.csv` |
| 23 | `outputs/application_predictions.csv`, `application_selection.json` |

## Scan PDF và chèn vào Markdown

Tự viết rồi scan bốn phần thành `calculations_scan.pdf`, `prediction_scan.pdf`, `error_analysis_scan.pdf`, `reflection_scan.pdf`. Đặt cạnh các file `.md` tương ứng trong `lab04/`. Mỗi Markdown đã có link tương đối và khối nhúng PDF bằng HTML. Trình xem không hỗ trợ nhúng PDF vẫn mở được link. Khi nộp phải gửi cả **Markdown và PDF scan**, vì link không chứa nội dung PDF.

Notebook đã chạy chỉ là phần code hoàn thành. Bài nộp chỉ đầy đủ sau khi người học bổ sung các scan. Không tạo PDF giả hoặc AI answers thay cho scan của người học.

## Deliverables theo mục 26

`README.md`, `calculations.md`, `prediction.md`, `classification.ipynb` (code + output), `preprocessing.py`, `results.csv`, `confusion_matrix.png`, `error_analysis.md`, `reflection.md`. Bổ sung bốn PDF scan theo quy cách người dùng yêu cầu. Giữ `outputs/` để tham khảo evidence và `requirements.txt` để chạy lại.

## Trạng thái lần chạy đã lưu

Notebook đã chạy lại thành công **14/14 code cells**, mỗi cell đúng một mục, thứ tự số mục tăng dần, có output và không có error output. Bản mới có bốn cấu hình model. Train/validation/test sau clean có lần lượt **19.920 / 4.981 / 24.675** mẫu. Metrics của cả bốn model được kiểm tra khớp với dự đoán đã lưu; các bảng validation/test ghi đúng split; document keys và sample IDs của split manifest không trùng. Các hình đã được kiểm tra hiển thị.

Chi tiết kiểm tra nằm ở `outputs/verification.json`; cấu hình và checksums ở `outputs/run_metadata.json`. Bốn PDF scan đang thiếu và phải được người học tự bổ sung. Các số liệu trong phần này chỉ xác nhận việc chạy và kiểm tra kỹ thuật, không thay phần diễn giải của người học.

## AI assistance statement — mục 24

- **Tool:** ChatGPT / Codex.
- **Purpose:** Hỗ trợ preprocessing, dùng sklearn, documentation, visualization, debugging và kiểm tra kỹ thuật.
- **Generated:** `preprocessing.py`; code cells trong `classification.ipynb`; biểu đồ và CSV do code sinh; `requirements.txt`; khung Markdown, bảng phân công, hướng dẫn nộp và `HUONG_DAN_VIET_TAY.md` (checklist/khung điền trống).
- **Modified:** Tạo thư mục `lab04/` mới; bổ sung hướng dẫn và liên kết trong các Markdown; tách cell theo từng mục và sắp xếp tăng dần; bỏ các thí nghiệm bổ sung ngoài cấu trúc mục chính. Không sửa nội dung các lab trước. Người học cần tự bổ sung các chỉnh sửa sau này nếu có.
- **Verified:** Chạy notebook từ đầu đến cuối; kiểm tra execution/output, cấu trúc code-only, splits không trùng token document, train-only fitting, metrics/CSV/hình. Người học cần tự kiểm tra nội dung, tự diễn giải số liệu và hoàn thành phần Verified của mình trước khi nộp.

Nội dung prediction, calculation, interpretation, nguyên nhân error, reflection và câu trả lời individual check không được AI tạo.
