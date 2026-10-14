# LAB 04 — Text Classification



## Cấu hình và cách chạy

Mỗi code cell gắn đúng một số mục, theo thứ tự **10 → 11 → 12 → 13 → 14 → 15 → 17 → 18 → 19 → 20 → 21 → 22 → 23 → 26**. 


1. Cài dependencies: `python -m pip install -r requirements.txt`.
2. Mở terminal tại `lab04/`. Mở `classification.ipynb` và chạy tất cả cell theo thứ tự, hoặc dùng lệnh bên dưới.
3. **Trước lần chạy đầu tiên**,  tự viết mục 5 và Prediction 1–5 ở mục 10. 


```powershell
$env:PYTHONHASHSEED = '42'
python -X utf8 -c "import nbformat; from nbclient import NotebookClient; from pathlib import Path; p=Path('classification.ipynb'); n=nbformat.read(p,as_version=4); c=NotebookClient(n,timeout=1800,kernel_name='python3',resources={'metadata':{'path':str(Path.cwd())}}); c.execute(); nbformat.write(n,p)"
```

Seed 42. Giữ official test, loại duplicate theo cùng tokenizer giữa các split; tách validation 20% từ official train đã clean, stratify theo nhãn. Giữ negation trong tokenizer. TF-IDF fit trên train; Word2Vec train trên train; StandardScaler của average embeddings fit trên train. C chọn trong `0.1/1/10` bằng F1 positive trên validation, test không dùng chọn cấu hình hoặc model application.

TF-IDF baseline unigram có cap 20.000, `min_df=2`, sublinear TF. Uni+bigram cap 40.000. Word2Vec Skip-gram: dimension 100, window 5, min_count 2, negative 5, epochs 10, workers 1. Các model dùng đúng cùng train/validation/test. Notebook giữ bốn model: unigram LR, Average Word2Vec LR, unigram Linear SVM và unigram+bigram LR. Các thí nghiệm vocabulary/stopwords và threshold bổ sung trước đây được lưu cùng bản notebook cũ ngoài thư mục nộp, tại `tmp/w4/before_section_order_fix/` ở thư mục project.

Các cột precision/recall/F1 chính là binary, positive label=1. Macro/weighted F1 và per-class metrics được lưu riêng. Training time = thời gian fit representation trên train (bao gồm average+scaler với Word2Vec) + thời gian fit classifier được chọn. Thời gian thử cả C grid lưu riêng; transform validation/test không tính vào training time. Các model dùng chung TF-IDF chia sẻ representation fit nên không cộng training time giữa các dòng để tính thời gian toàn bộ run.

Đây là so sánh pipeline đã ghi cấu hình, không phải bằng chứng mọi cấu hình TF-IDF/Word2Vec đều có thứ hạng tương tự. W2V train riêng trên tập train IMDb; không dùng pretrained embeddings. Synthetic probes ở mục 17 và application không phải benchmark có nhãn. Dữ liệu chỉ được loại trùng theo token, chưa thực hiện phát hiện near-duplicate.


## AI assistance statement — mục 24

- **Tool:** ChatGPT
- **Purpose:** Hỗ trợ preprocessing, dùng sklearn, documentation, visualization, debugging và kiểm tra kỹ thuật.
- **Generated:** `preprocessing.py`; code cells trong `classification.ipynb`;  `requirements.txt`; khung Markdown,
- **Modified:** Tạo thư mục `lab04/` mới; bổ sung hướng dẫn và liên kết trong các Markdown; tách cell theo từng mục và sắp xếp tăng dần; bỏ các thí nghiệm bổ sung ngoài cấu trúc mục chính. 
- **Verified:** kiểm tra execution/output, cấu trúc code-only, splits không trùng token document, train-only fitting, metrics/CSV/hình. 

