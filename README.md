# Data Mining HW3

本專案為資料探勘課程第三次作業，目標為二元分類（預測 ACTION 0/1），包含資料前處理、模型訓練、參數調整、特徵工程與多次 Kaggle submission。

## 專案結構

- `preprocess.md`：資料前處理、特徵分析、模型流程與改進紀錄
- `submission_results.md`：四次 submission 的模型設定與驗證集結果
- `submission1_code.py`：Submission 1（基礎 Random Forest）
- `submission2_code.py`：Submission 2（參數調優 Random Forest）
- `submission3_code.py`：Submission 3（類別權重調整 Random Forest）
- `submission4_code.py`：Submission 4（特徵工程 + 集成模型）
- `submission_v1.csv` ~ `submission_v4.csv`：各次 submission 的預測結果檔案
- `dm_hw3.py`：原始整合流程腳本

## 如何執行

1. 請先準備 `train.csv`、`test.csv` 於同目錄（已被 .gitignore 排除，不會上傳）
2. 執行對應的 submission code 產生預測結果
   ```bash
   python submission1_code.py
   python submission2_code.py
   python submission3_code.py
   python submission4_code.py
   ```
3. 產生的 `submission_v*.csv` 可直接上傳 Kaggle

## 內容說明

- 每個 submission code 對應一個模型版本，便於追蹤與分析
- `preprocess.md` 詳細記錄資料分析、前處理、模型選擇與調整過程
- `submission_results.md` 彙整每次模型的驗證集分數與預測分佈

## 注意事項

- 本 repo 不包含原始資料（train.csv, test.csv, sampleSubmission.csv, DM_Homework_3.pdf）
- 若需重現結果，請自行放置資料檔案於專案資料夾

---

如有問題請聯絡作者。
