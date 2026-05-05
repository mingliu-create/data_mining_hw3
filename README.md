# Data Mining HW3

本專案為資料探勘課程第三次作業，目標為二元分類（預測 ACTION 0/1），包含資料前處理、模型訓練、參數調整、特徵工程與多次 Kaggle submission。

## 專案結構

### 文件
- `preprocess.md`：資料前處理、特徵分析、類別不平衡處理與模型流程
- `submission_results.md`：所有 submission 的模型設定與驗證集結果
- `model_improve.md`：模型改進與參數選擇說明
- `README.md`：專案說明

### 程式碼
| 檔案 | 用途 | 輸出 |
|------|------|------|
| `dm_hw3.py` | 只做資料前處理與資料檢查：讀取 `train.csv` / `test.csv`、檢查缺失值、欄位型態、ACTION 分佈、唯一值數量，並切分 train/validation | 無 CSV 輸出 |
| `imbalanced_handling.py` | 比較資料不平衡處理方法：SMOTE、Random Undersampling、SMOTETomek、class_weight 等 | `submission_imbalanced.csv`, `submission_imbalanced_prob.csv` |
| `submission1_code.py` | Submission 1：基礎 Random Forest baseline | `submission_v1.csv` |
| `submission2_code.py` | Submission 2：Random Forest + GridSearchCV，使用 F1-Score 作為調參目標 | `submission_v2.csv` |
| `submission3_code.py` | Submission 3：Random Forest + threshold tuning，用驗證集 F1-Score 搜尋最佳 threshold | `submission_v3.csv` |
| `submission4_code.py` | Submission 4：加入 ID 出現次數特徵，並使用 Random Forest、Gradient Boosting、Logistic Regression 做機率平均集成 | `submission_v4.csv` |
| `submission_best_code.py` | Random Forest 最佳候選：沿用 v2_balanced 最佳參數與 `class_weight='balanced'`，用完整訓練資料重新訓練，並同時輸出 0/1 與機率版本 | `submission_best.csv`, `submission_best_prob.csv` |
| `submission_ohe_lr_prob.py` | 目前最高分版本：One-Hot Encoding + 兩兩交互特徵 + Logistic Regression，輸出 ACTION=1 的預測機率 | `submission_ohe_lr_prob.csv` |
| `submission_v1_balanced.py` | Balanced v1：基礎 Random Forest + `class_weight='balanced'` | `submission_v1_balanced.csv` |
| `submission_v2_balanced.py` | Balanced v2：Random Forest + GridSearchCV + `class_weight='balanced'`，使用 F1-Score 調參 | `submission_v2_balanced.csv` |
| `submission_v3_balanced.py` | Balanced v3：Random Forest + `class_weight='balanced'` + threshold tuning | `submission_v3_balanced.csv` |
| `submission_v4_balanced.py` | Balanced v4：特徵工程 + 集成模型，Random Forest 和 Logistic Regression 使用 class_weight 設定 | `submission_v4_balanced.csv` |

所有 submission CSV 欄位皆統一為 Kaggle sample 要求的 `Id, Action`。

## 如何執行

1. 請先準備 `train.csv`、`test.csv` 於同目錄（已被 .gitignore 排除，不會上傳）
2. 執行對應的 submission code 產生預測結果
   ```bash
   # 前處理與資料檢查，不產生 submission
   python dm_hw3.py

   # 原始版本
   python submission1_code.py
   python submission2_code.py
   python submission3_code.py
   python submission4_code.py
   python submission_best_code.py
   python submission_ohe_lr_prob.py

   # 類別不平衡處理版本
   python submission_v1_balanced.py
   python submission_v2_balanced.py
   python submission_v3_balanced.py
   python submission_v4_balanced.py
   ```
3. 產生的 `submission_*.csv` 可直接上傳 Kaggle

## 內容說明

- `dm_hw3.py` 只保留前處理與資料檢查用途，不包含模型訓練或 submission 輸出
- 每個 submission code 對應一個模型版本，便於追蹤與分析
- `preprocess.md` 詳細記錄資料分析、前處理、類別不平衡處理與模型選擇
- `submission_results.md` 彙整每次模型的驗證集分數與預測分佈
- `model_improve.md` 說明各模型設計差異與參數選擇依據

## 注意事項

- 本 repo 不包含原始資料（train.csv, test.csv, sampleSubmission.csv, DM_Homework_3.pdf）
- 若需重現結果，請自行放置資料檔案於專案資料夾

---

如有問題請聯絡作者。
