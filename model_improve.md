# 模型改進與參數選擇說明（model_improve.md）

本文件說明四個 submission 模型的設計差異、參數選擇依據，以及每次改進的重點與效果。

---

## Submission 1：基礎 Random Forest
- **設計**：直接使用 RandomForestClassifier 預設參數（n_estimators=100）。
- **目的**：建立 baseline，觀察資料本身的可分性。
- **優缺點**：
  - 優點：簡單、穩定，對高維特徵與類別型特徵友善。
  - 缺點：未針對資料不平衡與最佳參數進行調整。
- **結果**：
  - 準確率 94.72%，AUC-ROC 0.8390。

---

## Submission 2：參數調優 Random Forest
- **設計**：使用 GridSearchCV 搜尋最佳參數組合。
  - n_estimators: 150, 200
  - max_depth: 15, 20, 25
  - min_samples_split: 3, 5
  - min_samples_leaf: 1, 2
- **選擇依據**：
  - n_estimators 增加可提升穩定性
  - max_depth 控制過擬合
  - min_samples_split/leaf 防止小樣本分割過度
- **優缺點**：
  - 優點：模型針對資料特性調整，提升泛化能力
  - 缺點：搜尋耗時
- **結果**：
  - 準確率 94.87%，AUC-ROC 0.8610（最佳）

---

## Submission 3：類別權重調整 Random Forest
- **設計**：設定 class_weight='balanced'，讓模型更重視少數類別（ACTION=0）。
- **選擇依據**：
  - 資料極度不平衡（1:0 約 16:1），預設模型易忽略少數類別
  - 調整權重可提升少數類別的召回率
- **優缺點**：
  - 優點：改善不平衡資料下的預測偏誤
  - 缺點：可能降低整體準確率
- **結果**：
  - 準確率 94.63%，AUC-ROC 0.8542，預測為 0 的數量明顯增加

---

## Submission 4：特徵工程 + 集成模型
- **設計**：
  1. 新增五個計數特徵（ID 出現次數），提升模型對高基數特徵的辨識力
  2. 集成三種模型（Random Forest, Gradient Boosting, Logistic Regression），以平均機率投票
- **選擇依據**：
  - 計數特徵可補足單一 ID 編號資訊不足
  - 集成學習可降低單一模型過擬合風險
- **優缺點**：
  - 優點：理論上提升泛化能力，融合多模型優勢
  - 缺點：本題資料特性下，集成效果有限，AUC 反而下降
- **結果**：
  - 準確率 94.19%，AUC-ROC 0.7580（AUC 下降，可能因特徵泛化性不足）

---

## 綜合建議
- **最佳模型**：Submission 2（參數調優 RF）
- **參數調整重點**：
  - n_estimators 增加提升穩定性
  - max_depth、min_samples_split/leaf 控制過擬合
  - class_weight 處理不平衡
- **特徵工程**：可嘗試更多交互特徵、分箱、目標編碼等
- **集成學習**：適合模型間表現差異大時，否則效果有限

---

如需更進階改進，可考慮 XGBoost、LightGBM、SMOTE 等方法。
