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
- **設計**：使用 GridSearchCV 搜尋最佳參數組合，並以 F1-Score 作為 scoring。
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
  - 交叉驗證 F1=0.9741，驗證集準確率 94.87%，F1=0.9733，AUC-ROC 0.8610。

---

## Submission 3：Random Forest + threshold tuning
- **設計**：使用 Random Forest 輸出預測機率，再於驗證集上搜尋最佳 threshold。
- **選擇依據**：
  - 資料極度不平衡（1:0 約 16:1），預設 threshold=0.5 不一定能取得最佳 F1-Score
  - threshold tuning 可以在不改變模型與不加入 class_weight 的情況下調整預測分佈
- **優缺點**：
  - 優點：維持 Random Forest 架構，能直接優化 F1-Score
  - 缺點：threshold 由驗證集決定，仍需用 Kaggle 分數確認泛化效果
- **結果**：
  - 最佳 threshold=0.41，準確率 94.95%，F1=0.9738，AUC-ROC 0.8610。

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

## Submission 5：資料不平衡處理方法比較
- **設計**：比較多種資料不平衡處理方法（SMOTE、Random Undersampling、SMOTETomek、class_weight='balanced'）
- **選擇依據**：
  - 資料極度不平衡（1:0 約 16:1），需針對性處理
  - 測試不同過採樣/欠採樣方法的效果差異
- **測試方法**：
  - SMOTE：合成少數類別樣本
  - Random Undersampling：隨機減少多數類別樣本
  - SMOTETomek：結合過採樣與欠採樣
  - class_weight='balanced'：調整模型權重
- **結果**：
  - class_weight='balanced' 表現最佳，F1=0.9722，AUC=0.8628
  - 少數類別召回率從 26.65% 提升至 39.84%

---

## Balanced 版本：所有模型套用 class_weight='balanced'
- **設計**：將原本 v1~v4 四個模型全部加上 class_weight='balanced' 參數
- **目的**：在保持各模型特性的同時，改善資料不平衡問題
- **結果**：
  - v1 (balanced)：準確率 94.78%，F1=0.9727，AUC=0.8448
  - v2 (balanced)：準確率 94.64%，F1=0.9719，AUC=0.8606
  - v3 (balanced + threshold tuning)：最佳 threshold=0.39，準確率 95.03%，F1=0.9740，AUC=0.8628
  - v4 (balanced)：準確率 94.23%，F1=0.9703，AUC=0.7785

---

## 綜合建議
- **最佳驗證集模型**：Submission v3 balanced（class_weight + threshold tuning）
- **參數調整重點**：
  - n_estimators 增加提升穩定性
  - max_depth、min_samples_split/leaf 控制過擬合
  - class_weight 處理不平衡
- **特徵工程**：可嘗試更多交互特徵、分箱、目標編碼等
- **集成學習**：適合模型間表現差異大時，否則效果有限

---

如需更進階改進，可考慮 XGBoost、LightGBM、SMOTE 等方法。
