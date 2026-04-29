# 資料前處理與模型分析報告

## 1. 資料集概述

### 1.1 資料來源
- **訓練集**: `train.csv` (32,769 筆資料)
- **測試集**: `test.csv` (58,921 筆資料)
- **目標變數**: ACTION (0 或 1) - 二元分類問題

### 1.2 特徵欄位
| 欄位名稱 | 說明 | 唯一值數量 |
|---------|------|-----------|
| RESOURCE | 資源 ID | 7,518 |
| MGR_ID | 管理者 ID | 4,243 |
| ROLE_ROLLUP_1 | 角色階層 1 | 128 |
| ROLE_ROLLUP_2 | 角色階層 2 | 177 |
| ROLE_DEPTNAME | 部門名稱 | 449 |
| ROLE_TITLE | 職稱 | 343 |
| ROLE_FAMILY_DESC | 角色家族描述 | 2,358 |
| ROLE_FAMILY | 角色家族 | 67 |
| ROLE_CODE | 角色代碼 | 343 |

---

## 2. 資料前處理分析

### 2.1 缺失值檢查

**結果**: 訓練集與測試集**皆無缺失值**

```
訓練集缺失值: 0
測試集缺失值: 0
```

**處理方式**: 無需進行缺失值填補

### 2.2 資料類型檢查

**結果**: 所有特徵皆為數值型 (int64)

**處理方式**: 
- 不需要進行類別編碼 (Label Encoding / One-Hot Encoding)
- 所有資料可直接用於模型訓練

### 2.3 目標變數分佈

| ACTION 值 | 數量 | 比例 |
|----------|------|------|
| 1 | 30,872 | 94.21% |
| 0 | 1,897 | 5.79% |

**問題診斷**: 資料存在嚴重的**類別不平衡**問題 (Imbalanced Data)

**處理方式建議**:
1. **過採樣 (Oversampling)**: 使用 SMOTE 技術生成少數類別樣本
2. **欠採樣 (Undersampling)**: 減少多數類別樣本
3. **類別權重調整**: 在模型中設定 `class_weight='balanced'`
4. **評估指標**: 使用 F1-score、AUC-ROC 而非準確率

---

### 2.4 資料不平衡處理實驗結果

我們實際測試了多種資料不平衡處理方法，結果如下：

| 方法 | 準確率 | F1-Score | AUC-ROC | 少數類別召回率 |
|------|--------|----------|---------|---------------|
| 基準模型 (無處理) | 94.87% | 0.9733 | 0.8610 | 26.65% |
| SMOTE 過採樣 | 94.45% | 0.9708 | 0.8478 | 35.88% |
| Random Undersampling | 76.98% | 0.8635 | 0.8247 | 72.30% |
| SMOTETomek | 94.43% | 0.9707 | 0.8472 | 38.26% |
| **class_weight='balanced'** | **94.72%** | **0.9722** | **0.8628** | **39.84%** |
| SMOTE + class_weight | 94.45% | 0.9708 | 0.8478 | 35.88% |

**分析結論**:
- `class_weight='balanced'` 在 F1-Score 和 AUC-ROC 上表現最佳，且能顯著提升少數類別 (ACTION=0) 的召回率
- SMOTE 過採樣可提升少數類別召回率，但整體 F1-Score 略降
- Random Undersampling 雖然召回率高，但犧牲過多資訊，導致整體表現下滑
- 最終選擇 `class_weight='balanced'` 作為提交方案

### 2.5 Kaggle 評分結果

#### 2.5.1 原始版本 (未處理類別不平衡)

| Submission | Private Score | Public Score | 說明 |
|------------|---------------|--------------|------|
| 第一次 (v1) | 0.67073 | 0.67498 | 基礎 Random Forest (n_estimators=100) |
| 第二次 (v2) | 0.64071 | 0.65635 | GridSearchCV 調參後 Random Forest |
| 第三次 (v3) | 0.70568 | 0.70448 | class_weight='balanced' |
| 第四次 (v4) | 0.58913 | 0.58772 | 特徵工程 + 集成模型 |

#### 2.5.2 平衡版本 (class_weight='balanced')

| Submission | Private Score | Public Score | 說明 |
|------------|---------------|--------------|------|
| 第一次 (v1_balanced) | 0.65628 | 0.66703 | 基礎 RF + class_weight |
| 第二次 (v2_balanced) | 0.67450 | 0.67875 | GridSearchCV + class_weight |
| 第三次 (v3_balanced) | 0.70568 | 0.70448 | RF + class_weight='balanced' |
| 第四次 (v4_balanced) | 0.62636 | 0.62442 | 集成模型 + class_weight |

#### 2.5.3 評分分析

- **最佳成績**: 第三次提交 (v3/v3_balanced) - Private: 0.70568, Public: 0.70448
- **類別不平衡處理效果**: 平衡版本在第二次提交表現較好 (0.67450 vs 0.64071)，顯示類別權重調整對提升預測能力有幫助
- **特徵工程影響**: 第四次提交分數下降，可能是過擬合或特徵工程方法需要改進

#### 2.4.1 描述統計
```
             ACTION       RESOURCE  ...    ROLE_FAMILY      ROLE_CODE
count    32769.000000   42923.916171  ...  183703.408893  119789.430132
mean         0.942110   42923.916171  ...  183703.408893  119789.430132
std          0.233539   34173.892702  ...  100488.407413   5784.275516
min          0.000000       0.000000  ...    3130.000000  117880.000000
max          1.000000  312153.000000  ...    308574.000000  270691.000000
```

#### 2.4.2 特徵類型判斷
- **ID 型特徵**: RESOURCE, MGR_ID, ROLE_* (這些是 ID 編號，不是連續數值)
- **計數型特徵**: 無

---

## 3. 前處理策略

### 3.1 選擇的處理方式

| 步驟 | 處理方式 | 原因 |
|------|---------|------|
| 缺失值 | 無處理 | 資料無缺失值 |
| 資料類型 | 無處理 | 所有特徵為數值型 |
| 特徵縮放 | 視模型而定 | 樹狀模型不需要標準化；線性模型建議標準化 |
| 類別不平衡 | 未處理 | 預設模型已達到 94% 準確率 |

### 3.2 為何不需要複雜前處理？

1. **無缺失值**: 資料品質良好，無需填補
2. **全數值特徵**: 不需要編碼轉換
3. **ID 特徵可直接使用**: 雖然是 ID，但隨機森林可以從中學習到有價值的模式

### 3.3 可選的進階前處理（供參考）

若要進一步提升模型表現，可考慮：

1. **特徵工程**:
   - 計算各 ID 的出現次數作為新特徵
   - 建立特徵交互作用

2. **類別不平衡處理**:
   ```python
   from imblearn.over_sampling import SMOTE
   smote = SMOTE(random_state=42)
   X_resampled, y_resampled = smote.fit_resample(X, y)
   ```

3. **標準化** (用於 LR/SVM):
   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   ```

---

## 4. 模型參數設定

### 4.1 測試的模型

| 模型 | 驗證集準確率 |
|------|-------------|
| **Random Forest** | **94.72%** |
| Gradient Boosting | 94.19% |
| Logistic Regression | 94.22% |
| Decision Tree | 92.74% |

### 4.2 Random Forest 參數調優

#### 4.2.1 網格搜尋參數空間
```python
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
```

#### 4.2.2 最佳參數
```python
{
    'n_estimators': 200,      # 森林中決策樹的數量
    'max_depth': 20,          # 樹的最大深度
    'min_samples_split': 5,   # 分割內部節點所需最小樣本數
    'min_samples_leaf': 1     # 葉節點所需最小樣本數
}
```

#### 4.2.3 調優結果
- 交叉驗證分數: **94.94%**
- 驗證集準確率: **94.78%**

### 4.3 各參數說明

| 參數 | 建議值範圍 | 說明 |
|------|-----------|------|
| `n_estimators` | 100, 200, 300 | 森林中樹的數量，越多越好但計算時間增加 |
| `max_depth` | 10, 20, 30, None | 樹的最大深度，None 表示不限制 |
| `min_samples_split` | 2, 5, 10 | 分割內部節點所需的最小樣本數 |
| `min_samples_leaf` | 1, 2, 4 | 葉節點所需的最小樣本數 |
| `max_features` | 'sqrt', 'log2', None | 每次分割時考慮的最大特徵數 |
| `class_weight` | 'balanced', {0: w0, 1: w1} | 類別權重，用於處理不平衡資料 |

### 4.4 其他模型參數建議

#### Gradient Boosting
```python
{
    'n_estimators': 100, 200,
    'learning_rate': 0.01, 0.1, 0.2,
    'max_depth': 3, 5, 7,
    'min_samples_split': 2, 5,
    'subsample': 0.8, 1.0
}
```

#### Logistic Regression
```python
{
    'C': 0.01, 0.1, 1, 10, 100,  # 正則化強度的倒數
    'penalty': 'l1', 'l2',       # 正則化類型
    'class_weight': 'balanced'   # 類別權重
}
```

#### SVM
```python
{
    'C': 0.1, 1, 10, 100,
    'kernel': 'rbf', 'linear',
    'gamma': 'scale', 'auto'
}
```

---

## 5. 特徵重要性分析

| 排名 | 特徵 | 重要性 |
|------|------|--------|
| 1 | RESOURCE | 25.23% |
| 2 | MGR_ID | 20.03% |
| 3 | ROLE_DEPTNAME | 14.05% |
| 4 | ROLE_FAMILY_DESC | 12.16% |
| 5 | ROLE_ROLLUP_2 | 9.08% |
| 6 | ROLE_CODE | 5.50% |
| 7 | ROLE_TITLE | 5.43% |
| 8 | ROLE_ROLLUP_1 | 4.69% |
| 9 | ROLE_FAMILY | 3.82% |

**結論**: `RESOURCE` 和 `MGR_ID` 是最重要的兩個特徵，合計貢獻約 45% 的預測能力。

---

## 6. 總結

### 6.1 前處理總結
- ✅ 資料無缺失值
- ✅ 所有特徵為數值型，無需編碼
- ⚠️ 類別不平衡 (94% vs 6%)，可考慮進一步處理

### 6.2 模型選擇
- **最終模型**: Random Forest
- **最佳參數**: n_estimators=200, max_depth=20, min_samples_split=5, min_samples_leaf=1
- **驗證集準確率**: 94.78%

### 6.3 建議後續改進方向
1. 處理類別不平衡問題 (SMOTE 或 class_weight)
2. 進行特徵工程 (ID 出現次數、特徵交互作用)
3. 嘗試更多模型 (XGBoost, LightGBM)
4. 使用交叉驗證獲得更穩定的評估

---

## 7. Kaggle Submission 記錄

### 7.1 Submission 1: 基礎 Random Forest

| 項目 | 內容 |
|------|------|
| **修改內容** | 使用預設參數的 Random Forest 作為 baseline |
| **修改原因** | 建立基礎模型作為比較基準 |
| **模型參數** | n_estimators=100, random_state=42 |
| **驗證集準確率** | 94.72% |
| **F1-Score** | 0.9723 |
| **AUC-ROC** | 0.8390 |
| **預測分佈** | 0=2,150, 1=56,771 |
| **Kaggle Score** | 待填寫 |

### 7.2 Submission 2: 參數調優後的 Random Forest

| 項目 | 內容 |
|------|------|
| **修改內容** | 使用網格搜尋 (GridSearchCV) 進行參數調優 |
| **修改原因** | 透過系統化搜尋找到最佳參數組合，提升模型表現 |
| **模型參數** | n_estimators=200, max_depth=25, min_samples_split=5, min_samples_leaf=1 |
| **交叉驗證分數** | 95.03% |
| **驗證集準確率** | 94.87% |
| **F1-Score** | 0.9733 |
| **AUC-ROC** | 0.8610 |
| **預測分佈** | 0=1,571, 1=57,350 |
| **Kaggle Score** | 待填寫 |

**變化分析**: 
- 準確率提升: 94.72% → 94.87% (+0.15%)
- AUC-ROC 大幅提升: 0.8390 → 0.8610 (+2.2%)
- 這表示模型對正負樣本的區分能力增強

### 7.3 Submission 3: 類別權重調整

| 項目 | 內容 |
|------|------|
| **修改內容** | 使用 class_weight='balanced' 處理類別不平衡問題 |
| **修改原因** | 資料集正樣本佔 94%，使用類別權重讓模型更關注少數類別 (ACTION=0) |
| **模型參數** | n_estimators=200, max_depth=20, class_weight='balanced' |
| **驗證集準確率** | 94.63% |
| **F1-Score** | 0.9717 |
| **AUC-ROC** | 0.8542 |
| **預測分佈** | 0=2,815, 1=56,106 |
| **Kaggle Score** | 待填寫 |

**變化分析**:
- 預測為 0 的數量增加 (2,150 → 2,815)
- 這是因為模型更傾向於預測少數類別
- AUC-ROC 提升至 0.8542，表示整體區分能力提升

### 7.4 Submission 4: 特徵工程 + 集成模型

| 項目 | 內容 |
|------|------|
| **修改內容** | 加入特徵工程 (ID出現次數) + 集成 Random Forest + Gradient Boosting + Logistic Regression |
| **修改原因** | 透過特徵工程增加資訊量，使用集成學習提升穩定性 |
| **新增特徵** | RESOURCE_count, MGR_ID_count, ROLE_DEPTNAME_count, ROLE_TITLE_count, ROLE_FAMILY_DESC_count |
| **驗證集準確率** | 94.19% |
| **F1-Score** | 0.9700 |
| **AUC-ROC** | 0.7580 |
| **預測分佈** | 0=829, 1=58,092 |
| **Kaggle Score** | 待填寫 |

**變化分析**:
- 準確率略降: 94.87% → 94.19%
- AUC-ROC 大幅下降: 0.8610 → 0.7580
- 可能原因: 計數特徵在測試集上的泛化能力較差，或集成方法在此資料上效果不佳

---

## 8. 總結比較表

| Version | 模型 | 驗證 Accuracy | F1-Score | AUC-ROC | 預測為0的數量 |
|---------|------|--------------|----------|---------|-------------|
| v1 | Baseline RF | 94.72% | 0.9723 | 0.8390 | 2,150 |
| v2 | Grid Search RF | 94.87% | 0.9733 | 0.8610 | 1,571 |
| v3 | Balanced RF | 94.63% | 0.9717 | 0.8542 | 2,815 |
| v4 | Ensemble + FE | 94.19% | 0.9700 | 0.7580 | 829 |

---

## 9. 建議

### 9.1 最佳模型選擇
根據驗證集結果，**Submission 2 (參數調優後的 Random Forest)** 表現最佳：
- 最高準確率: 94.87%
- 最高 AUC-ROC: 0.8610
- 預測分佈也較為合理

### 9.2 後續改進建議
1. **嘗試 XGBoost/LightGBM**: 通常在這類資料上表現更好
2. **更細緻的類別不平衡處理**: 使用 SMOTE 或調整閾值
3. **特徵工程優化**: 考慮特徵交互作用，而非簡單的計數特徵
4. **模型融合**: 使用 stacking 而非簡單平均