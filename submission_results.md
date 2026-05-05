# 三次 Submission 結果紀錄

> 所有 submission CSV 欄位已統一為 Kaggle sample submission 要求的 `Id, Action`。

## Submission 1: 基礎 Random Forest
- 修改內容：使用預設參數的 Random Forest 作為 baseline
- 驗證集準確率：0.9472
- F1-Score：0.9723
- AUC-ROC：0.8390
- 預測分佈：0=2,150, 1=56,771

## Submission 2: 參數調優後的 Random Forest
- 修改內容：使用 GridSearchCV 進行參數調優，scoring 改為 F1-Score
- 最佳參數：{'n_estimators': 200, 'max_depth': 25, 'min_samples_split': 5, 'min_samples_leaf': 1}
- 交叉驗證 F1-Score：0.9741
- 驗證集準確率：0.9487
- F1-Score：0.9733
- AUC-ROC：0.8610
- 預測分佈：0=1,571, 1=57,350


## Submission 3: Random Forest + threshold tuning
- 修改內容：使用 Random Forest 預測機率，並在驗證集上調整 threshold 以最大化 F1-Score
- 模型參數：n_estimators=200, max_depth=25, min_samples_split=5, min_samples_leaf=1
- 最佳 threshold：0.41
- 驗證集準確率：0.9495
- F1-Score：0.9738
- AUC-ROC：0.8610
- 預測分佈：0=1,088, 1=57,833

## Submission 4: 特徵工程 + 集成模型
- 修改內容：加入特徵工程 (ID出現次數) + 集成 Random Forest、Gradient Boosting、Logistic Regression
- 驗證集準確率：0.9419
- F1-Score：0.9700
- AUC-ROC：0.7580
- 預測分佈：0=829, 1=58,092

## Submission 5: 資料不平衡處理 (Random Forest + class_weight='balanced')
- **模型**：Random Forest
- 修改內容：使用 class_weight='balanced' 處理類別不平衡問題
- 驗證集準確率：0.9472
- F1-Score：0.9722
- AUC-ROC：0.8628
- 少數類別召回率：39.84% (相比基準模型 26.65% 大幅提升)
- 預測分佈：0=2,679, 1=56,242
- 說明：此方法在保持高準確率與 F1-Score 的同時，顯著提升了少數類別 (ACTION=0) 的識別能力

---

## Balanced 版本 (所有模型套用 class_weight='balanced')

### Submission v1 (balanced): 基礎 Random Forest + class_weight
- 驗證集準確率：0.9478
- F1-Score：0.9727
- AUC-ROC：0.8448
- 預測分佈：0=1,943, 1=56,978

### Submission v2 (balanced): GridSearchCV 調參 + class_weight
- 最佳參數：{'n_estimators': 200, 'max_depth': 25, 'min_samples_split': 3, 'min_samples_leaf': 1, 'class_weight': 'balanced'}
- 交叉驗證 F1-Score：0.9735
- 驗證集準確率：0.9464
- F1-Score：0.9719
- AUC-ROC：0.8606
- 預測分佈：0=2,202, 1=56,719

### Submission v3 (balanced): Random Forest + class_weight='balanced' + threshold tuning
- 模型參數：n_estimators=200, max_depth=25, min_samples_split=5, min_samples_leaf=1, class_weight='balanced'
- 最佳 threshold：0.39
- 驗證集準確率：0.9503
- F1-Score：0.9740
- AUC-ROC：0.8628
- 預測分佈：0=1,735, 1=57,186

### Submission v4 (balanced): 特徵工程 + 集成模型 + class_weight
- 驗證集準確率：0.9423
- F1-Score：0.9703
- AUC-ROC：0.7785
- 預測分佈：0=1,256, 1=57,665

---

## Best Candidate: full-data Random Forest + class_weight
- 檔案：`submission_best.csv`
- 程式：`submission_best_code.py`
- 修改內容：沿用 v2_balanced 最佳參數與 `class_weight='balanced'`，並使用完整訓練資料重新訓練最終模型
- 模型參數：n_estimators=200, max_depth=25, min_samples_split=3, min_samples_leaf=1, class_weight='balanced'
- 驗證集準確率：0.9464
- F1-Score：0.9719
- AUC-ROC：0.8606
- 預測分佈：0=2,259, 1=56,662
- Kaggle Score：Private=0.68505, Public=0.69099

### Probability version: full-data Random Forest + class_weight
- 檔案：`submission_best_prob.csv`
- 程式：`submission_best_code.py`
- 修改內容：與 submission_best 使用相同模型，但提交 ACTION=1 的預測機率
- 驗證 AUC-ROC：0.8606
- Kaggle Score：Private=0.86584

### Probability version: One-Hot Logistic Regression
- 檔案：`submission_ohe_lr_prob.csv`
- 程式：`submission_ohe_lr_prob.py`
- 修改內容：使用 One-Hot Encoding 表示 ID 類別特徵，加入兩兩交互特徵，再以 Logistic Regression 輸出 ACTION=1 的預測機率
- 驗證 AUC-ROC：0.85375
- Kaggle Score：Private=0.87610, Public=0.88328

### Probability blend versions
- 程式：`submission_blend_prob.py`
- 修改內容：融合 `submission_ohe_lr_prob.csv` 與 `submission_best_prob.csv` 的機率輸出

| 檔案 | 融合方式 | Private Score | Public Score |
|------|----------|---------------|--------------|
| `submission_blend_ohe90_rf10.csv` | 90% OHE-LR + 10% RF | 0.88187 | **0.88941** |
| `submission_blend_ohe80_rf20.csv` | 80% OHE-LR + 20% RF | **0.88221** | 0.88920 |
| `submission_blend_rank_ohe80_rf20.csv` | 80% OHE-LR rank + 20% RF rank | 0.88207 | 0.88871 |
| `submission_blend_ohe70_rf30.csv` | 70% OHE-LR + 30% RF | 0.88168 | 0.88803 |

- 說明：目前 Private 最高為 `submission_blend_ohe80_rf20.csv`，Public 最高為 `submission_blend_ohe90_rf10.csv`
