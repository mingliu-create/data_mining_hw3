# 三次 Submission 結果紀錄

## Submission 1: 基礎 Random Forest
- 修改內容：使用預設參數的 Random Forest 作為 baseline
- 驗證集準確率：0.9472
- F1-Score：0.9723
- AUC-ROC：0.8390
- 預測分佈：0=2,150, 1=56,771

## Submission 2: 參數調優後的 Random Forest
- 修改內容：使用 GridSearchCV 進行參數調優
- 最佳參數：{'n_estimators': 200, 'max_depth': 25, 'min_samples_split': 5, 'min_samples_leaf': 1}
- 交叉驗證分數：0.9503
- 驗證集準確率：0.9487
- F1-Score：0.9733
- AUC-ROC：0.8610
- 預測分佈：0=1,571, 1=57,350


## Submission 3: 類別權重調整 Random Forest
- 修改內容：使用 class_weight='balanced' 處理類別不平衡
- 驗證集準確率：0.9463
- F1-Score：0.9717
- AUC-ROC：0.8542
- 預測分佈：0=2,815, 1=56,106

## Submission 4: 特徵工程 + 集成模型
- 修改內容：加入特徵工程 (ID出現次數) + 集成 Random Forest、Gradient Boosting、Logistic Regression
- 驗證集準確率：0.9419
- F1-Score：0.9700
- AUC-ROC：0.7580
- 預測分佈：0=829, 1=58,092

## Submission 5: 資料不平衡處理 (class_weight='balanced')
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
- 交叉驗證分數：0.9495
- 驗證集準確率：0.9464
- F1-Score：0.9719
- AUC-ROC：0.8606
- 預測分佈：0=2,202, 1=56,719

### Submission v3 (balanced): Random Forest + class_weight='balanced'
- 驗證集準確率：0.9463
- F1-Score：0.9717
- AUC-ROC：0.8542
- 預測分佈：0=2,815, 1=56,106

### Submission v4 (balanced): 特徵工程 + 集成模型 + class_weight
- 驗證集準確率：0.9423
- F1-Score：0.9703
- AUC-ROC：0.7785
- 預測分佈：0=1,256, 1=57,665
