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
