"""
資料探勘 HW3 - 多版本模型與 Kaggle Submission
==============================================
目標：至少 3 次 submission，並記錄每次的修改與分數
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. 載入資料
# ============================================================
print("=" * 70)
print("載入資料")
print("=" * 70)

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

print(f"訓練集: {train_df.shape}")
print(f"測試集: {test_df.shape}")

# ============================================================
# 2. 資料前處理
# ============================================================
print("\n" + "=" * 70)
print("資料前處理")
print("=" * 70)

# 檢查缺失值
print(f"訓練集缺失值: {train_df.isnull().sum().sum()}")
print(f"測試集缺失值: {test_df.isnull().sum().sum()}")

# 準備特徵與目標
X = train_df.drop('ACTION', axis=1)
y = train_df['ACTION']
X_test = test_df.drop('id', axis=1)
test_ids = test_df['id']

print(f"\n目標變數分佈:")
print(y.value_counts())
print(f"正樣本比例: {y.mean():.4f}")

# 分割訓練/驗證集
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================================
# 3. Submission 1: 基礎 Random Forest
# ============================================================
print("\n" + "=" * 70)
print("Submission 1: 基礎 Random Forest")
print("=" * 70)

print("修改內容: 使用預設參數的 Random Forest 作為 baseline")

rf_v1 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_v1.fit(X_train, y_train)

y_pred_v1 = rf_v1.predict(X_val)
y_prob_v1 = rf_v1.predict_proba(X_val)[:, 1]

acc_v1 = accuracy_score(y_val, y_pred_v1)
f1_v1 = f1_score(y_val, y_pred_v1)
auc_v1 = roc_auc_score(y_val, y_prob_v1)

print(f"驗證集準確率: {acc_v1:.4f}")
print(f"F1-Score: {f1_v1:.4f}")
print(f"AUC-ROC: {auc_v1:.4f}")

# 生成預測
pred_v1 = rf_v1.predict(X_test)
submission_v1 = pd.DataFrame({'Id': test_ids, 'Action': pred_v1})
submission_v1.to_csv('submission_v1.csv', index=False)
print(f"預測結果已儲存: submission_v1.csv")
print(f"預測分佈: 0={sum(pred_v1==0)}, 1={sum(pred_v1==1)}")

# ============================================================
# 4. Submission 2: 參數調優後的 Random Forest
# ============================================================
print("\n" + "=" * 70)
print("Submission 2: 參數調優後的 Random Forest")
print("=" * 70)

print("修改內容: 使用網格搜尋進行參數調優")

param_grid = {
    'n_estimators': [150, 200],
    'max_depth': [15, 20, 25],
    'min_samples_split': [3, 5],
    'min_samples_leaf': [1, 2]
}

rf_v2 = RandomForestClassifier(random_state=42, n_jobs=-1)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    rf_v2, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1
)
grid_search.fit(X_train, y_train)

print(f"最佳參數: {grid_search.best_params_}")
print(f"交叉驗證分數: {grid_search.best_score_:.4f}")

best_rf = grid_search.best_estimator_
y_pred_v2 = best_rf.predict(X_val)
y_prob_v2 = best_rf.predict_proba(X_val)[:, 1]

acc_v2 = accuracy_score(y_val, y_pred_v2)
f1_v2 = f1_score(y_val, y_pred_v2)
auc_v2 = roc_auc_score(y_val, y_prob_v2)

print(f"驗證集準確率: {acc_v2:.4f}")
print(f"F1-Score: {f1_v2:.4f}")
print(f"AUC-ROC: {auc_v2:.4f}")

# 生成預測
pred_v2 = best_rf.predict(X_test)
submission_v2 = pd.DataFrame({'Id': test_ids, 'Action': pred_v2})
submission_v2.to_csv('submission_v2.csv', index=False)
print(f"預測結果已儲存: submission_v2.csv")
print(f"預測分佈: 0={sum(pred_v2==0)}, 1={sum(pred_v2==1)}")

# ============================================================
# 5. Submission 3: 加入類別權重 + Gradient Boosting
# ============================================================
print("\n" + "=" * 70)
print("Submission 3: 類別權重調整 + Gradient Boosting")
print("=" * 70)

print("修改內容: 使用 class_weight='balanced' 處理類別不平衡")

# Random Forest with class weight
rf_v3 = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=1,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_v3.fit(X_train, y_train)

y_pred_v3_rf = rf_v3.predict(X_val)
y_prob_v3_rf = rf_v3.predict_proba(X_val)[:, 1]

acc_v3_rf = accuracy_score(y_val, y_pred_v3_rf)
f1_v3_rf = f1_score(y_val, y_pred_v3_rf)
auc_v3_rf = roc_auc_score(y_val, y_prob_v3_rf)

print(f"Random Forest (balanced):")
print(f"  準確率: {acc_v3_rf:.4f}, F1: {f1_v3_rf:.4f}, AUC: {auc_v3_rf:.4f}")

# Gradient Boosting
gb_v3 = GradientBoostingClassifier(
    n_estimators=150,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
gb_v3.fit(X_train, y_train)

y_pred_v3_gb = gb_v3.predict(X_val)
y_prob_v3_gb = gb_v3.predict_proba(X_val)[:, 1]

acc_v3_gb = accuracy_score(y_val, y_pred_v3_gb)
f1_v3_gb = f1_score(y_val, y_pred_v3_gb)
auc_v3_gb = roc_auc_score(y_val, y_prob_v3_gb)

print(f"Gradient Boosting:")
print(f"  準確率: {acc_v3_gb:.4f}, F1: {f1_v3_gb:.4f}, AUC: {auc_v3_gb:.4f}")

# 選擇較好的模型
if auc_v3_rf > auc_v3_gb:
    print("選擇: Random Forest (balanced)")
    pred_v3 = rf_v3.predict(X_test)
    acc_v3, f1_v3, auc_v3 = acc_v3_rf, f1_v3_rf, auc_v3_rf
else:
    print("選擇: Gradient Boosting")
    pred_v3 = gb_v3.predict(X_test)
    acc_v3, f1_v3, auc_v3 = acc_v3_gb, f1_v3_gb, auc_v3_gb

submission_v3 = pd.DataFrame({'Id': test_ids, 'Action': pred_v3})
submission_v3.to_csv('submission_v3.csv', index=False)
print(f"預測結果已儲存: submission_v3.csv")
print(f"預測分佈: 0={sum(pred_v3==0)}, 1={sum(pred_v3==1)}")

# ============================================================
# 6. Submission 4 (額外): 特徵工程 + 集成模型
# ============================================================
print("\n" + "=" * 70)
print("Submission 4: 特徵工程 + 集成模型")
print("=" * 70)

print("修改內容: 加入特徵工程 (ID出現次數) + 集成多個模型")

# 特徵工程: 計算各 ID 在訓練集中的出現次數
def add_count_features(train, test, columns):
    """為指定欄位添加計數特徵"""
    train_new = train.copy()
    test_new = test.copy()
    
    for col in columns:
        # 計算訓練集中每個值的出現次數
        count_map = train[col].value_counts().to_dict()
        
        # 添加計數特徵
        train_new[f'{col}_count'] = train[col].map(count_map)
        test_new[f'{col}_count'] = test[col].map(count_map).fillna(0)
    
    return train_new, test_new

# 選擇高基數欄位進行計數特徵工程
count_cols = ['RESOURCE', 'MGR_ID', 'ROLE_DEPTNAME', 'ROLE_TITLE', 'ROLE_FAMILY_DESC']

X_train_fe, X_test_fe = add_count_features(X_train, X_test, count_cols)
X_val_fe, _ = add_count_features(X_val, X_test, count_cols)

print(f"特徵工程後維度: {X_train_fe.shape[1]}")

# 使用多個模型進行集成
# Model 1: Random Forest
rf_fe = RandomForestClassifier(
    n_estimators=200, max_depth=20,
    min_samples_split=5, min_samples_leaf=1,
    random_state=42, n_jobs=-1
)
rf_fe.fit(X_train_fe, y_train)
prob_rf = rf_fe.predict_proba(X_val_fe)[:, 1]

# Model 2: Gradient Boosting
gb_fe = GradientBoostingClassifier(
    n_estimators=150, max_depth=5, learning_rate=0.1,
    random_state=42
)
gb_fe.fit(X_train_fe, y_train)
prob_gb = gb_fe.predict_proba(X_val_fe)[:, 1]

# Model 3: Logistic Regression (需要標準化)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_fe)
X_val_scaled = scaler.transform(X_val_fe)

lr_fe = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr_fe.fit(X_train_scaled, y_train)
prob_lr = lr_fe.predict_proba(X_val_scaled)[:, 1]

# 集成預測 (平均機率)
prob_ensemble = (prob_rf + prob_gb + prob_lr) / 3
y_pred_ensemble = (prob_ensemble >= 0.5).astype(int)

acc_v4 = accuracy_score(y_val, y_pred_ensemble)
f1_v4 = f1_score(y_val, y_pred_ensemble)
auc_v4 = roc_auc_score(y_val, prob_ensemble)

print(f"集成模型 (RF + GB + LR):")
print(f"  準確率: {acc_v4:.4f}")
print(f"  F1-Score: {f1_v4:.4f}")
print(f"  AUC-ROC: {auc_v4:.4f}")

# 生成測試集預測
X_test_fe_scaled = scaler.transform(X_test_fe)
prob_rf_test = rf_fe.predict_proba(X_test_fe)[:, 1]
prob_gb_test = gb_fe.predict_proba(X_test_fe)[:, 1]
prob_lr_test = lr_fe.predict_proba(X_test_fe_scaled)[:, 1]

prob_ensemble_test = (prob_rf_test + prob_gb_test + prob_lr_test) / 3
pred_v4 = (prob_ensemble_test >= 0.5).astype(int)

submission_v4 = pd.DataFrame({'Id': test_ids, 'Action': pred_v4})
submission_v4.to_csv('submission_v4.csv', index=False)
print(f"預測結果已儲存: submission_v4.csv")
print(f"預測分佈: 0={sum(pred_v4==0)}, 1={sum(pred_v4==1)}")

# ============================================================
# 7. 總結比較
# ============================================================
print("\n" + "=" * 70)
print("所有 Submission 總結")
print("=" * 70)

summary = pd.DataFrame({
    'Version': ['v1 (Baseline RF)', 'v2 (Grid Search)', 'v3 (Class Weight)', 'v4 (Feature Eng + Ensemble)'],
    'Accuracy': [acc_v1, acc_v2, acc_v3, acc_v4],
    'F1-Score': [f1_v1, f1_v2, f1_v3, f1_v4],
    'AUC-ROC': [auc_v1, auc_v2, auc_v3, auc_v4]
})

print(summary.to_string(index=False))

print("\n" + "=" * 70)
print("完成！所有 submission 檔案已生成")
print("=" * 70)
print("\n生成的檔案:")
print("  - submission_v1.csv: 基礎 Random Forest")
print("  - submission_v2.csv: 參數調優後的 Random Forest")
print("  - submission_v3.csv: 類別權重調整 + Gradient Boosting")
print("  - submission_v4.csv: 特徵工程 + 集成模型")