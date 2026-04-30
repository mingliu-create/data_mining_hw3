"""
Submission v2 (class_weight='balanced' 版本)
-----------------------------
原始 Submission 2: GridSearchCV 調參 Random Forest
本次修正：加上 class_weight='balanced'，並以 F1-Score 作為 GridSearchCV 搜尋目標
"""
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# 載入資料
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
X = train_df.drop('ACTION', axis=1)
y = train_df['ACTION']
X_test = test_df.drop('id', axis=1)
test_ids = test_df['id']

# 分割訓練/驗證集
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 參數搜尋 
param_grid = {
    'n_estimators': [150, 200],
    'max_depth': [15, 20, 25],
    'min_samples_split': [3, 5],
    'min_samples_leaf': [1, 2],
    'class_weight': ['balanced']
}
rf = RandomForestClassifier(random_state=42, n_jobs=-1)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
gs = GridSearchCV(rf, param_grid, cv=cv, scoring='f1', n_jobs=-1, verbose=1)
gs.fit(X_train, y_train)

print(f"最佳參數: {gs.best_params_}")
print(f"交叉驗證分數: {gs.best_score_:.4f}")

best_rf = gs.best_estimator_
y_pred = best_rf.predict(X_val)
y_prob = best_rf.predict_proba(X_val)[:, 1]

acc = accuracy_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
auc = roc_auc_score(y_val, y_prob)

print(f"驗證集準確率: {acc:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"AUC-ROC: {auc:.4f}")
print(f"預測分佈: 0={sum(best_rf.predict(X_test)==0)}, 1={sum(best_rf.predict(X_test)==1)}")

# 生成預測
submission = pd.DataFrame({'Id': test_ids, 'Action': best_rf.predict(X_test)})
submission.to_csv('submission_v2_balanced.csv', index=False)
