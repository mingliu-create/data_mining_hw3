"""
Submission v4 (class_weight='balanced' 版本)
-----------------------------
原始 Submission 4: 特徵工程 + 集成模型
本次修正：所有模型皆加上 class_weight='balanced' 處理資料不平衡
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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

def add_count_features(train, test, columns):
    train_new = train.copy()
    test_new = test.copy()
    for col in columns:
        count_map = train[col].value_counts().to_dict()
        train_new[f'{col}_count'] = train[col].map(count_map)
        test_new[f'{col}_count'] = test[col].map(count_map).fillna(0)
    return train_new, test_new

count_cols = ['RESOURCE', 'MGR_ID', 'ROLE_DEPTNAME', 'ROLE_TITLE', 'ROLE_FAMILY_DESC']
X_train_fe, X_test_fe = add_count_features(X_train, X_test, count_cols)
X_val_fe, _ = add_count_features(X_val, X_test, count_cols)

# Model 1: Random Forest (加上 class_weight)
rf = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=5, min_samples_leaf=1, random_state=42, n_jobs=-1, class_weight='balanced')
rf.fit(X_train_fe, y_train)
prob_rf = rf.predict_proba(X_val_fe)[:, 1]

# Model 2: Gradient Boosting (scikit-learn 1.8.0 支援 class_weight)
gb = GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42)
gb.fit(X_train_fe, y_train)
prob_gb = gb.predict_proba(X_val_fe)[:, 1]

# Model 3: Logistic Regression (加上 class_weight)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_fe)
X_val_scaled = scaler.transform(X_val_fe)
lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr.fit(X_train_scaled, y_train)
prob_lr = lr.predict_proba(X_val_scaled)[:, 1]

# 集成預測 (平均機率)
prob_ensemble = (prob_rf + prob_gb + prob_lr) / 3
y_pred_ensemble = (prob_ensemble >= 0.5).astype(int)

acc = accuracy_score(y_val, y_pred_ensemble)
f1 = f1_score(y_val, y_pred_ensemble)
auc = roc_auc_score(y_val, prob_ensemble)

print(f"驗證集準確率: {acc:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"AUC-ROC: {auc:.4f}")

# 測試集預測
X_test_fe_scaled = scaler.transform(X_test_fe)
prob_rf_test = rf.predict_proba(X_test_fe)[:, 1]
prob_gb_test = gb.predict_proba(X_test_fe)[:, 1]
prob_lr_test = lr.predict_proba(X_test_fe_scaled)[:, 1]
prob_ensemble_test = (prob_rf_test + prob_gb_test + prob_lr_test) / 3
pred_test = (prob_ensemble_test >= 0.5).astype(int)

print(f"預測分佈: 0={sum(pred_test==0)}, 1={sum(pred_test==1)}")

submission = pd.DataFrame({'id': test_ids, 'ACTION': pred_test})
submission.to_csv('submission_v4_balanced.csv', index=False)
