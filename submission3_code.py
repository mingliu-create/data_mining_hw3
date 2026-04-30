"""
Submission 3: Random Forest + threshold tuning
-----------------------------
修改內容：使用 Random Forest 預測機率，並在驗證集上調整 threshold 以最大化 F1-Score
說明：此版本不加入 class_weight，保留原始資料不平衡狀態
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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

# 建立模型
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=25,
    min_samples_split=5,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)

y_prob = rf.predict_proba(X_val)[:, 1]
thresholds = np.arange(0.10, 0.91, 0.01)
scores = [(threshold, f1_score(y_val, (y_prob >= threshold).astype(int))) for threshold in thresholds]
best_threshold, best_f1 = max(scores, key=lambda item: item[1])
y_pred = (y_prob >= best_threshold).astype(int)

acc = accuracy_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
auc = roc_auc_score(y_val, y_prob)

print(f"最佳 threshold: {best_threshold:.2f}")
print(f"驗證集準確率: {acc:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"AUC-ROC: {auc:.4f}")
test_prob = rf.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= best_threshold).astype(int)
print(f"預測分佈: 0={sum(test_pred==0)}, 1={sum(test_pred==1)}")

# 生成預測
submission = pd.DataFrame({'Id': test_ids, 'Action': test_pred})
submission.to_csv('submission_v3.csv', index=False)
