"""
Submission v1 (class_weight='balanced' 版本)
-----------------------------
原始 Submission 1: 預設 Random Forest
本次修正：加上 class_weight='balanced' 處理資料不平衡
"""
import pandas as pd
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

# 建立模型 (加上 class_weight)
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
rf.fit(X_train, y_train)

y_pred = rf.predict(X_val)
y_prob = rf.predict_proba(X_val)[:, 1]

acc = accuracy_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
auc = roc_auc_score(y_val, y_prob)

print(f"驗證集準確率: {acc:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"AUC-ROC: {auc:.4f}")
print(f"預測分佈: 0={sum(rf.predict(X_test)==0)}, 1={sum(rf.predict(X_test)==1)}")

# 生成預測
submission = pd.DataFrame({'id': test_ids, 'ACTION': rf.predict(X_test)})
submission.to_csv('submission_v1_balanced.csv', index=False)
