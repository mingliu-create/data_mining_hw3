"""
Submission v3 (class_weight='balanced' 版本)
-----------------------------
原始 Submission 3: Random Forest + class_weight='balanced'
本次修正：參數與原本一致，確保與新資料一致
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

# 建立模型 (與原本一致)
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=1,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
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
submission.to_csv('submission_v3_balanced.csv', index=False)
