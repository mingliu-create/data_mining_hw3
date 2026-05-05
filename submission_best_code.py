"""
Best candidate submission: full-data Random Forest with class_weight
-------------------------------------------------------------------
This version is based on the best Kaggle result so far: v2_balanced.

Key idea:
1. Use the best v2_balanced hyperparameters.
2. Keep class_weight='balanced' because it gave the best Kaggle score.
3. Validate once on the same 80/20 split for reference.
4. Refit the final model on the full training set before predicting test.csv.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

X = train_df.drop("ACTION", axis=1)
y = train_df["ACTION"]
X_test = test_df.drop("id", axis=1)
test_ids = test_df["id"]

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

best_params = {
    "n_estimators": 200,
    "max_depth": 25,
    "min_samples_split": 3,
    "min_samples_leaf": 1,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}

validation_model = RandomForestClassifier(**best_params)
validation_model.fit(X_train, y_train)

y_pred = validation_model.predict(X_val)
y_prob = validation_model.predict_proba(X_val)[:, 1]

print(f"Validation accuracy: {accuracy_score(y_val, y_pred):.4f}")
print(f"Validation F1-Score: {f1_score(y_val, y_pred):.4f}")
print(f"Validation AUC-ROC: {roc_auc_score(y_val, y_prob):.4f}")

final_model = RandomForestClassifier(**best_params)
final_model.fit(X, y)

test_pred = final_model.predict(X_test)
test_prob = final_model.predict_proba(X_test)[:, 1]
print(f"Test prediction distribution: 0={sum(test_pred == 0)}, 1={sum(test_pred == 1)}")

submission = pd.DataFrame({"Id": test_ids, "Action": test_pred})
submission.to_csv("submission_best.csv", index=False)
print("Saved: submission_best.csv")

submission_prob = pd.DataFrame({"Id": test_ids, "Action": test_prob})
submission_prob.to_csv("submission_best_prob.csv", index=False)
print("Saved: submission_best_prob.csv")
