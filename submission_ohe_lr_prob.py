"""
High-AUC candidate: one-hot Logistic Regression with pairwise interactions.

Why this version:
- The dataset features are categorical IDs, not continuous numeric values.
- AUC-based Kaggle tasks usually benefit from probability submissions.
- One-hot encoding preserves categorical meaning better than treating IDs as
  ordered numbers.
- Pairwise interactions capture combinations such as RESOURCE x MGR_ID.

Output:
- submission_ohe_lr_prob.csv with Kaggle columns: Id, Action
  where Action is the predicted probability of ACTION=1.
"""

from itertools import combinations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder


RANDOM_STATE = 42
INTERACTION_COLS = [
    "RESOURCE",
    "MGR_ID",
    "ROLE_ROLLUP_1",
    "ROLE_ROLLUP_2",
    "ROLE_DEPTNAME",
    "ROLE_TITLE",
    "ROLE_FAMILY_DESC",
    "ROLE_FAMILY",
    "ROLE_CODE",
]


def add_pairwise_interactions(df):
    """Create categorical pairwise interaction columns."""
    out = df.astype(str).copy()
    for left, right in combinations(INTERACTION_COLS, 2):
        out[f"{left}__{right}"] = out[left] + "_" + out[right]
    return out


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
    random_state=RANDOM_STATE,
    stratify=y,
)

X_train_fe = add_pairwise_interactions(X_train)
X_val_fe = add_pairwise_interactions(X_val)
X_full_fe = add_pairwise_interactions(X)
X_test_fe = add_pairwise_interactions(X_test)

model = make_pipeline(
    OneHotEncoder(handle_unknown="ignore", min_frequency=2),
    LogisticRegression(
        C=1.0,
        solver="saga",
        penalty="l2",
        max_iter=1000,
        random_state=RANDOM_STATE,
        n_jobs=1,
    ),
)

model.fit(X_train_fe, y_train)
val_prob = model.predict_proba(X_val_fe)[:, 1]
val_auc = roc_auc_score(y_val, val_prob)
print(f"Validation AUC-ROC: {val_auc:.5f}")

final_model = make_pipeline(
    OneHotEncoder(handle_unknown="ignore", min_frequency=2),
    LogisticRegression(
        C=1.0,
        solver="saga",
        penalty="l2",
        max_iter=1000,
        random_state=RANDOM_STATE,
        n_jobs=1,
    ),
)

final_model.fit(X_full_fe, y)
test_prob = final_model.predict_proba(X_test_fe)[:, 1]

submission = pd.DataFrame({"Id": test_ids, "Action": test_prob})
submission.to_csv("submission_ohe_lr_prob.csv", index=False)

print("Saved: submission_ohe_lr_prob.csv")
print(submission["Action"].describe())
