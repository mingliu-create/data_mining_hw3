"""
Blend probability submissions for AUC-oriented Kaggle scoring.

Inputs:
- submission_ohe_lr_prob.csv: current best single model
- submission_best_prob.csv: full-data RF probability model

Outputs:
- weighted probability blends
- rank-average blend
"""

import pandas as pd
from scipy.stats import rankdata


ohe = pd.read_csv("submission_ohe_lr_prob.csv")
rf = pd.read_csv("submission_best_prob.csv")

if not ohe["Id"].equals(rf["Id"]):
    raise ValueError("Input submission Id columns do not match.")

candidates = {
    "submission_blend_ohe90_rf10.csv": 0.90 * ohe["Action"] + 0.10 * rf["Action"],
    "submission_blend_ohe80_rf20.csv": 0.80 * ohe["Action"] + 0.20 * rf["Action"],
    "submission_blend_ohe70_rf30.csv": 0.70 * ohe["Action"] + 0.30 * rf["Action"],
    "submission_blend_ohe60_rf40.csv": 0.60 * ohe["Action"] + 0.40 * rf["Action"],
}

ohe_rank = rankdata(ohe["Action"], method="average") / len(ohe)
rf_rank = rankdata(rf["Action"], method="average") / len(rf)
candidates["submission_blend_rank_ohe80_rf20.csv"] = 0.80 * ohe_rank + 0.20 * rf_rank
candidates["submission_blend_rank_ohe70_rf30.csv"] = 0.70 * ohe_rank + 0.30 * rf_rank

for filename, action in candidates.items():
    submission = pd.DataFrame({"Id": ohe["Id"], "Action": action})
    submission.to_csv(filename, index=False)
    print(filename)
    print(submission["Action"].describe())
    print()
