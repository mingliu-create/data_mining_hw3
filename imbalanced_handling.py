"""
DM Homework 3 - 資料不平衡處理
使用多種技術處理類別不平衡問題：
1. SMOTE (Synthetic Minority Over-sampling Technique)
2. Random Undersampling
3. SMOTETomek (結合過採樣與欠採樣)
4. class_weight='balanced' (模型層級)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 1. 載入資料
# ============================================
print("=" * 60)
print("載入資料...")
print("=" * 60)

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

print(f"訓練集大小: {train_df.shape}")
print(f"測試集大小: {test_df.shape}")

# 準備特徵與目標變數
X = train_df.drop('ACTION', axis=1)
y = train_df['ACTION']

# 原始類別分佈
print("\n原始類別分佈:")
print(y.value_counts())
print(f"類別比例: {y.value_counts(normalize=True).values}")

# 分割訓練集與驗證集
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n訓練集大小: {X_train.shape[0]}")
print(f"驗證集大小: {X_val.shape[0]}")

# ============================================
# 2. 定義評估函數
# ============================================
def evaluate_model(model, X_test, y_test, model_name):
    """評估模型並輸出指標"""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    print(f"\n{'='*40}")
    print(f"模型: {model_name}")
    print(f"{'='*40}")
    print(f"準確率 (Accuracy): {acc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    print(f"\n混淆矩陣:")
    print(confusion_matrix(y_test, y_pred))
    print(f"\n分類報告:")
    print(classification_report(y_test, y_pred, digits=4))
    
    return {'model': model_name, 'accuracy': acc, 'f1': f1, 'auc': auc}

# ============================================
# 3. 基準模型 (無不平衡處理)
# ============================================
print("\n" + "=" * 60)
print("基準模型: 無不平衡處理")
print("=" * 60)

rf_baseline = RandomForestClassifier(
    n_estimators=200,
    max_depth=25,
    min_samples_split=5,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)
rf_baseline.fit(X_train, y_train)
baseline_results = evaluate_model(rf_baseline, X_val, y_val, "基準模型 (無處理)")

# ============================================
# 4. 方法一: SMOTE 過採樣
# ============================================
print("\n" + "=" * 60)
print("方法一: SMOTE 過採樣")
print("=" * 60)

print(f"SMOTE 前類別分佈: {np.bincount(y_train)}")

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"SMOTE 後類別分佈: {np.bincount(y_train_smote)}")
print(f"訓練樣本數: {len(X_train_smote)}")

rf_smote = RandomForestClassifier(
    n_estimators=200,
    max_depth=25,
    min_samples_split=5,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)
rf_smote.fit(X_train_smote, y_train_smote)
smote_results = evaluate_model(rf_smote, X_val, y_val, "SMOTE")

# ============================================
# 5. 方法二: Random Undersampling
# ============================================
print("\n" + "=" * 60)
print("方法二: Random Undersampling")
print("=" * 60)

print(f"Undersampling 前類別分佈: {np.bincount(y_train)}")

rus = RandomUnderSampler(random_state=42)
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)

print(f"Undersampling 後類別分佈: {np.bincount(y_train_rus)}")
print(f"訓練樣本數: {len(X_train_rus)}")

rf_rus = RandomForestClassifier(
    n_estimators=200,
    max_depth=25,
    min_samples_split=5,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)
rf_rus.fit(X_train_rus, y_train_rus)
rus_results = evaluate_model(rf_rus, X_val, y_val, "Random Undersampling")

# ============================================
# 6. 方法三: SMOTETomek (混合方法)
# ============================================
print("\n" + "=" * 60)
print("方法三: SMOTETomek (過採樣 + 欠採樣)")
print("=" * 60)

print(f"SMOTETomek 前類別分佈: {np.bincount(y_train)}")

smote_tomek = SMOTETomek(random_state=42)
X_train_st, y_train_st = smote_tomek.fit_resample(X_train, y_train)

print(f"SMOTETomek 後類別分佈: {np.bincount(y_train_st)}")
print(f"訓練樣本數: {len(X_train_st)}")

rf_st = RandomForestClassifier(
    n_estimators=200,
    max_depth=25,
    min_samples_split=5,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)
rf_st.fit(X_train_st, y_train_st)
st_results = evaluate_model(rf_st, X_val, y_val, "SMOTETomek")

# ============================================
# 7. 方法四: class_weight='balanced'
# ============================================
print("\n" + "=" * 60)
print("方法四: class_weight='balanced'")
print("=" * 60)

rf_balanced = RandomForestClassifier(
    n_estimators=200,
    max_depth=25,
    min_samples_split=5,
    min_samples_leaf=1,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_balanced.fit(X_train, y_train)
balanced_results = evaluate_model(rf_balanced, X_val, y_val, "class_weight='balanced'")

# ============================================
# 8. 方法五: SMOTE + class_weight
# ============================================
print("\n" + "=" * 60)
print("方法五: SMOTE + class_weight='balanced'")
print("=" * 60)

rf_smote_balanced = RandomForestClassifier(
    n_estimators=200,
    max_depth=25,
    min_samples_split=5,
    min_samples_leaf=1,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_smote_balanced.fit(X_train_smote, y_train_smote)
smote_balanced_results = evaluate_model(rf_smote_balanced, X_val, y_val, "SMOTE + class_weight")

# ============================================
# 9. 結果比較
# ============================================
print("\n" + "=" * 60)
print("所有方法結果比較")
print("=" * 60)

results_df = pd.DataFrame([
    baseline_results,
    smote_results,
    rus_results,
    st_results,
    balanced_results,
    smote_balanced_results
])

print(results_df.to_string(index=False))

# 找出最佳模型
best_idx = results_df['f1'].idxmax()
best_model_name = results_df.loc[best_idx, 'model']
print(f"\n最佳 F1-Score 模型: {best_model_name}")

# ============================================
# 10. 生成提交檔案
# ============================================
print("\n" + "=" * 60)
print("生成提交檔案...")
print("=" * 60)

# 選擇最佳模型生成提交檔案
# 使用 class_weight='balanced' 模型 (少數類別召回率較高)

# 重新在全量訓練資料上訓練最佳模型
print("\n在全量訓練資料上重新訓練 class_weight='balanced' 模型...")

rf_final = RandomForestClassifier(
    n_estimators=200,
    max_depth=25,
    min_samples_split=5,
    min_samples_leaf=1,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf_final.fit(X, y)

# 準備測試資料 (移除 id 欄位)
test_X = test_df.drop('id', axis=1) if 'id' in test_df.columns else test_df

# 預測測試集
test_predictions = rf_final.predict(test_X)
test_probabilities = rf_final.predict_proba(test_X)[:, 1]

# 生成提交檔案
submission = pd.DataFrame({
    'Id': test_df['id'] if 'id' in test_df.columns else range(len(test_df)),
    'Action': test_predictions
})

submission.to_csv('submission_imbalanced.csv', index=False)
print(f"提交檔案已生成: submission_imbalanced.csv")
print(f"預測分佈: {pd.Series(test_predictions).value_counts().to_dict()}")

# 同時生成機率版本 (可用於 threshold tuning)
submission_prob = pd.DataFrame({
    'Id': test_df['id'] if 'id' in test_df.columns else range(len(test_df)),
    'Action': test_probabilities
})
submission_prob.to_csv('submission_imbalanced_prob.csv', index=False)
print(f"機率版本已生成: submission_imbalanced_prob.csv")

print("\n" + "=" * 60)
print("完成!")
print("=" * 60)
