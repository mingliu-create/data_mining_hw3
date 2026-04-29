"""
資料探勘 HW3 - 資料前處理與預測模型
=====================================
任務：預測 ACTION (0 或 1)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. 載入資料
# ============================================================
print("=" * 60)
print("1. 載入資料")
print("=" * 60)

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

print(f"訓練集大小: {train_df.shape}")
print(f"測試集大小: {test_df.shape}")

# ============================================================
# 2. 資料前處理 - 特徵值分析
# ============================================================
print("\n" + "=" * 60)
print("2. 資料前處理 - 特徵值分析")
print("=" * 60)

# 2.1 基本資訊
print("\n--- 訓練集基本資訊 ---")
print(train_df.info())

print("\n--- 訓練集描述統計 ---")
print(train_df.describe())

# 2.2 缺失值檢查
print("\n--- 缺失值檢查 ---")
print("訓練集缺失值:")
print(train_df.isnull().sum())
print("\n測試集缺失值:")
print(test_df.isnull().sum())

# 2.3 目標變數分佈
print("\n--- 目標變數 ACTION 分佈 ---")
print(train_df['ACTION'].value_counts())
print(f"\n正樣本比例: {train_df['ACTION'].mean():.4f}")

# 2.4 各特徵的唯一值數量
print("\n--- 各特徵唯一值數量 ---")
for col in train_df.columns:
    print(f"{col}: {train_df[col].nunique()}")

# 2.5 檢查是否有類別型特徵需要編碼
print("\n--- 特徵資料類型 ---")
print(train_df.dtypes)

# ============================================================
# 3. 資料前處理 - 處理策略
# ============================================================
print("\n" + "=" * 60)
print("3. 資料前處理 - 處理策略")
print("=" * 60)

# 策略說明
print("""
根據分析結果，建議的前處理策略：

1. 缺失值處理:
   - 如果有缺失值，使用中位數填補（數值型）或眾數填補（類別型）

2. 特徵類型:
   - 所有特徵都是數值型（ID 或計數），可直接使用
   - 不需要進行類別編碼

3. 特徵縮放:
   - 可選擇是否標準化（StandardScaler）
   - 樹狀模型（RF, GB, DT）不需要標準化
   - 線性模型（LR, SVM）建議標準化

4. 特徵工程建議:
   - 可考慮特徵交互作用
   - 可計算各 ID 的出現次數作為新特徵
""")

# ============================================================
# 4. 準備訓練資料
# ============================================================
print("\n" + "=" * 60)
print("4. 準備訓練資料")
print("=" * 60)

# 分離特徵與目標變數
X = train_df.drop('ACTION', axis=1)
y = train_df['ACTION']

# 測試集（去掉 id 欄位）
X_test = test_df.drop('id', axis=1)
test_ids = test_df['id']

print(f"特徵矩陣形狀: {X.shape}")
print(f"目標變數形狀: {y.shape}")
print(f"測試集形狀: {X_test.shape}")

# 分割訓練/驗證集
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n訓練子集: {X_train.shape[0]} 筆")
print(f"驗證集: {X_val.shape[0]} 筆")

# ============================================================
# 5. 模型訓練與比較
# ============================================================
print("\n" + "=" * 60)
print("5. 模型訓練與比較")
print("=" * 60)

# 標準化（用於 LR 和 SVM）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# 定義多個模型
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
}

# 訓練與評估
results = {}
for name, model in models.items():
    print(f"\n--- {name} ---")
    
    # 樹狀模型用原始資料，線性模型用標準化資料
    if name in ['Logistic Regression']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_val_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
    
    accuracy = accuracy_score(y_val, y_pred)
    results[name] = accuracy
    
    print(f"準確率: {accuracy:.4f}")
    print(f"混淆矩陣:\n{confusion_matrix(y_val, y_pred)}")

# 找出最佳模型
best_model_name = max(results, key=results.get)
print(f"\n>>> 最佳模型: {best_model_name} (準確率: {results[best_model_name]:.4f})")

# ============================================================
# 6. 參數調優 - Random Forest 為例
# ============================================================
print("\n" + "=" * 60)
print("6. 參數調優 - Random Forest")
print("=" * 60)

print("""
Random Forest 可調整的主要參數：

1. n_estimators: 森林中決策樹的數量
   - 建議值: 100, 200, 300
   - 越多越好但計算時間增加

2. max_depth: 樹的最大深度
   - 建議值: 10, 20, 30, None（不限制）
   - 避免過擬合

3. min_samples_split: 分割內部節點所需的最小樣本數
   - 建議值: 2, 5, 10

4. min_samples_leaf: 葉節點所需的最小樣本數
   - 建議值: 1, 2, 4

5. max_features: 每次分割時考慮的最大特徵數
   - 建議值: 'sqrt', 'log2', None
""")

# 網格搜尋範例（較小的參數空間以加快執行）
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

print("執行網格搜尋...")
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1
)
grid_search.fit(X_train, y_train)

print(f"\n最佳參數: {grid_search.best_params_}")
print(f"最佳交叉驗證分數: {grid_search.best_score_:.4f}")

# 使用最佳參數預測
best_rf = grid_search.best_estimator_
y_pred_best = best_rf.predict(X_val)
print(f"驗證集準確率: {accuracy_score(y_val, y_pred_best):.4f}")

# ============================================================
# 7. 特徵重要性
# ============================================================
print("\n" + "=" * 60)
print("7. 特徵重要性分析")
print("=" * 60)

feature_importance = pd.DataFrame({
    '特徵': X.columns,
    '重要性': best_rf.feature_importances_
}).sort_values('重要性', ascending=False)

print(feature_importance)

# ============================================================
# 8. 生成預測結果
# ============================================================
print("\n" + "=" * 60)
print("8. 生成預測結果")
print("=" * 60)

# 使用最佳模型預測測試集
test_predictions = best_rf.predict(X_test)

# 建立提交檔案
submission = pd.DataFrame({
    'Id': test_ids,
    'Action': test_predictions
})

submission.to_csv('submission.csv', index=False)
print("預測結果已儲存至 submission.csv")
print(f"\n預測結果分佈:")
print(submission['Action'].value_counts())

print("\n" + "=" * 60)
print("完成！")
print("=" * 60)