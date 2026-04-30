"""
Data Mining HW3 - data preprocessing
====================================
Task: predict ACTION (0 or 1)

This script only performs data loading, inspection, and train/validation
preparation. Model training and submission generation are handled by the
submission*.py scripts.
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def print_section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ============================================================
# 1. Load data
# ============================================================
print_section("1. Load data")

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

print(f"Training set shape: {train_df.shape}")
print(f"Test set shape: {test_df.shape}")


# ============================================================
# 2. Basic data inspection
# ============================================================
print_section("2. Basic data inspection")

print("\n--- Training set info ---")
train_df.info()

print("\n--- Training set descriptive statistics ---")
print(train_df.describe())

print("\n--- Missing values: training set ---")
print(train_df.isnull().sum())

print("\n--- Missing values: test set ---")
print(test_df.isnull().sum())


# ============================================================
# 3. Target and feature inspection
# ============================================================
print_section("3. Target and feature inspection")

print("\n--- ACTION distribution ---")
print(train_df["ACTION"].value_counts())
print("\n--- ACTION ratio ---")
print(train_df["ACTION"].value_counts(normalize=True).round(4))

print("\n--- Unique values per column ---")
for col in train_df.columns:
    print(f"{col}: {train_df[col].nunique()}")

print("\n--- Training set data types ---")
print(train_df.dtypes)

print("\n--- Test set data types ---")
print(test_df.dtypes)


# ============================================================
# 4. Prepare feature matrices
# ============================================================
print_section("4. Prepare feature matrices")

X = train_df.drop("ACTION", axis=1)
y = train_df["ACTION"]
X_test = test_df.drop("id", axis=1)
test_ids = test_df["id"]

print(f"Feature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")
print(f"Test feature matrix shape: {X_test.shape}")
print(f"Test id count: {test_ids.shape[0]}")


# ============================================================
# 5. Train/validation split
# ============================================================
print_section("5. Train/validation split")

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print(f"Training subset shape: {X_train.shape}")
print(f"Validation subset shape: {X_val.shape}")
print("\n--- Training subset ACTION ratio ---")
print(y_train.value_counts(normalize=True).round(4))
print("\n--- Validation subset ACTION ratio ---")
print(y_val.value_counts(normalize=True).round(4))

print_section("Preprocessing complete")
