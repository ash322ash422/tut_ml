"""
train_model.py
--------------
Run this once to produce:  credit_model.pkl  and  sample_data.csv

Usage:
    python train_model.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# ── Reproducibility
SEED = 42
np.random.seed(SEED)

# 1. Load the dataset
df = pd.read_csv("sample_data.csv")

# ── 2. Train / Test Split 
FEATURES = [
    "annual_income", "loan_amount", "credit_history_yrs",
    "num_late_payments", "debt_to_income", "employment_yrs", "num_open_accounts",
]

X = df[FEATURES]
y = df["approved"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

# ── 3. Train RandomForestClassifier 
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=8,
    min_samples_leaf=10,
    class_weight="balanced",
    random_state=SEED,
    n_jobs=-1,
)
model.fit(X_train, y_train)

# ── 4. Evaluation 
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n── Test-set Performance ")
print(classification_report(y_test, y_pred, target_names=["Denied", "Approved"]))
print(f"ROC-AUC : {roc_auc_score(y_test, y_proba):.4f}")

importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\n── Feature Importances ")
print(importances.to_string())

# ── 5. Save Model
joblib.dump({"model": model, "features": FEATURES}, "credit_model.pkl")
print("\n✅  credit_model.pkl saved")
