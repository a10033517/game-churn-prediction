"""
Step 3: train and evaluate a churn classifier.

Churn rate turned out to be ~93.7% -- heavily imbalanced. That has two
consequences for how this script is written:
  1. accuracy is a useless metric here (predicting "everyone churns"
     scores ~94% accuracy while being completely unhelpful), so the
     script reports a naive-baseline comparison explicitly.
  2. both models use class_weight="balanced" so they don't just learn
     to always predict the majority class.
"""
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("features.csv")

FEATURES = [
    "days_active_14d",
    "active_ratio_14d",
    "recency_days",
    "first_week_sessions",
    "second_week_sessions",
    "trend",
    "max_gap_days",
]
X = df[FEATURES]
y = df["churned"]

print(f"Churn rate: {y.mean():.4f}  (positive class = churned)")
print(f"Retained users (the minority class we actually care about): {(y == 0).sum()}")
print()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# --- naive baseline: what if we just guessed the majority class every time? ---
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_pred = baseline.predict(X_test)
print("=== Naive baseline (always predict 'churned') ===")
print(classification_report(y_test, baseline_pred, digits=3, zero_division=0))

# --- logistic regression ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logreg = LogisticRegression(class_weight="balanced", random_state=42)
logreg.fit(X_train_scaled, y_train)
logreg_proba = logreg.predict_proba(X_test_scaled)[:, 1]
logreg_pred = logreg.predict(X_test_scaled)

print("=== Logistic Regression (class_weight=balanced) ===")
print(classification_report(y_test, logreg_pred, digits=3, zero_division=0))
print(f"ROC-AUC: {roc_auc_score(y_test, logreg_proba):.3f}")
print(f"Average Precision (PR-AUC): {average_precision_score(y_test, logreg_proba):.3f}")
print("Confusion matrix [[TN, FP], [FN, TP]]:")
print(confusion_matrix(y_test, logreg_pred))
print()
print("Coefficients (standardized features, positive = pushes toward churn):")
for name, coef in sorted(zip(FEATURES, logreg.coef_[0]), key=lambda t: -abs(t[1])):
    print(f"  {name:22s} {coef:+.3f}")
print()

# --- random forest ---
rf = RandomForestClassifier(
    n_estimators=300, class_weight="balanced", random_state=42, max_depth=6
)
rf.fit(X_train, y_train)
rf_proba = rf.predict_proba(X_test)[:, 1]
rf_pred = rf.predict(X_test)

print("=== Random Forest (class_weight=balanced) ===")
print(classification_report(y_test, rf_pred, digits=3, zero_division=0))
print(f"ROC-AUC: {roc_auc_score(y_test, rf_proba):.3f}")
print(f"Average Precision (PR-AUC): {average_precision_score(y_test, rf_proba):.3f}")
print("Confusion matrix [[TN, FP], [FN, TP]]:")
print(confusion_matrix(y_test, rf_pred))
print()
print("Feature importance:")
for name, imp in sorted(zip(FEATURES, rf.feature_importances_), key=lambda t: -t[1]):
    print(f"  {name:22s} {imp:.3f}")

results = pd.DataFrame(
    {
        "model": ["baseline", "logistic_regression", "random_forest"],
        "roc_auc": [0.5, roc_auc_score(y_test, logreg_proba), roc_auc_score(y_test, rf_proba)],
        "pr_auc": [
            y_test.mean(),
            average_precision_score(y_test, logreg_proba),
            average_precision_score(y_test, rf_proba),
        ],
    }
)
results.to_csv("model_results.csv", index=False)
print("\nSaved model_results.csv")
