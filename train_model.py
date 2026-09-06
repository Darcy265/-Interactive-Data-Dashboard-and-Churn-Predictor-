"""
train_model.py
--------------
Generates a synthetic telecom churn dataset, preprocesses it with an
sklearn Pipeline, trains a Random Forest classifier, evaluates it, and
saves both the preprocessor and model to disk using joblib.

Run:
    python train_model.py
"""

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report, confusion_matrix
)

# ─────────────────────────────────────────────
# 1. Synthetic Data Generation
# ─────────────────────────────────────────────

def generate_churn_data(n_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """Generate a realistic synthetic telecom churn dataset."""
    rng = np.random.default_rng(random_state)

    tenure          = rng.integers(1, 73, n_samples)           # months 1-72
    monthly_charges = rng.uniform(20, 120, n_samples).round(2)
    total_charges   = (tenure * monthly_charges * rng.uniform(0.85, 1.05, n_samples)).round(2)
    num_services    = rng.integers(1, 8, n_samples)
    support_calls   = rng.integers(0, 11, n_samples)

    contract_type   = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        n_samples, p=[0.55, 0.25, 0.20]
    )
    payment_method  = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        n_samples
    )
    internet_service = rng.choice(
        ["DSL", "Fiber optic", "No"],
        n_samples, p=[0.35, 0.45, 0.20]
    )

    # Inject ~10 % missing values in total_charges and support_calls
    missing_idx_tc = rng.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    missing_idx_sc = rng.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    total_charges_with_nan = total_charges.copy().astype(float)
    support_calls_with_nan = support_calls.copy().astype(float)
    total_charges_with_nan[missing_idx_tc] = np.nan
    support_calls_with_nan[missing_idx_sc] = np.nan

    # Churn probability driven by domain logic
    churn_score = (
        0.40 * (contract_type == "Month-to-month").astype(float)
        + 0.20 * (monthly_charges / 120)
        + 0.15 * (support_calls / 10)
        - 0.25 * (tenure / 72)
        - 0.10 * (num_services / 7)
        + rng.normal(0, 0.05, n_samples)
    )
    churn_prob = 1 / (1 + np.exp(-churn_score * 4))   # sigmoid squash
    churn = (rng.uniform(0, 1, n_samples) < churn_prob).astype(int)

    df = pd.DataFrame({
        "tenure":           tenure,
        "monthly_charges":  monthly_charges,
        "total_charges":    total_charges_with_nan,
        "num_services":     num_services,
        "support_calls":    support_calls_with_nan,
        "contract_type":    contract_type,
        "payment_method":   payment_method,
        "internet_service": internet_service,
        "churn":            churn,
    })
    return df


# ─────────────────────────────────────────────
# 2. Preprocessing Pipeline
# ─────────────────────────────────────────────

NUMERIC_FEATURES = [
    "tenure", "monthly_charges", "total_charges",
    "num_services", "support_calls"
]
CATEGORICAL_FEATURES = [
    "contract_type", "payment_method", "internet_service"
]

def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


# ─────────────────────────────────────────────
# 3. Train & Evaluate
# ─────────────────────────────────────────────

def train_and_evaluate(df: pd.DataFrame):
    X = df.drop(columns=["churn"])
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc  = preprocessor.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_proc, y_train)

    y_pred  = model.predict(X_test_proc)
    y_proba = model.predict_proba(X_test_proc)[:, 1]

    print("\n" + "=" * 50)
    print("  MODEL EVALUATION")
    print("=" * 50)
    print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  ROC-AUC   : {roc_auc_score(y_test, y_proba):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("=" * 50 + "\n")

    return preprocessor, model


# ─────────────────────────────────────────────
# 4. Save Artifacts
# ─────────────────────────────────────────────

def save_artifacts(preprocessor, model, df: pd.DataFrame):
    os.makedirs("data",  exist_ok=True)
    os.makedirs("model", exist_ok=True)

    df.to_csv("data/churn_data.csv", index=False)
    joblib.dump(preprocessor, "model/preprocessor.joblib")
    joblib.dump(model,        "model/churn_model.joblib")

    print("[OK] Saved:")
    print("    data/churn_data.csv")
    print("    model/preprocessor.joblib")
    print("    model/churn_model.joblib")


# ─────────────────────────────────────────────
# 5. Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("[*] Generating synthetic churn dataset ...")
    df = generate_churn_data(n_samples=5000)
    print(f"    {len(df):,} rows x {df.shape[1]} columns  |  "
          f"Churn rate: {df['churn'].mean():.1%}")

    print("[*] Training Random Forest ...")
    preprocessor, model = train_and_evaluate(df)

    print("[*] Saving artifacts ...")
    save_artifacts(preprocessor, model, df)

    print("\n[OK] All done!  Run:  streamlit run app.py")
