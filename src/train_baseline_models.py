"""
AMLGuard AI — Baseline Model Training
---------------------------------------
Loads account_features from MySQL, trains four baseline models
(Logistic Regression, Random Forest, XGBoost, Isolation Forest),
and evaluates them with imbalance-aware metrics.

IMPORTANT: label_* columns are targets, NEVER features. This script
explicitly excludes them from X to avoid the data leakage bug we
already caught and fixed in build_features.py.

Run this from your project root (AMLGuard-AI/), with your venv active:
    python src/train_baseline_models.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, precision_recall_curve
)

import joblib
import os

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost not installed. Run: pip install xgboost")

# ---------------------------------------------------------------
# 1. Connection settings — same as previous scripts
# ---------------------------------------------------------------
DB_USER = "aml_user"
DB_PASSWORD = "yourpassword"
DB_HOST = "localhost"
DB_NAME = "amlguard"

engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}")

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

RANDOM_SEED = 42


def load_features():
    print("Loading account_features from MySQL ...")
    df = pd.read_sql("SELECT * FROM account_features", engine)
    print(f"  Loaded shape: {df.shape}")
    return df


def prepare_data(df: pd.DataFrame):
    label_cols = [c for c in df.columns if c.startswith("label_")]
    print(f"Excluding label columns from features: {label_cols}")

    y = df["label_is_flagged_account"]
    X = df.drop(columns=label_cols + ["acct_id"])

    # Encode categorical columns
    categorical_cols = X.select_dtypes(include="object").columns.tolist()
    print(f"Encoding categorical columns: {categorical_cols}")
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    print(f"Final X shape: {X.shape}, y shape: {y.shape}")
    print(f"Positive class rate: {y.mean():.4f} ({y.sum()} / {len(y)})")

    return X, y


def stratified_split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )
    print(f"\nTrain set: {X_train.shape[0]} rows ({y_train.sum()} positive)")
    print(f"Test set:  {X_test.shape[0]} rows ({y_test.sum()} positive)")
    return X_train, X_test, y_train, y_test


def evaluate(name, y_true, y_pred, y_score):
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc_pr = average_precision_score(y_true, y_score)

    print(f"\n--- {name} ---")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  AUC-PR:    {auc_pr:.4f}")

    return {
        "model": name,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc_pr": auc_pr,
    }


def main():
    df = load_features()
    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = stratified_split(X, y)

    # Scale features (helps LogReg; harmless for tree models)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = []

    # ---------------------------------------------------------
    # 1. Logistic Regression
    # ---------------------------------------------------------
    print("\n" + "=" * 50)
    print("Training Logistic Regression ...")
    logreg = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED
    )
    logreg.fit(X_train_scaled, y_train)
    y_pred = logreg.predict(X_test_scaled)
    y_score = logreg.predict_proba(X_test_scaled)[:, 1]
    results.append(evaluate("Logistic Regression", y_test, y_pred, y_score))
    joblib.dump(logreg, f"{MODELS_DIR}/logistic_regression.pkl")

    # ---------------------------------------------------------
    # 2. Random Forest
    # ---------------------------------------------------------
    print("\n" + "=" * 50)
    print("Training Random Forest ...")
    rf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_score = rf.predict_proba(X_test)[:, 1]
    results.append(evaluate("Random Forest", y_test, y_pred, y_score))
    joblib.dump(rf, f"{MODELS_DIR}/random_forest.pkl")

    # ---------------------------------------------------------
    # 3. XGBoost
    # ---------------------------------------------------------
    if HAS_XGB:
        print("\n" + "=" * 50)
        print("Training XGBoost ...")
        pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        xgb = XGBClassifier(
            n_estimators=300,
            scale_pos_weight=pos_weight,
            random_state=RANDOM_SEED,
            eval_metric="logloss",
        )
        xgb.fit(X_train, y_train)
        y_pred = xgb.predict(X_test)
        y_score = xgb.predict_proba(X_test)[:, 1]
        results.append(evaluate("XGBoost", y_test, y_pred, y_score))
        joblib.dump(xgb, f"{MODELS_DIR}/xgboost.pkl")
    else:
        print("\nSkipping XGBoost (not installed).")

    # ---------------------------------------------------------
    # 4. Isolation Forest (unsupervised)
    # ---------------------------------------------------------
    print("\n" + "=" * 50)
    print("Training Isolation Forest (unsupervised) ...")
    # Train only on data labeled normal, to simulate not knowing the fraud cases
    X_train_normal = X_train[y_train == 0]
    iso = IsolationForest(
        n_estimators=300, contamination=y_train.mean(), random_state=RANDOM_SEED
    )
    iso.fit(X_train_normal)

    # Isolation Forest gives -1 (anomaly) / 1 (normal) — convert to 0/1 with 1=flagged
    raw_pred = iso.predict(X_test)
    y_pred_iso = (raw_pred == -1).astype(int)
    # decision_function: lower score = more abnormal. Flip sign so higher = more suspicious.
    y_score_iso = -iso.decision_function(X_test)
    results.append(evaluate("Isolation Forest", y_test, y_pred_iso, y_score_iso))
    joblib.dump(iso, f"{MODELS_DIR}/isolation_forest.pkl")

    # ---------------------------------------------------------
    # Final comparison table
    # ---------------------------------------------------------
    print("\n" + "=" * 50)
    print("FINAL MODEL COMPARISON (minority class = flagged accounts)")
    print("=" * 50)
    results_df = pd.DataFrame(results).sort_values("auc_pr", ascending=False)
    print(results_df.to_string(index=False))

    results_df.to_csv(f"{MODELS_DIR}/baseline_results.csv", index=False)
    joblib.dump(scaler, f"{MODELS_DIR}/scaler.pkl")
    print(f"\nModels saved to {MODELS_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
