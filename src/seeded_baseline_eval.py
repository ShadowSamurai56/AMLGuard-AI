"""
AMLGuard AI — Seeded Baseline Evaluation (Week 1 of Research Rigor Plan)
--------------------------------------------------------------------------
Re-runs Logistic Regression, Random Forest, and XGBoost across 5 seeds on:
  1. AMLSim (account_features from MySQL, using the graph masks from
     models/aml_graph.pt for a 60/20/20 train/val/test split that matches
     the GNN evaluation — fixes an earlier 80/20 mismatch)
  2. Elliptic, random split
  3. Elliptic, time-respecting split

Reports mean ± std for precision, recall, F1, AUC-PR for every model on
every dataset/split combination. This replaces all single-seed baseline
numbers used so far with statistically defensible ones.

Run from project root, venv active:
    python src/seeded_baseline_eval.py [--only AMLSim|Elliptic|all]
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from xgboost import XGBClassifier

SEEDS = [0, 1, 42, 123, 2024]

DB_USER = "aml_user"
DB_PASSWORD = "yourpassword"
DB_HOST = "localhost"
DB_NAME = "amlguard"
engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}")


def evaluate_once(y_true, y_pred, y_score):
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc_pr": average_precision_score(y_true, y_score),
    }


def run_seeded_models(X_train, y_train, X_test, y_test, dataset_label, seeds=SEEDS):
    """Trains LogReg, RF, XGBoost across `seeds`, returns a long-form results list."""
    rows = []

    for seed in seeds:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # NOTE: LogisticRegression with the default lbfgs solver is a convex
        # optimization with a unique solution given fixed data — random_state
        # has effectively no influence on the converged result. Zero variance
        # here is expected and correct, not a seeding failure (unlike XGBoost
        # above, which required an explicit fix).
        logreg = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
        logreg.fit(X_train_scaled, y_train)
        y_pred = logreg.predict(X_test_scaled)
        y_score = logreg.predict_proba(X_test_scaled)[:, 1]
        m = evaluate_once(y_test, y_pred, y_score)
        m.update({"dataset": dataset_label, "model": "Logistic Regression", "seed": seed})
        rows.append(m)

        rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        y_score = rf.predict_proba(X_test)[:, 1]
        m = evaluate_once(y_test, y_pred, y_score)
        m.update({"dataset": dataset_label, "model": "Random Forest", "seed": seed})
        rows.append(m)

        # NOTE: subsample/colsample_bytree < 1.0 are required for random_state
        # to have any effect on XGBoost. With full subsampling (the default),
        # XGBoost is deterministic given the data, so seeding is a no-op and
        # produces identical results every run — this was verified empirically
        # (all-zero std across 5 seeds) before this fix was added.
        pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        xgb = XGBClassifier(
            n_estimators=300,
            scale_pos_weight=pos_weight,
            random_state=seed,
            eval_metric="logloss",
            subsample=0.8,
            colsample_bytree=0.8,
        )
        xgb.fit(X_train, y_train)
        y_pred = xgb.predict(X_test)
        y_score = xgb.predict_proba(X_test)[:, 1]
        m = evaluate_once(y_test, y_pred, y_score)
        m.update({"dataset": dataset_label, "model": "XGBoost", "seed": seed})
        rows.append(m)

        print(f"  [{dataset_label}] seed {seed} done")

    return rows


def load_amlsim():
    """Load AMLSim features from MySQL, split using graph masks from
    models/aml_graph.pt (60/20/20, same split as GNN evaluation).

    The graph's node ordering matches the account_features table row order
    (node i = row i), so train_mask/test_mask index directly into X and y.
    Categorical encoding is identical to the original load (LabelEncoder on
    object columns). Features are NOT pre-scaled here — the caller
    (run_seeded_models) applies StandardScaler per seed as before."""
    print("Loading AMLSim account_features from MySQL ...")
    df = pd.read_sql("SELECT * FROM account_features", engine)
    label_cols = [c for c in df.columns if c.startswith("label_")]
    y = df["label_is_flagged_account"].values
    X = df.drop(columns=label_cols + ["acct_id"])

    categorical_cols = X.select_dtypes(include="object").columns.tolist()
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Use the same graph masks as the GNNs (60/20/20 split from build_graph.py)
    # to ensure baselines and GNNs are evaluated on the exact same test set.
    data = torch.load("models/aml_graph.pt", weights_only=False)
    train_mask = data.train_mask.cpu().numpy()
    test_mask = data.test_mask.cpu().numpy()

    X_vals = X.values
    return X_vals[train_mask], y[train_mask], X_vals[test_mask], y[test_mask]


def load_elliptic(split_type):
    """split_type: 'random' or 'timesplit'"""
    print(f"Loading Elliptic ({split_type} split) ...")
    path = "models/elliptic_graph.pt" if split_type == "random" else "models/elliptic_graph_timesplit.pt"
    data = torch.load(path, weights_only=False)

    X_all = data.x.cpu().numpy()
    y_all = data.y.cpu().numpy()
    train_mask = data.train_mask.cpu().numpy()
    test_mask = data.test_mask.cpu().numpy()

    return X_all[train_mask], y_all[train_mask], X_all[test_mask], y_all[test_mask]


def summarize(results_df):
    summary = results_df.groupby(["dataset", "model"]).agg(
        precision_mean=("precision", "mean"), precision_std=("precision", "std"),
        recall_mean=("recall", "mean"), recall_std=("recall", "std"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        auc_pr_mean=("auc_pr", "mean"), auc_pr_std=("auc_pr", "std"),
    ).reset_index()
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["AMLSim", "Elliptic", "all"], default="all",
                        help="Run only AMLSim, only Elliptic, or all (default).")
    args = parser.parse_args()

    all_rows = []

    if args.only in ("AMLSim", "all"):
        X_train, y_train, X_test, y_test = load_amlsim()
        all_rows += run_seeded_models(X_train, y_train, X_test, y_test, "AMLSim")

    if args.only in ("Elliptic", "all"):
        X_train, y_train, X_test, y_test = load_elliptic("random")
        all_rows += run_seeded_models(X_train, y_train, X_test, y_test, "Elliptic (random split)")

        X_train, y_train, X_test, y_test = load_elliptic("timesplit")
        all_rows += run_seeded_models(X_train, y_train, X_test, y_test, "Elliptic (time-respecting split)")

    results_df = pd.DataFrame(all_rows)

    if args.only == "all":
        results_df.to_csv("models/seeded_baseline_results_raw.csv", index=False)
        summary = summarize(results_df)
        summary.to_csv("models/seeded_baseline_results_summary.csv", index=False)
    else:
        # When running a subset, print but do not overwrite the full CSV.
        # Caller is responsible for merging rows into the existing file.
        summary = summarize(results_df)

    print("\n" + "=" * 70)
    print("SEEDED BASELINE RESULTS (mean ± std across 5 seeds)")
    print("=" * 70)
    for _, row in summary.iterrows():
        print(f"\n{row['dataset']} — {row['model']}")
        print(f"  Precision: {row['precision_mean']:.4f} ± {row['precision_std']:.4f}")
        print(f"  Recall:    {row['recall_mean']:.4f} ± {row['recall_std']:.4f}")
        print(f"  F1:        {row['f1_mean']:.4f} ± {row['f1_std']:.4f}")
        print(f"  AUC-PR:    {row['auc_pr_mean']:.4f} ± {row['auc_pr_std']:.4f}")

    print("\nSaved: models/seeded_baseline_results_raw.csv (all 15 runs per dataset)")
    print("Saved: models/seeded_baseline_results_summary.csv (mean ± std)")


if __name__ == "__main__":
    main()
