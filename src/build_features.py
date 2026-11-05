"""
AMLGuard AI — Feature Engineering
----------------------------------
Reads `transactions` and `accounts` from MySQL, computes per-account
behavioral features (used later by both the ML models and the graph
construction step), and writes the result to a new `account_features`
table in the same database.

Run this from your project root (AMLGuard-AI/), with your venv active:
    python src/build_features.py
"""

import pandas as pd
from sqlalchemy import create_engine

# ---------------------------------------------------------------
# 1. Connection settings — same as load_to_mysql.py
# ---------------------------------------------------------------
DB_USER = "aml_user"
DB_PASSWORD = "yourpassword"
DB_HOST = "localhost"
DB_NAME = "amlguard"

engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}")


def load_tables():
    print("Loading tables from MySQL ...")
    transactions = pd.read_sql("SELECT * FROM transactions", engine)
    accounts = pd.read_sql("SELECT * FROM accounts", engine)
    print(f"  transactions: {transactions.shape}")
    print(f"  accounts: {accounts.shape}")
    return transactions, accounts


def build_outgoing_features(tx: pd.DataFrame) -> pd.DataFrame:
    """Features describing an account's behavior as a SENDER."""
    g = tx.groupby("orig_acct")

    out = g.agg(
        out_tx_count=("tran_id", "count"),
        out_total_amt=("base_amt", "sum"),
        out_avg_amt=("base_amt", "mean"),
        out_max_amt=("base_amt", "max"),
        out_std_amt=("base_amt", "std"),
        out_unique_counterparties=("bene_acct", "nunique"),
        out_sar_count=("is_sar", "sum"),
    ).reset_index().rename(columns={"orig_acct": "acct_id"})

    out["out_std_amt"] = out["out_std_amt"].fillna(0)
    return out


def build_incoming_features(tx: pd.DataFrame) -> pd.DataFrame:
    """Features describing an account's behavior as a RECEIVER."""
    g = tx.groupby("bene_acct")

    inc = g.agg(
        in_tx_count=("tran_id", "count"),
        in_total_amt=("base_amt", "sum"),
        in_avg_amt=("base_amt", "mean"),
        in_max_amt=("base_amt", "max"),
        in_std_amt=("base_amt", "std"),
        in_unique_counterparties=("orig_acct", "nunique"),
        in_sar_count=("is_sar", "sum"),
    ).reset_index().rename(columns={"bene_acct": "acct_id"})

    inc["in_std_amt"] = inc["in_std_amt"].fillna(0)
    return inc


def build_timing_features(tx: pd.DataFrame) -> pd.DataFrame:
    """Average time (in hours) between an account's consecutive outgoing transactions."""
    tx_sorted = tx.sort_values(["orig_acct", "tran_timestamp"])
    tx_sorted["prev_ts"] = tx_sorted.groupby("orig_acct")["tran_timestamp"].shift(1)
    tx_sorted["gap_hours"] = (
        tx_sorted["tran_timestamp"] - tx_sorted["prev_ts"]
    ).dt.total_seconds() / 3600

    timing = tx_sorted.groupby("orig_acct")["gap_hours"].mean().reset_index()
    timing = timing.rename(columns={"orig_acct": "acct_id", "gap_hours": "avg_hours_between_out_tx"})
    return timing


def build_account_features(transactions: pd.DataFrame, accounts: pd.DataFrame) -> pd.DataFrame:
    print("Building outgoing features ...")
    out_feats = build_outgoing_features(transactions)

    print("Building incoming features ...")
    in_feats = build_incoming_features(transactions)

    print("Building timing features ...")
    timing_feats = build_timing_features(transactions)

    # Base: every account, even ones with zero transactions
    base = accounts[["acct_id", "type"]].copy()

    print("Merging feature sets ...")
    feats = base.merge(out_feats, on="acct_id", how="left")
    feats = feats.merge(in_feats, on="acct_id", how="left")
    feats = feats.merge(timing_feats, on="acct_id", how="left")

    # Fill accounts with no outgoing/incoming activity
    numeric_cols = feats.select_dtypes(include="number").columns
    feats[numeric_cols] = feats[numeric_cols].fillna(0)

    # Derived features (behavioral only — no label leakage)
    feats["total_tx_count"] = feats["out_tx_count"] + feats["in_tx_count"]
    feats["net_flow"] = feats["in_total_amt"] - feats["out_total_amt"]
    feats["total_unique_counterparties"] = (
        feats["out_unique_counterparties"] + feats["in_unique_counterparties"]
    )

    # --- LABEL (kept separate — NEVER a feature) ---
    # These come directly from is_sar and must not be used as model inputs.
    label_total_sar_tx_count = feats["out_sar_count"] + feats["in_sar_count"]
    label_is_flagged_account = (label_total_sar_tx_count > 0).astype(int)

    # Drop the leakage-source columns from the feature table entirely so
    # they can never accidentally be passed to a model as X.
    feats = feats.drop(columns=["out_sar_count", "in_sar_count"])

    # Attach label columns with an explicit "label_" prefix so they're
    # unmistakable and easy to exclude when building the X matrix.
    feats["label_total_sar_tx_count"] = label_total_sar_tx_count
    feats["label_is_flagged_account"] = label_is_flagged_account

    print(f"Final feature table shape: {feats.shape}")
    print("NOTE: columns prefixed 'label_' are targets, not features — exclude them from X.")
    return feats


def main():
    transactions, accounts = load_tables()

    # Ensure timestamp is proper datetime (MySQL round-trip can return strings)
    transactions["tran_timestamp"] = pd.to_datetime(transactions["tran_timestamp"], errors="coerce", utc=True)

    features = build_account_features(transactions, accounts)

    print("\nWriting account_features table to MySQL ...")
    features.to_sql("account_features", engine, if_exists="replace", index=False, chunksize=5000)
    print(f"  -> account_features table written ({len(features)} rows, {features.shape[1]} columns)")

    print("\nSample of engineered features:")
    print(features.head())

    label_cols = [c for c in features.columns if c.startswith("label_")]
    feature_cols = [c for c in features.columns if not c.startswith("label_") and c != "acct_id"]
    print(f"\nLabel columns (exclude from X): {label_cols}")
    print(f"Feature columns (safe for X): {len(feature_cols)} columns")

    print("\nDone.")


if __name__ == "__main__":
    main()
