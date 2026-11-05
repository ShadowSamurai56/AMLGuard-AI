"""
AMLGuard AI — Load AMLSim data into MySQL
------------------------------------------
Reads the raw AMLSim CSVs from data/, cleans them, and loads them
into the `amlguard` MySQL database as proper tables.

Run this from your project root (AMLGuard-AI/), with your venv active:
    python src/load_to_mysql.py
"""

import pandas as pd
from sqlalchemy import create_engine

# ---------------------------------------------------------------
# 1. Connection settings — edit these if your setup differs
# ---------------------------------------------------------------
DB_USER = "aml_user"
DB_PASSWORD = "yourpassword"
DB_HOST = "localhost"
DB_NAME = "amlguard"

engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}")

DATA_DIR = "data"


def load_and_clean_transactions():
    print("Reading transactions.csv ...")
    df = pd.read_csv(f"{DATA_DIR}/transactions.csv")

    print(f"  Raw shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    # Standardize column names (lowercase, no spaces)
    df.columns = [c.strip().lower() for c in df.columns]

    # Drop exact duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    print(f"  Dropped {before - len(df)} duplicate rows")

    # Drop rows missing critical fields
    critical_cols = ["tran_id", "orig_acct", "bene_acct", "base_amt"]
    critical_cols = [c for c in critical_cols if c in df.columns]
    before = len(df)
    df = df.dropna(subset=critical_cols)
    print(f"  Dropped {before - len(df)} rows missing critical fields")

    # Parse timestamp
    if "tran_timestamp" in df.columns:
        df["tran_timestamp"] = pd.to_datetime(df["tran_timestamp"], errors="coerce", utc=True)

    # Ensure numeric amount
    if "base_amt" in df.columns:
        df["base_amt"] = pd.to_numeric(df["base_amt"], errors="coerce")
        df = df.dropna(subset=["base_amt"])

    # Normalize is_sar to boolean/int
    if "is_sar" in df.columns:
        df["is_sar"] = df["is_sar"].astype(str).str.strip().str.lower().map(
            {"true": 1, "false": 0, "1": 1, "0": 0}
        ).fillna(0).astype(int)

    print(f"  Clean shape: {df.shape}")
    return df


def load_and_clean_accounts():
    print("Reading accounts.csv ...")
    df = pd.read_csv(f"{DATA_DIR}/accounts.csv")

    print(f"  Raw shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    df.columns = [c.strip().lower() for c in df.columns]

    before = len(df)
    df = df.drop_duplicates()
    print(f"  Dropped {before - len(df)} duplicate rows")

    print(f"  Clean shape: {df.shape}")
    return df


def load_and_clean_alert_transactions():
    print("Reading alert_transactions.csv ...")
    df = pd.read_csv(f"{DATA_DIR}/alert_transactions.csv")
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.drop_duplicates()
    print(f"  Clean shape: {df.shape}")
    return df


def load_and_clean_alert_accounts():
    print("Reading alert_accounts.csv ...")
    df = pd.read_csv(f"{DATA_DIR}/alert_accounts.csv")
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.drop_duplicates()
    print(f"  Clean shape: {df.shape}")
    return df


def main():
    transactions = load_and_clean_transactions()
    accounts = load_and_clean_accounts()
    alert_transactions = load_and_clean_alert_transactions()
    alert_accounts = load_and_clean_alert_accounts()

    print("\nConnecting to MySQL and writing tables ...")

    transactions.to_sql("transactions", engine, if_exists="replace", index=False, chunksize=5000)
    print(f"  -> transactions table written ({len(transactions)} rows)")

    accounts.to_sql("accounts", engine, if_exists="replace", index=False, chunksize=5000)
    print(f"  -> accounts table written ({len(accounts)} rows)")

    alert_transactions.to_sql("alert_transactions", engine, if_exists="replace", index=False)
    print(f"  -> alert_transactions table written ({len(alert_transactions)} rows)")

    alert_accounts.to_sql("alert_accounts", engine, if_exists="replace", index=False)
    print(f"  -> alert_accounts table written ({len(alert_accounts)} rows)")

    print("\nDone. All tables loaded into the 'amlguard' MySQL database.")


if __name__ == "__main__":
    main()
