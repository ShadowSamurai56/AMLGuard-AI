"""
AMLGuard AI — Data Management & Ingestion Engine
Handles live MySQL querying with seamless parquet cache fallback and PyG graph loading.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine
import torch

from src.app.config import DB_CONFIG, CACHE_DIR, MODELS_DIR, FEATURE_NAMES_AMLSIM


@st.cache_data(show_spinner=False)
def load_amlsim_features():
    """Loads AMLSim engineered account features from MySQL or local cache."""
    # 1. Try local parquet cache first for instant load
    cache_path = os.path.join(CACHE_DIR, "amlsim_features.parquet")
    if os.path.exists(cache_path):
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            pass

    # 2. Fallback to MySQL if cache is missing
    try:
        url = f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}"
        engine = create_engine(url)
        df = pd.read_sql("SELECT * FROM account_features", engine)
        os.makedirs(CACHE_DIR, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        return df
    except Exception as e:
        st.error(f"Failed to load AMLSim features from MySQL and cache: {e}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_amlsim_transactions():
    """Loads raw transactions from MySQL or local cache."""
    cache_path = os.path.join(CACHE_DIR, "amlsim_transactions.parquet")
    if os.path.exists(cache_path):
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            pass

    try:
        url = f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}"
        engine = create_engine(url)
        df = pd.read_sql("SELECT * FROM transactions", engine)
        os.makedirs(CACHE_DIR, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        return df
    except Exception as e:
        st.error(f"Failed to load transactions: {e}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_amlsim_alerts():
    """Loads alert typologies from MySQL or local cache."""
    cache_path = os.path.join(CACHE_DIR, "amlsim_alerts.parquet")
    if os.path.exists(cache_path):
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            pass

    try:
        url = f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}"
        engine = create_engine(url)
        df = pd.read_sql("SELECT acct_id, alert_type, alert_id FROM alert_accounts", engine)
        os.makedirs(CACHE_DIR, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        return df
    except Exception as e:
        return pd.DataFrame(columns=["acct_id", "alert_type", "alert_id"])


@st.cache_resource(show_spinner=False)
def load_aml_graph():
    """Loads PyTorch Geometric AML graph."""
    graph_path = os.path.join(MODELS_DIR, "aml_graph.pt")
    if os.path.exists(graph_path):
        return torch.load(graph_path, weights_only=False)
    return None


@st.cache_data(show_spinner=False)
def load_elliptic_sample():
    """Loads cached sample of Elliptic Bitcoin transactions."""
    feat_path = os.path.join(CACHE_DIR, "elliptic_sample_features.parquet")
    edge_path = os.path.join(CACHE_DIR, "elliptic_sample_edges.parquet")

    if os.path.exists(feat_path) and os.path.exists(edge_path):
        df_feat = pd.read_parquet(feat_path)
        df_edges = pd.read_parquet(edge_path)
        return df_feat, df_edges
    return None, None


@st.cache_data(show_spinner=False)
def get_accounts_catalog(dataset="AMLSim", filter_mode="All"):
    """
    Returns a curated list of accounts with metadata for user selection.
    Filters: All, Confirmed SAR Only, Cycles, Fan-In, Fan-Out, Normal
    """
    if dataset == "AMLSim":
        features = load_amlsim_features()
        alerts = load_amlsim_alerts()

        if features.empty:
            return []

        # Merge alert types
        cols_needed = ["acct_id", "type", "label_is_flagged_account", "out_total_amt", "total_tx_count"]
        if "in_total_amt" in features.columns:
            cols_needed.append("in_total_amt")
        merged = features[cols_needed].copy()
        merged = merged.merge(alerts[["acct_id", "alert_type"]].drop_duplicates("acct_id"), on="acct_id", how="left")
        merged["alert_type"] = merged["alert_type"].fillna("normal")

        if filter_mode == "Confirmed SAR Only":
            merged = merged[merged["label_is_flagged_account"] == 1]
        elif filter_mode == "Cycle Typology":
            merged = merged[merged["alert_type"] == "cycle"]
        elif filter_mode == "Fan-In Typology":
            merged = merged[merged["alert_type"] == "fan_in"]
        elif filter_mode == "Fan-Out Typology":
            merged = merged[merged["alert_type"] == "fan_out"]
        elif filter_mode == "Normal Only":
            merged = merged[merged["label_is_flagged_account"] == 0]

        items = []
        for _, row in merged.iterrows():
            aid = int(row["acct_id"])
            typ = row["alert_type"].upper()
            is_sar = "⚠️ SAR" if row["label_is_flagged_account"] == 1 else "✅ NORMAL"
            gross_flow = float(row.get("out_total_amt", 0.0) + row.get("in_total_amt", 0.0))
            label = f"ID #{aid:<4} | {is_sar:<10} | [{typ:<7}] | Total Gross Flow: ${gross_flow:,.0f}"
            items.append({"id": aid, "label": label, "is_sar": int(row["label_is_flagged_account"]), "type": row["alert_type"]})

        return items

    elif dataset == "Elliptic":
        feat, edges = load_elliptic_sample()
        if feat is None:
            return []

        if filter_mode == "Confirmed SAR Only":
            feat = feat[feat["class"] == "1"]
        elif filter_mode == "Normal Only":
            feat = feat[feat["class"] == "2"]

        items = []
        for _, row in feat.iterrows():
            tx_id = int(row["tx_id"])
            status = "⚠️ ILLICIT" if row["class"] == "1" else ("✅ LICIT" if row["class"] == "2" else "UNKNOWN")
            label = f"Tx #{tx_id} | {status} | Step: {int(row.get('feat_0', 0))}"
            items.append({"id": tx_id, "label": label, "is_sar": 1 if row["class"] == "1" else 0, "type": "bitcoin"})

        return items

    return []

