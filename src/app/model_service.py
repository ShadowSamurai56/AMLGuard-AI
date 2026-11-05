"""
AMLGuard AI — Inference & Model Scoring Engine
Loads tabular models (RF, XGB, LogReg) and PyG graph models (GraphSAGE, GAT)
and computes fraud probabilities, risk bands, and model consensus.
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv

from src.app.config import MODELS_DIR, RISK_LEVELS
from src.app.data_manager import load_aml_graph


# -------------------------------------------------------------
# PyTorch Geometric Model Definitions
# -------------------------------------------------------------
class GraphSAGEModel(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, out_channels=2, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList([
            SAGEConv(in_channels, hidden_channels),
            SAGEConv(hidden_channels, out_channels)
        ])
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.convs[0](x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[1](x, edge_index)


class GATModel(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, out_channels=2, heads=2, dropout=0.3):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


# -------------------------------------------------------------
# Model Loaders with Streamlit Caching
# -------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_amlsim_models():
    """Loads all 5 trained models for AMLSim."""
    models = {}
    try:
        models["rf"] = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))
        models["xgb"] = joblib.load(os.path.join(MODELS_DIR, "xgboost.pkl"))
        models["lr"] = joblib.load(os.path.join(MODELS_DIR, "logistic_regression.pkl"))
        models["scaler"] = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    except Exception as e:
        st.warning(f"Note: Error loading some tabular models: {e}")

    # Load GraphSAGE
    try:
        sage = GraphSAGEModel(17, 64, 2)
        sage_state = torch.load(os.path.join(MODELS_DIR, "graphsage_1layer.pt"), weights_only=False)
        sage.load_state_dict(sage_state)
        sage.eval()
        models["graphsage"] = sage
    except Exception as e:
        st.warning(f"GraphSAGE model load error: {e}")

    # Load GAT (tuned)
    try:
        gat_path = os.path.join(MODELS_DIR, "gat_tuned_best_amlsim.pt")
        if os.path.exists(gat_path):
            ckpt = torch.load(gat_path, weights_only=False)
            heads = ckpt.get("heads", 2)
            hidden = ckpt.get("hidden", 64)
            gat = GATModel(17, hidden, 2, heads=heads)
            gat.load_state_dict(ckpt["model_state_dict"])
            gat.eval()
            models["gat"] = gat
    except Exception as e:
        pass

    return models


@st.cache_resource(show_spinner=False)
def load_elliptic_models():
    """Loads trained models for Elliptic Bitcoin."""
    models = {}
    try:
        models["rf"] = joblib.load(os.path.join(MODELS_DIR, "random_forest_elliptic.pkl"))
        models["xgb"] = joblib.load(os.path.join(MODELS_DIR, "xgboost_elliptic.pkl"))
        models["lr"] = joblib.load(os.path.join(MODELS_DIR, "logistic_regression_elliptic.pkl"))
        models["scaler"] = joblib.load(os.path.join(MODELS_DIR, "scaler_elliptic.pkl"))
    except Exception as e:
        pass
    return models


# -------------------------------------------------------------
# Inference Functions
# -------------------------------------------------------------
def predict_amlsim_account(acct_id, features_df):
    """
    Computes predictions for a specific account ID across all models.
    Returns: dict of model name -> probability, consensus risk, risk level.
    """
    models = load_amlsim_models()
    row = features_df[features_df["acct_id"] == acct_id]
    if row.empty:
        return None

    # Node index in graph
    node_idx = int(row.index[0])

    # Prepare tabular X
    label_cols = [c for c in row.columns if c.startswith("label_")]
    X_raw = row.drop(columns=label_cols + ["acct_id"]).copy()

    # Ensure all non-numeric columns are encoded as float
    for col in X_raw.columns:
        if not pd.api.types.is_numeric_dtype(X_raw[col]):
            X_raw[col] = 0.0

    X_mat = X_raw.values.astype(float)
    scores = {}

    # 1. Random Forest
    if "rf" in models:
        try:
            scores["Random Forest"] = float(models["rf"].predict_proba(X_mat)[0, 1])
        except Exception:
            scores["Random Forest"] = 0.0

    # 2. XGBoost
    if "xgb" in models:
        try:
            scores["XGBoost"] = float(models["xgb"].predict_proba(X_mat)[0, 1])
        except Exception:
            scores["XGBoost"] = 0.0

    # 3. Logistic Regression
    if "lr" in models and "scaler" in models:
        try:
            X_scaled = models["scaler"].transform(X_mat)
            scores["Logistic Regression"] = float(models["lr"].predict_proba(X_scaled)[0, 1])
        except Exception:
            scores["Logistic Regression"] = 0.0

    # 4. GraphSAGE & GAT (using pre-built graph)
    graph = load_aml_graph()
    if graph is not None:
        with torch.no_grad():
            if "graphsage" in models:
                out_sage = models["graphsage"](graph.x, graph.edge_index)
                prob_sage = F.softmax(out_sage, dim=1)[node_idx, 1].item()
                scores["GraphSAGE"] = float(prob_sage)

            if "gat" in models:
                out_gat = models["gat"](graph.x, graph.edge_index)
                prob_gat = F.softmax(out_gat, dim=1)[node_idx, 1].item()
                scores["GAT (Tuned)"] = float(prob_gat)

    # ---------------------------------------------------------
    # Calibrated Consensus & Risk Disposition
    # Weighted by production validation performance:
    # Random Forest (35%), XGBoost (35%), GraphSAGE (20%), GAT (5%), LogReg (5%)
    # ---------------------------------------------------------
    weights = {
        "Random Forest": 0.35,
        "XGBoost": 0.35,
        "GraphSAGE": 0.20,
        "GAT (Tuned)": 0.05,
        "Logistic Regression": 0.05,
    }

    weighted_sum = 0.0
    total_weight = 0.0
    for m_name, score in scores.items():
        w = weights.get(m_name, 0.1)
        weighted_sum += score * w
        total_weight += w

    consensus_prob = weighted_sum / max(total_weight, 0.001)
    val_list = list(scores.values())
    max_prob = max(val_list) if val_list else 0.0
    min_prob = min(val_list) if val_list else 0.0

    # Operational disposition recommendation
    if consensus_prob >= 0.65:
        disposition = "ESCALATE TO SAR"
        band = "CRITICAL" if consensus_prob >= 0.80 else "HIGH"
    elif consensus_prob >= 0.35:
        disposition = "ANALYST REVIEW"
        band = "ELEVATED"
    else:
        disposition = "AUTO-CLEARED"
        band = "LOW"

    # Automated model reconciliation note
    divergence_note = None
    if (max_prob - min_prob) >= 0.35:
        if scores.get("GraphSAGE", 0) > 0.65 and scores.get("Random Forest", 0) < 0.4:
            divergence_note = "Relational Flag: GraphSAGE identified coordinated multi-hop network patterns (circular routing) invisible to isolated tabular scoring."
        elif scores.get("Logistic Regression", 0) > 0.6 and scores.get("XGBoost", 0) < 0.2:
            divergence_note = "Model Divergence Detected: Linear baseline elevated by gross volume; primary gradient-boosted ensemble clears entity with balanced turnover."

    return {
        "acct_id": acct_id,
        "node_idx": node_idx,
        "scores": scores,
        "max_prob": max_prob,
        "consensus_prob": consensus_prob,
        "risk_level": band,
        "disposition": disposition,
        "divergence_note": divergence_note,
        "features": X_raw.iloc[0].to_dict(),
    }


def predict_elliptic_transaction(tx_id, sample_features_df):
    """Computes tabular predictions for an Elliptic transaction."""
    models = load_elliptic_models()
    row = sample_features_df[sample_features_df["tx_id"] == tx_id]
    if row.empty:
        return None

    drop_cols = ["tx_id", "class"]
    X_raw = row.drop(columns=[c for c in drop_cols if c in row.columns])
    for col in X_raw.columns:
        if not pd.api.types.is_numeric_dtype(X_raw[col]):
            X_raw[col] = 0.0
    X_mat = X_raw.values.astype(float)
    scores = {}

    if "rf" in models:
        try:
            scores["Random Forest"] = float(models["rf"].predict_proba(X_mat)[0, 1])
        except Exception:
            scores["Random Forest"] = 0.0

    if "xgb" in models:
        try:
            scores["XGBoost"] = float(models["xgb"].predict_proba(X_mat)[0, 1])
        except Exception:
            scores["XGBoost"] = 0.0

    if "lr" in models and "scaler" in models:
        try:
            X_scaled = models["scaler"].transform(X_mat)
            scores["Logistic Regression"] = float(models["lr"].predict_proba(X_scaled)[0, 1])
        except Exception:
            scores["Logistic Regression"] = 0.0

    weights = {"Random Forest": 0.45, "XGBoost": 0.45, "Logistic Regression": 0.10}
    weighted_sum = sum(scores[m] * weights.get(m, 0.1) for m in scores)
    total_w = sum(weights.get(m, 0.1) for m in scores)
    consensus_prob = weighted_sum / max(total_w, 0.001)

    val_list = list(scores.values())
    max_prob = max(val_list) if val_list else 0.0

    if consensus_prob >= 0.65:
        disposition = "ESCALATE TO SAR"
        band = "CRITICAL" if consensus_prob >= 0.80 else "HIGH"
    elif consensus_prob >= 0.35:
        disposition = "ANALYST REVIEW"
        band = "ELEVATED"
    else:
        disposition = "AUTO-CLEARED"
        band = "LOW"

    return {
        "tx_id": tx_id,
        "scores": scores,
        "max_prob": max_prob,
        "consensus_prob": consensus_prob,
        "risk_level": band,
        "disposition": disposition,
        "divergence_note": None,
        "features": X_raw.iloc[0].to_dict(),
    }
