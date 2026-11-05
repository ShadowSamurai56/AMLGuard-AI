"""
AMLGuard AI — Explainability Service (SHAP & GNNExplainer)
Provides transparent, auditable feature and topological attributions.
"""

import os
import shap
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn.functional as F
from torch_geometric.explain import Explainer, GNNExplainer

from src.app.config import FEATURE_LABELS_AMLSIM, FEATURE_NAMES_AMLSIM
from src.app.model_service import load_amlsim_models, load_aml_graph


# -------------------------------------------------------------
# 1. SHAP Tabular Feature Attribution (TreeSHAP)
# -------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_rf_tree_explainer():
    """Initializes and caches TreeExplainer for Random Forest."""
    models = load_amlsim_models()
    if "rf" in models:
        try:
            return shap.TreeExplainer(models["rf"])
        except Exception:
            return None
    return None


def compute_local_shap(features_dict, top_k=8):
    """
    Computes exact Shapley values for the selected account's features.
    Returns: Plotly horizontal bar chart of top positive and negative drivers.
    """
    explainer = get_rf_tree_explainer()
    if explainer is None:
        fig = go.Figure()
        fig.update_layout(title="SHAP Explainer not available.")
        return fig

    # Build ordered feature array
    feat_names = FEATURE_NAMES_AMLSIM
    x_vals = []
    for col in feat_names:
        val = features_dict.get(col, 0.0)
        if isinstance(val, str):
            val = 0.0
        x_vals.append(float(val))

    X_mat = np.array(x_vals).reshape(1, -1)
    shap_vals = explainer.shap_values(X_mat)

    # Scikit-learn binary RF yields shap_vals of shape [1, n_features, 2] or list of 2 arrays
    if isinstance(shap_vals, list):
        # Class 1 (SAR)
        vals = shap_vals[1][0]
    elif len(shap_vals.shape) == 3:
        vals = shap_vals[0, :, 1]
    else:
        vals = shap_vals[0]

    # Create DataFrame for ranking
    df_shap = pd.DataFrame({
        "feature_key": feat_names,
        "feature_name": [FEATURE_LABELS_AMLSIM.get(k, k) for k in feat_names],
        "value": x_vals,
        "shap": vals,
        "abs_shap": np.abs(vals)
    })

    df_shap = df_shap.sort_values("abs_shap", ascending=True).tail(top_k)

    # Plotly horizontal bar chart with refined slim bars
    colors = ["#f43f5e" if s > 0 else "#10b981" for s in df_shap["shap"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df_shap["feature_name"],
        x=df_shap["shap"],
        orientation="h",
        width=0.55,
        marker=dict(
            color=colors,
            line=dict(color="rgba(255, 255, 255, 0.15)", width=1)
        ),
        text=[f"Val: {v:,.1f}" if abs(v) > 10 else f"Val: {v:.2f}" for v in df_shap["value"]],
        textposition="outside",
        textfont=dict(size=10, family="JetBrains Mono, monospace", color="#94a3b8"),
        hoverinfo="text",
        hovertext=[
            f"<b>{row['feature_name']}</b><br>"
            f"Account Value: {row['value']:,.2f}<br>"
            f"SHAP Impact: {row['shap']:+.4f} "
            f"({'Pushes to Suspicious' if row['shap'] > 0 else 'Pushes to Legitimate'})"
            for _, row in df_shap.iterrows()
        ]
    ))

    fig.update_layout(
        paper_bgcolor="#0b0f17",
        plot_bgcolor="#0b0f17",
        height=320,
        margin=dict(l=10, r=40, t=30, b=30),
        xaxis=dict(
            title=dict(text="SHAP Attribution (Shapley Value on Log-Odds)", font=dict(size=11, color="#64748b", family="DM Sans")),
            showgrid=True,
            gridcolor="#162032",
            zeroline=True,
            zerolinecolor="#334155",
            tickfont=dict(size=10, color="#94a3b8", family="JetBrains Mono, monospace")
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(size=11, color="#cbd5e1", family="DM Sans")
        ),
        annotations=[
            dict(
                text="<span style='color: #fb7185;'>■ Suspicious Contribution</span> &nbsp;&nbsp;|&nbsp;&nbsp; <span style='color: #34d399;'>■ Legitimate Contribution</span>",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=1.12,
                font=dict(color="#94a3b8", size=11, family="DM Sans")
            )
        ]
    )

    return fig


# -------------------------------------------------------------
# 2. GNNExplainer Topological Edge & Subgraph Attribution
# -------------------------------------------------------------
@st.cache_data(show_spinner=False)
def compute_gnn_explanation(node_idx, epochs=25):
    """
    Computes GNNExplainer edge importance weights for the 1-hop ego network.
    Returns: dict of (u, v) -> edge_importance_score.
    """
    models = load_amlsim_models()
    graph = load_aml_graph()

    if "graphsage" not in models or graph is None:
        return {}

    try:
        model = models["graphsage"]
        model.eval()

        explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=epochs),
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type="object",
            model_config=dict(mode="multiclass_classification", task_level="node", return_type="raw")
        )

        explanation = explainer(graph.x, graph.edge_index, index=node_idx)
        edge_mask = explanation.edge_mask.detach().cpu().numpy()

        # Map edge mask back to (u, v) pairs
        edge_index = graph.edge_index.cpu().numpy()
        edge_weights = {}

        # Only retain edges connected to node_idx
        for i in range(edge_index.shape[1]):
            u, v = int(edge_index[0, i]), int(edge_index[1, i])
            if u == node_idx or v == node_idx:
                edge_weights[(u, v)] = float(edge_mask[i])

        return edge_weights
    except Exception as e:
        return {}
