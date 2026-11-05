"""
AMLGuard AI — Model Performance & Governance Analytics
Comprehensive evaluation across multi-seed benchmarks, temporal stability, and typologies.
"""

import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.app.config import RESULTS_DIR


def render_benchmarks_view():
    """Renders enterprise performance telemetry, drift stability, and typology metrics."""
    st.markdown("## 📊 Model Performance & Validation Analytics")
    st.markdown(
        """
        Independent evaluation metrics across **AMLSim (Core Banking)** and **Elliptic (Blockchain Forensics)**,
        validating multi-seed statistical significance, resistance to temporal concept drift, and typology-specific recall.
        """
    )

    tab_master, tab_temporal, tab_typology, tab_significance = st.tabs([
        "🏆 Model Benchmark Matrix",
        "⏳ Temporal Drift & Stability",
        "🔄 Typology Detection Recall",
        "📐 Hypothesis Testing & Significance"
    ])

    # ---------------------------------------------------------
    # TAB 1: Master Results
    # ---------------------------------------------------------
    with tab_master:
        master_csv_path = os.path.join(RESULTS_DIR, "tables", "master_results.csv")
        if os.path.exists(master_csv_path):
            df_master = pd.read_csv(master_csv_path)

            ds_choice = st.radio(
                "Filter Evaluation Domain",
                ["All", "AMLSim", "Elliptic (random split)", "Elliptic (time-respecting split)"],
                horizontal=True
            )

            df_filtered = df_master if ds_choice == "All" else df_master[df_master["dataset"] == ds_choice]

            fig_bar = go.Figure()
            colors = {
                "Logistic Regression": "#64748b",
                "Random Forest": "#10b981",
                "XGBoost": "#3b82f6",
                "GraphSAGE": "#8b5cf6",
                "GAT": "#f59e0b",
                "GAT (tuned)": "#ec4899"
            }

            for model_name in df_filtered["model"].unique():
                sub = df_filtered[df_filtered["model"] == model_name]
                fig_bar.add_trace(go.Bar(
                    name=model_name,
                    x=sub["dataset"],
                    y=sub["auc_pr_mean"],
                    error_y=dict(type="data", array=sub["auc_pr_std"], visible=True),
                    marker_color=colors.get(model_name, "#94a3b8"),
                    text=[f"{v:.3f}" for v in sub["auc_pr_mean"]],
                    textposition="outside"
                ))

            fig_bar.update_layout(
                title=dict(text="Model Discriminative Power (AUC-PR ± Std Over 5 Seeds)", font=dict(family="DM Sans", size=14)),
                paper_bgcolor="#0b0f17",
                plot_bgcolor="#0b0f17",
                barmode="group",
                height=380,
                yaxis=dict(title="AUC-PR", range=[0, 1.05], gridcolor="#1e293b"),
                xaxis=dict(title="Evaluation Configuration", gridcolor="#1e293b"),
                font=dict(family="DM Sans", color="#cbd5e1"),
                legend=dict(bgcolor="rgba(15, 23, 42, 0.7)")
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # Formatted Table
            st.markdown("### 📋 Calibrated Performance Summary (Seeds: 0, 1, 42, 123, 2024)")
            candidate_cols = ["dataset", "model", "precision_mean", "recall_mean", "f1_mean", "auc_pr_mean", "test_N", "test_n", "test_positives"]
            display_cols = [c for c in candidate_cols if c in df_filtered.columns]

            df_table = df_filtered[display_cols].copy()
            for col in ["precision_mean", "recall_mean", "f1_mean", "auc_pr_mean"]:
                if col in df_table.columns:
                    df_table[col] = df_table[col].apply(lambda v: f"{v:.4f}" if isinstance(v, (int, float)) else v)

            rename_map = {
                "dataset": "Domain / Dataset",
                "model": "Architecture",
                "precision_mean": "Precision",
                "recall_mean": "Recall",
                "f1_mean": "F1-Score",
                "auc_pr_mean": "AUC-PR",
                "test_N": "Evaluation Size (N)",
                "test_n": "Evaluation Size (N)",
                "test_positives": "Positive Targets"
            }
            st.dataframe(df_table.rename(columns={k: v for k, v in rename_map.items() if k in df_table.columns}), use_container_width=True, hide_index=True)

    # ---------------------------------------------------------
    # TAB 2: Elliptic Temporal Drift
    # ---------------------------------------------------------
    with tab_temporal:
        split_csv_path = os.path.join(RESULTS_DIR, "tables", "split_comparison.csv")
        if os.path.exists(split_csv_path):
            df_split = pd.read_csv(split_csv_path)

            col1, col2 = st.columns([1.2, 0.8])
            with col1:
                fig_split = go.Figure()
                fig_split.add_trace(go.Bar(
                    name="Random Partition (AUC-PR)",
                    x=df_split["model"],
                    y=df_split["random_auc_pr_mean"],
                    text=[f"{v:.3f}" for v in df_split["random_auc_pr_mean"]],
                    textposition="outside",
                    marker_color="#3b82f6"
                ))
                fig_split.add_trace(go.Bar(
                    name="Time-Respecting Partition (AUC-PR)",
                    x=df_split["model"],
                    y=df_split["timesplit_auc_pr_mean"],
                    text=[f"{v:.3f}" for v in df_split["timesplit_auc_pr_mean"]],
                    textposition="outside",
                    marker_color="#f43f5e"
                ))
                fig_split.update_layout(
                    title=dict(text="Concept Drift Degradation: Random Split vs Time-Respecting Split", font=dict(family="DM Sans", size=14)),
                    paper_bgcolor="#0b0f17",
                    plot_bgcolor="#0b0f17",
                    barmode="group",
                    height=380,
                    yaxis=dict(title="AUC-PR", range=[0, 1.15], gridcolor="#1e293b"),
                    font=dict(family="DM Sans", color="#cbd5e1"),
                    legend=dict(bgcolor="rgba(15, 23, 42, 0.7)")
                )
                st.plotly_chart(fig_split, use_container_width=True)

            with col2:
                st.markdown("### 📉 Operational Drift Analysis")
                st.markdown(
                    """
                    - **Ensemble Stability:** Random Forest and XGBoost maintained superior temporal stability, with an AUC-PR drop of only **~20%** between time steps.
                    - **Graph Neural Network Decay:** GraphSAGE and GAT architectures degraded by **51% to 62%**, demonstrating that message passing models require continuous dynamic retraining in live streams.
                    """
                )
                df_split_display = df_split.copy()
                df_split_display["random_auc_pr_mean"] = df_split_display["random_auc_pr_mean"].apply(lambda v: f"{v:.4f}")
                df_split_display["timesplit_auc_pr_mean"] = df_split_display["timesplit_auc_pr_mean"].apply(lambda v: f"{v:.4f}")
                df_split_display["relative_drop_pct"] = df_split_display["relative_drop_pct"].apply(lambda v: f"{v:.1f}%")

                st.dataframe(df_split_display[["model", "random_auc_pr_mean", "timesplit_auc_pr_mean", "relative_drop_pct"]].rename(columns={
                    "model": "Model",
                    "random_auc_pr_mean": "Random Partition",
                    "timesplit_auc_pr_mean": "Time Partition",
                    "relative_drop_pct": "Degradation (%)"
                }), use_container_width=True, hide_index=True)

    # ---------------------------------------------------------
    # TAB 3: Typology Detection Recall
    # ---------------------------------------------------------
    with tab_typology:
        typo_csv_path = os.path.join(RESULTS_DIR, "tables", "typology_counts_v2.csv")
        if os.path.exists(typo_csv_path):
            df_typo = pd.read_csv(typo_csv_path)

            fig_typo = go.Figure()
            for model_name in df_typo["model"].unique():
                sub = df_typo[df_typo["model"] == model_name]
                fig_typo.add_trace(go.Bar(
                    name=model_name,
                    x=sub["alert_type"].str.upper(),
                    y=sub["mean_recall"] * 100,
                    text=[f"{v*100:.1f}%" for v in sub["mean_recall"]],
                    textposition="outside"
                ))

            fig_typo.update_layout(
                title=dict(text="Typology Detection Rate (Mean Recall % on Test Partition)", font=dict(family="DM Sans", size=14)),
                paper_bgcolor="#0b0f17",
                plot_bgcolor="#0b0f17",
                barmode="group",
                height=380,
                yaxis=dict(title="Recall (%)", range=[0, 110], gridcolor="#1e293b"),
                xaxis=dict(title="Money Laundering Typology", gridcolor="#1e293b"),
                font=dict(family="DM Sans", color="#cbd5e1")
            )
            st.plotly_chart(fig_typo, use_container_width=True)

            st.markdown(
                """
                > **Typology Sample Context ($N=16$ Test Positives):** Evaluations are conducted on held-out test set positives ($n=4$ cycles, $n=6$ fan-in, $n=6$ fan-out) averaged across 5 random seeds. While Random Forest achieves higher precision on tabular structuring, GraphSAGE demonstrates structural sensitivity on relational topologies, detecting **80% of circular routing test rings (3.2 / 4 rings on average)** compared to only **25% (1 / 4 rings)** detected by Random Forest.
                """
            )
            df_typo_display = df_typo.copy()
            df_typo_display["mean_recall"] = df_typo_display["mean_recall"].apply(lambda v: f"{v*100:.1f}%")
            df_typo_display["min_recall"] = df_typo_display["min_recall"].apply(lambda v: f"{v*100:.1f}%")
            df_typo_display["max_recall"] = df_typo_display["max_recall"].apply(lambda v: f"{v*100:.1f}%")
            st.dataframe(
                df_typo_display[["model", "alert_type", "n", "mean_recall", "min_recall", "max_recall"]].rename(columns={
                    "model": "Model",
                    "alert_type": "Typology",
                    "n": "n (Test Positives)",
                    "mean_recall": "Mean Recall",
                    "min_recall": "Min Recall",
                    "max_recall": "Max Recall"
                }),
                use_container_width=True,
                hide_index=True
            )

    # ---------------------------------------------------------
    # TAB 4: Statistical Significance
    # ---------------------------------------------------------
    with tab_significance:
        sig_csv_path = os.path.join(RESULTS_DIR, "tables", "significance_test_results.csv")
        if os.path.exists(sig_csv_path):
            df_sig = pd.read_csv(sig_csv_path)
            st.markdown("### 📐 Formal Statistical Hypothesis Testing ($N=5$ Seeds)")
            st.markdown(
                "Welch's two-sample $t$-test and Mann-Whitney $U$ test validating that model ranking differences are statistically significant ($p < 0.05$) rather than initialization artifacts."
            )
            st.dataframe(df_sig, use_container_width=True, hide_index=True)
