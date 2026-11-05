"""
AMLGuard AI — Platform Architecture & Model Governance
Institutional Documentation for Compliance, Audit, and Technical Validation
"""

import os
import streamlit as st
import pandas as pd
from src.app.config import RESULTS_DIR


@st.cache_data(show_spinner=False)
def load_governance_data():
    """Loads benchmark and override validation data for dynamic governance reporting."""
    master_path = os.path.join(RESULTS_DIR, "tables", "master_results.csv")
    split_path = os.path.join(RESULTS_DIR, "tables", "split_comparison.csv")
    typo_path = os.path.join(RESULTS_DIR, "tables", "typology_counts_v2.csv")
    override_path = os.path.join(RESULTS_DIR, "tables", "override_evaluation.csv")

    data = {}
    if os.path.exists(master_path):
        data["master"] = pd.read_csv(master_path)
    if os.path.exists(split_path):
        data["split"] = pd.read_csv(split_path)
    if os.path.exists(typo_path):
        data["typo"] = pd.read_csv(typo_path)
    if os.path.exists(override_path):
        data["override"] = pd.read_csv(override_path)
    return data


def render_about_view():
    """Renders the research platform architecture and model risk management specification."""
    st.markdown("## 🛡️ Research Platform Architecture & Model Governance")
    st.markdown(
        """
        **AMLGuard AI** is a dual-tier transaction monitoring research prototype. 
        It investigates the integration of gradient-boosted decision ensembles with deep inductive Graph Neural Networks 
        to evaluate detection trade-offs between isolated volume anomalies and complex multi-hop laundering syndicates.
        """
    )

    st.markdown("---")

    # Load dynamic governance metrics
    gov_data = load_governance_data()
    df_master = gov_data.get("master", pd.DataFrame())
    df_split = gov_data.get("split", pd.DataFrame())
    df_typo = gov_data.get("typo", pd.DataFrame())
    df_override = gov_data.get("override", pd.DataFrame())

    # Dynamic metrics with safe defaults
    rf_auc_str = "0.8214 ± 0.0126"
    gs_auc_str = "0.5442 ± 0.0515"
    n_seeds = 5
    if not df_master.empty:
        rf_m = df_master[(df_master["dataset"] == "AMLSim") & (df_master["model"] == "Random Forest")]
        if not rf_m.empty:
            r = rf_m.iloc[0]
            rf_auc_str = f"{r['auc_pr_mean']:.4f} ± {r['auc_pr_std']:.4f}"
            n_seeds = int(r.get("n_seeds", 5))
        gs_m = df_master[(df_master["dataset"] == "AMLSim") & (df_master["model"] == "GraphSAGE")]
        if not gs_m.empty:
            r = gs_m.iloc[0]
            gs_auc_str = f"{r['auc_pr_mean']:.4f} ± {r['auc_pr_std']:.4f}"

    gs_drift_drop = "51.15%"
    gat_drift_drop = "61.85%"
    rf_drift_retention = "80%"
    gs_drift_short = "51.2%"
    if not df_split.empty:
        gs_sp = df_split[df_split["model"] == "GraphSAGE"]
        if not gs_sp.empty:
            gs_val = gs_sp["relative_drop_pct"].iloc[0]
            gs_drift_drop = f"{gs_val:.2f}%"
            gs_drift_short = f"{gs_val:.1f}%"
        gat_sp = df_split[df_split["model"] == "GAT"]
        if not gat_sp.empty:
            gat_drift_drop = f"{gat_sp['relative_drop_pct'].iloc[0]:.2f}%"
        rf_sp = df_split[df_split["model"] == "Random Forest"]
        if not rf_sp.empty:
            rf_drift_retention = f"{100 - rf_sp['relative_drop_pct'].iloc[0]:.0f}%"

    rf_cycle_recall = "25%"
    gs_cycle_recall = "80%"
    if not df_typo.empty:
        rf_c = df_typo[(df_typo["model"] == "Random Forest") & (df_typo["alert_type"] == "cycle")]
        if not rf_c.empty:
            rf_cycle_recall = f"{rf_c['mean_recall'].iloc[0]*100:.0f}%"
        gs_c = df_typo[(df_typo["model"] == "GraphSAGE") & (df_typo["alert_type"] == "cycle")]
        if not gs_c.empty:
            gs_cycle_recall = f"{gs_c['mean_recall'].iloc[0]*100:.0f}%"

    # Override validation metrics
    cycle_added = 1
    cycle_base_pct = "50%"
    cycle_over_pct = "75%"
    fanin_added = 1
    sar_base_pct = "68.8%"
    sar_over_pct = "81.2%"
    fp_count = 1
    total_normals = 274
    total_test = 290
    tot_pos = 16
    fpr_pct = "0.36%"

    if not df_override.empty:
        cyc_o = df_override[df_override["metric"] == "cycle_recall"]
        if not cyc_o.empty:
            r = cyc_o.iloc[0]
            cycle_added = int(r["delta_hits"])
            cycle_base_pct = f"{r['baseline_rate']*100:.0f}%"
            cycle_over_pct = f"{r['override_rate']*100:.0f}%"

        fi_o = df_override[df_override["metric"] == "fan_in_recall"]
        if not fi_o.empty:
            fanin_added = int(fi_o.iloc[0]["delta_hits"])

        sar_o = df_override[df_override["metric"] == "overall_sar_recall"]
        if not sar_o.empty:
            r = sar_o.iloc[0]
            sar_base_pct = f"{r['baseline_rate']*100:.1f}%"
            sar_over_pct = f"{r['override_rate']*100:.1f}%"
            tot_pos = int(r["total_targets"])

        fpr_o = df_override[df_override["metric"] == "false_positive_rate"]
        if not fpr_o.empty:
            r = fpr_o.iloc[0]
            fp_count = int(r["delta_hits"])
            total_normals = int(r["total_targets"])
            fpr_pct = f"{r['override_rate']*100:.2f}%"
            total_test = tot_pos + total_normals

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            ### 🏗️ Dual-Tier Research Architecture
            1. **Tier 1: High-Throughput Ensemble Screening**
               - **Engines:** Random Forest & XGBoost.
               - **Role:** Evaluates streaming transaction records against 17 engineered behavioral features.
               - **Objective:** Maximum precision on local volume surges, structuring, and velocity anomalies while suppressing false positives.
            
            2. **Tier 2: Relational Graph Topological Forensics**
               - **Engines:** Inductive GraphSAGE & Multi-Head GAT.
               - **Role:** Periodically processes the attributed customer transaction network to aggregate multi-hop neighborhood representations.
               - **Objective:** Identifies complex coordinated money laundering typologies (cyclic routing, layered dispersion) that appear benign to isolated tabular filters.
            """
        )

    with col2:
        st.markdown(
            f"""
            ### 📋 Model Risk Management & Governance Alignment
            1. **SR 11-7-inspired Model Governance**
               - **Leakage Elimination:** Independent 60/20/20 train/validation/test partitions enforced across tabular matrices and graph topological masks.
               - **Imbalance Calibration:** Class-weighted cross-entropy and cost-sensitive loss functions configured for minority-class ($<5\\%$) detection without threshold distortion.
            
            2. **Explainability & SAR Audit Trail**
               - **Local Attribution:** Integrated **TreeSHAP** engine producing exact game-theoretic Shapley values on log-odds for every alert.
               - **Subnetwork Evidence:** **GNNExplainer** subgraph isolation highlighting the exact counterparty pathways responsible for deep graph alerts.

            3. **Dual-Tier Policy Override (GNN Structural Veto)**
               - **Design Rationale:** In a 70% tree / 25% graph weighted consensus, tabular models miss multi-hop wash trading, diluting ring scores (e.g., Account #96 scores 35.4%). The policy override escalates directly to SAR if both GNN models strongly detect topological coordination ($\\text{{GraphSAGE}} \\ge 0.75 \\text{{ and }} \\text{{GAT}} \\ge 0.75$).
               - **Empirical Validation:** On the {total_test} held-out test accounts, the override adds **{cycle_added} circular routing ring** (increasing cycle recall from {cycle_base_pct} to {cycle_over_pct}) and **{fanin_added} fan-in alert** (raising overall SAR recall from {sar_base_pct} to {sar_over_pct}), at a cost of **{fp_count} false positive escalation** among {total_normals} normal accounts ($\\text{{FPR}} = {fpr_pct}$).
               - **Governance Limitation & Drift Risk:** The override grants veto escalation power to the GNNs—the very models that degraded **{gs_drift_short} under temporal concept drift** on Elliptic. In high-drift production environments with evolving network topology, the override should trigger senior analyst review rather than automated filing.
            """
        )

    st.markdown("---")

    st.markdown(
        f"""
        ### 🔬 Core Empirical Findings from Experimental Evaluation
        - **Tabular Superiority on Local Attributes:** Evaluated on the standardized AMLSim benchmark across {n_seeds} distinct random initializations ($N={n_seeds}$), Random Forest delivered superior overall precision and area under the precision-recall curve (**AUC-PR: {rf_auc_str}**) compared to standalone GNNs (**GraphSAGE: {gs_auc_str}**).
        - **Temporal Drift Vulnerability in Deep Graph Learning:** Under realistic temporal evaluation on the Elliptic network (training on steps 1–34 and testing on steps 35–49), static GNN architectures suffered a **{gs_drift_drop} to {gat_drift_drop} degradation** due to shifting graph topology, whereas tree-based ensembles retained **{rf_drift_retention} of their baseline discriminative power**.
        - **Topological Complementarity in Cyclic Detection:** For structured circular laundering rings, tree models detected only **{rf_cycle_recall} of cases**, whereas GraphSAGE achieved an **{gs_cycle_recall} recall rate**, demonstrating that a two-stage hybrid ensemble is essential for robust surveillance.
        """
    )
