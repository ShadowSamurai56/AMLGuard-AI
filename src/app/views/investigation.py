"""
AMLGuard AI — Account Investigation View
The primary interactive workbench for exploring accounts, multi-model predictions,
ego networks, and local explainability.
"""

import streamlit as st
import pandas as pd
import textwrap

from src.app.data_manager import (
    load_amlsim_features,
    load_amlsim_transactions,
    load_amlsim_alerts,
    load_elliptic_sample,
    get_accounts_catalog
)
from src.app.model_service import (
    predict_amlsim_account,
    predict_elliptic_transaction
)
from src.app.graph_engine import (
    build_amlsim_ego_network,
    build_elliptic_ego_network,
    render_interactive_graph
)
from src.app.explainer_service import (
    compute_local_shap,
    compute_gnn_explanation
)
from src.app.components import (
    render_kpi_bar,
    render_model_comparison_cards,
    render_transaction_ledger
)
from src.app.config import FEATURE_LABELS_AMLSIM


def render_investigation_view():
    """Renders the comprehensive investigation console."""
    # Top Filter controls
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1.5, 2.0, 3.5])

    with col_ctrl1:
        dataset = st.selectbox(
            "Dataset Domain",
            ["AMLSim (Synthetic Banking)", "Elliptic (Bitcoin Network)"],
            index=0
        )

    ds_code = "AMLSim" if "AMLSim" in dataset else "Elliptic"

    with col_ctrl2:
        if ds_code == "AMLSim":
            filter_mode = st.selectbox(
                "Filter Accounts By",
                ["All", "Confirmed SAR Only", "Cycle Typology", "Fan-In Typology", "Fan-Out Typology", "Normal Only"],
                index=1
            )
        else:
            filter_mode = st.selectbox(
                "Filter Transactions By",
                ["All", "Confirmed SAR Only", "Normal Only"],
                index=1
            )

    # Get account options
    accounts = get_accounts_catalog(dataset=ds_code, filter_mode=filter_mode)

    if not accounts:
        st.warning(f"No accounts found for dataset: {ds_code} with filter: {filter_mode}")
        return

    # Account selector dropdown
    with col_ctrl3:
        acct_labels = [a["label"] for a in accounts]
        
        # Default to a flagged account in a circular routing ring (Account #96 where GraphSAGE catches circular flow)
        default_idx = 0
        target_default_id = 96 if ds_code == "AMLSim" else None
        if target_default_id is not None:
            for idx, a in enumerate(accounts):
                if a["id"] == target_default_id:
                    default_idx = idx
                    break

        selected_label = st.selectbox(
            "Select Account to Investigate",
            acct_labels,
            index=default_idx
        )
        selected_item = next(a for a in accounts if a["label"] == selected_label)
        selected_id = selected_item["id"]

    st.markdown("---")

    # ---------------------------------------------------------
    # AMLSim Workflow
    # ---------------------------------------------------------
    if ds_code == "AMLSim":
        features_df = load_amlsim_features()
        transactions_df = load_amlsim_transactions()
        alerts_df = load_amlsim_alerts()

        # Look up alert metadata first
        alert_row = alerts_df[alerts_df["acct_id"] == selected_id]
        typology = alert_row["alert_type"].iloc[0] if not alert_row.empty else "normal"
        ground_truth = selected_item["is_sar"]

        # Run inference with typology-aware divergence modeling
        prediction = predict_amlsim_account(selected_id, features_df, typology=typology)
        if prediction is None:
            st.error(f"Account #{selected_id} not found in feature matrix.")
            return

        feat_dict = prediction["features"]
        total_vol = feat_dict.get("out_total_amt", 0.0) + feat_dict.get("in_total_amt", 0.0)
        tx_cnt = int(feat_dict.get("total_tx_count", 0))

        # 1. Executive Summary KPI Bar
        render_kpi_bar(
            acct_id=selected_id,
            ground_truth=ground_truth,
            typology=typology,
            consensus_prob=prediction["consensus_prob"],
            risk_level=prediction["risk_level"],
            disposition=prediction["disposition"],
            total_volume=total_vol,
            tx_count=tx_cnt,
            divergence_note=prediction.get("divergence_note"),
            split=selected_item.get("split", "TEST")
        )

        # 2. Multi-Model Prediction Cards
        st.markdown("### Multi-Model Consensus & Inference Matrix")
        render_model_comparison_cards(prediction["scores"], ground_truth)

        st.markdown("<br>", unsafe_allow_html=True)

        # 3. Two-Column Layout: Ego Graph & Explainability
        col_graph, col_shap = st.columns([1.1, 0.9])

        with col_graph:
            st.markdown("### Transaction Flow & Counterparty Network")

            col_g1, col_g2 = st.columns([1, 1])
            with col_g1:
                enable_gnn_explainer = st.checkbox(
                    "⚡ GNNExplainer Weights",
                    value=False,
                    help="Calculates topological message-passing importance weights for edges."
                )
            with col_g2:
                include_inter = st.checkbox(
                    "Show Peer-to-Peer Cross Edges",
                    value=False,
                    help="Displays transactions occurring between counterparties (useful for detecting cycles)."
                )

            edge_importance = None
            if enable_gnn_explainer:
                with st.spinner("Calculating GNNExplainer gradients..."):
                    edge_importance = compute_gnn_explanation(prediction["node_idx"])

            G = build_amlsim_ego_network(
                selected_id,
                transactions_df,
                features_df,
                alerts_df,
                max_counterparties=12,
                include_inter_edges=include_inter
            )
            fig_graph = render_interactive_graph(
                G,
                target_id=selected_id,
                dataset="AMLSim",
                gnn_edge_importance=edge_importance
            )
            st.plotly_chart(fig_graph, use_container_width=True)

            # Clean legend
            legend_html = textwrap.dedent("""
            <div style="font-size: 0.78rem; color: #94a3b8; display: flex; gap: 1.2rem; justify-content: center; margin-top: -8px;">
                <span>🔷 <b>Investigated Account</b></span>
                <span><span style="color: #f43f5e;">●</span> <b>SAR Counterparty</b></span>
                <span><span style="color: #64748b;">●</span> <b>Normal Peer</b></span>
            </div>
            """).strip()
            st.markdown(legend_html, unsafe_allow_html=True)

        with col_shap:
            st.markdown("### Model Explainability & Feature Attribution")

            tab_shap, tab_features = st.tabs(["TreeSHAP Feature Drivers", "Raw Behavioral Metrics"])

            with tab_shap:
                fig_shap = compute_local_shap(feat_dict, top_k=7)
                st.plotly_chart(fig_shap, use_container_width=True)

            with tab_features:
                # Tabular feature inspection
                feat_table_data = []
                for k in feat_dict.keys():
                    lbl = FEATURE_LABELS_AMLSIM.get(k, k)
                    val = feat_dict[k]
                    val_str = f"${val:,.2f}" if "amt" in k else (f"{val:,.0f}" if isinstance(val, (int, float)) and val.is_integer() else f"{val:.4f}")
                    feat_table_data.append({"Feature Metric": lbl, "Account Value": val_str})

                df_ft = pd.DataFrame(feat_table_data)
                st.dataframe(df_ft, use_container_width=True, height=330, hide_index=True)

        # 4. Atomic Transaction Ledger
        st.markdown("### Account Transaction History (Audit Ledger)")
        render_transaction_ledger(transactions_df, selected_id, limit=8)

    # ---------------------------------------------------------
    # Elliptic Workflow
    # ---------------------------------------------------------
    else:
        sample_feat, sample_edges = load_elliptic_sample()
        if sample_feat is None or sample_edges is None:
            st.error("Elliptic sample cache not found.")
            return

        prediction = predict_elliptic_transaction(selected_id, sample_feat)
        if prediction is None:
            st.error(f"Transaction #{selected_id} not found in sample.")
            return

        ground_truth = selected_item["is_sar"]

        # 1. Executive Summary
        render_kpi_bar(
            acct_id=selected_id,
            ground_truth=ground_truth,
            typology="bitcoin",
            consensus_prob=prediction["consensus_prob"],
            risk_level=prediction["risk_level"],
            disposition=prediction["disposition"],
            total_volume=1.0,
            tx_count=1,
            split=selected_item.get("split", "TEST")
        )

        # 2. Model Cards
        st.markdown("### Multi-Model Consensus on Entity Transaction")
        render_model_comparison_cards(prediction["scores"], ground_truth)

        st.markdown("<br>", unsafe_allow_html=True)

        # 3. Ego Graph
        col_graph_el, col_info_el = st.columns([1.2, 0.8])
        with col_graph_el:
            st.markdown("### Bitcoin 1-Hop Transaction Tree")
            G_el = build_elliptic_ego_network(selected_id, sample_edges, sample_feat)
            fig_el = render_interactive_graph(G_el, target_id=selected_id, dataset="Elliptic")
            st.plotly_chart(fig_el, use_container_width=True)

        with col_info_el:
            st.markdown("### Elliptic Forensics Details")
            st.info(
                f"""
                **Transaction ID:** `{selected_id}`  
                **Ground Truth Class:** `{'1 (Illicit)' if ground_truth == 1 else '2 (Licit)'}`  
                **Network Context:** Bitcoin graph nodes represent individual transactions, and edges represent BTC payment flows.
                
                Tree models (Random Forest and XGBoost) exhibit high accuracy on local transaction features, but drop significantly under temporal split drift.
                """
            )

