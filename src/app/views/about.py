"""
AMLGuard AI — Platform Architecture & Model Governance
Institutional Documentation for Compliance, Audit, and Technical Validation
"""

import streamlit as st


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
            """
            ### 📋 Model Risk Management & Governance Alignment
            1. **SR 11-7-inspired Model Governance**
               - **Leakage Elimination:** Independent 60/20/20 train/validation/test partitions enforced across tabular matrices and graph topological masks.
               - **Imbalance Calibration:** Class-weighted cross-entropy and cost-sensitive loss functions configured for minority-class ($<5\%$) detection without threshold distortion.
            
            2. **Explainability & SAR Audit Trail**
               - **Local Attribution:** Integrated **TreeSHAP** engine producing exact game-theoretic Shapley values on log-odds for every alert.
               - **Subnetwork Evidence:** **GNNExplainer** subgraph isolation highlighting the exact counterparty pathways responsible for deep graph alerts.

            3. **Dual-Tier Policy Override (GNN Structural Veto)**
               - **Design Rationale:** In a 70% tree / 25% graph weighted consensus, tabular models miss multi-hop wash trading, diluting ring scores (e.g., Account #96 scores 35.4%). The policy override escalates directly to SAR if both GNN models strongly detect topological coordination ($\text{GraphSAGE} \ge 0.75 \text{ and } \text{GAT} \ge 0.75$).
               - **Empirical Validation:** On the 290 held-out test accounts, the override adds **1 circular routing ring** (increasing cycle recall from 50% to 75%) and **1 fan-in alert** (raising overall SAR recall from 68.8% to 81.2%), at a cost of **1 false positive escalation** among 274 normal accounts ($\text{FPR} = 0.36\%$).
               - **Governance Limitation & Drift Risk:** The override grants veto escalation power to the GNNs—the very models that degraded **51.2% under temporal concept drift** on Elliptic. In high-drift production environments with evolving network topology, the override should trigger senior analyst review rather than automated filing.
            """
        )

    st.markdown("---")

    st.markdown(
        """
        ### 🔬 Core Empirical Findings from Experimental Evaluation
        - **Tabular Superiority on Local Attributes:** Evaluated on the standardized AMLSim benchmark across 5 distinct random initializations ($N=5$), Random Forest delivered superior overall precision and area under the precision-recall curve (**AUC-PR: 0.8214 ± 0.0126**) compared to standalone GNNs (**GraphSAGE: 0.5442 ± 0.0515**).
        - **Temporal Drift Vulnerability in Deep Graph Learning:** Under realistic temporal evaluation on the Elliptic network (training on steps 1–34 and testing on steps 35–49), static GNN architectures suffered a **51.15% to 61.85% degradation** due to shifting graph topology, whereas tree-based ensembles retained **80% of their baseline discriminative power**.
        - **Topological Complementarity in Cyclic Detection:** For structured circular laundering rings, tree models detected only **25% of cases**, whereas GraphSAGE achieved an **80% recall rate**, demonstrating that a two-stage hybrid ensemble is essential for robust surveillance.
        """
    )
