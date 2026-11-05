# AMLGuard AI Application Architecture (`src/app/`)

This package powers the interactive Streamlit transaction surveillance workbench. It is engineered with a modular separation of concerns between data ingestion, inference serving, graph visualization, feature explainability, and presentation views.

---

## 1. Module Overview

| Module | Purpose |
|---|---|
| `theme.py` | Professional dark-mode design system (`#0b0f17` slate background), card containers, typography rules (`Inter`, `JetBrains Mono`), and SVG status badges. |
| `config.py` | Central configuration defining file paths, model feature definitions, risk threshold bands (Low $<0.35$, Medium $<0.70$, High $\ge 0.70$), and heuristic consensus weights. |
| `data_manager.py` | High-performance cached data layer (`@st.cache_data`). Connects to local MySQL when available, automatically falling back to optimized Apache Parquet tables (`data/cache/`) on cloud deployments. |
| `model_service.py` | Core inference engine running simultaneous forward passes across 5 models (RF, XGBoost, LogReg, GraphSAGE, GAT). Implements the **Dual-Tier Structural Policy Override** and automated volume vs. structure divergence explanations. |
| `graph_engine.py` | Interactive 1-hop Plotly ego-network generator displaying concentric transaction flows, directed links, counterparty risk annotations, and GNN attention weights. |
| `explainer_service.py` | Regulatory explainability engine producing exact TreeSHAP feature attribution waterfall charts and GNN edge attribution masks. |
| `components.py` | Modular UI renderers: Entity Dossier card, Top KPI metrics bar, Model Score Matrix cards, and Transaction Audit Ledgers. Enforces strict zero-raw-HTML escaping via `textwrap.dedent`. |

---

## 2. Views Subpackage (`src/app/views/`)

The application is structured into three dedicated multi-page views:

- **`investigation.py` (Case Investigation Console):**
  - Account selector filtering by risk category (Confirmed SAR, Cycle, Fan-in, Fan-out, Normal).
  - Multi-model risk score matrix with dynamic model reconciliation banners.
  - Interactive concentric transaction network with hover tooltips and edge flow directions.
  - TreeSHAP local feature attribution and immutable audit ledger.

- **`benchmarks.py` (Model Performance Analytics):**
  - Master Benchmark Matrix covering all 17 evaluated model/split combinations.
  - Temporal Concept Drift Analysis comparing random vs. time-respecting partitions.
  - Typology Detection Recall broken down across $n=4$ test cycles, $n=6$ fan-in, and $n=6$ fan-out patterns.
  - Formal Statistical Hypothesis Testing ($N=5$ seeds) displaying Welch's $t$-test and Wilcoxon signed-rank results ($p < 0.05$).

- **`about.py` (Architecture & Governance):**
  - End-to-end data pipeline flowchart and model topology diagrams.
  - SR 11-7 Regulatory Model Risk Governance documentation.
  - Heuristic consensus weighting framework and dual-tier structural override logic.

