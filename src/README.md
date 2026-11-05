# Source Code Pipeline (`src/`)

This directory contains the Python execution scripts, training harnesses, statistical evaluation tools, and Streamlit workbench modules powering AMLGuard-AI.

---

## 1. Script Architecture & Responsibilities

| Script | Responsibility | Primary Inputs & Outputs |
|---|---|---|
| `load_to_mysql.py` | Ingests AMLSim synthetic simulation CSVs into a local MySQL instance (`amlguard` database). | `data/*.csv` $\rightarrow$ MySQL `accounts`, `transactions`, `alert_accounts` |
| `build_features.py` | Extracts 17 node-level behavioral attributes without label leakage (velocity, turnover ratios, counterparty entropy). | MySQL $\rightarrow$ MySQL `account_features` / `data/cache/` |
| `build_graph.py` | Constructs stratified PyTorch Geometric `Data` graph objects with aligned train/val/test masks (60/20/20). | MySQL $\rightarrow$ `models/aml_graph.pt` |
| `build_elliptic_graph.py` | Parses raw Elliptic Bitcoin CSVs into random-split and temporal time-respecting PyG graphs. | `elliptic_data/*.csv` $\rightarrow$ `models/elliptic_graph*.pt` |
| `seeded_baseline_eval.py` | Runs 5-seed baseline training and evaluation across Logistic Regression, Random Forest, and XGBoost. | PyG graphs $\rightarrow$ `models/seeded_baseline_results_raw.csv` |
| `gat_tuning.py` | Grid-search hyperparameter tuner for Graph Attention Networks (GAT heads, hidden dimensions, learning rate). | PyG graphs $\rightarrow$ `models/gat_tuned_*.pt`, `results/tables/gat_grid_search_results.csv` |
| `typology_eval.py` | Evaluates multi-seed detection recall across specific money laundering typologies (Cycle, Fan-in, Fan-out). | Checkpoints $\rightarrow$ `results/tables/typology_counts_v2.csv` |
| `significance_tests.py` | Computes paired two-sample Welch's $t$-tests and Wilcoxon signed-rank tests across model seed pairs ($\alpha = 0.05$). | Raw CSVs $\rightarrow$ `results/tables/significance_test_results.csv` |
| `generate_outputs.py` | Compiles master benchmark tables, temporal degradation matrices, and publication-ready 200 DPI PNG figures. | Raw CSVs $\rightarrow$ `results/tables/`, `results/figures/` |
| `generate_pdf_report.py` | Generates a 14-page regulatory and academic project report in PDF and HTML format fulfilling SR 11-7 model risk governance. | Code & results $\rightarrow$ `docs/PROJECT_REPORT.pdf` |
| `generate_resume_pdf.py` | Compiles a single-page ATS-compliant resume PDF incorporating audited metrics and live project links. | HTML template $\rightarrow$ `Raghav_Krishan_Aggarwal_Resume.pdf` |

---

## 2. Interactive Application Subpackage (`src/app/`)

The modular Streamlit workbench is organized inside `src/app/`:
- **`theme.py`**: Design system tokens, dark-mode CSS injection, and status badges.
- **`config.py`**: Model weights, thresholds, file paths, and runtime constants.
- **`data_manager.py`**: Caching data layer with dual MySQL and Parquet fallback modes.
- **`model_service.py`**: Multi-model inference engine featuring tabular scoring, GNN forward passes, heuristic consensus weights, and the dual-tier structural policy override.
- **`graph_engine.py`**: Plotly concentric 1-hop transaction network visualizer with GNNExplainer attention overlays.
- **`explainer_service.py`**: TreeSHAP and GNN feature attribution explainers.
- **`components.py`**: Reusable UI cards, KPI header bars, and transaction audit ledgers.
- **`views/`**: Page views for Case Investigation, Model Performance Analytics, and Model Architecture & Governance.
