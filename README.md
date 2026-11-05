# AMLGuard-AI: Graph Neural Networks vs. Tabular Tree Ensembles for Anti-Money Laundering Detection

AMLGuard-AI is an experimental research framework comparing **Graph Neural Networks (GraphSAGE, Graph Attention Networks)** against **classical tree ensembles (Random Forest, XGBoost) and linear baselines** for detecting illicit financial flows and money laundering typologies.

Evaluations are conducted across 5 fixed random seeds (`[0, 1, 42, 123, 2024]`) under both synthetic banking simulation (AMLSim) and large-scale real-world cryptocurrency transaction networks (Elliptic Bitcoin dataset).

---

## 1. Problem Definition & Datasets

### Problem Formulation
Anti-Money Laundering (AML) detection is formulated as a semi-supervised / supervised node classification task on directed transaction graphs under extreme class imbalance. Models are evaluated using minority-class **Precision**, **Recall**, **F1-score**, and **Area Under the Precision-Recall Curve (AUC-PR)**.

### Datasets

#### 1. AMLSim (IBM Synthetic Multi-Agent Simulation)
- **Graph Topology**: 1,446 accounts (nodes) and 41,657 transactions (directed edges).
- **Features**: 17 node-level behavioral features engineered from transaction histories (in/out counts, mean/std amounts, counterparty diversity, flow velocity, account type). Label leakage prevention is strictly enforced.
- **Split**: Stratified 60% train (867 accounts, 48 positives), 20% validation (289 accounts, 16 positives), 20% test (290 accounts, 16 positives).
- **Laundering Typologies**: Known synthetic patterns including **Cycles** ($n=4$), **Fan-in** / Gathering ($n=6$), and **Fan-out** / Scattering ($n=6$).

#### 2. Elliptic Bitcoin Dataset
- **Graph Topology**: 203,769 Bitcoin transactions (nodes) and 234,355 payments (directed edges) spanning 49 distinct time steps (~2-week intervals).
- **Features**: 166 features per node (94 local transaction attributes + 72 aggregated 1-hop neighborhood features).
- **Evaluated Splits**:
  - **Random Split (Transductive)**: 60/20/20 random partition ($N=9,313$ test nodes, 909 illicit, 9.8% base rate).
  - **Time-Respecting Split (Inductive / Temporal)**: Time steps 1–34 for training, time steps 35–49 for testing ($N=16,670$ test nodes, 1,083 illicit, 6.5% base rate). Tests resilience to distribution drift.

---

## 2. Environment Setup

### Prerequisites
- **Python**: 3.10 – 3.13 (tested on Python 3.13)
- **MySQL**: 8.0+ (database `amlguard` running on `localhost:3306`)

### Installation

```bash
# Clone repository and navigate to root
cd AMLGuard-AI

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install core dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Database Configuration (AMLSim)
Ensure MySQL is running with database `amlguard` and user credentials configured in `src/seeded_baseline_eval.py`:
- User: `aml_user`
- Password: `yourpassword`
- Host: `localhost`
- Database: `amlguard`

The tables `account_features`, `transactions`, and `alert_accounts` are populated via `src/load_to_mysql.py` and `src/build_features.py`.

---

## 3. Reproduction Guide

To run scripts without threading collisions or library conflicts, export the standard environment variables:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/matplotlib
```

### One-Command Full Pipeline

```bash
# 1. Run 5-seed baseline evaluations on the aligned graph split
python -u src/seeded_baseline_eval.py --only AMLSim

# 2. Run GAT architecture & learning-rate hyperparameter tuning (skipped if already completed)
python -u src/gat_tuning.py

# 3. Run multi-seed alert typology evaluation (Cycle, Fan-in, Fan-out)
python -u src/typology_eval.py

# 4. Generate consolidated master tables and 200 DPI figures
python -u src/generate_outputs.py

# 5. Run paired statistical significance tests (t-test and Wilcoxon)
python -u src/significance_tests.py
```

---

## 4. Master Results Summary

All statistics represent **Mean ± Sample Standard Deviation (`ddof=1`) across 5 fixed random seeds**.

### Master Table (`results/tables/master_results.md`)

| Dataset | Model | Precision | Recall | F1 | AUC-PR | Seeds | Test $N$ | Positives |
|---|---|---|---|---|---|---|---|---|
| **AMLSim** | Logistic Regression | 0.1667 ± 0.0000 | 0.7500 ± 0.0000 | 0.2727 ± 0.0000 | 0.3623 ± 0.0000 | 5 | 290 | 16 |
| **AMLSim** | Random Forest | 0.9778 ± 0.0497 | 0.5375 ± 0.0342 | 0.6933 ± 0.0377 | **0.8214 ± 0.0126** | 5 | 290 | 16 |
| **AMLSim** | XGBoost | 0.9250 ± 0.1118 | 0.4500 ± 0.0685 | 0.6041 ± 0.0786 | 0.7186 ± 0.0347 | 5 | 290 | 16 |
| **AMLSim** | GraphSAGE | 0.2677 ± 0.0245 | 0.7125 ± 0.0713 | 0.3891 ± 0.0358 | 0.5442 ± 0.0515 | 5 | 290 | 16 |
| **AMLSim** | GAT (untuned) | 0.1576 ± 0.0269 | 0.5875 ± 0.0559 | 0.2463 ± 0.0298 | 0.2701 ± 0.0531 | 5 | 290 | 16 |
| **AMLSim** | GAT (tuned) | 0.1742 ± 0.0588 | 0.6000 ± 0.0559 | 0.2661 ± 0.0738 | 0.2774 ± 0.1085 | 5 | 290 | 16 |
| **Elliptic (random)** | Logistic Regression | 0.4430 ± 0.0000 | 0.9274 ± 0.0000 | 0.5996 ± 0.0000 | 0.7720 ± 0.0000 | 5 | 9,313 | 909 |
| **Elliptic (random)** | Random Forest | 0.9915 ± 0.0009 | 0.8946 ± 0.0009 | 0.9406 ± 0.0009 | 0.9805 ± 0.0003 | 5 | 9,313 | 909 |
| **Elliptic (random)** | XGBoost | 0.9829 ± 0.0043 | 0.9250 ± 0.0036 | 0.9531 ± 0.0032 | **0.9830 ± 0.0007** | 5 | 9,313 | 909 |
| **Elliptic (random)** | GraphSAGE | 0.7714 ± 0.0091 | 0.9039 ± 0.0048 | 0.8324 ± 0.0061 | 0.9361 ± 0.0017 | 5 | 9,313 | 909 |
| **Elliptic (random)** | GAT (untuned) | 0.3815 ± 0.0077 | 0.9512 ± 0.0036 | 0.5445 ± 0.0075 | 0.8616 ± 0.0058 | 5 | 9,313 | 909 |
| **Elliptic (random)** | GAT (tuned) | 0.3893 ± 0.0163 | 0.9525 ± 0.0031 | 0.5525 ± 0.0163 | 0.8679 ± 0.0054 | 5 | 9,313 | 909 |
| **Elliptic (temporal)** | Logistic Regression | 0.1342 ± 0.0000 | 0.9566 ± 0.0000 | 0.2354 ± 0.0000 | 0.1984 ± 0.0000 | 5 | 16,670 | 1,083 |
| **Elliptic (temporal)** | Random Forest | 0.9673 ± 0.0037 | 0.6419 ± 0.0221 | 0.7715 ± 0.0151 | **0.7831 ± 0.0008** | 5 | 16,670 | 1,083 |
| **Elliptic (temporal)** | XGBoost | 0.8338 ± 0.0123 | 0.7097 ± 0.0051 | 0.7667 ± 0.0041 | 0.7805 ± 0.0027 | 5 | 16,670 | 1,083 |
| **Elliptic (temporal)** | GraphSAGE | 0.3340 ± 0.0494 | 0.6842 ± 0.0242 | 0.4462 ± 0.0404 | 0.4573 ± 0.0486 | 5 | 16,670 | 1,083 |
| **Elliptic (temporal)** | GAT (untuned) | 0.1254 ± 0.0027 | 0.9435 ± 0.0052 | 0.2213 ± 0.0042 | 0.3287 ± 0.0108 | 5 | 16,670 | 1,083 |

---

### Elliptic Temporal Degradation (`results/tables/split_comparison.md`)

When evaluating models under temporal distribution shift (time-respecting split) versus random split:

| Model | Random Split AUC-PR | Time-Respecting Split AUC-PR | Absolute Drop | Relative Drop (%) |
|---|---|---|---|---|
| **Random Forest** | 0.9805 ± 0.0003 | 0.7831 ± 0.0008 | 0.1973 | **20.13%** |
| **XGBoost** | 0.9830 ± 0.0007 | 0.7805 ± 0.0027 | 0.2025 | **20.60%** |
| **GraphSAGE** | 0.9361 ± 0.0017 | 0.4573 ± 0.0486 | 0.4788 | **51.15%** |
| **GAT** | 0.8616 ± 0.0058 | 0.3287 ± 0.0108 | 0.5329 | **61.85%** |
| **Logistic Regression** | 0.7720 ± 0.0000 | 0.1984 ± 0.0000 | 0.5736 | **74.30%** |

*Tree ensembles exhibit substantially greater temporal robustness (dropping ~20%) compared to GNNs (dropping 51–62%) and linear models (dropping 74%).*

---

### Typology Detection Recall (`results/tables/typology_counts_v2.md`)

Multi-seed evaluation on AMLSim test positives ($N=16$):

| Model | Cycle ($n=4$) | Fan-in ($n=6$) | Fan-out ($n=6$) |
|---|---|---|---|
| **GraphSAGE** | **0.8000** (min 0.75, max 1.00) | **0.7000** (min 0.67, max 0.83) | **0.6667** (min 0.50, max 0.83) |
| **Random Forest** | 0.2500 (min 0.25, max 0.25) | 0.6667 (min 0.67, max 0.67) | 0.6000 (min 0.50, max 0.67) |
| **XGBoost** | 0.4000 (min 0.25, max 0.50) | 0.5333 (min 0.33, max 0.67) | 0.4000 (min 0.33, max 0.67) |

*GraphSAGE demonstrates higher recall on structural flow typologies (particularly cyclic routing at 80% recall vs. 25% for Random Forest), despite tree ensembles achieving higher global precision and AUC-PR.*

---

## 5. Interactive Demo / UI (Streamlit Workbench)

AMLGuard AI includes an industry-grade interactive workbench designed for compliance examiners and academic viva presentations:

```bash
# Launch the interactive Streamlit application
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

### Key Capabilities:
- **Account Investigation Console:** Select any account ID from AMLSim or Elliptic (filter by Confirmed SAR, Cycle, Fan-In, Fan-Out, or Normal).
- **Multi-Model Consensus:** Real-time side-by-side fraud probabilities across Random Forest, XGBoost, Logistic Regression, GraphSAGE, and GAT.
- **Interactive 1-Hop Ego Network:** Dynamic concentric transaction graph with color-coded risk nodes, directed arrows, amounts, and optional **GNNExplainer** topological attention overlays.
- **Local Explainability:** Exact **TreeSHAP** feature attribution waterfall/bar charts showing which behavioral metrics triggered the alert.
- **Research Benchmark Explorer:** Interactive comparisons of master results, temporal drift degradation, and typology recall.

---

## 6. Directory Structure

```text
AMLGuard-AI/
├── README.md                      # Primary project documentation
├── app.py                         # Streamlit interactive application entrypoint
├── requirements.txt               # Pinned core dependencies
├── AMLSim/                        # Java multi-agent transaction simulation engine
├── data/                          # Raw simulation CSVs & offline parquet cache
├── elliptic_data/                 # Raw Elliptic benchmark CSVs (classes, edgelist, features)
├── models/                        # Serialized graph objects, model checkpoints, raw CSVs
│   └── README.md                  # Detailed models directory index
├── notebooks/                     # Exploratory Jupyter notebooks
│   └── exploratory_experiments.ipynb
├── results/                       # Final consolidated tables and figures
│   ├── README.md                  # Results directory index
│   ├── figures/                   # 200 DPI publication charts
│   └── tables/                    # CSV and Markdown result tables
└── src/                           # Execution scripts & UI modules
    ├── app/                       # Modular Streamlit application architecture
    │   ├── theme.py               # FinTech design system CSS, cards, and badges
    │   ├── config.py              # Configuration & constants
    │   ├── data_manager.py        # Cached data loader (MySQL + Parquet fallback)
    │   ├── model_service.py       # Multi-model inference engine (RF, XGB, GraphSAGE, GAT)
    │   ├── graph_engine.py        # 1-hop interactive Plotly ego-network visualizer
    │   ├── explainer_service.py   # TreeSHAP & GNNExplainer attribution
    │   ├── components.py          # UI cards, KPI bars, and transaction ledger
    │   └── views/                 # Navigation pages (investigation, benchmarks, about)
    ├── build_features.py          # Account-level feature engineering
    ├── build_graph.py             # AMLSim PyG graph construction
    ├── build_elliptic_graph.py    # Elliptic PyG graph construction (random & time split)
    ├── load_to_mysql.py           # MySQL data ingestion pipeline
    ├── seeded_baseline_eval.py    # 5-seed baseline training & evaluation
    ├── gat_tuning.py              # GAT grid search & final seed evaluation
    ├── typology_eval.py           # Multi-seed alert typology detection evaluation
    ├── generate_outputs.py        # Master table & publication figure generation
    └── significance_tests.py      # Paired t-test and Wilcoxon statistical testing
```

