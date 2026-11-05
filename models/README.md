# Models & Intermediate Artifacts Index

This directory contains trained model checkpoints, serialized graph datasets, raw evaluation results across random seeds, and execution logs.

---

## 1. Graph Datasets (`.pt`)

| File | Description | Nodes | Features | Splits |
|---|---|---|---|---|
| `aml_graph.pt` | PyTorch Geometric `Data` object for AMLSim synthetic accounts | 1,446 | 17 | Stratified 60/20/20 (`train_mask`, `val_mask`, `test_mask`) |
| `elliptic_graph.pt` | PyTorch Geometric `Data` object for Elliptic Bitcoin transactions | 203,769 | 166 | Random 60/20/20 (`train_mask`, `val_mask`, `test_mask`) |
| `elliptic_graph_timesplit.pt` | PyTorch Geometric `Data` object for Elliptic with temporal split | 203,769 | 166 | Temporal split (`train_mask`, `val_mask`, `test_mask`) |

---

## 2. Model Checkpoints (`.pt` & `.pkl`)

### Graph Neural Networks
- `graphsage_1layer.pt`: 1-layer GraphSAGE state dict on AMLSim (`hidden=64, dropout=0.3`).
- `graphsage_2layer.pt`: 2-layer GraphSAGE state dict on AMLSim.
- `graphsage_elliptic.pt`: GraphSAGE trained on Elliptic (random split).
- `gat.pt`: Baseline untuned GAT state dict on AMLSim (`heads=4, hidden=32, lr=0.005`).
- `gat_elliptic.pt`: Baseline GAT on Elliptic (random split).
- `gat_tuned_best_amlsim.pt`: Best tuned GAT on AMLSim (`heads=2, hidden=64, lr=0.01`, selected by validation F1).
- `gat_tuned_best_elliptic.pt`: Best tuned GAT on Elliptic (`heads=4, hidden=32, lr=0.01`, selected by validation F1).

### Tabular Baselines & Scalers
- `random_forest.pkl`, `random_forest_elliptic.pkl`, `random_forest_elliptic_timesplit.pkl`: Random Forest classifiers.
- `xgboost.pkl`, `xgboost_elliptic.pkl`, `xgboost_elliptic_timesplit.pkl`: XGBoost classifiers.
- `logistic_regression.pkl`, `logistic_regression_elliptic.pkl`: Logistic Regression classifiers.
- `isolation_forest.pkl`: Unsupervised Isolation Forest on normal AMLSim accounts.
- `scaler.pkl`, `scaler_elliptic.pkl`: Fitted `StandardScaler` instances.

---

## 3. Seeded Results (Source of Truth) & Tuning Logs

- `seeded_baseline_results_raw.csv`: 45 rows (5 seeds $\times$ 3 models $\times$ 3 dataset splits). AMLSim rows evaluated on the exact graph test mask.
- `seeded_baseline_results_raw.BACKUP_before_amlsim_splitfix.csv`: Archival copy of the baseline raw CSV before the 80/20 $\rightarrow$ graph mask split alignment.
- `seeded_gnn_results_raw.csv`: 30 rows (5 seeds $\times$ 2 GNN models $\times$ 3 dataset splits).
- `gat_tuned_final_seeds.csv`: 10 rows (5 seeds $\times$ tuned GAT on AMLSim and Elliptic random split).
- `gat_grid_search_results.csv`: 35 rows (27 AMLSim + 8 Elliptic) capturing validation F1 and AUC-PR across architectural grids.
- `gat_tuning.log`: Full stdout/stderr log of the GAT hyperparameter tuning job.
- `build_graph.log`: Execution log for graph construction.

---

## 4. Explainability & Legacy Artifacts

- `shap_feature_importance.csv`: Mean absolute SHAP values for Random Forest features on AMLSim.
- `graphsage_typology_results.csv`, `rf_typology_results.csv`: Earlier single-run typology predictions (superseded by multi-seed evaluation in `results/tables/typology_counts_v2.csv`).
- `baseline_results*.csv`: Legacy single-seed runs (superseded by 5-seed evaluations).

