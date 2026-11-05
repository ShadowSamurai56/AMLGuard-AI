# Results Index

This directory contains the final consolidated experimental tables and publication-ready figures.

All standard deviations reported follow the sample standard deviation convention (`ddof=1`, pandas default) across 5 fixed random seeds (`[0, 1, 42, 123, 2024]`).

---

## 1. Tables (`results/tables/`)

| File | Markdown Version | Description |
|---|---|---|
| `master_results.csv` | `master_results.md` | Master results table covering all 17 evaluated `(dataset, model)` pairs. Reports Precision, Recall, F1, and AUC-PR (mean ± sample std), seed counts ($n=5$), test set sizes ($N$), and positive counts. |
| `split_comparison.csv` | `split_comparison.md` | Comparison of Elliptic models evaluated on both Random Split ($N=9,313$) and Time-Respecting Split ($N=16,670$). Reports AUC-PR mean ± std, absolute drop, and relative drop (%). |
| `significance_test_results.csv` | — | Statistical significance tests for 13 paired comparisons using Paired $t$-test and Wilcoxon signed-rank test ($\alpha = 0.05$). |
| `typology_counts_v2.csv` | `typology_counts_v2.md` | Multi-seed alert typology detection breakdown (cycle, fan_in, fan_out) on AMLSim graph test mask ($N=16$ positives) for Random Forest, XGBoost, and GraphSAGE. |
| `typology_counts_superseded.csv` | `typology_counts_superseded.md` | Archived single-run typology results (kept for traceability; superseded by v2). |

---

## 2. Figures (`results/figures/`, PNG at 200 DPI)

| Figure | Description |
|---|---|
| `auc_pr_by_model.png` | 3-panel grouped bar chart showing AUC-PR (mean ± 1 sample std error bars) across AMLSim, Elliptic (random split), and Elliptic (time-respecting split). Consistent model colors maintained throughout. |
| `elliptic_split_comparison.png` | Paired bar chart displaying the drop in AUC-PR from random split to temporal split across models on the Elliptic benchmark. |
| `gat_grid_search_f1_heatmap.png` | Combined 2-panel heatmap of validation F1 across GAT architecture rows `(heads, hidden)` and learning rate columns `lr`, with the optimal configuration cell highlighted. |
| `gat_grid_search_f1_heatmap_amlsim.png` | Standalone GAT validation F1 heatmap for AMLSim (27 grid cells). |
| `gat_grid_search_f1_heatmap_elliptic.png` | Standalone GAT validation F1 heatmap for Elliptic random split (8 grid cells). |
| `shap_top10_features.png` | Horizontal bar chart of the top 10 features ranked by mean absolute SHAP value from Random Forest on AMLSim accounts. |

