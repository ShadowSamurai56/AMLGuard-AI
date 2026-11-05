# Elliptic Split Comparison: Random Split vs Time-Respecting Split (AUC-PR)

| Model | Elliptic (random split) AUC-PR | Elliptic (time-respecting split) AUC-PR | Absolute Drop | Relative Drop (%) |
| --- | --- | --- | --- | --- |
| Logistic Regression | 0.7720 ± 0.0000 | 0.1984 ± 0.0000 | 0.5736 | 74.30% |
| Random Forest | 0.9805 ± 0.0003 | 0.7831 ± 0.0008 | 0.1973 | 20.13% |
| XGBoost | 0.9830 ± 0.0007 | 0.7805 ± 0.0027 | 0.2025 | 20.60% |
| GraphSAGE | 0.9361 ± 0.0017 | 0.4573 ± 0.0486 | 0.4788 | 51.15% |
| GAT | 0.8616 ± 0.0058 | 0.3287 ± 0.0108 | 0.5329 | 61.85% |
