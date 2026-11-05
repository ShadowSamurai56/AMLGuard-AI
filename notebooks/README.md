# Notebooks Directory (`notebooks/`)

This directory contains Jupyter notebooks used during the initial research, exploratory data analysis, and prototyping phases of the project.

---

## Contents

- **`exploratory_experiments.ipynb`**:
  - Initial exploratory data analysis (EDA) on the raw AMLSim banking data and Elliptic Bitcoin dataset.
  - Prototyping account-level behavioral features (turnover ratios, in/out count balances, velocity).
  - Exploratory graph topology analysis using NetworkX (degree distributions, connected components, cycle discovery).
  - Initial single-run model prototyping (Scikit-Learn Random Forest, XGBoost, PyTorch Geometric GraphSAGE) before migration to the multi-seed modular pipeline in `src/`.

---

## Reproduction Note

For audited, reproducible results across the 5 fixed random seeds (`[0, 1, 42, 123, 2024]`), please execute the modular scripts located in `src/` as documented in the primary [README.md](../README.md).
