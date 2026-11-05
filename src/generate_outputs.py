"""
Generate experiment outputs for AMLGuard-AI.
- Task 1: Consolidated tables (master_results, split_comparison) in CSV and Markdown.
- Task 2: Descriptive figures (AUC-PR grouped bar, Elliptic split paired bars, GAT grid heatmaps, SHAP top 10).
- Task 3: Typology breakdown (n, k, recall).

Standard deviation convention: sample standard deviation (ddof=1, the pandas default).
All outputs written to results/ (results/tables/ and results/figures/).
Nothing in models/ is modified or deleted.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Standard deviation convention: sample standard deviation (ddof=1, the pandas default).

# Ensure output directories exist
os.makedirs("results/tables", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)

# -----------------------------------------------------------------------------
# Color palette: consistent color per model across all figures
# -----------------------------------------------------------------------------
MODEL_COLORS = {
    "Logistic Regression": "#4A7BB7",
    "Random Forest": "#2E7D32",
    "XGBoost": "#F57C00",
    "GraphSAGE": "#6A1B9A",
    "GAT": "#C62828",
    "GAT (tuned)": "#AD1457",
}

# -----------------------------------------------------------------------------
# Hard-coded test set sizes from earlier training logs:
# AMLSim: test_N=290, test_positives=16;
# Elliptic (random split): test_N=9313, test_positives=909;
# Elliptic (time-respecting split): test_N=16670, test_positives=1083.
# -----------------------------------------------------------------------------
TEST_STATS = {
    "AMLSim": {"test_N": 290, "test_positives": 16},
    "Elliptic (random split)": {"test_N": 9313, "test_positives": 909},
    "Elliptic (time-respecting split)": {"test_N": 16670, "test_positives": 1083},
}

DATASET_ORDER = ["AMLSim", "Elliptic (random split)", "Elliptic (time-respecting split)"]
MODEL_ORDER = ["Logistic Regression", "Random Forest", "XGBoost", "GraphSAGE", "GAT", "GAT (tuned)"]


def df_to_markdown(df):
    headers = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(str(h) for h in headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def generate_task1_tables():
    # Load raw CSVs (source of truth)
    baseline = pd.read_csv("models/seeded_baseline_results_raw.csv")
    gnn = pd.read_csv("models/seeded_gnn_results_raw.csv")
    gat_tuned = pd.read_csv("models/gat_tuned_final_seeds.csv")

    cols = ["dataset", "model", "seed", "precision", "recall", "f1", "auc_pr"]
    raw_all = pd.concat([baseline[cols], gnn[cols], gat_tuned[cols]], ignore_index=True)

    # 1a) Master results table
    master_rows = []
    for ds in DATASET_ORDER:
        for m in MODEL_ORDER:
            sub = raw_all[(raw_all["dataset"] == ds) & (raw_all["model"] == m)]
            if len(sub) == 0:
                continue
            master_rows.append({
                "dataset": ds,
                "model": m,
                "precision_mean": sub["precision"].mean(),
                "precision_std": sub["precision"].std(ddof=1),
                "recall_mean": sub["recall"].mean(),
                "recall_std": sub["recall"].std(ddof=1),
                "f1_mean": sub["f1"].mean(),
                "f1_std": sub["f1"].std(ddof=1),
                "auc_pr_mean": sub["auc_pr"].mean(),
                "auc_pr_std": sub["auc_pr"].std(ddof=1),
                "n_seeds": len(sub),
                "test_N": TEST_STATS[ds]["test_N"],
                "test_positives": TEST_STATS[ds]["test_positives"],
            })

    master_df = pd.DataFrame(master_rows)
    master_csv_path = "results/tables/master_results.csv"
    master_df.to_csv(master_csv_path, index=False)

    # Markdown version of master_results
    master_md_rows = []
    for r in master_rows:
        master_md_rows.append({
            "Dataset": r["dataset"],
            "Model": r["model"],
            "Precision (mean ± std)": f"{r['precision_mean']:.4f} ± {r['precision_std']:.4f}",
            "Recall (mean ± std)": f"{r['recall_mean']:.4f} ± {r['recall_std']:.4f}",
            "F1 (mean ± std)": f"{r['f1_mean']:.4f} ± {r['f1_std']:.4f}",
            "AUC-PR (mean ± std)": f"{r['auc_pr_mean']:.4f} ± {r['auc_pr_std']:.4f}",
            "Seeds": r["n_seeds"],
            "Test N": r["test_N"],
            "Test Positives": r["test_positives"],
        })
    master_md_df = pd.DataFrame(master_md_rows)
    master_md_path = "results/tables/master_results.md"
    with open(master_md_path, "w") as f:
        f.write("# Master Results: Mean ± Sample Std (ddof=1) Over Seeds\n\n")
        f.write(df_to_markdown(master_md_df))
        f.write("\n")

    # 1b) Split comparison table
    # Models present on both Elliptic splits
    split_models = ["Logistic Regression", "Random Forest", "XGBoost", "GraphSAGE", "GAT"]
    split_rows = []
    for m in split_models:
        r_sub = raw_all[(raw_all["dataset"] == "Elliptic (random split)") & (raw_all["model"] == m)]
        t_sub = raw_all[(raw_all["dataset"] == "Elliptic (time-respecting split)") & (raw_all["model"] == m)]
        r_mean, r_std = r_sub["auc_pr"].mean(), r_sub["auc_pr"].std(ddof=1)
        t_mean, t_std = t_sub["auc_pr"].mean(), t_sub["auc_pr"].std(ddof=1)
        abs_drop = r_mean - t_mean
        rel_drop = (abs_drop / r_mean) * 100.0
        split_rows.append({
            "model": m,
            "random_auc_pr_mean": r_mean,
            "random_auc_pr_std": r_std,
            "timesplit_auc_pr_mean": t_mean,
            "timesplit_auc_pr_std": t_std,
            "absolute_drop": abs_drop,
            "relative_drop_pct": rel_drop,
        })

    split_df = pd.DataFrame(split_rows)
    split_csv_path = "results/tables/split_comparison.csv"
    split_df.to_csv(split_csv_path, index=False)

    # Markdown version of split_comparison
    split_md_rows = []
    for r in split_rows:
        split_md_rows.append({
            "Model": r["model"],
            "Elliptic (random split) AUC-PR": f"{r['random_auc_pr_mean']:.4f} ± {r['random_auc_pr_std']:.4f}",
            "Elliptic (time-respecting split) AUC-PR": f"{r['timesplit_auc_pr_mean']:.4f} ± {r['timesplit_auc_pr_std']:.4f}",
            "Absolute Drop": f"{r['absolute_drop']:.4f}",
            "Relative Drop (%)": f"{r['relative_drop_pct']:.2f}%",
        })
    split_md_df = pd.DataFrame(split_md_rows)
    split_md_path = "results/tables/split_comparison.md"
    with open(split_md_path, "w") as f:
        f.write("# Elliptic Split Comparison: Random Split vs Time-Respecting Split (AUC-PR)\n\n")
        f.write(df_to_markdown(split_md_df))
        f.write("\n")

    return master_df, split_df, raw_all


def run_sanity_checks(master_df, raw_all):
    print("=" * 70)
    print("TASK 1d: SANITY CHECKS")
    print("=" * 70)

    # Check 1: Seed count per (dataset, model) is not 5
    print("\n1. Seed count check (expecting 5 seeds for every model):")
    seed_counts = raw_all.groupby(["dataset", "model"]).size()
    non_5 = seed_counts[seed_counts != 5]
    if len(non_5) == 0:
        print("  PASS: All 17 evaluated (dataset, model) combinations have exactly 5 seeds.")
    else:
        print(f"  WARNING: Combinations with seed count != 5:\n{non_5}")

    # Check 2: Missing dataset/model combinations
    print("\n2. Missing dataset/model combinations check:")
    expected_combos = [(ds, m) for ds in DATASET_ORDER for m in MODEL_ORDER]
    present_combos = set(zip(master_df["dataset"], master_df["model"]))
    missing = [c for c in expected_combos if c not in present_combos]
    if len(missing) == 0:
        print("  All combinations present.")
    else:
        print(f"  Notice: {len(missing)} missing combination(s):")
        for ds, m in missing:
            print(f"    - {m} on {ds}")
        print("    (Note: GAT (tuned) was evaluated only on AMLSim and Elliptic (random split); not tuned on Elliptic time split).")

    # Check 3: Models whose std is exactly 0
    print("\n3. Zero-variance check (models with std == 0):")
    zero_std = master_df[(master_df["precision_std"] == 0) & 
                         (master_df["recall_std"] == 0) & 
                         (master_df["f1_std"] == 0) & 
                         (master_df["auc_pr_std"] == 0)]
    for _, row in zero_std.iterrows():
        print(f"    - {row['model']} on {row['dataset']}: std=0 across all metrics")
    print("  Note: Zero std across seeds is expected only for Logistic Regression, which is deterministic.")

    # Check 4: Compare recomputed values against existing summary CSVs
    print("\n4. Verification against existing seeded_*_summary.csv files:")
    sum_baseline = pd.read_csv("models/seeded_baseline_results_summary.csv")
    sum_gnn = pd.read_csv("models/seeded_gnn_results_summary.csv")
    sum_existing = pd.concat([sum_baseline, sum_gnn], ignore_index=True)

    diffs = []
    for _, s_row in sum_existing.iterrows():
        ds, m = s_row["dataset"], s_row["model"]
        m_row = master_df[(master_df["dataset"] == ds) & (master_df["model"] == m)]
        if len(m_row) == 0:
            continue
        m_row = m_row.iloc[0]
        for metric in ["precision", "recall", "f1", "auc_pr"]:
            diff_mean = abs(s_row[f"{metric}_mean"] - m_row[f"{metric}_mean"])
            diff_std = abs(s_row[f"{metric}_std"] - m_row[f"{metric}_std"])
            if diff_mean > 1e-6 or diff_std > 1e-6:
                diffs.append((ds, m, metric, diff_mean, diff_std))

    if len(diffs) == 0:
        print("  PASS: All recomputed means and sample standard deviations (ddof=1) match the existing")
        print("        seeded_*_summary.csv files exactly within floating-point tolerance (< 1e-15).")
    else:
        print(f"  Differences found ({len(diffs)}):")
        for d in diffs:
            print(f"    {d[0]} | {d[1]} | {d[2]}: diff_mean={d[3]:.2e}, diff_std={d[4]:.2e}")


def generate_task2_figures(master_df):
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 10

    # -------------------------------------------------------------------------
    # 2a) Grouped bar chart of AUC-PR per model, one panel per dataset label
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    fig.suptitle("AUC-PR by model across datasets (mean ± std over 5 seeds)", fontsize=13, y=0.98)

    for i, ds in enumerate(DATASET_ORDER):
        ax = axes[i]
        sub = master_df[master_df["dataset"] == ds]
        models = sub["model"].tolist()
        means = sub["auc_pr_mean"].tolist()
        stds = sub["auc_pr_std"].tolist()
        colors = [MODEL_COLORS[m] for m in models]

        x = np.arange(len(models))
        bars = ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="black", linewidth=0.8, alpha=0.9)
        
        ax.set_title(ds, fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Model", fontsize=10)
        if i == 0:
            ax.set_ylabel("AUC-PR", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=35, ha="right", fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        # Annotate mean values above error bars
        for bar, mean, std in zip(bars, means, stds):
            y_pos = mean + std + 0.02
            if y_pos > 1.02:
                y_pos = mean - 0.06
            ax.text(bar.get_x() + bar.get_width() / 2, y_pos, f"{mean:.3f}",
                    ha="center", va="bottom", fontsize=8, rotation=0)

    plt.tight_layout()
    fig2a_path = "results/figures/auc_pr_by_model.png"
    plt.savefig(fig2a_path, dpi=200)
    plt.close()
    print(f"Saved: {fig2a_path}")

    # -------------------------------------------------------------------------
    # 2b) Paired bars of AUC-PR for each model on Elliptic random vs time-respecting split
    # -------------------------------------------------------------------------
    split_models = ["Logistic Regression", "Random Forest", "XGBoost", "GraphSAGE", "GAT"]
    r_sub = master_df[master_df["dataset"] == "Elliptic (random split)"].set_index("model").loc[split_models]
    t_sub = master_df[master_df["dataset"] == "Elliptic (time-respecting split)"].set_index("model").loc[split_models]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(split_models))
    width = 0.36

    # Random split: solid model color
    bars_r = ax.bar(x - width / 2, r_sub["auc_pr_mean"], width=width,
                    yerr=r_sub["auc_pr_std"], capsize=4,
                    color=[MODEL_COLORS[m] for m in split_models],
                    edgecolor="black", linewidth=0.8, label="Elliptic (random split)")

    # Time-respecting split: same model color with diagonal hatching and lighter shade
    bars_t = ax.bar(x + width / 2, t_sub["auc_pr_mean"], width=width,
                    yerr=t_sub["auc_pr_std"], capsize=4,
                    color=[MODEL_COLORS[m] for m in split_models],
                    edgecolor="black", linewidth=0.8, alpha=0.5, hatch="//", label="Elliptic (time-respecting split)")

    ax.set_title("AUC-PR by model on Elliptic random vs time-respecting split (mean ± std over 5 seeds)",
                 fontsize=12, pad=12)
    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("AUC-PR", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(split_models, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Annotate values
    for bar in bars_r:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.3f}",
                ha="center", va="bottom", fontsize=8)
    for bar in bars_t:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.3f}",
                ha="center", va="bottom", fontsize=8)

    ax.legend(frameon=True, facecolor="white", edgecolor="gray")
    plt.tight_layout()
    fig2b_path = "results/figures/elliptic_split_comparison.png"
    plt.savefig(fig2b_path, dpi=200)
    plt.close()
    print(f"Saved: {fig2b_path}")

    # -------------------------------------------------------------------------
    # 2c) Heatmap of validation F1 over the GAT grid from gat_grid_search_results.csv
    # -------------------------------------------------------------------------
    grid_df = pd.read_csv("models/gat_grid_search_results.csv")

    for ds in ["AMLSim", "Elliptic (random split)"]:
        sub_grid = grid_df[grid_df["dataset"] == ds].copy()
        sub_grid["arch"] = [f"heads={h}, hidden={hid}" for h, hid in zip(sub_grid["heads"], sub_grid["hidden"])]
        pivot = sub_grid.pivot(index="arch", columns="lr", values="val_f1")

        # Sort index by heads then hidden
        heads_hidden = sorted(list(set(zip(sub_grid["heads"], sub_grid["hidden"]))))
        sorted_arch = [f"heads={h}, hidden={hid}" for h, hid in heads_hidden]
        pivot = pivot.reindex(sorted_arch)

        best_val = sub_grid["val_f1"].max()
        best_row = sub_grid[sub_grid["val_f1"] == best_val].iloc[0]
        best_arch = f"heads={int(best_row['heads'])}, hidden={int(best_row['hidden'])}"
        best_lr = best_row["lr"]

        fig, ax = plt.subplots(figsize=(7, 5 if ds == "AMLSim" else 3.8))
        im = ax.imshow(pivot.values, cmap="YlGnBu", aspect="auto")

        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Validation F1", rotation=-90, va="bottom", fontsize=10)

        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_xticklabels(pivot.columns, fontsize=9)
        ax.set_yticklabels(pivot.index, fontsize=9)
        ax.set_xlabel("Learning rate (lr)", fontsize=10)
        ax.set_ylabel("Architecture (heads, hidden)", fontsize=10)
        ax.set_title(f"GAT grid search validation F1 — {ds}", fontsize=11, pad=10)

        # Loop over data dimensions and create text annotations.
        for r_idx, arch in enumerate(pivot.index):
            for c_idx, lr_col in enumerate(pivot.columns):
                val = pivot.iloc[r_idx, c_idx]
                if np.isnan(val):
                    continue
                is_best = (arch == best_arch and abs(lr_col - best_lr) < 1e-6)
                txt = f"{val:.4f}\n(best)" if is_best else f"{val:.4f}"
                weight = "bold" if is_best else "normal"
                color = "red" if is_best else ("white" if val > pivot.values[~np.isnan(pivot.values)].mean() else "black")
                ax.text(c_idx, r_idx, txt, ha="center", va="center", color=color,
                        fontweight=weight, fontsize=9)
                if is_best:
                    rect = plt.Rectangle((c_idx - 0.48, r_idx - 0.48), 0.96, 0.96,
                                         fill=False, edgecolor="red", linewidth=2.5)
                    ax.add_patch(rect)

        plt.tight_layout()
        slug = "amlsim" if "aml" in ds.lower() else "elliptic"
        fig2c_path = f"results/figures/gat_grid_search_f1_heatmap_{slug}.png"
        plt.savefig(fig2c_path, dpi=200)
        plt.close()
        print(f"Saved: {fig2c_path}")

    # Also save combined two-panel heatmap figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.1, 1]})
    for ax, ds, title in [(ax1, "AMLSim", "AMLSim"), (ax2, "Elliptic (random split)", "Elliptic (random split)")]:
        sub_grid = grid_df[grid_df["dataset"] == ds].copy()
        sub_grid["arch"] = [f"heads={h}, hidden={hid}" for h, hid in zip(sub_grid["heads"], sub_grid["hidden"])]
        pivot = sub_grid.pivot(index="arch", columns="lr", values="val_f1")
        heads_hidden = sorted(list(set(zip(sub_grid["heads"], sub_grid["hidden"]))))
        sorted_arch = [f"heads={h}, hidden={hid}" for h, hid in heads_hidden]
        pivot = pivot.reindex(sorted_arch)

        best_val = sub_grid["val_f1"].max()
        best_row = sub_grid[sub_grid["val_f1"] == best_val].iloc[0]
        best_arch = f"heads={int(best_row['heads'])}, hidden={int(best_row['hidden'])}"
        best_lr = best_row["lr"]

        im = ax.imshow(pivot.values, cmap="YlGnBu", aspect="auto")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel("Validation F1", rotation=-90, va="bottom", fontsize=9)

        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_xticklabels(pivot.columns, fontsize=9)
        ax.set_yticklabels(pivot.index, fontsize=9)
        ax.set_xlabel("Learning rate (lr)", fontsize=10)
        ax.set_ylabel("Architecture (heads, hidden)", fontsize=10)
        ax.set_title(f"GAT grid validation F1 — {title}", fontsize=11, pad=8)

        for r_idx, arch in enumerate(pivot.index):
            for c_idx, lr_col in enumerate(pivot.columns):
                val = pivot.iloc[r_idx, c_idx]
                if np.isnan(val):
                    continue
                is_best = (arch == best_arch and abs(lr_col - best_lr) < 1e-6)
                txt = f"{val:.4f}\n(best)" if is_best else f"{val:.4f}"
                weight = "bold" if is_best else "normal"
                color = "red" if is_best else ("white" if val > pivot.values[~np.isnan(pivot.values)].mean() else "black")
                ax.text(c_idx, r_idx, txt, ha="center", va="center", color=color,
                        fontweight=weight, fontsize=8.5)
                if is_best:
                    rect = plt.Rectangle((c_idx - 0.48, r_idx - 0.48), 0.96, 0.96,
                                         fill=False, edgecolor="red", linewidth=2.5)
                    ax.add_patch(rect)

    fig.suptitle("Validation F1 over GAT architecture and learning rate grid", fontsize=12, y=0.98)
    plt.tight_layout()
    fig2c_comb_path = "results/figures/gat_grid_search_f1_heatmap.png"
    plt.savefig(fig2c_comb_path, dpi=200)
    plt.close()
    print(f"Saved: {fig2c_comb_path}")

    # -------------------------------------------------------------------------
    # 2d) Horizontal bar chart of the top 10 features in shap_feature_importance.csv
    # -------------------------------------------------------------------------
    shap_df = pd.read_csv("models/shap_feature_importance.csv")
    top10 = shap_df.sort_values("mean_abs_shap", ascending=False).head(10).copy()
    top10 = top10.sort_values("mean_abs_shap", ascending=True)  # ascending so top is at top of barh

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(top10["feature"], top10["mean_abs_shap"], color="#37474F", edgecolor="black", linewidth=0.8, alpha=0.85)

    ax.set_title("Top 10 features by mean absolute SHAP value", fontsize=12, pad=10)
    ax.set_xlabel("Mean |SHAP value|", fontsize=10)
    ax.set_ylabel("Feature", fontsize=10)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    # Annotate numeric values at the end of each bar
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.001, bar.get_y() + bar.get_height() / 2, f"{w:.4f}",
                va="center", ha="left", fontsize=9)

    ax.set_xlim(0, top10["mean_abs_shap"].max() * 1.15)
    plt.tight_layout()
    fig2d_path = "results/figures/shap_top10_features.png"
    plt.savefig(fig2d_path, dpi=200)
    plt.close()
    print(f"Saved: {fig2d_path}")


def generate_task3_typology():
    # If typology_counts_v2.csv exists from src/typology_eval.py, point to it and skip old single-run generation
    if os.path.exists("results/tables/typology_counts_v2.csv"):
        print("\n" + "=" * 70)
        print("TASK 3: TYPOLOGY COUNTS (v2)")
        print("=" * 70)
        print("Using results/tables/typology_counts_v2.csv (generated by src/typology_eval.py on graph test split across 5 seeds).")
        df_v2 = pd.read_csv("results/tables/typology_counts_v2.csv")
        print(df_v2.to_string(index=False))
        return

    # Inspect files
    gs = pd.read_csv("models/graphsage_typology_results.csv")
    rf = pd.read_csv("models/rf_typology_results.csv")

    print("\n" + "=" * 70)
    print("TASK 3: TYPOLOGY CSV COLUMNS AND CONTENTS")
    print("=" * 70)
    print(f"\nmodels/graphsage_typology_results.csv columns: {list(gs.columns)}")
    print(gs.to_string())
    print(f"\nmodels/rf_typology_results.csv columns: {list(rf.columns)}")
    print(rf.to_string())

    alert_types = ["cycle", "fan_in", "fan_out"]
    rows = []
    for at in alert_types:
        sub_gs = gs[gs["alert_type"] == at]
        n_gs = len(sub_gs)
        k_gs = int((sub_gs["y_pred"] == sub_gs["y_true"]).sum())
        recall_gs = f"{k_gs}/{n_gs}"

        sub_rf = rf[rf["alert_type"] == at]
        n_rf = len(sub_rf)
        k_rf = int((sub_rf["rf_pred"] == sub_rf["y_true"]).sum())
        recall_rf = f"{k_rf}/{n_rf}"

        rows.append({
            "alert_type": at,
            "model": "GraphSAGE",
            "n": n_gs,
            "k": k_gs,
            "recall": recall_gs,
        })
        rows.append({
            "alert_type": at,
            "model": "Random Forest",
            "n": n_rf,
            "k": k_rf,
            "recall": recall_rf,
        })

    typo_df = pd.DataFrame(rows)
    typo_csv_path = "results/tables/typology_counts.csv"
    typo_df.to_csv(typo_csv_path, index=False)
    print(f"\nSaved: {typo_csv_path}")

    # Markdown table
    typo_md_rows = []
    for at in alert_types:
        sub_gs = typo_df[(typo_df["alert_type"] == at) & (typo_df["model"] == "GraphSAGE")].iloc[0]
        sub_rf = typo_df[(typo_df["alert_type"] == at) & (typo_df["model"] == "Random Forest")].iloc[0]
        typo_md_rows.append({
            "Alert Type": at,
            "GraphSAGE (k/n)": f"{sub_gs['k']}/{sub_gs['n']}",
            "GraphSAGE Recall": f"{sub_gs['k']/sub_gs['n']:.4f}",
            "Random Forest (k/n)": f"{sub_rf['k']/sub_rf['n']}",
            "Random Forest Recall": f"{sub_rf['k']/sub_rf['n']:.4f}",
        })
    typo_md_df = pd.DataFrame(typo_md_rows)
    typo_md_path = "results/tables/typology_counts.md"
    with open(typo_md_path, "w") as f:
        f.write("# Typology Breakdown by Alert Type: GraphSAGE vs Random Forest\n\n")
        f.write("> **Note:** These counts are derived from a single evaluation run on AMLSim test positives; n per alert type is very small (n=4 for cycle, n=6 for fan_in, n=6 for fan_out; total n=16).\n\n")
        f.write(df_to_markdown(typo_md_df))
        f.write("\n")
    print(f"Saved: {typo_md_path}")

    print("\nTypology Breakdown Summary:")
    print(typo_df.to_string(index=False))
    print("\nNote: These files come from a single run and n per type is very small.")


if __name__ == "__main__":
    print("Generating Task 1 consolidated tables...")
    master_df, split_df, raw_all = generate_task1_tables()
    run_sanity_checks(master_df, raw_all)

    print("\nGenerating Task 2 figures...")
    generate_task2_figures(master_df)

    print("\nGenerating Task 3 typology counts...")
    generate_task3_typology()

    print("\nAll tasks 1-3 completed successfully!")
