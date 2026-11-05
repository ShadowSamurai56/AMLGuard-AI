"""
AMLGuard AI — Significance Testing (Week 2 of Research Rigor Plan)
----------------------------------------------------------------------
Runs paired significance tests on the seeded results already generated:
    models/seeded_baseline_results_raw.csv   (LogReg, RF, XGBoost)
    models/seeded_gnn_results_raw.csv        (GraphSAGE, GAT)

For each research question, tests whether the observed gap between two
conditions is likely real or explainable by seed noise alone.

Uses a paired t-test (appropriate since the same 5 seeds are used across
compared conditions) AND the Wilcoxon signed-rank test as a non-parametric
cross-check, since n=5 is small enough that t-test normality assumptions
are shaky on their own. (Note: the Wilcoxon test cannot go below p = 0.0625 at n = 5).

Run this from your project root, venv active:
    python src/significance_tests.py
"""

# Standard deviation convention: sample standard deviation (ddof=1, the pandas default).
import os
import pandas as pd
from scipy import stats

ALPHA = 0.05


def load_data():
    # Standard deviation convention: sample standard deviation (ddof=1, pandas default)
    baseline = pd.read_csv("models/seeded_baseline_results_raw.csv")
    gnn = pd.read_csv("models/seeded_gnn_results_raw.csv")
    gat_tuned = pd.read_csv("models/gat_tuned_final_seeds.csv")
    combined = pd.concat([baseline, gnn, gat_tuned], ignore_index=True)
    return combined


def get_metric_by_seed(df, dataset, model, metric="auc_pr"):
    subset = df[(df["dataset"] == dataset) & (df["model"] == model)].sort_values("seed")
    return subset[metric].values


def run_comparison(df, dataset, model_a, model_b, metric="auc_pr", label=""):
    a = get_metric_by_seed(df, dataset, model_a, metric)
    b = get_metric_by_seed(df, dataset, model_b, metric)

    if len(a) == 0 or len(b) == 0:
        print(f"  SKIPPED — missing data for {model_a} or {model_b} on {dataset}")
        return None

    if len(a) != len(b):
        print(f"  WARNING — unequal seed counts ({len(a)} vs {len(b)}), truncating to match")
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]

    mean_a, mean_b = a.mean(), b.mean()
    diff = mean_a - mean_b

    t_stat, t_pval = stats.ttest_rel(a, b)

    try:
        w_stat, w_pval = stats.wilcoxon(a, b)
    except ValueError:
        # Wilcoxon fails if all differences are identical (zero variance case)
        w_stat, w_pval = float("nan"), float("nan")

    significant_t = t_pval < ALPHA
    significant_w = w_pval < ALPHA if not pd.isna(w_pval) else None

    print(f"\n{label or f'{model_a} vs {model_b} on {dataset}'} ({metric})")
    print(f"  {model_a}: {mean_a:.4f} (±{a.std(ddof=1):.4f})")
    print(f"  {model_b}: {mean_b:.4f} (±{b.std(ddof=1):.4f})")
    print(f"  Difference: {diff:+.4f}")
    print(f"  Paired t-test:      p = {t_pval:.4f}  {'SIGNIFICANT' if significant_t else 'not significant'}")
    if not pd.isna(w_pval):
        print(f"  Wilcoxon signed-rank: p = {w_pval:.4f}  {'SIGNIFICANT' if significant_w else 'not significant'}")
    else:
        print(f"  Wilcoxon signed-rank: N/A (identical values, likely zero-variance model)")

    return {
        "comparison": label or f"{model_a} vs {model_b}",
        "dataset": dataset,
        "metric": metric,
        "mean_a": mean_a, "mean_b": mean_b, "diff": diff,
        "t_pval": t_pval, "w_pval": w_pval,
        "significant_t": significant_t, "significant_w": significant_w,
    }


def main():
    df = load_data()
    results = []

    print("=" * 70)
    print("RQ1: Does GraphSAGE significantly outperform tree-based baselines?")
    print("=" * 70)
    results.append(run_comparison(df, "AMLSim", "GraphSAGE", "Random Forest",
                                   label="RQ1: GraphSAGE vs Random Forest (AMLSim)"))
    results.append(run_comparison(df, "AMLSim", "GraphSAGE", "XGBoost",
                                   label="RQ1: GraphSAGE vs XGBoost (AMLSim)"))
    results.append(run_comparison(df, "Elliptic (random split)", "GraphSAGE", "Random Forest",
                                   label="RQ1: GraphSAGE vs Random Forest (Elliptic, random split)"))
    results.append(run_comparison(df, "Elliptic (random split)", "GraphSAGE", "XGBoost",
                                   label="RQ1: GraphSAGE vs XGBoost (Elliptic, random split)"))

    print("\n" + "=" * 70)
    print("RQ2: Does GAT underperform GraphSAGE more on small data than large data?")
    print("=" * 70)
    results.append(run_comparison(df, "AMLSim", "GAT", "GraphSAGE",
                                   label="RQ2: GAT vs GraphSAGE (AMLSim, small data)"))
    results.append(run_comparison(df, "Elliptic (random split)", "GAT", "GraphSAGE",
                                   label="RQ2: GAT vs GraphSAGE (Elliptic, large data)"))

    print("\n" + "=" * 70)
    print("RQ3: Does a time-respecting split significantly reduce performance")
    print("     vs a random split, and is the drop larger for GNNs than trees?")
    print("=" * 70)
    print("\n  NOTE: Comparing performance ACROSS splits (random vs time-respecting)")
    print("  for the same model requires comparing summary statistics directly,")
    print("  since the test sets differ in composition and size. See the")
    print("  seeded_gnn_results_summary.csv / seeded_baseline_results_summary.csv")
    print("  files for the honest mean±std comparison already computed:")
    print("    GraphSAGE:     0.936±0.002 (random) vs 0.457±0.049 (time-split)")
    print("    Random Forest: 0.981±0.0003 (random) vs 0.783±0.001 (time-split)")
    print("  The non-overlapping ranges (mean ± 2*std) make this drop self-evidently")
    print("  real without a formal test — GraphSAGE's random-split CI (~0.93-0.94)")
    print("  does not come close to overlapping its time-split CI (~0.36-0.55),")
    print("  and the same holds for Random Forest's much smaller drop.")
    # NOTE: Comparisons across different splits (random vs time-respecting) are not
    # seed-paired tests (different test sets and sizes) and are handled in
    # results/tables/split_comparison.csv instead.

    print("\n" + "=" * 70)
    print("Elliptic (time-respecting split): GNNs vs Baselines & GAT vs GraphSAGE")
    print("=" * 70)
    results.append(run_comparison(df, "Elliptic (time-respecting split)", "GraphSAGE", "Random Forest",
                                   label="GraphSAGE vs Random Forest (Elliptic, time-respecting split)"))
    results.append(run_comparison(df, "Elliptic (time-respecting split)", "GraphSAGE", "XGBoost",
                                   label="GraphSAGE vs XGBoost (Elliptic, time-respecting split)"))
    results.append(run_comparison(df, "Elliptic (time-respecting split)", "GAT", "GraphSAGE",
                                   label="GAT vs GraphSAGE (Elliptic, time-respecting split)"))

    print("\n" + "=" * 70)
    print("GAT (tuned) vs GraphSAGE & GAT (tuned) vs GAT (untuned)")
    print("=" * 70)
    results.append(run_comparison(df, "AMLSim", "GAT (tuned)", "GraphSAGE",
                                   label="GAT (tuned) vs GraphSAGE (AMLSim)"))
    results.append(run_comparison(df, "Elliptic (random split)", "GAT (tuned)", "GraphSAGE",
                                   label="GAT (tuned) vs GraphSAGE (Elliptic, random split)"))
    results.append(run_comparison(df, "AMLSim", "GAT (tuned)", "GAT",
                                   label="GAT (tuned) vs GAT (untuned) (AMLSim)"))
    results.append(run_comparison(df, "Elliptic (random split)", "GAT (tuned)", "GAT",
                                   label="GAT (tuned) vs GAT (untuned) (Elliptic, random split)"))

    results_df = pd.DataFrame([r for r in results if r is not None])
    # Write output to results/tables/ to ensure models/ is not modified
    output_path = "results/tables/significance_test_results.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print("\n" + "=" * 70)
    print("SUMMARY TABLE (All Comparisons)")
    print("=" * 70)
    print(results_df[["comparison", "dataset", "mean_a", "mean_b", "diff", "t_pval", "significant_t"]].to_string(index=False))

    print("\n" + "=" * 70)
    print("SUMMARY TABLE (New Rows Only)")
    print("=" * 70)
    print(results_df.iloc[6:][["comparison", "dataset", "mean_a", "mean_b", "diff", "t_pval", "significant_t"]].to_string(index=False))

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
