"""
AMLGuard AI — GAT Hyperparameter Tuning (Week 3 of Research Rigor Plan)
-------------------------------------------------------------------------
Purpose: check whether GAT's weak results (especially on AMLSim) are a
data-scarcity effect (supports RQ2) or just an under-tuned configuration.

Protocol:
  1. Grid search over heads / hidden dim / learning rate, scored on the
     VALIDATION set only. The test set is never touched during tuning.
  2. Take the best config per dataset (by validation F1) and re-run it
     across 5 seeds, reporting test metrics as mean +- std.
  3. The saved checkpoint per dataset is the seed with the best VALIDATION
     F1 (never test F1), and a resumed run only overwrites an existing
     checkpoint if it beats that checkpoint's validation F1.

Results are appended to a CSV after every config, so an interruption
does not lose finished work, and re-running skips configs already done.

Run from the project root, venv active:
    python -u src/gat_tuning.py 2>&1 | tee models/gat_tuning.log
"""

import itertools
import os
import sys
import time
import traceback
import argparse

print("[INFO] AMLGuard AI — GAT Hyperparameter Tuning", flush=True)
print(f"[INFO] Python executable: {sys.executable}", flush=True)
print("[INFO] Loading core dependencies (PyTorch, PyG, scikit-learn)...", flush=True)
_t_import_start = time.time()

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score

print(f"[INFO] Dependencies loaded in {time.time() - _t_import_start:.2f}s", flush=True)

DEVICE = torch.device("cpu")
SEEDS = [0, 1, 42, 123, 2024]
GRID_SEED = 42          # single seed for the search itself (cheap); final eval uses all 5
DEFAULT_EPOCHS = 200
DEFAULT_EVAL_EVERY = 20

# AMLSim is small, so it gets the full 27-config grid.
# Elliptic is large and slow on CPU, so it gets a reduced 8-config grid.
GRIDS = {
    "AMLSim": dict(heads=(2, 4, 8), hidden=(16, 32, 64), lr=(0.001, 0.005, 0.01)),
    "Elliptic (random split)": dict(heads=(2, 4), hidden=(16, 32), lr=(0.005, 0.01)),
}
GRAPH_PATHS = {
    "AMLSim": "models/aml_graph.pt",
    "Elliptic (random split)": "models/elliptic_graph.pt",
}

GRID_CSV = "models/gat_grid_search_results.csv"
FINAL_CSV = "models/gat_tuned_final_seeds.csv"


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.3):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


def metrics(model, data, mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = F.softmax(out, dim=1)[:, 1]
        preds = out.argmax(dim=1)
        y_true = data.y[mask].cpu().numpy()
        return {
            "precision": float(precision_score(y_true, preds[mask].cpu().numpy(), zero_division=0)),
            "recall": float(recall_score(y_true, preds[mask].cpu().numpy(), zero_division=0)),
            "f1": float(f1_score(y_true, preds[mask].cpu().numpy(), zero_division=0)),
            "auc_pr": float(average_precision_score(y_true, probs[mask].cpu().numpy())),
        }


def train_gat(data, heads, hidden, lr, seed, use_test, epochs=DEFAULT_EPOCHS, eval_every=DEFAULT_EVAL_EVERY):
    """Train one GAT. Model selection is always by validation F1.
    Test metrics are computed only when use_test=True (final evaluation)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = GAT(data.num_node_features, hidden, 2, heads=heads).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    y_train = data.y[data.train_mask]
    n_pos = (y_train == 1).sum().item()
    n_neg = (y_train == 0).sum().item()
    class_weights = torch.tensor([1.0, n_neg / max(n_pos, 1)], dtype=torch.float)

    best_val_f1, best_val, best_test = -1.0, None, None
    best_state_dict = None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask], weight=class_weights)
        loss.backward()
        optimizer.step()

        if epoch % eval_every == 0 or epoch == epochs:
            val = metrics(model, data, data.val_mask)
            if val["f1"] >= best_val_f1:
                best_val_f1, best_val = val["f1"], val
                best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                if use_test:
                    best_test = metrics(model, data, data.test_mask)

    return best_val, best_test, best_state_dict


def already_done(csv_path, key_cols, key_vals):
    if not os.path.exists(csv_path):
        return False
    df = pd.read_csv(csv_path)
    mask = np.ones(len(df), dtype=bool)
    for c, v in zip(key_cols, key_vals):
        mask &= (df[c] == v).values
    return bool(mask.any())


def append_row(csv_path, row):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    pd.DataFrame([row]).to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)


def load_existing_checkpoint_val_f1(ckpt_path, dataset, heads, hidden, lr):
    """Return the validation F1 of an existing checkpoint, or -1.0 if there is
    none, it cannot be read, or it was produced by a different config/dataset
    (in which case it is stale and should be replaced)."""
    if not os.path.exists(ckpt_path):
        return -1.0
    try:
        ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
        same = (ckpt.get("dataset") == dataset and ckpt.get("heads") == heads
                and ckpt.get("hidden") == hidden and abs(ckpt.get("lr", -1) - lr) < 1e-12)
        if not same:
            print(f"[WARN] Existing checkpoint {ckpt_path} was made with a different config; it will be replaced.", flush=True)
            return -1.0
        return float(ckpt["val_metrics"]["f1"])
    except Exception as exc:
        print(f"[WARN] Could not read existing checkpoint {ckpt_path} ({exc}); it will be replaced.", flush=True)
        return -1.0


def main():
    parser = argparse.ArgumentParser(description="AMLGuard AI — GAT Hyperparameter Tuning")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run a quick smoke test (1 AMLSim config, 2 epochs, no files written). "
                             "Overrides --dataset.")
    parser.add_argument("--dataset", choices=["AMLSim", "Elliptic", "all"], default="all", help="Target dataset to run")
    args = parser.parse_args()

    epochs = 2 if args.smoke_test else DEFAULT_EPOCHS
    eval_every = 1 if args.smoke_test else DEFAULT_EVAL_EVERY
    seeds = [42] if args.smoke_test else SEEDS

    print(f"Device: {DEVICE} | threads: {torch.get_num_threads()} | seeds: {seeds} | epochs: {epochs}", flush=True)
    t_start = time.time()

    active_grids = GRIDS
    if args.smoke_test:
        active_grids = {
            "AMLSim": dict(heads=(2,), hidden=(16,), lr=(0.01,))
        }
    elif args.dataset == "AMLSim":
        active_grids = {"AMLSim": GRIDS["AMLSim"]}
    elif args.dataset == "Elliptic":
        active_grids = {"Elliptic (random split)": GRIDS["Elliptic (random split)"]}

    for dataset, grid in active_grids.items():
        print(f"\n{'=' * 70}\nDATASET: {dataset}\n{'=' * 70}", flush=True)
        graph_path = GRAPH_PATHS[dataset]
        if not os.path.exists(graph_path):
            raise FileNotFoundError(f"Required graph file does not exist: {graph_path}")

        print(f"[INFO] Loading graph data from {graph_path} ...", flush=True)
        t_load = time.time()
        data = torch.load(graph_path, weights_only=False).to(DEVICE)
        print(f"[INFO] Graph loaded in {time.time() - t_load:.2f}s", flush=True)
        print(f"  Nodes: {data.num_nodes} | Edges: {data.edge_index.shape[1]} | Features: {data.num_node_features}", flush=True)
        print(f"  Train: {data.train_mask.sum().item()} | Val: {data.val_mask.sum().item()} | Test: {data.test_mask.sum().item()}", flush=True)

        # ---- Phase 1: grid search, validation only ----
        combos = list(itertools.product(grid["heads"], grid["hidden"], grid["lr"]))
        print(f"\nPhase 1: grid search over {len(combos)} configs (validation only)", flush=True)
        for i, (heads, hidden, lr) in enumerate(combos, 1):
            if not args.smoke_test and already_done(GRID_CSV, ["dataset", "heads", "hidden", "lr"], [dataset, heads, hidden, lr]):
                print(f"  [{i}/{len(combos)}] heads={heads} hidden={hidden} lr={lr}: already done, skipping", flush=True)
                continue
            t0 = time.time()
            val, _, _ = train_gat(data, heads, hidden, lr, GRID_SEED, use_test=False, epochs=epochs, eval_every=eval_every)
            if not args.smoke_test:
                append_row(GRID_CSV, {"dataset": dataset, "heads": heads, "hidden": hidden, "lr": lr,
                                      "val_f1": val["f1"], "val_auc_pr": val["auc_pr"],
                                      "seconds": round(time.time() - t0, 1)})
            print(f"  [{i}/{len(combos)}] heads={heads} hidden={hidden} lr={lr}: "
                  f"val F1={val['f1']:.4f} val AUC-PR={val['auc_pr']:.4f} ({time.time() - t0:.1f}s)", flush=True)

        if args.smoke_test:
            best_heads, best_hidden, best_lr = combos[0]
            print(f"\n[Smoke-test] Using test config: heads={best_heads}, hidden={best_hidden}, lr={best_lr}", flush=True)
        else:
            grid_df = pd.read_csv(GRID_CSV)
            grid_df = grid_df[grid_df["dataset"] == dataset].sort_values("val_f1", ascending=False)
            best = grid_df.iloc[0]
            best_heads, best_hidden, best_lr = int(best["heads"]), int(best["hidden"]), float(best["lr"])
            print(f"\nBest config for {dataset} (by validation F1): heads={best_heads}, hidden={best_hidden}, lr={best_lr} "
                  f"(val F1={best['val_f1']:.4f})", flush=True)

        # ---- Phase 2: multi-seed final evaluation of best config ----
        print(f"\nPhase 2: {len(seeds)}-seed test evaluation of best config (heads={best_heads}, hidden={best_hidden}, lr={best_lr})", flush=True)
        dataset_slug = "amlsim" if "aml" in dataset.lower() else "elliptic"
        ckpt_path = f"models/gat_tuned_best_{dataset_slug}.pt"

        # FIX 2: seed the comparison with any existing checkpoint's validation F1,
        # so a resumed run cannot overwrite a better model from an earlier run.
        if args.smoke_test:
            best_overall_val_f1 = -1.0
        else:
            best_overall_val_f1 = load_existing_checkpoint_val_f1(ckpt_path, dataset, best_heads, best_hidden, best_lr)
            if best_overall_val_f1 >= 0:
                print(f"[INFO] Existing checkpoint val F1 = {best_overall_val_f1:.4f}; "
                      f"it is replaced only if a new seed beats it.", flush=True)
        best_checkpoint_state = None

        for seed in seeds:
            if not args.smoke_test and already_done(FINAL_CSV, ["dataset", "seed"], [dataset, seed]):
                print(f"  seed {seed}: already done, skipping", flush=True)
                continue
            val, test, state_dict = train_gat(data, best_heads, best_hidden, best_lr, seed, use_test=True, epochs=epochs, eval_every=eval_every)
            row = {"dataset": dataset, "model": "GAT (tuned)", "seed": seed,
                   "heads": best_heads, "hidden": best_hidden, "lr": best_lr, **test}
            if not args.smoke_test:
                append_row(FINAL_CSV, row)
            print(f"  seed {seed}: val F1={val['f1']:.4f} -> test F1={test['f1']:.4f} AUC-PR={test['auc_pr']:.4f}", flush=True)

            # FIX 1: select the checkpoint by VALIDATION F1, never test F1.
            if val["f1"] > best_overall_val_f1 and state_dict is not None:
                best_overall_val_f1 = val["f1"]
                best_checkpoint_state = {
                    "model_state_dict": state_dict,
                    "dataset": dataset,
                    "heads": best_heads,
                    "hidden": best_hidden,
                    "lr": best_lr,
                    "seed": seed,
                    "val_metrics": val,
                    "test_metrics": test,
                }

        # Save only if a new seed beat the existing checkpoint (or none existed)
        if best_checkpoint_state is not None and not args.smoke_test:
            torch.save(best_checkpoint_state, ckpt_path)
            print(f"[INFO] Saved tuned model checkpoint to {ckpt_path} "
                  f"(seed {best_checkpoint_state['seed']}, val F1={best_overall_val_f1:.4f})", flush=True)
        elif not args.smoke_test:
            print(f"[INFO] Checkpoint {ckpt_path} unchanged.", flush=True)

    # ---- Summary ----
    if not args.smoke_test and os.path.exists(FINAL_CSV):
        final = pd.read_csv(FINAL_CSV)
        print(f"\n{'=' * 70}\nTUNED GAT — mean +- std over seeds (test set)\n{'=' * 70}", flush=True)
        for dataset, g in final.groupby("dataset"):
            print(f"\n{dataset}  (heads={g['heads'].iloc[0]}, hidden={g['hidden'].iloc[0]}, lr={g['lr'].iloc[0]})")
            for m in ["precision", "recall", "f1", "auc_pr"]:
                std_str = f" +- {g[m].std():.4f}" if len(g) > 1 else ""
                print(f"  {m:<9}: {g[m].mean():.4f}{std_str}")
        print(f"\nTotal time: {(time.time() - t_start) / 60:.1f} min", flush=True)
        print(f"Saved: {GRID_CSV}, {FINAL_CSV}", flush=True)
    elif args.smoke_test:
        print(f"\n[Smoke-test] Completed successfully in {time.time() - t_start:.2f}s!", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[FATAL ERROR] Script failed with exception: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)
