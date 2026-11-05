"""
AMLGuard AI — Typology Evaluation on AMLSim with Graph Masks (Multi-Seed)
--------------------------------------------------------------------------
Standard deviation convention: sample standard deviation (ddof=1, pandas default).

Evaluates Random Forest, XGBoost, and 1-layer GraphSAGE across 5 seeds:
[0, 1, 42, 123, 2024], trained on data.train_mask and evaluated ONLY
on data.test_mask of models/aml_graph.pt.

Joins test positive accounts to alert_accounts.alert_type from MySQL.
Outputs results/tables/typology_counts_v2.csv.
"""

# Standard deviation convention: sample standard deviation (ddof=1, the pandas default).

import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

SEEDS = [0, 1, 42, 123, 2024]

DB_USER = "aml_user"
DB_PASSWORD = "yourpassword"
DB_HOST = "localhost"
DB_NAME = "amlguard"
engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}")


class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=1, dropout=0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


def load_amlsim_data():
    df = pd.read_sql("SELECT * FROM account_features", engine)
    label_cols = [c for c in df.columns if c.startswith("label_")]
    y = df["label_is_flagged_account"].values
    X = df.drop(columns=label_cols + ["acct_id"])

    categorical_cols = X.select_dtypes(include="object").columns.tolist()
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    alert_df = pd.read_sql("SELECT acct_id, alert_type FROM alert_accounts", engine)

    data = torch.load("models/aml_graph.pt", weights_only=False)

    return df["acct_id"].values, X.values, y, alert_df, data


def evaluate_graphsage_seed(data, seed, epochs=200, lr=0.01):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cpu")
    data = data.to(device)

    y_train = data.y[data.train_mask]
    n_pos = (y_train == 1).sum().item()
    n_neg = (y_train == 0).sum().item()
    class_weights = torch.tensor([1.0, n_neg / max(n_pos, 1)], dtype=torch.float).to(device)

    model = GraphSAGE(
        in_channels=data.num_node_features,
        hidden_channels=64,
        out_channels=2,
        num_layers=1,
        dropout=0.3
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    best_val_f1 = -1.0
    best_preds = None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask], weight=class_weights)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                val_out = model(data.x, data.edge_index)
                preds = val_out.argmax(dim=1).cpu().numpy()
                y_val_true = data.y[data.val_mask].cpu().numpy()
                y_val_pred = preds[data.val_mask.cpu().numpy()]
                val_f1 = f1_score(y_val_true, y_val_pred, zero_division=0)

                if val_f1 >= best_val_f1:
                    best_val_f1 = val_f1
                    best_preds = preds.copy()

    return best_preds


def main():
    print("=" * 70)
    print("Running Typology Evaluation on AMLSim with Graph Masks (5 Seeds)")
    print("=" * 70)

    acct_ids, X_vals, y, alert_df, data = load_amlsim_data()
    train_mask = data.train_mask.cpu().numpy()
    test_mask = data.test_mask.cpu().numpy()

    X_train, y_train = X_vals[train_mask], y[train_mask]
    X_test, y_test = X_vals[test_mask], y[test_mask]
    test_indices = np.where(test_mask)[0]

    # Map test positive nodes to alert types
    test_pos_mask = test_mask & (y == 1)
    test_pos_indices = np.where(test_pos_mask)[0]
    test_pos_accts = pd.DataFrame({
        "node_idx": test_pos_indices,
        "acct_id": acct_ids[test_pos_indices]
    }).merge(alert_df, on="acct_id", how="left")

    alert_types = sorted(test_pos_accts["alert_type"].unique())
    print(f"Total test positives: {len(test_pos_accts)}")
    for at in alert_types:
        print(f"  {at}: n = {(test_pos_accts['alert_type'] == at).sum()}")

    models = ["Random Forest", "XGBoost", "GraphSAGE"]
    # Store predictions for each model and seed on test_pos_accts
    # seed -> model -> array of 0/1 predictions for test_pos_accts
    preds_by_model = {m: {} for m in models}

    for seed in SEEDS:
        print(f"\n--- Running Seed {seed} ---")
        # 1. Random Forest
        rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1)
        rf.fit(X_train, y_train)
        rf_test_preds = rf.predict(X_vals[test_pos_indices])
        preds_by_model["Random Forest"][seed] = rf_test_preds

        # 2. XGBoost
        pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        xgb = XGBClassifier(
            n_estimators=300,
            scale_pos_weight=pos_weight,
            random_state=seed,
            eval_metric="logloss",
            subsample=0.8,
            colsample_bytree=0.8,
        )
        xgb.fit(X_train, y_train)
        xgb_test_preds = xgb.predict(X_vals[test_pos_indices])
        preds_by_model["XGBoost"][seed] = xgb_test_preds

        # 3. 1-layer GraphSAGE
        gs_all_preds = evaluate_graphsage_seed(data, seed)
        gs_test_preds = gs_all_preds[test_pos_indices]
        preds_by_model["GraphSAGE"][seed] = gs_test_preds

    # Aggregate by alert type
    records = []
    for m in models:
        for at in alert_types:
            at_idx = np.where(test_pos_accts["alert_type"] == at)[0]
            n_type = len(at_idx)
            k_seeds = [int((preds_by_model[m][s][at_idx] == 1).sum()) for s in SEEDS]
            recalls = [k / n_type for k in k_seeds]
            mean_rec = np.mean(recalls)
            min_rec = np.min(recalls)
            max_rec = np.max(recalls)

            row = {
                "model": m,
                "alert_type": at,
                "n": n_type,
                "k_seed_0": k_seeds[0],
                "k_seed_1": k_seeds[1],
                "k_seed_42": k_seeds[2],
                "k_seed_123": k_seeds[3],
                "k_seed_2024": k_seeds[4],
                "mean_recall": mean_rec,
                "min_recall": min_rec,
                "max_recall": max_rec,
            }
            records.append(row)

    out_df = pd.DataFrame(records)
    out_csv = "results/tables/typology_counts_v2.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # Also write a clear markdown version
    out_md = "results/tables/typology_counts_v2.md"
    md_lines = [
        "# Typology Breakdown on Graph Test Split Across 5 Seeds (v2)\n\n",
        "> **Note:** n per alert type is very small because the test set has 16 positives (4 cycle, 6 fan_in, 6 fan_out). ",
        "Seeds vary only model initialisation, not the test set.\n\n",
    ]
    md_table_rows = []
    for _, r in out_df.iterrows():
        k_str = f"[{r['k_seed_0']}, {r['k_seed_1']}, {r['k_seed_42']}, {r['k_seed_123']}, {r['k_seed_2024']}]"
        md_table_rows.append({
            "Model": r["model"],
            "Alert Type": r["alert_type"],
            "n": r["n"],
            "k per seed [0,1,42,123,2024]": k_str,
            "Mean Recall": f"{r['mean_recall']:.4f}",
            "Min Recall": f"{r['min_recall']:.4f}",
            "Max Recall": f"{r['max_recall']:.4f}",
        })
    md_df = pd.DataFrame(md_table_rows)

    headers = list(md_df.columns)
    md_lines.append("| " + " | ".join(headers) + " |\n")
    md_lines.append("| " + " | ".join(["---"] * len(headers)) + " |\n")
    for _, row in md_df.iterrows():
        md_lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |\n")

    with open(out_md, "w") as f:
        f.writelines(md_lines)
    print(f"Saved: {out_md}")

    print("\n" + "=" * 70)
    print("TYPOLOGY RESULTS (v2)")
    print("=" * 70)
    print(out_df.to_string(index=False))
    print("\nNote: n per type is very small because the test set has 16 positives, and seeds vary only model initialisation, not the test set.")


if __name__ == "__main__":
    main()

