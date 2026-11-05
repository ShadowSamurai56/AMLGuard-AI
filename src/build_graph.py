"""
AMLGuard AI — Build Transaction Graph
----------------------------------------
Reads accounts, transactions, and account_features from MySQL, builds a
directed graph (accounts = nodes, transactions = edges), attaches node
features, and saves it as a PyTorch Geometric Data object ready for
GraphSAGE training.

IMPORTANT — graph-safe train/test split:
Unlike a plain tabular split, a graph model can "see" test-node labels
indirectly through message passing if we're not careful. This script
uses the standard, correct approach for node classification: ALL nodes
and ALL edges stay in the graph (the model needs the full structure to
compute neighborhood aggregations correctly), but we mark which nodes
are train/val/test via boolean masks. During training, loss is computed
ONLY on train-masked nodes; test-masked nodes' labels are never used for
loss or backprop, only for final evaluation. This is the standard
"transductive" node classification setup and is not a leak, because the
model never sees test LABELS, only test node FEATURES (which are
legitimate to know, same as in the real world you'd know an account's
own transaction history without knowing whether it's fraud).

Run this from your project root (AMLGuard-AI/), with your venv active:
    python src/build_graph.py

Requires: pip install torch torch_geometric
"""

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import torch
from torch_geometric.data import Data
print("CHECKPOINT: imports done (torch, torch_geometric)")

# ---------------------------------------------------------------
# 1. Connection settings — same as previous scripts
# ---------------------------------------------------------------
DB_USER = "aml_user"
DB_PASSWORD = "yourpassword"
DB_HOST = "localhost"
DB_NAME = "amlguard"

engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}")
print("CHECKPOINT: engine = create_engine(...) done")

GRAPH_DIR = "models"
RANDOM_SEED = 42


def load_data():
    print("Loading data from MySQL ...")
    accounts_features = pd.read_sql("SELECT * FROM account_features", engine)
    print("CHECKPOINT: account_features loaded from MySQL")
    transactions = pd.read_sql(
        "SELECT orig_acct, bene_acct, base_amt, tran_timestamp FROM transactions", engine
    )
    print("CHECKPOINT: transactions loaded from MySQL")
    print(f"  account_features: {accounts_features.shape}")
    print(f"  transactions: {transactions.shape}")
    return accounts_features, transactions


def build_node_index(accounts_features: pd.DataFrame):
    """Map each acct_id to a contiguous integer node index [0, N)."""
    acct_ids = accounts_features["acct_id"].values
    acct_to_idx = {acct: i for i, acct in enumerate(acct_ids)}
    print(f"Built node index for {len(acct_to_idx)} accounts.")
    return acct_to_idx


def build_edges(transactions: pd.DataFrame, acct_to_idx: dict):
    """Convert transactions into an edge_index tensor of node indices."""
    print("Building edges from transactions ...")

    # Drop transactions referencing accounts not in our node set (shouldn't
    # normally happen, but guards against orphan references)
    before = len(transactions)
    transactions = transactions[
        transactions["orig_acct"].isin(acct_to_idx) & transactions["bene_acct"].isin(acct_to_idx)
    ]
    dropped = before - len(transactions)
    if dropped > 0:
        print(f"  Dropped {dropped} transactions referencing unknown accounts")

    src = transactions["orig_acct"].map(acct_to_idx).values
    dst = transactions["bene_acct"].map(acct_to_idx).values

    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)

    # Edge attribute: transaction amount (log-scaled to tame skew)
    edge_attr = torch.tensor(
        np.log1p(transactions["base_amt"].values), dtype=torch.float
    ).unsqueeze(1)

    print(f"  Built {edge_index.shape[1]} directed edges")
    return edge_index, edge_attr


def build_node_features(accounts_features: pd.DataFrame):
    """Build the node feature matrix X, excluding acct_id and label_* columns."""
    label_cols = [c for c in accounts_features.columns if c.startswith("label_")]
    feature_cols = [c for c in accounts_features.columns if c not in label_cols + ["acct_id"]]

    print(f"Node feature columns ({len(feature_cols)}): {feature_cols}")
    print(f"Label columns (kept separate, used only as y): {label_cols}")

    X = accounts_features[feature_cols].copy()

    # Encode categorical columns
    categorical_cols = X.select_dtypes(include="object").columns.tolist()
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # Scale all features (GNNs are sensitive to feature scale)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    x = torch.tensor(X_scaled, dtype=torch.float)
    y = torch.tensor(accounts_features["label_is_flagged_account"].values, dtype=torch.long)

    print(f"  Node feature matrix shape: {x.shape}")
    print(f"  Label vector shape: {y.shape}, positive rate: {y.float().mean():.4f}")

    return x, y, feature_cols


def build_masks(y: torch.Tensor, n_nodes: int):
    """Stratified train/val/test masks — 60/20/20 split preserving positive rate."""
    indices = np.arange(n_nodes)
    labels = y.numpy()

    train_idx, temp_idx = train_test_split(
        indices, test_size=0.4, stratify=labels, random_state=RANDOM_SEED
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=labels[temp_idx], random_state=RANDOM_SEED
    )

    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    print(f"\nTrain nodes: {train_mask.sum().item()} ({labels[train_idx].sum()} positive)")
    print(f"Val nodes:   {val_mask.sum().item()} ({labels[val_idx].sum()} positive)")
    print(f"Test nodes:  {test_mask.sum().item()} ({labels[test_idx].sum()} positive)")

    return train_mask, val_mask, test_mask


def main():
    accounts_features, transactions = load_data()

    acct_to_idx = build_node_index(accounts_features)
    print("CHECKPOINT: acct_to_idx = build_node_index(...) done")
    edge_index, edge_attr = build_edges(transactions, acct_to_idx)
    print("CHECKPOINT: edge_index, edge_attr = build_edges(...) done")
    x, y, feature_cols = build_node_features(accounts_features)
    print("CHECKPOINT: x, y, feature_cols = build_node_features(...) done")
    train_mask, val_mask, test_mask = build_masks(y, n_nodes=x.shape[0])
    print("CHECKPOINT: train_mask, val_mask, test_mask = build_masks(...) done")

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    print("CHECKPOINT: data = Data(...) done")

    print("\nFinal graph object:")
    print(data)

    out_path = f"{GRAPH_DIR}/aml_graph.pt"
    torch.save(data, out_path)
    print("CHECKPOINT: torch.save(data, out_path) done")
    print(f"\nSaved graph to {out_path}")
    print("Done. Ready for GraphSAGE training.")


if __name__ == "__main__":
    main()
