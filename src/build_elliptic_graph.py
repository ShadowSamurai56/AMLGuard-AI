"""
AMLGuard AI — Elliptic Dataset: Build Graph
----------------------------------------------
Loads the Elliptic Bitcoin dataset (already pre-featurized, no feature
engineering needed) and builds a PyTorch Geometric graph, ready for the
same GraphSAGE/GAT training loop already used on AMLSim data.

Files expected in elliptic_data/:
    elliptic_txs_features.csv   (203,769 rows x 167 cols: tx_id + 166 features)
    elliptic_txs_classes.csv    (tx_id, class: '1'=illicit, '2'=licit, 'unknown')
    elliptic_txs_edgelist.csv   (txId1, txId2 — directed edges)

IMPORTANT — Elliptic label convention:
    class '1' = illicit  -> we map to label 1 (positive / flagged)
    class '2' = licit    -> we map to label 0 (negative / normal)
    class 'unknown'      -> EXCLUDED from train/val/test masks entirely
                            (still present as graph nodes for message passing,
                            since dropping them would break the graph structure,
                            but their labels are never used for loss or eval)

Run this from your project root (AMLGuard-AI/), with your venv active:
    python -u src/build_elliptic_graph.py

Requires: pandas, torch, torch_geometric (all already installed)
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_DIR = "elliptic_data"
OUT_DIR = "models"
RANDOM_SEED = 42


def load_data():
    print("Loading Elliptic features (this is a large file, ~690MB — may take a bit) ...")
    # No header row in the official release; first col is tx_id, then 165 feature cols
    features = pd.read_csv(f"{DATA_DIR}/elliptic_txs_features.csv", header=None)
    n_features = features.shape[1] - 1
    features.columns = ["tx_id"] + [f"feat_{i}" for i in range(n_features)]
    print(f"  features: {features.shape}")

    classes = pd.read_csv(f"{DATA_DIR}/elliptic_txs_classes.csv")
    classes.columns = ["tx_id", "class"]
    print(f"  classes: {classes.shape}")
    print(f"  class value counts:\n{classes['class'].value_counts()}")

    edgelist = pd.read_csv(f"{DATA_DIR}/elliptic_txs_edgelist.csv")
    edgelist.columns = ["txId1", "txId2"]
    print(f"  edgelist: {edgelist.shape}")

    return features, classes, edgelist


def build_node_index(features: pd.DataFrame):
    tx_ids = features["tx_id"].values
    tx_to_idx = {tx: i for i, tx in enumerate(tx_ids)}
    print(f"Built node index for {len(tx_to_idx)} transactions.")
    return tx_to_idx


def build_edges(edgelist: pd.DataFrame, tx_to_idx: dict):
    print("Building edges ...")
    before = len(edgelist)
    edgelist = edgelist[
        edgelist["txId1"].isin(tx_to_idx) & edgelist["txId2"].isin(tx_to_idx)
    ]
    dropped = before - len(edgelist)
    if dropped > 0:
        print(f"  Dropped {dropped} edges referencing unknown transactions")

    src = edgelist["txId1"].map(tx_to_idx).values
    dst = edgelist["txId2"].map(tx_to_idx).values
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)

    print(f"  Built {edge_index.shape[1]} directed edges")
    return edge_index


def build_node_features_and_labels(features: pd.DataFrame, classes: pd.DataFrame, tx_to_idx: dict):
    print("Building node features and labels ...")

    # Merge, preserving the tx_id order used for node indices
    merged = features.merge(classes, on="tx_id", how="left")
    merged = merged.sort_values("tx_id", key=lambda s: s.map(tx_to_idx)).reset_index(drop=True)

    feature_cols = [c for c in merged.columns if c.startswith("feat_")]
    X = merged[feature_cols].values.astype(np.float32)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    x = torch.tensor(X_scaled, dtype=torch.float)

    # Map class labels: '1' -> 1 (illicit), '2' -> 0 (licit), 'unknown' -> -1 (excluded)
    label_map = {"1": 1, "2": 0, "unknown": -1}
    y_raw = merged["class"].astype(str).map(label_map).values
    y = torch.tensor(y_raw, dtype=torch.long)

    print(f"  Node feature matrix shape: {x.shape}")
    print(f"  Labeled illicit: {(y == 1).sum().item()}, licit: {(y == 0).sum().item()}, unknown: {(y == -1).sum().item()}")

    return x, y


def build_masks(y: torch.Tensor):
    """Only labeled nodes (y != -1) get train/val/test masks. Unknown nodes
    stay in the graph for message passing but are never used for loss/eval."""
    labeled_idx = (y != -1).nonzero(as_tuple=True)[0].numpy()
    labels = y[labeled_idx].numpy()

    train_idx, temp_idx = train_test_split(
        labeled_idx, test_size=0.4, stratify=labels, random_state=RANDOM_SEED
    )
    temp_labels = y[temp_idx].numpy()
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=temp_labels, random_state=RANDOM_SEED
    )

    n_nodes = y.shape[0]
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    print(f"\nTrain nodes: {train_mask.sum().item()} ({(y[train_idx] == 1).sum().item()} illicit)")
    print(f"Val nodes:   {val_mask.sum().item()} ({(y[val_idx] == 1).sum().item()} illicit)")
    print(f"Test nodes:  {test_mask.sum().item()} ({(y[test_idx] == 1).sum().item()} illicit)")

    return train_mask, val_mask, test_mask


def main():
    features, classes, edgelist = load_data()

    tx_to_idx = build_node_index(features)
    edge_index = build_edges(edgelist, tx_to_idx)
    x, y = build_node_features_and_labels(features, classes, tx_to_idx)
    train_mask, val_mask, test_mask = build_masks(y)

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )

    print("\nFinal Elliptic graph object:")
    print(data)

    out_path = f"{OUT_DIR}/elliptic_graph.pt"
    torch.save(data, out_path)
    print(f"\nSaved graph to {out_path}")
    print("Done. Ready for GraphSAGE/GAT training on Elliptic.")


if __name__ == "__main__":
    main()
