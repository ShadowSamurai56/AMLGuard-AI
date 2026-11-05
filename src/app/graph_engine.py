"""
AMLGuard AI — Interactive Ego-Network Graph Engine (Refined & Clean)
Provides clean, readable radial transaction flows without clutter.
"""

import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import networkx as nx


def build_amlsim_ego_network(acct_id, transactions_df, features_df, alerts_df, max_counterparties=14, include_inter_edges=False):
    """
    Extracts a clean, legible 1-hop ego network for an account.
    Defaults to direct transactions to/from the target to avoid chaotic hairballs.
    Limits to top counterparties by transaction volume for clarity.
    """
    G = nx.DiGraph()

    # Find direct 1-hop transactions
    direct_tx = transactions_df[
        (transactions_df["orig_acct"] == acct_id) | (transactions_df["bene_acct"] == acct_id)
    ].copy()

    if direct_tx.empty:
        G.add_node(acct_id, role="Target", is_target=True, is_sar=0, typology="normal", label=f"Acct #{acct_id}", volume=0.0)
        return G

    # Group direct transactions by counterparty to find top volume partners
    direct_tx["counterparty"] = direct_tx.apply(
        lambda r: r["bene_acct"] if r["orig_acct"] == acct_id else r["orig_acct"], axis=1
    )
    cp_volumes = direct_tx.groupby("counterparty")["base_amt"].sum().sort_values(ascending=False)

    # Select top counterparties (ensuring any confirmed SAR neighbor is always kept)
    sar_lookup = dict(zip(features_df["acct_id"], features_df["label_is_flagged_account"]))
    alert_lookup = dict(zip(alerts_df["acct_id"], alerts_df["alert_type"]))

    sar_counterparties = [cp for cp in cp_volumes.index if sar_lookup.get(cp, 0) == 1]
    top_counterparties = list(cp_volumes.head(max_counterparties).index)
    selected_counterparties = set(top_counterparties).union(set(sar_counterparties[:4]))

    # Filter transactions to selected counterparties
    filtered_direct = direct_tx[direct_tx["counterparty"].isin(selected_counterparties)].copy()

    # Pre-build neighbor node set
    neighbors = set(selected_counterparties)
    all_nodes = neighbors.union({acct_id})

    # Add investigated central node
    is_target_sar = int(sar_lookup.get(acct_id, 0))
    G.add_node(
        acct_id,
        role="Investigated Target",
        is_target=True,
        is_sar=is_target_sar,
        typology=alert_lookup.get(acct_id, "normal"),
        label=f"Acct #{acct_id}",
        volume=float(cp_volumes.sum())
    )

    # Add neighbor nodes
    for cp in neighbors:
        is_sar = int(sar_lookup.get(cp, 0))
        vol = float(cp_volumes.get(cp, 0.0))
        role = "SAR Counterparty" if is_sar else "Peer Account"
        G.add_node(
            cp,
            role=role,
            is_target=False,
            is_sar=is_sar,
            typology=alert_lookup.get(cp, "normal"),
            label=f"Acct #{cp}",
            volume=vol
        )

    # Add direct edges
    edge_groups = filtered_direct.groupby(["orig_acct", "bene_acct"]).agg(
        total_amt=("base_amt", "sum"),
        tx_count=("base_amt", "count")
    ).reset_index()

    for _, row in edge_groups.iterrows():
        u, v = int(row["orig_acct"]), int(row["bene_acct"])
        G.add_edge(u, v, total_amt=float(row["total_amt"]), tx_count=int(row["tx_count"]), is_direct=True)

    # Optionally add inter-counterparty transactions if requested (e.g. for cycle inspection)
    if include_inter_edges and len(neighbors) <= 12:
        inter_tx = transactions_df[
            (transactions_df["orig_acct"].isin(neighbors)) & (transactions_df["bene_acct"].isin(neighbors))
        ]
        if not inter_tx.empty:
            inter_groups = inter_tx.groupby(["orig_acct", "bene_acct"]).agg(
                total_amt=("base_amt", "sum"),
                tx_count=("base_amt", "count")
            ).reset_index()
            for _, row in inter_groups.iterrows():
                u, v = int(row["orig_acct"]), int(row["bene_acct"])
                G.add_edge(u, v, total_amt=float(row["total_amt"]), tx_count=int(row["tx_count"]), is_direct=False)

    return G


def build_elliptic_ego_network(tx_id, edges_df, sample_features_df, max_counterparties=12):
    """Builds clean ego network for an Elliptic transaction."""
    G = nx.DiGraph()
    direct_edges = edges_df[(edges_df["from_tx"] == tx_id) | (edges_df["to_tx"] == tx_id)]

    if direct_edges.empty:
        G.add_node(tx_id, role="Target", is_target=True, is_sar=0, label=f"Tx #{tx_id}")
        return G

    # Limit to top counterparties
    direct_edges = direct_edges.head(max_counterparties)
    neighbors = set(direct_edges["from_tx"]).union(set(direct_edges["to_tx"]))
    class_lookup = dict(zip(sample_features_df["tx_id"], sample_features_df["class"]))

    for node in neighbors:
        is_target = (node == tx_id)
        cls_val = str(class_lookup.get(node, "2"))
        is_sar = 1 if cls_val == "1" else 0
        role = "Target (Investigated)" if is_target else ("Illicit Tx" if is_sar else "Licit Tx")
        G.add_node(node, role=role, is_target=is_target, is_sar=is_sar, label=f"Tx #{node}")

    for _, row in direct_edges.iterrows():
        G.add_edge(int(row["from_tx"]), int(row["to_tx"]), total_amt=1.0, tx_count=1, is_direct=True)

    return G


def compute_ego_layout(G, center_node):
    """
    Computes a clean, aesthetic circular radial layout centered at (0, 0).
    Senders on left arc, Receivers on right arc.
    """
    pos = {center_node: np.array([0.0, 0.0])}
    other_nodes = [n for n in G.nodes() if n != center_node]

    if not other_nodes:
        return pos

    senders = [n for n in other_nodes if G.has_edge(n, center_node) and not G.has_edge(center_node, n)]
    receivers = [n for n in other_nodes if G.has_edge(center_node, n) and not G.has_edge(n, center_node)]
    mutual = [n for n in other_nodes if n not in senders and n not in receivers]

    radius = 1.0

    # Inbound senders on left semi-circle: theta in [3pi/4, 5pi/4]
    if senders:
        thetas = np.linspace(math.pi * 0.72, math.pi * 1.28, len(senders))
        for i, n in enumerate(senders):
            pos[n] = np.array([radius * math.cos(thetas[i]), radius * math.sin(thetas[i])])

    # Outbound receivers on right semi-circle: theta in [-pi/3.2, pi/3.2]
    if receivers:
        thetas = np.linspace(-math.pi * 0.28, math.pi * 0.28, len(receivers))
        for i, n in enumerate(receivers):
            pos[n] = np.array([radius * math.cos(thetas[i]), radius * math.sin(thetas[i])])

    # Mutual / bi-directional nodes on top and bottom
    if mutual:
        for i, n in enumerate(mutual):
            angle = math.pi * 0.5 if i % 2 == 0 else -math.pi * 0.5
            pos[n] = np.array([0.2 * (i + 1) * math.cos(angle), radius * math.sin(angle)])

    missing = [n for n in G.nodes() if n not in pos]
    if missing:
        full_pos = nx.spring_layout(G, pos=pos, fixed=[center_node], k=0.8, seed=42)
        return full_pos

    return pos


def render_interactive_graph(G, target_id, dataset="AMLSim", gnn_edge_importance=None):
    """
    Renders an uncluttered, publication-quality Plotly graph.
    """
    if len(G.nodes) == 0:
        fig = go.Figure()
        fig.update_layout(title="No transaction graph available for this account.")
        return fig

    pos = compute_ego_layout(G, target_id)

    # 1. Edge traces
    edge_x = []
    edge_y = []
    edge_hover = []

    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]

        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

        amt_str = f"${data.get('total_amt', 0):,.0f}" if dataset == "AMLSim" else "BTC Transfer"
        tx_cnt = data.get("tx_count", 1)
        direction_str = f"#{u} &rarr; #{v}"
        hover = f"<b>Flow:</b> {direction_str}<br><b>Amount:</b> {amt_str}<br><b>Tx Count:</b> {tx_cnt}"
        if gnn_edge_importance and (u, v) in gnn_edge_importance:
            hover += f"<br><b>GNN Importance:</b> {gnn_edge_importance[(u, v)]:.3f}"

        edge_hover.extend([hover, hover, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1.2, color="rgba(100, 116, 139, 0.35)"),
        hoverinfo="text",
        text=edge_hover,
        mode="lines",
        showlegend=False
    )

    # 2. Node traces
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    node_size = []
    node_symbol = []
    node_line_color = []

    for n, data in G.nodes(data=True):
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)

        is_target = data.get("is_target", False)
        is_sar = data.get("is_sar", 0)
        role = data.get("role", "Peer")
        typology = data.get("typology", "normal")

        if is_target:
            color = "#3b82f6"  # Professional blue focus
            size = 26
            symbol = "diamond"
            line_c = "#ffffff"
        elif is_sar == 1:
            color = "#f43f5e"  # Subtle rose for known SAR
            size = 16
            symbol = "circle"
            line_c = "rgba(244, 63, 94, 0.4)"
        else:
            color = "#64748b"  # Muted slate for normal accounts
            size = 14
            symbol = "circle"
            line_c = "rgba(148, 163, 184, 0.3)"

        node_color.append(color)
        node_size.append(size)
        node_symbol.append(symbol)
        node_line_color.append(line_c)

        hover = (
            f"<b>{data.get('label', n)}</b><br>"
            f"<b>Role:</b> {role}<br>"
            f"<b>SAR Flag:</b> {'YES' if is_sar == 1 else 'NO'}<br>"
            f"<b>Typology:</b> {typology.upper()}<br>"
            f"<b>In / Out Degree:</b> {G.in_degree(n)} in / {G.out_degree(n)} out"
        )
        node_text.append(hover)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=[f"#{n}" if G.nodes[n].get("is_target") or G.nodes[n].get("is_sar") == 1 else "" for n in G.nodes()],
        textposition="top center",
        textfont=dict(color="#cbd5e1", size=10, family="JetBrains Mono, monospace"),
        hoverinfo="text",
        hovertext=node_text,
        marker=dict(
            symbol=node_symbol,
            color=node_color,
            size=node_size,
            line=dict(color=node_line_color, width=1.5)
        ),
        showlegend=False
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        paper_bgcolor="#0b0f17",
        plot_bgcolor="#0b0f17",
        margin=dict(l=15, r=15, t=15, b=15),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=420,
        hoverlabel=dict(
            bgcolor="#1e293b",
            font_size=12,
            font_family="DM Sans, sans-serif",
            font_color="#f8fafc"
        )
    )

    return fig
