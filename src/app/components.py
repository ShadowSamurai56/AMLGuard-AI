"""
AMLGuard AI — UI Component Library (Refined Enterprise Design)
Clean, calm, high-contrast aesthetics without visual fatigue.
"""

import streamlit as st
import pandas as pd
import textwrap
from src.app.config import RISK_LEVELS, TYPOLOGY_INFO


def render_kpi_bar(acct_id, ground_truth, typology, consensus_prob, risk_level, disposition, total_volume, tx_count, divergence_note=None, split="TEST"):
    """Renders the executive summary card with refined enterprise badges and consensus metrics."""
    cfg = RISK_LEVELS.get(risk_level, RISK_LEVELS["LOW"])
    typ_cfg = TYPOLOGY_INFO.get(typology, TYPOLOGY_INFO["normal"])
    
    # Ground truth indicator
    sar_badge = (
        '<span style="background: rgba(244, 63, 94, 0.12); color: #fb7185; border: 1px solid rgba(244, 63, 94, 0.3); font-size: 0.72rem; font-weight: 600; padding: 0.2rem 0.6rem; border-radius: 4px; font-family: \'JetBrains Mono\', monospace;">CONFIRMED SAR</span>'
        if ground_truth == 1
        else '<span style="background: rgba(16, 185, 129, 0.1); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.25); font-size: 0.72rem; font-weight: 600; padding: 0.2rem 0.6rem; border-radius: 4px; font-family: \'JetBrains Mono\', monospace;">VERIFIED NORMAL</span>'
    )

    # Split indicator
    if split == "TEST":
        split_badge = '<span style="background: rgba(59, 130, 246, 0.12); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); font-size: 0.72rem; font-weight: 600; padding: 0.2rem 0.6rem; border-radius: 4px; font-family: \'JetBrains Mono\', monospace;">HELD-OUT TEST SET</span>'
    elif split == "VAL":
        split_badge = '<span style="background: rgba(148, 163, 184, 0.12); color: #94a3b8; border: 1px solid rgba(148, 163, 184, 0.25); font-size: 0.72rem; font-weight: 600; padding: 0.2rem 0.6rem; border-radius: 4px; font-family: \'JetBrains Mono\', monospace;">VALIDATION SET</span>'
    else:
        split_badge = '<span style="background: rgba(148, 163, 184, 0.12); color: #94a3b8; border: 1px solid rgba(148, 163, 184, 0.25); font-size: 0.72rem; font-weight: 600; padding: 0.2rem 0.6rem; border-radius: 4px; font-family: \'JetBrains Mono\', monospace;">TRAINING SET</span>'

    # Disposition styling
    if "ESCALATE" in disposition:
        disp_style = "background: rgba(244, 63, 94, 0.15); color: #f43f5e; border: 1px solid rgba(244, 63, 94, 0.35);"
    elif "REVIEW" in disposition:
        disp_style = "background: rgba(234, 179, 8, 0.15); color: #facc15; border: 1px solid rgba(234, 179, 8, 0.35);"
    else:
        disp_style = "background: rgba(16, 185, 129, 0.12); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3);"

    disp_badge = f'<span style="{disp_style} font-size: 0.72rem; font-weight: 600; padding: 0.2rem 0.6rem; border-radius: 4px; font-family: \'JetBrains Mono\', monospace;">{disposition}</span>'
    typology_badge = f'<span style="background: rgba(139, 92, 246, 0.12); color: #c084fc; border: 1px solid rgba(139, 92, 246, 0.3); font-size: 0.72rem; font-weight: 600; padding: 0.2rem 0.6rem; border-radius: 4px; font-family: \'JetBrains Mono\', monospace;">{typ_cfg["badge"]}</span>'

    note_html = ""
    if divergence_note:
        note_html = f'<div style="margin-top: 0.65rem; padding: 0.5rem 0.75rem; background: rgba(59, 130, 246, 0.08); border-left: 3px solid #3b82f6; border-radius: 4px; font-size: 0.78rem; color: #93c5fd;">ℹ️ {divergence_note}</div>'

    html = textwrap.dedent(f"""<div style="background: #0f172a; border: 1px solid #1e293b; border-radius: 12px; padding: 1.25rem 1.5rem; margin-bottom: 1.5rem;">
<div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 1rem;">
<div>
<div style="font-size: 0.7rem; text-transform: uppercase; color: #64748b; font-weight: 700; letter-spacing: 0.06em;">MONITORED ENTITY DOSSIER</div>
<div style="font-size: 1.65rem; font-weight: 700; color: #f8fafc; font-family: 'JetBrains Mono', monospace; margin: 0.15rem 0;">ACCOUNT #{acct_id}</div>
<div style="display: flex; gap: 0.45rem; margin-top: 0.35rem; align-items: center; flex-wrap: wrap;">
{sar_badge}
{split_badge}
{disp_badge}
{typology_badge}
</div>
</div>
<div style="display: flex; gap: 2.2rem; text-align: right; flex-wrap: wrap;">
<div>
<div style="font-size: 0.7rem; color: #64748b; text-transform: uppercase; font-weight: 600; letter-spacing: 0.04em;" title="Heuristic weights set from validation AUC-PR ranking: RF (35%), XGB (35%), GraphSAGE (20%), GAT (5%), LogReg (5%)">CONSENSUS RISK SCORE <span style="font-size: 0.72rem; color: #94a3b8; cursor: help;">ⓘ</span></div>
<div style="font-size: 1.55rem; font-weight: 700; color: #f8fafc; font-family: 'JetBrains Mono', monospace;">{consensus_prob * 100:.1f} <span style="font-size: 0.85rem; color: #64748b; font-weight: 500;">/ 100</span></div>
<div style="font-size: 0.66rem; color: #64748b; margin-top: 1px;">Weights set from validation AUC-PR ranking</div>
</div>
<div>
<div style="font-size: 0.7rem; color: #64748b; text-transform: uppercase; font-weight: 600; letter-spacing: 0.04em;">TOTAL GROSS FLOW</div>
<div style="font-size: 1.55rem; font-weight: 700; color: #f8fafc; font-family: 'JetBrains Mono', monospace;">${total_volume:,.0f}</div>
</div>
<div>
<div style="font-size: 0.7rem; color: #64748b; text-transform: uppercase; font-weight: 600; letter-spacing: 0.04em;">TRANSACTION EVENTS</div>
<div style="font-size: 1.55rem; font-weight: 700; color: #f8fafc; font-family: 'JetBrains Mono', monospace;">{tx_count}</div>
</div>
</div>
</div>
{note_html}
<div style="margin-top: 0.65rem; padding-top: 0.65rem; border-top: 1px solid #1e293b; font-size: 0.8rem; color: #94a3b8;">
<b>Typology Classification:</b> {typ_cfg['description']}
</div>
</div>""").strip()
    st.markdown(html, unsafe_allow_html=True)


def render_model_comparison_cards(scores, ground_truth):
    """Renders sleek, clean cards with high-contrast text and subtle accents."""
    cols = st.columns(len(scores))

    for idx, (model_name, prob) in enumerate(scores.items()):
        pct = prob * 100
        
        # Subtle status indicator
        if prob >= 0.5:
            badge_color = "#fb7185"
            badge_bg = "rgba(244, 63, 94, 0.12)"
            badge_border = "rgba(244, 63, 94, 0.3)"
            badge_text = "FLAGGED"
            bar_color = "#f43f5e"
        elif prob >= 0.25:
            badge_color = "#facc15"
            badge_bg = "rgba(234, 179, 8, 0.12)"
            badge_border = "rgba(234, 179, 8, 0.3)"
            badge_text = "REVIEW"
            bar_color = "#eab308"
        else:
            badge_color = "#34d399"
            badge_bg = "rgba(16, 185, 129, 0.12)"
            badge_border = "rgba(16, 185, 129, 0.3)"
            badge_text = "CLEAR"
            bar_color = "#10b981"

        if "Forest" in model_name or "XGBoost" in model_name:
            family = "Tabular Ensemble"
            icon = "🌲"
        elif "SAGE" in model_name or "GAT" in model_name:
            family = "Graph Neural Net"
            icon = "🕸️"
        else:
            family = "Linear Baseline"
            icon = "📈"

        with cols[idx]:
            card_html = textwrap.dedent(f"""<div style="background: #0f172a; border: 1px solid #1e293b; border-radius: 10px; padding: 1rem; height: 100%;">
<div style="display: flex; justify-content: space-between; align-items: center;">
<span style="font-size: 0.68rem; color: #64748b; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em;">{family}</span>
<span style="font-size: 0.68rem; font-family: 'JetBrains Mono', monospace; font-weight: 600; padding: 0.15rem 0.45rem; border-radius: 4px; background: {badge_bg}; color: {badge_color}; border: 1px solid {badge_border};">{badge_text}</span>
</div>
<div style="font-size: 0.95rem; font-weight: 600; color: #e2e8f0; margin-top: 0.3rem;">
{icon} {model_name}
</div>
<div style="margin-top: 0.5rem;">
<div style="font-size: 1.65rem; font-weight: 700; color: #f8fafc; font-family: 'JetBrains Mono', monospace; line-height: 1.1;">{pct:.1f}%</div>
<div style="font-size: 0.72rem; color: #64748b; margin-top: 0.15rem;">Fraud Probability</div>
</div>
<div style="margin-top: 0.65rem; background: #1e293b; border-radius: 9999px; height: 4px; overflow: hidden;">
<div style="background: {bar_color}; width: {min(pct, 100)}%; height: 100%; border-radius: 9999px;"></div>
</div>
</div>""").strip()
            st.markdown(card_html, unsafe_allow_html=True)


def render_transaction_ledger(transactions_df, acct_id, limit=8):
    """Renders a sleek transaction ledger table."""
    direct_tx = transactions_df[
        (transactions_df["orig_acct"] == acct_id) | (transactions_df["bene_acct"] == acct_id)
    ].copy()

    if direct_tx.empty:
        st.info("No atomic transactions recorded for this account.")
        return

    direct_tx["Direction"] = direct_tx.apply(
        lambda r: "OUTBOUND ↗" if r["orig_acct"] == acct_id else "INBOUND ↙", axis=1
    )
    direct_tx["Counterparty"] = direct_tx.apply(
        lambda r: f"Acct #{r['bene_acct']}" if r["orig_acct"] == acct_id else f"Acct #{r['orig_acct']}", axis=1
    )
    direct_tx["Amount ($)"] = direct_tx["base_amt"].apply(lambda v: f"${v:,.2f}")
    direct_tx["Timestamp / Step"] = direct_tx.get("tran_timestamp", direct_tx.get("step", "N/A"))
    direct_tx["SAR Flag"] = direct_tx.get("is_sar", 0).apply(lambda v: "YES" if v == 1 else "NO")

    display_df = direct_tx[["Direction", "Counterparty", "Amount ($)", "Timestamp / Step", "SAR Flag"]].head(limit)
    st.dataframe(display_df, use_container_width=True, hide_index=True)
