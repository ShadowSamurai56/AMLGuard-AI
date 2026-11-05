"""
AMLGuard AI — Custom Design System & CSS Styling
Professional FinTech / RegTech Dark & Light Theme
"""

import streamlit as st
import textwrap


def apply_theme():
    """Injects high-end, responsive custom styling into Streamlit."""
    custom_css = """
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&display=swap');

        /* Root color tokens */
        :root {
            --bg-main: #0b0f17;
            --bg-card: #111827;
            --bg-card-hover: #1e293b;
            --border-card: #1f293d;
            --border-subtle: #162032;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --brand-primary: #3b82f6;
            --brand-glow: rgba(59, 130, 246, 0.25);
            --danger: #ef4444;
            --warning: #f59e0b;
            --success: #10b981;
            --purple: #8b5cf6;
            --cyan: #06b6d4;
        }

        /* Streamlit global resets */
        html, body, [data-testid="stAppViewContainer"] {
            font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
            background-color: var(--bg-main) !important;
            color: var(--text-primary) !important;
        }

        /* Top header bar clean up */
        header[data-testid="stHeader"] {
            background-color: transparent !important;
        }

        /* Main container width */
        .block-container {
            max-width: 1440px !important;
            padding-top: 1.5rem !important;
            padding-bottom: 3rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }

        /* Navigation Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #0d121d !important;
            border-right: 1px solid var(--border-subtle) !important;
        }

        /* AMLGuard Brand Header */
        .brand-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 1.25rem 1.75rem;
            background: linear-gradient(135deg, #131d31 0%, #0d1424 100%);
            border: 1px solid #1e2d4a;
            border-radius: 14px;
            margin-bottom: 1.75rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        }

        .brand-title {
            font-size: 1.5rem;
            font-weight: 700;
            color: #ffffff;
            letter-spacing: -0.02em;
            display: flex;
            align-items: center;
            gap: 0.6rem;
        }

        .brand-pill {
            font-size: 0.75rem;
            font-family: 'JetBrains Mono', monospace;
            padding: 0.2rem 0.6rem;
            background: rgba(59, 130, 246, 0.15);
            color: #60a5fa;
            border: 1px solid rgba(59, 130, 246, 0.3);
            border-radius: 20px;
            font-weight: 600;
        }

        /* Metric / Summary Cards */
        .aml-card {
            background-color: var(--bg-card);
            border: 1px solid var(--border-card);
            border-radius: 12px;
            padding: 1.25rem 1.5rem;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.25);
            transition: all 0.2s ease-in-out;
        }

        .aml-card:hover {
            border-color: #2b3a55;
            background-color: var(--bg-card-hover);
        }

        .aml-card-title {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: var(--text-muted);
            font-weight: 600;
            margin-bottom: 0.4rem;
        }

        .aml-card-value {
            font-size: 1.6rem;
            font-weight: 700;
            color: #ffffff;
            font-family: 'JetBrains Mono', monospace;
        }

        .aml-card-subtext {
            font-size: 0.78rem;
            color: var(--text-secondary);
            margin-top: 0.35rem;
        }

        /* Risk Pill Badges */
        .risk-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.3rem 0.85rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            font-family: 'JetBrains Mono', monospace;
        }

        .badge-critical {
            background: rgba(239, 68, 68, 0.18);
            color: #f87171;
            border: 1px solid rgba(239, 68, 68, 0.45);
        }

        .badge-high {
            background: rgba(249, 115, 22, 0.18);
            color: #fb923c;
            border: 1px solid rgba(249, 115, 22, 0.45);
        }

        .badge-elevated {
            background: rgba(234, 179, 8, 0.18);
            color: #facc15;
            border: 1px solid rgba(234, 179, 8, 0.45);
        }

        .badge-low {
            background: rgba(16, 185, 129, 0.18);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.45);
        }

        .badge-typology {
            background: rgba(139, 92, 246, 0.18);
            color: #c084fc;
            border: 1px solid rgba(139, 92, 246, 0.45);
        }

        /* Model Consensus Score Bar */
        .consensus-container {
            background: #0f172a;
            border: 1px solid #1e293b;
            border-radius: 12px;
            padding: 1.25rem;
            margin-bottom: 1.5rem;
        }

        /* Tabs styling */
        button[data-baseweb="tab"] {
            font-family: 'DM Sans', sans-serif !important;
            font-weight: 600 !important;
            font-size: 0.92rem !important;
            color: var(--text-muted) !important;
            border-radius: 8px !important;
            padding: 0.5rem 1.25rem !important;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            background: rgba(59, 130, 246, 0.12) !important;
            color: #60a5fa !important;
            border: 1px solid rgba(59, 130, 246, 0.3) !important;
        }

        /* DataTables and Code */
        code {
            font-family: 'JetBrains Mono', monospace !important;
            background: #172033 !important;
            color: #93c5fd !important;
            padding: 0.15rem 0.4rem !important;
            border-radius: 4px !important;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


def render_header(current_dataset="AMLSim"):
    """Renders the top branding navigation bar."""
    ds_badge = "SYNTHETIC BANKING (AMLSim)" if current_dataset == "AMLSim" else "BITCOIN FORENSICS (Elliptic)"
    html = textwrap.dedent(f"""
    <div class="brand-header">
        <div>
            <div class="brand-title">
                🛡️ AMLGuard AI <span style="font-weight: 300; color: #94a3b8; font-size: 1.15rem;">| Transaction Monitoring Workbench</span>
            </div>
            <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 0.25rem;">
                Dual-Tier Relational AI: Tree Ensembles & Inductive Graph Neural Networks (GraphSAGE / GAT)
            </div>
        </div>
        <div style="display: flex; gap: 0.6rem; align-items: center;">
            <span class="brand-pill">{ds_badge}</span>
            <span class="brand-pill" style="border-color: #10b981; color: #34d399; background: rgba(16, 185, 129, 0.1);">LIVE DEMO</span>
        </div>
    </div>
    """).strip()
    st.markdown(html, unsafe_allow_html=True)

