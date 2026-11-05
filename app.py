"""
AMLGuard AI — Anti-Money Laundering Transaction Monitoring Workbench
Interactive Web Application & Compliance Investigation Console
Author: Raghav Krishan Aggarwal
"""

import streamlit as st
import os
import textwrap

# 1. Page Configuration (must be called first)
st.set_page_config(
    page_title="AMLGuard AI — Transaction Monitoring Workbench",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.app.config import APP_VERSION
from src.app.theme import apply_theme, render_header
from src.app.views.investigation import render_investigation_view
from src.app.views.benchmarks import render_benchmarks_view
from src.app.views.about import render_about_view


def main():
    # 2. Inject Custom Theme & Design System
    apply_theme()

    # 3. Sidebar Navigation & System Telemetry
    with st.sidebar:
        header_html = textwrap.dedent(f"""
        <div style="padding: 0.5rem 0 1rem 0;">
            <div style="font-size: 1.35rem; font-weight: 700; color: #ffffff; letter-spacing: -0.02em;">
                🛡️ AMLGuard AI
            </div>
            <div style="font-size: 0.78rem; color: #64748b; font-family: 'JetBrains Mono', monospace; margin-top: 0.2rem;">
                {APP_VERSION} · Research Prototype
            </div>
        </div>
        """).strip()
        st.markdown(header_html, unsafe_allow_html=True)

        st.markdown("### Navigation")
        nav_choice = st.radio(
            "Go to",
            [
                "🔍 Case Investigation Console",
                "📊 Model Performance Analytics",
                "🏛️ Architecture & Governance",
            ],
            index=0,
            label_visibility="collapsed"
        )

        st.markdown("---")

        st.markdown("### System Telemetry")
        telemetry_html = textwrap.dedent("""
        <div style="background: #111827; border: 1px solid #1f293d; border-radius: 10px; padding: 0.85rem; font-size: 0.78rem;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.4rem;">
                <span style="color: #94a3b8;">Tabular Engines:</span>
                <span style="color: #10b981; font-weight: 600;">RF, XGB, LogReg</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.4rem;">
                <span style="color: #94a3b8;">Graph Engines:</span>
                <span style="color: #8b5cf6; font-weight: 600;">GraphSAGE, GAT</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.4rem;">
                <span style="color: #94a3b8;">Audit Explainers:</span>
                <span style="color: #3b82f6; font-weight: 600;">TreeSHAP, GNNExplainer</span>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span style="color: #94a3b8;">Governance:</span>
                <span style="color: #10b981; font-weight: 600;">SR 11-7-inspired</span>
            </div>
        </div>
        """).strip()
        st.markdown(telemetry_html, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        footer_html = textwrap.dedent(f"""
        <div style="font-size: 0.72rem; color: #64748b; line-height: 1.4;">
            <b>AMLGuard AI Platform</b><br>
            {APP_VERSION} · Research Prototype<br>
            Financial Crime Analytics Prototype
        </div>
        """).strip()
        st.markdown(footer_html, unsafe_allow_html=True)

    # 4. Top Header
    current_ds = "AMLSim"
    render_header(current_dataset=current_ds)

    # 5. Route to Selected View
    if nav_choice == "🔍 Case Investigation Console":
        render_investigation_view()
    elif nav_choice == "📊 Model Performance Analytics":
        render_benchmarks_view()
    elif nav_choice == "🏛️ Architecture & Governance":
        render_about_view()


if __name__ == "__main__":
    main()

