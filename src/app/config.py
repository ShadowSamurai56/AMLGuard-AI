"""
AMLGuard AI — App Configuration & Constants
"""

import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS_DIR = os.path.join(ROOT_DIR, "models")
DATA_DIR = os.path.join(ROOT_DIR, "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

# Database configuration (with graceful fallback to cached parquet)
DB_CONFIG = {
    "user": "aml_user",
    "password": "yourpassword",
    "host": "localhost",
    "database": "amlguard",
}

# Feature display mappings for AMLSim
FEATURE_NAMES_AMLSIM = [
    "type",
    "out_tx_count",
    "out_total_amt",
    "out_avg_amt",
    "out_max_amt",
    "out_std_amt",
    "out_unique_counterparties",
    "in_tx_count",
    "in_total_amt",
    "in_avg_amt",
    "in_max_amt",
    "in_std_amt",
    "in_unique_counterparties",
    "avg_hours_between_out_tx",
    "total_tx_count",
    "net_flow",
    "total_unique_counterparties",
]

FEATURE_LABELS_AMLSIM = {
    "type": "Account Type (Retail/Biz)",
    "out_tx_count": "Outbound Tx Count",
    "out_total_amt": "Total Outbound ($)",
    "out_avg_amt": "Avg Outbound Amount ($)",
    "out_max_amt": "Max Outbound Amount ($)",
    "out_std_amt": "Std Dev Outbound ($)",
    "out_unique_counterparties": "Unique Receivers",
    "in_tx_count": "Inbound Tx Count",
    "in_total_amt": "Total Inbound ($)",
    "in_avg_amt": "Avg Inbound Amount ($)",
    "in_max_amt": "Max Inbound Amount ($)",
    "in_std_amt": "Std Dev Inbound ($)",
    "in_unique_counterparties": "Unique Senders",
    "avg_hours_between_out_tx": "Avg Hours Between Tx",
    "total_tx_count": "Total Transaction Count",
    "net_flow": "Net Flow ($ In - Out)",
    "total_unique_counterparties": "Total Distinct Peers",
}

# Risk Thresholds
RISK_LEVELS = {
    "CRITICAL": {"min": 0.75, "color": "#ef4444", "bg": "rgba(239, 68, 68, 0.15)", "badge": "CRITICAL RISK"},
    "HIGH": {"min": 0.50, "color": "#f97316", "bg": "rgba(249, 115, 22, 0.15)", "badge": "HIGH RISK"},
    "ELEVATED": {"min": 0.25, "color": "#eab308", "bg": "rgba(234, 179, 8, 0.15)", "badge": "ELEVATED RISK"},
    "LOW": {"min": 0.00, "color": "#10b981", "bg": "rgba(16, 185, 129, 0.15)", "badge": "LOW RISK"},
}

# Typology Descriptions
TYPOLOGY_INFO = {
    "cycle": {
        "title": "Circular Fund Routing (Cycle)",
        "badge": "CYCLE",
        "color": "#a855f7",
        "description": "Funds leave an account and route through 3–5 intermediary shell accounts before returning to the originator to disguise origin.",
    },
    "fan_in": {
        "title": "Smurfing / Structuring (Fan-In)",
        "badge": "FAN-IN",
        "color": "#3b82f6",
        "description": "Multiple small deposits from disparate accounts consolidate rapidly into a single gathering account below reporting thresholds.",
    },
    "fan_out": {
        "title": "Layered Dispersion (Fan-Out)",
        "badge": "FAN-OUT",
        "color": "#06b6d4",
        "description": "Large inbound sum is rapidly fractured and dispersed across dozens of outbound recipient wallets/accounts to avoid tracing.",
    },
    "normal": {
        "title": "Standard Commercial Activity",
        "badge": "NORMAL",
        "color": "#64748b",
        "description": "Standard business or retail banking pattern with balanced inflows, typical transaction frequencies, and predictable counterparties.",
    },
}

