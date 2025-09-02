"""
ui/diagnostics_ui.py
--------------------
Lightweight diagnostics panel (intended for the Streamlit **sidebar**).

Responsibilities
- Check directory existence and readability for models/scalers/output
- Load model/scaler inventory (cached) and show health
- Surface remediation tips when something is missing
"""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import streamlit as st

from core import model_manager


# ----------------------------
# Cached loaders
# ----------------------------

@st.cache_resource(show_spinner=False)
def _cached_inventory(models_dir: str, scalers_dir: str):
    return model_manager.load_all_models(models_dir=models_dir, scalers_dir=scalers_dir)


# ----------------------------
# Public rendering
# ----------------------------

def render_sidebar_diagnostics(
    *,
    models_dir: str = "models/trained_models",
    scalers_dir: str = "models/scalers",
    output_dir: str = "outputs/reports",
) -> Dict[str, Any]:
    """
    Render a compact diagnostics block in the sidebar.
    Returns diagnostics dict from model_manager.
    """
    st.sidebar.markdown("### 🔎 Diagnostics")

    # Paths status
    _path_badge("Models dir", models_dir, is_dir=True)
    _path_badge("Scalers dir", scalers_dir, is_dir=True)
    _path_badge("Output dir", output_dir, is_dir=True, create_if_missing=True)

    # Inventory (cached)
    models, scalers, diags = _cached_inventory(models_dir, scalers_dir)

    # Models summary
    st.sidebar.markdown("**Models**")
    if models:
        st.sidebar.write(", ".join(sorted(models.keys())))
    else:
        st.sidebar.caption("_No models loaded_")

    # Scalers summary
    st.sidebar.markdown("**Scalers**")
    if scalers:
        st.sidebar.write(", ".join(sorted(scalers.keys())))
    else:
        st.sidebar.caption("_No scalers loaded_")

    # Diagnostics details
    if diags.get("errors"):
        st.sidebar.error("Errors: " + " | ".join(map(str, diags["errors"])))
    if diags.get("warnings"):
        st.sidebar.warning(
            "Warnings: " + " | ".join(map(str, diags["warnings"])))
    if diags.get("missing"):
        st.sidebar.info("Missing: " + " | ".join(map(str, diags["missing"])))
    if diags.get("versions"):
        st.sidebar.caption(
            "Versions: " + ", ".join([f"{k} {v}" for k, v in diags["versions"].items()]))

    # Tips
    _remediation_tips(diags, models_dir, scalers_dir)

    st.sidebar.divider()
    return diags


# ----------------------------
# Helpers
# ----------------------------

def _path_badge(label: str, path: str, *, is_dir: bool, create_if_missing: bool = False) -> None:
    exists = os.path.isdir(path) if is_dir else os.path.exists(path)
    if not exists and create_if_missing and is_dir:
        try:
            os.makedirs(path, exist_ok=True)
            exists = True
        except Exception:
            pass

    icon = "✅" if exists else "❌"
    st.sidebar.caption(f"{icon} {label}: `{path}`")


def _remediation_tips(diags: Dict[str, Any], models_dir: str, scalers_dir: str) -> None:
    """Show friendly next steps if common items are missing."""
    missing = set(map(str, diags.get("missing") or []))
    warnings = " ".join(map(str, diags.get("warnings") or []))

    tips = []

    # Models folder empty
    if any("models_dir not found" in m for m in missing) or "No models loaded" in " ".join(missing):
        tips.append(
            f"Place trained models in `{models_dir}` (e.g., `tabpfn_model.pkl`, `attention_model.pth`).")

    # Scalers missing
    if any("scaler not found" in m for m in missing):
        tips.append(
            f"Place scalers in `{scalers_dir}` (e.g., `chest_scaler.pkl`, `wrist_scaler.pkl`, `demo_scaler.pkl`, `label_encoder.pkl`).")

    # Package availability
    if "TabPFN model present but 'tabpfn' package not installed." in warnings:
        tips.append("Install TabPFN: `pip install tabpfn`.")
    if "Attention model present but 'torch' not installed." in warnings:
        tips.append(
            "Install PyTorch compatible with your system (CPU is fine): see pytorch.org for the exact command.")

    if tips:
        with st.sidebar.expander("How to fix", expanded=False):
            for t in tips:
                st.sidebar.write("• " + t)
