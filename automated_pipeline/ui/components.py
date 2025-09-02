"""
ui/components.py
----------------
Small, reusable Streamlit UI widgets and layout helpers.
Keep these presentational and state-light; business logic lives in core/ and ui/display_handlers.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import streamlit as st


# ----------------------------
# Layout & Headers
# ----------------------------

def render_app_header(title: str, subtitle: Optional[str] = None, icon: str = "🩺") -> None:
    """
    Render the app header with a subtle subtitle.
    """
    st.markdown(f"## {icon} {title}")
    if subtitle:
        st.caption(subtitle)
    st.divider()


def section(title: str, help_text: Optional[str] = None) -> None:
    """
    Section heading with optional help tooltip.
    """
    st.markdown(f"### {title}")
    if help_text:
        st.caption(help_text)


def sub_section(title: str) -> None:
    st.markdown(f"**{title}**")


def hr() -> None:
    st.markdown("---")


# ----------------------------
# Alerts & Status
# ----------------------------

def info(msg: str) -> None:
    st.info(msg, icon="ℹ️")


def warning(msg: str) -> None:
    st.warning(msg, icon="⚠️")


def error(msg: str) -> None:
    st.error(msg, icon="❌")


def success(msg: str) -> None:
    st.success(msg, icon="✅")


def debug_expander(title: str, content: Any) -> None:
    with st.expander(title):
        st.write(content)


# ----------------------------
# Dataset Widgets
# ----------------------------

def file_uploader_card(key: str = "csv_upload", label: str = "Upload processed WESAD CSV") -> Optional[bytes]:
    """
    Returns file bytes or None. Do not parse here; pass bytes to core.data_manager.load_csv.
    """
    st.markdown("#### Dataset")
    uploaded = st.file_uploader(label, type=["csv"], key=key)
    if uploaded is not None:
        st.caption(
            f"Selected: `{uploaded.name}` · {uploaded.size/1024:.1f} KB")
        return uploaded.getvalue()
    return None


def dataset_metrics_tiles(summary: Dict[str, Any]) -> None:
    """
    Show a compact row of dataset metrics.
    Expects keys: n_rows, n_cols, n_subjects, condition_col, subject_col
    """
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{summary.get('n_rows', 0):,}")
    c2.metric("Columns", f"{summary.get('n_cols', 0):,}")
    c3.metric("Subjects", f"{summary.get('n_subjects', 0):,}")
    cond = summary.get("condition_col") or "—"
    subj = summary.get("subject_col") or "—"
    c4.metric("Columns (ID / Label)", f"{subj} / {cond}")


def subject_selector(subjects: List[str], key: str = "subject_select", label: str = "Choose Subject") -> Optional[str]:
    """
    Return the chosen subject or None if list is empty.
    """
    if not subjects:
        warning("No subjects detected. Please upload a valid processed CSV.")
        return None
    default_idx = 0
    return st.selectbox(label, subjects, index=default_idx, key=key)


def conditions_pill_counts(counts: Dict[str, int]) -> None:
    """
    Render condition counts as pills.
    """
    if not counts:
        return
    st.markdown(
        " ".join(
            [f"<span style='padding:4px 8px;border-radius:999px;background:#F3F4F6;margin-right:6px;font-size:12px'>{k}: <b>{v}</b></span>"
             for k, v in counts.items()]
        ),
        unsafe_allow_html=True
    )


# ----------------------------
# Model & Diagnostics Widgets
# ----------------------------

def model_inventory_badges(models_present: Iterable[str], attention_available: bool) -> None:
    """
    Show which models were loaded.
    """
    tags = list(models_present) or []
    if attention_available and "attention" not in tags:
        tags.append("attention")
    if not tags:
        warning("No models loaded.")
        return
    st.markdown("**Models available:** " + " ".join(_pill(t)
                for t in tags), unsafe_allow_html=True)


def diagnostics_list(diags: Dict[str, Any]) -> None:
    """
    Render diagnostics dict from core.model_manager.load_all_models.
    """
    if not diags:
        return
    if diags.get("errors"):
        error("Model errors: " + "; ".join(map(str, diags["errors"])))
    if diags.get("warnings"):
        warning("Model warnings: " + "; ".join(map(str, diags["warnings"])))
    if diags.get("missing"):
        info("Missing: " + "; ".join(map(str, diags["missing"])))
    if diags.get("versions"):
        st.caption(
            "Versions: " + ", ".join([f"{k} {v}" for k, v in diags["versions"].items()]))


def primary_model_selector(default: str = "tabpfn", key: str = "primary_model") -> str:
    """
    Small selector for primary model. The caller can disable/enable options based on inventory.
    """
    return st.selectbox("Primary model", options=["tabpfn", "attention", "ensemble"], index=["tabpfn", "attention", "ensemble"].index(default), key=key)


def toggles_row(use_attention_default: bool = True, include_legacy_default: bool = False, key_prefix: str = "tog") -> Tuple[bool, bool]:
    cols = st.columns(2)
    use_attention = cols[0].toggle(
        "Use Attention insights", value=use_attention_default, key=f"{key_prefix}_attn")
    include_legacy = cols[1].toggle(
        "Include legacy models (ensemble)", value=include_legacy_default, key=f"{key_prefix}_legacy")
    return use_attention, include_legacy


# ----------------------------
# Actions & Progress
# ----------------------------

def run_button(label: str = "Analyze & Generate Report", key: str = "run") -> bool:
    return st.button(label, type="primary", key=key)


class Progress:
    """
    Lightweight progress/phase indicator:
        with Progress() as pg:
            pg.update(0.1, "Loading models")
            ...
            pg.update(1.0, "Done")
    """

    def __init__(self, key: str = "pg"):
        self._bar = None
        self._text = None
        self._key = key

    def __enter__(self):
        self._bar = st.progress(0, text="Starting…")
        self._text = st.empty()
        return self

    def update(self, fraction: float, message: str) -> None:
        fraction = max(0.0, min(1.0, float(fraction)))
        if self._bar is not None:
            self._bar.progress(fraction, text=message)
        if self._text is not None:
            self._text.caption(message)

    def __exit__(self, exc_type, exc, tb):
        if self._bar is not None:
            self._bar.empty()
        if self._text is not None:
            self._text.empty()


# ----------------------------
# Results & Downloads
# ----------------------------

def prediction_summary(adapter_meta: Dict[str, Any]) -> None:
    """
    Compact summary for predictions (uses adapter_meta returned by orchestrator).
    """
    if not adapter_meta:
        warning("No prediction metadata available.")
        return
    cols = st.columns(3)
    cols[0].metric("Primary model", str(
        adapter_meta.get("primary_model", "—")).upper())
    cols[1].metric("Windows analyzed", f"{adapter_meta.get('n_windows', 0)}")
    cols[2].metric("Attention", "Yes" if adapter_meta.get(
        "has_attention") else "No")
    st.caption("Models used: " +
               ", ".join(adapter_meta.get("models_present", []) or ["—"]))


def timings_table(timings: Dict[str, float]) -> None:
    if not timings:
        return
    st.markdown("**Timings (s)**")
    rows = "\n".join([f"- {k}: `{v:.2f}`" for k, v in timings.items()])
    st.markdown(rows)


def pdf_download_button(pdf_path: str, label: str = "Download PDF report") -> None:
    """
    Renders a download button if the file exists.
    """
    if not pdf_path:
        warning("Report path not available.")
        return
    p = Path(pdf_path)
    if not p.exists():
        error(f"Report not found at: {pdf_path}")
        return
    with p.open("rb") as f:
        st.download_button(label, data=f.read(),
                           file_name=p.name, mime="application/pdf")


# ----------------------------
# Tiny helpers
# ----------------------------

def _pill(text: str) -> str:
    return f"<span style='padding:3px 8px;border:1px solid #e5e7eb;border-radius:999px;background:#fafafa;margin-right:6px;font-size:12px'>{text}</span>"
