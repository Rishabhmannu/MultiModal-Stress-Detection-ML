"""
ui/display_handlers.py
----------------------
UI glue that connects reusable components to core orchestration.

Responsibilities
- Render dataset overview (summary, subjects, basic checks)
- Render model diagnostics (inventory, versions, warnings)
- Drive a single-subject analysis run (progress, timings, downloads)
- Keep business logic in core/; keep presentation in ui/components.py

This module does *not* hold model logic. It delegates to:
  - core.data_manager
  - core.model_manager
  - core.report_orchestrator
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from . import components as C

# core modules
from core import data_manager
from core import model_manager
from core.report_orchestrator import (
    analyze_and_generate,
    OrchestratorConfig,
    OrchestratorResult,
)


# ----------------------------
# Dataset page
# ----------------------------

def render_dataset_overview(df) -> Tuple[Dict[str, Any], List[str], Optional[str]]:
    """
    Render the dataset summary and subject picker.
    Returns: (summary_dict, subjects_list, selected_subject_or_None)
    """
    if df is None or len(df) == 0:
        C.warning("No dataset loaded yet. Please upload a processed WESAD CSV.")
        return {}, [], None

    # Validate + summarize
    findings = data_manager.validate_dataset(df)
    summary = data_manager.summarize_dataset(df)

    # Header + metrics
    C.section("Dataset Overview", "Quick summary of the uploaded CSV.")
    C.dataset_metrics_tiles(summary)
    if summary.get("conditions"):
        C.conditions_pill_counts(summary["conditions"])

    # Show validation messages
    if findings.get("errors"):
        C.error(" • ".join(findings["errors"]))
    if findings.get("warnings"):
        C.warning(" • ".join(findings["warnings"]))

    # Subjects
    C.sub_section("Subjects")
    subjects = data_manager.list_subjects(df)
    selected = C.subject_selector(
        subjects, key="subject_select", label="Select a subject to analyze")
    return summary, subjects, selected


# ----------------------------
# Models & diagnostics
# ----------------------------

@st.cache_resource(show_spinner=False)
def _cached_load_models(models_dir: str, scalers_dir: str):
    return model_manager.load_all_models(models_dir=models_dir, scalers_dir=scalers_dir)


def render_models_diagnostics(models_dir: str, scalers_dir: str) -> Dict[str, Any]:
    """
    Load (cached) model/scaler inventory for display only.
    Returns diagnostics dict.
    """
    C.section("Models & Scalers",
              "Inventory and health checks for loaded artifacts.")
    models, scalers, diags = _cached_load_models(models_dir, scalers_dir)

    # Inventory badges
    C.model_inventory_badges(models_present=models.keys(
    ), attention_available=("attention" in models))

    # Diagnostics list
    C.diagnostics_list(diags)

    # Small details
    with st.expander("Scalers loaded", expanded=False):
        if scalers:
            st.write(sorted(list(scalers.keys())))
        else:
            st.caption("No scalers loaded.")

    return diags


# ----------------------------
# Analyze + Report
# ----------------------------

def render_analysis_runner(
    df,
    subject_id: Optional[str],
    cfg: OrchestratorConfig,
    *,
    show_controls: bool = True,
) -> Optional[OrchestratorResult]:
    """
    Render the controls to run analysis and generate a report for the selected subject.
    Returns OrchestratorResult (or None if not run).
    """
    if df is None or len(df) == 0:
        return None
    if not subject_id:
        C.info("Pick a subject above to enable analysis.")
        return None

    C.section("Analyze & Generate Report",
              "Runs preprocessing → inference → 4-page PDF.")
    # Optional UI toggles (caller may choose to hide these and set cfg from sidebar)
    if show_controls:
        # Primary model + toggles
        st.caption("Configuration")
        cols = st.columns([1, 1, 2])
        with cols[0]:
            primary = C.primary_model_selector(
                default=cfg.primary_model, key="primary_model_sel")
        with cols[1]:
            use_attn, include_legacy = C.toggles_row(
                use_attention_default=cfg.use_attention,
                include_legacy_default=cfg.include_legacy_models,
                key_prefix="runner",
            )
        # Mutate a shallow copy of cfg for this run
        run_cfg = OrchestratorConfig(
            primary_model=primary,
            use_attention=use_attn,
            include_legacy_models=include_legacy,
            models_dir=cfg.models_dir,
            scalers_dir=cfg.scalers_dir,
            output_dir=cfg.output_dir,
            figures_subdir=cfg.figures_subdir,
            enable_pdf_validation=cfg.enable_pdf_validation,
            page_style=cfg.page_style,
            file_prefix=cfg.file_prefix,
            class_names=cfg.class_names,
        )
    else:
        run_cfg = cfg

    # Run button
    btn_label = f"Analyze & Generate report for {subject_id}"
    run_clicked = C.run_button(btn_label, key="analyze_btn")
    if not run_clicked:
        return None

    # Execute with progress indicator
    result: Optional[OrchestratorResult] = None
    with C.Progress() as pg:
        try:
            pg.update(0.10, "Slicing subject & reading metadata…")
            # (Slicing done inside orchestrator; UI progress is illustrative)
            pg.update(0.30, "Loading models & scalers…")
            pg.update(0.50, "Preprocessing & feature scaling…")
            pg.update(0.75, "Running predictions & building adapter…")
            result = analyze_and_generate(
                dataset_df=df, subject_id=subject_id, cfg=run_cfg)
            pg.update(1.00, "Report generated.")
        except Exception as e:
            C.error(f"Run failed: {e}")
            return None

    if result is None:
        C.error("No result returned.")
        return None

    # Surface results
    if result.ok:
        C.success("Analysis complete.")
        C.prediction_summary(result.adapter_meta)
        C.timings_table(result.timings_sec)
        if result.warnings:
            with st.expander("Warnings", expanded=False):
                for w in result.warnings:
                    st.write("• " + str(w))
        if result.errors:
            with st.expander("Errors", expanded=False):
                for er in result.errors:
                    st.write("• " + str(er))
        # Download
        C.pdf_download_button(result.report_path, label="Download PDF report")
        # Minimal log download (if available)
        if result.logs_path:
            try:
                with open(result.logs_path, "rb") as f:
                    st.download_button(
                        "Download run log",
                        data=f.read(),
                        file_name=result.logs_path.split("/")[-1],
                        mime="application/json",
                    )
            except Exception:
                pass
    else:
        C.error("Analysis failed.")
        if result.errors:
            with st.expander("Errors", expanded=True):
                for er in result.errors:
                    st.write("• " + str(er))
        if result.warnings:
            with st.expander("Warnings", expanded=True):
                for w in result.warnings:
                    st.write("• " + str(w))

    return result
