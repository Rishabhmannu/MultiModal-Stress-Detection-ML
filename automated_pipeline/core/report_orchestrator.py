"""
Report Orchestrator
-------------------
Bridges dataset slices, preprocessing, model inference, and the 4-page PDF generator.

Responsibilities
- Validate inputs and pick a subject
- Slice the dataset for that subject and extract metadata
- Prepare features with the same ordering/scalers as training
- Run predictions (TabPFN + optional Attention, and optional legacy ensemble)
- Build a model-agnostic "prediction adapter"
- Call the EnhancedReportGenerator to produce a 4-page clinical PDF
- Return paths, timings, and lightweight run logs to the caller (Streamlit UI)

This module assumes the following peer modules exist (provided in later steps):
- core/data_manager.py       -> load_csv(...), get_subject_df(...), get_subject_meta(...)
- core/preprocessing.py      -> prepare_subject_features(...)
- core/model_manager.py      -> load_all_models(...), predict_all(...)
- core/enhanced_report_generator.py -> EnhancedReportGenerator

Author: you + ChatGPT
"""

from __future__ import annotations

import os
import time
import json
import traceback
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

from .enhanced_report_generator import EnhancedReportGenerator, ReportStyle
# These will be implemented in subsequent files:
from . import data_manager
from . import preprocessing
from . import model_manager


# ----------------------------
# Config & result structures
# ----------------------------

@dataclass
class OrchestratorConfig:
    """User/system options that control a single report run."""
    primary_model: str = "tabpfn"          # 'tabpfn' | 'ensemble' | 'rf' (legacy) etc.
    use_attention: bool = True             # include attention insights if available
    # optional: include legacy models for comparison
    include_legacy_models: bool = False
    models_dir: str = "models/trained_models"
    scalers_dir: str = "models/scalers"
    output_dir: str = "outputs/reports"
    # where generator writes PNGs (under output_dir)
    figures_subdir: str = "figures"
    enable_pdf_validation: bool = True     # size/exists checks
    page_style: Dict[str, Any] = None      # optional overrides for ReportStyle
    file_prefix: Optional[str] = None      # override default filename
    # class name ordering must match label_encoder used during training
    class_names: Tuple[str, ...] = (
        "Baseline", "Stress", "Amusement", "Meditation")


@dataclass
class OrchestratorResult:
    """Return to UI after a run."""
    ok: bool
    report_path: Optional[str]
    adapter_meta: Dict[str, Any]
    timings_sec: Dict[str, float]
    warnings: Tuple[str, ...]
    errors: Tuple[str, ...]
    subject_id: str
    logs_path: Optional[str]


# ----------------------------
# Public API
# ----------------------------

def analyze_and_generate(
    dataset_df: pd.DataFrame,
    subject_id: str,
    cfg: OrchestratorConfig,
) -> OrchestratorResult:
    """
    Main entry point used by the Streamlit app.

    Parameters
    ----------
    dataset_df : DataFrame
        The full processed/integrated WESAD dataframe (all subjects).
    subject_id : str
        Subject identifier present in dataset_df (e.g., 'S4').
    cfg : OrchestratorConfig
        Settings (model choice, output dirs, etc.)

    Returns
    -------
    OrchestratorResult
    """
    t0 = time.time()
    timings: Dict[str, float] = {}
    warnings = []
    errors = []
    report_path = None
    logs_path = None
    adapter_meta: Dict[str, Any] = {}

    # 1) Slice subject & metadata
    try:
        t = time.time()
        subj_df = data_manager.get_subject_df(dataset_df, subject_id)
        subj_meta = data_manager.get_subject_meta(dataset_df, subject_id)
        _ensure_required_columns(subj_df)
        timings["slice_subject"] = time.time() - t
    except Exception as e:
        errors.append(f"Subject slicing failed: {e}")
        return OrchestratorResult(
            ok=False, report_path=None, adapter_meta={}, timings_sec=timings,
            warnings=tuple(warnings), errors=tuple(errors),
            subject_id=subject_id, logs_path=None
        )

    # 2) Load models & scalers
    try:
        t = time.time()
        models, scalers, model_diag = model_manager.load_all_models(
            models_dir=cfg.models_dir,
            scalers_dir=cfg.scalers_dir,
        )
        timings["load_models"] = time.time() - t

        # surface diagnostics as warnings (missing models/scalers, versions, etc.)
        for k in ("warnings", "missing", "errors"):
            if model_diag.get(k):
                msg = f"Model diagnostics ({k}): {model_diag[k]}"
                (warnings if k != "errors" else errors).append(msg)
    except Exception as e:
        errors.append(f"Model loading failed: {e}")
        return OrchestratorResult(
            ok=False, report_path=None, adapter_meta={}, timings_sec=timings,
            warnings=tuple(warnings), errors=tuple(errors),
            subject_id=subject_id, logs_path=None
        )

    # 3) Preprocess → features
    try:
        t = time.time()
        feat_pack = preprocessing.prepare_subject_features(
            subject_df=subj_df,
            scalers=scalers,
            class_names=list(cfg.class_names),
        )
        timings["preprocess_features"] = time.time() - t

        # sanity checks
        if not feat_pack.get("X_combined") is None:
            _validate_feature_shapes(feat_pack)
        else:
            warnings.append(
                "X_combined missing; predictions will rely on available modalities only.")
    except Exception as e:
        tb = traceback.format_exc(limit=1)
        errors.append(f"Preprocessing failed: {e} | {tb}")
        return OrchestratorResult(
            ok=False, report_path=None, adapter_meta={}, timings_sec=timings,
            warnings=tuple(warnings), errors=tuple(errors),
            subject_id=subject_id, logs_path=None
        )

    # 4) Predict → build model-agnostic adapter
    try:
        t = time.time()
        adapter = model_manager.predict_all(
            features=feat_pack,
            models=models,
            class_names=list(cfg.class_names),
            primary_model=cfg.primary_model,
            use_attention=cfg.use_attention,
            include_legacy=cfg.include_legacy_models,
        )
        timings["predict_and_adapter"] = time.time() - t

        # Adapter meta we surface back to UI/logs (no giant arrays)
        adapter_meta = {
            "primary_model": adapter.get("primary_model"),
            "n_windows": adapter.get("n_windows"),
            "class_names": adapter.get("class_names"),
            "meta": adapter.get("meta", {}),
            "has_attention": bool((adapter.get("interpretability") or {}).get("attention_summary")),
            "models_present": list((adapter.get("window_preds") or {}).keys()),
        }
    except Exception as e:
        tb = traceback.format_exc(limit=1)
        errors.append(f"Inference/adapter build failed: {e} | {tb}")
        return OrchestratorResult(
            ok=False, report_path=None, adapter_meta={}, timings_sec=timings,
            warnings=tuple(warnings), errors=tuple(errors),
            subject_id=subject_id, logs_path=None
        )

    # 5) Generate 4-page PDF
    try:
        t = time.time()
        style = ReportStyle(**(cfg.page_style or {}))
        out_dir = cfg.output_dir
        os.makedirs(out_dir, exist_ok=True)

        gen = EnhancedReportGenerator(output_dir=out_dir, style=style)
        report_path = gen.create_subject_report(
            subject_df=_canonize_subject_for_report(subj_df),
            meta={
                "subject_id": subject_id,
                "age": subj_meta.get("age"),
                "gender": subj_meta.get("gender"),
                "bmi": subj_meta.get("bmi"),
                "session_date": subj_meta.get("session_date"),
            },
            adapter=adapter,
            file_prefix=cfg.file_prefix,
        )
        timings["generate_pdf"] = time.time() - t
    except Exception as e:
        tb = traceback.format_exc(limit=1)
        errors.append(f"Report generation failed: {e} | {tb}")
        return OrchestratorResult(
            ok=False, report_path=None, adapter_meta=adapter_meta, timings_sec=timings,
            warnings=tuple(warnings), errors=tuple(errors),
            subject_id=subject_id, logs_path=None
        )

    # 6) Save a small run log
    try:
        t = time.time()
        logs_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        logs_path = os.path.join(logs_dir, f"{subject_id}_run_log.json")
        with open(logs_path, "w") as f:
            json.dump({
                "subject_id": subject_id,
                "adapter_meta": adapter_meta,
                "timings_sec": timings,
                "warnings": warnings,
                "errors": errors,
                "config": _public_cfg_dict(cfg),
            }, f, indent=2)
        timings["write_logs"] = time.time() - t
    except Exception as e:
        warnings.append(f"Could not save logs: {e}")

    total = time.time() - t0
    timings["total"] = total

    return OrchestratorResult(
        ok=True,
        report_path=report_path,
        adapter_meta=adapter_meta,
        timings_sec=timings,
        warnings=tuple(warnings),
        errors=tuple(errors),
        subject_id=subject_id,
        logs_path=logs_path,
    )


# ----------------------------
# Helpers
# ----------------------------

def _ensure_required_columns(df: pd.DataFrame) -> None:
    """
    Minimal guards so the report figures won't be blank.
    We expect at least a 'condition' (or 'label') column and a few numeric series.
    """
    cols_lower = {c.lower() for c in df.columns}
    if not ({"condition", "label", "state"} & cols_lower):
        raise ValueError(
            "Dataset must contain a 'condition'/'label'/'state' column.")

    # Warn (not fail) if classic signals are missing; the generator shows placeholders
    numerics = df.select_dtypes(include="number").columns
    if len(numerics) == 0:
        raise ValueError("No numeric columns found for plotting.")


def _validate_feature_shapes(feat_pack: Dict[str, Any]) -> None:
    """
    Ensure all feature arrays have consistent number of rows (windows).
    """
    n = None
    for key in ("X_chest", "X_wrist", "X_demo", "X_combined"):
        arr = feat_pack.get(key)
        if arr is None:
            continue
        if n is None:
            n = arr.shape[0]
        elif arr.shape[0] != n:
            raise ValueError(
                f"Feature pack mismatch: {key} has {arr.shape[0]} rows vs expected {n}.")

    # class indices/probas must align to class_names elsewhere (checked in model_manager)


def _canonize_subject_for_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Light canonicalization for the report figures:
    - Ensure 'condition' column exists with title-cased classes
    - Keep only relevant columns to reduce accidental plot clutter
    """
    df = df.copy()
    # condition harmonization
    cond_col = next((c for c in df.columns if c.lower()
                    in ("condition", "label", "state")), None)
    if cond_col is None:
        return df

    mapping = {
        "baseline": "Baseline",
        "stress": "Stress",
        "amusement": "Amusement",
        "meditation": "Meditation",
        # also allow numeric encodings 0..3
        0: "Baseline", 1: "Stress", 2: "Amusement", 3: "Meditation"
    }
    df["condition"] = df[cond_col].map(
        lambda x: mapping.get(str(x).lower(), mapping.get(x, str(x))))
    return df


def _public_cfg_dict(cfg: OrchestratorConfig) -> Dict[str, Any]:
    d = asdict(cfg)
    # do not leak giant style dicts in logs
    if d.get("page_style"):
        d["page_style"] = {k: d["page_style"][k]
                           for k in d["page_style"].keys()}
    return d
