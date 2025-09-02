"""
core/data_manager.py
--------------------
Data access utilities for the WESAD Streamlit app.

Responsibilities
- Load a processed/integrated CSV (from disk or uploaded file-like object)
- Validate required columns and basic integrity
- List subjects and slice a single subject's rows
- Extract lightweight subject metadata (age, gender, BMI, session_date)
- Summarize the dataset for UI display

This module is intentionally "pure data IO" and contains **no model logic**.
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Dict, Any, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


# ----------------------------
# Public API
# ----------------------------

def load_csv(
    src: Union[str, bytes, io.BytesIO],
    encoding: str = "utf-8",
    low_memory: bool = False,
) -> pd.DataFrame:
    """
    Load a CSV from a path or an uploaded file buffer.

    Parameters
    ----------
    src : path string, raw bytes, or BytesIO
    encoding : file encoding
    low_memory : forward to pandas.read_csv

    Returns
    -------
    DataFrame
    """
    if isinstance(src, (bytes, bytearray)):
        buf = io.BytesIO(src)
        df = pd.read_csv(buf, encoding=encoding, low_memory=low_memory)
    elif isinstance(src, io.BytesIO):
        df = pd.read_csv(src, encoding=encoding, low_memory=low_memory)
    elif isinstance(src, str):
        df = pd.read_csv(src, encoding=encoding, low_memory=low_memory)
    else:
        raise TypeError(
            "Unsupported type for src; expected path, bytes, or BytesIO.")
    return df


def validate_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Basic structural checks; returns a dictionary of findings instead of raising,
    so the Streamlit UI can present warnings cleanly.

    Expected columns (case-insensitive, flexible):
      - subject id: one of {'subject', 'subject_id', 'id', 'participant'}
      - condition label: one of {'condition', 'label', 'state'}
    """
    findings: Dict[str, Any] = {"ok": True, "warnings": [], "errors": []}

    subj_col, cond_col = _find_subject_and_condition(df)

    if subj_col is None:
        findings["errors"].append(
            "Missing subject identifier column (expected one of: subject, subject_id, id).")
    if cond_col is None:
        findings["errors"].append(
            "Missing condition/label column (expected one of: condition, label, state).")

    # at least some numeric columns to plot
    n_numeric = df.select_dtypes(include="number").shape[1]
    if n_numeric == 0:
        findings["errors"].append("No numeric feature columns detected.")

    # NaN ratio (not an error, but useful)
    nan_ratio = float(df.isna().mean().mean())
    if nan_ratio > 0.25:
        findings["warnings"].append(
            f"High overall missingness (~{nan_ratio:.0%}). Consider inspecting preprocessing.")

    # Subject balance
    if subj_col:
        counts = df[subj_col].value_counts(dropna=True)
        if counts.min() < 10:
            findings["warnings"].append(
                "Some subjects have <10 rows; plots may look sparse.")

    findings["ok"] = len(findings["errors"]) == 0
    findings["subject_col"] = subj_col
    findings["condition_col"] = cond_col
    return findings


def list_subjects(df: pd.DataFrame) -> List[str]:
    """Return a sorted list of subject IDs (stringified)."""
    subj_col, _ = _find_subject_and_condition(df)
    if subj_col is None:
        return []
    subs = df[subj_col].dropna().astype(str).unique().tolist()
    # normalize like "S4" rather than "4"
    subs = [_normalize_subject_id(s) for s in subs]
    subs = sorted(set(subs), key=lambda s: (
        s[0].upper(), int(s[1:]) if s[1:].isdigit() else 1e9))
    return subs


def summarize_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Lightweight summary object for UI.
    """
    subj_col, cond_col = _find_subject_and_condition(df)
    out: Dict[str, Any] = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": list(df.columns),
        "subject_col": subj_col,
        "condition_col": cond_col,
        "n_subjects": int(len(df[subj_col].unique())) if subj_col else 0,
        "conditions": {},
    }
    if cond_col:
        vc = df[cond_col].astype(str).str.title().value_counts().to_dict()
        out["conditions"] = {k: int(v) for k, v in vc.items()}
    return out


def get_subject_df(df: pd.DataFrame, subject_id: str) -> pd.DataFrame:
    """
    Slice rows for a given subject id. Accepts "S4" or "4" or "s4".
    """
    subj_col, _ = _find_subject_and_condition(df)
    if subj_col is None:
        raise ValueError("Cannot slice subject: subject column not found.")

    target = _normalize_subject_id(subject_id)
    # try exact match first
    m = df[subj_col].astype(str)
    mask = (m == target) | (m.str.upper() == target.upper())
    # also allow numeric part match (e.g., "4")
    numeric = target[1:] if target[:1].upper() == "S" else target
    if numeric.isdigit():
        mask = mask | (m == numeric)

    subset = df.loc[mask].copy()
    if subset.empty:
        # attempt a last-chance match if original data used "S-4" etc.
        mask2 = m.str.replace(
            "-", "", regex=False).str.upper() == target.upper().replace("-", "")
        subset = df.loc[mask2].copy()

    if subset.empty:
        raise ValueError(
            f"Subject '{subject_id}' not found in column '{subj_col}'.")
    return subset


def get_subject_meta(df: pd.DataFrame, subject_id: str) -> Dict[str, Any]:
    """
    Extract age/gender/BMI and session date if present. If multiple rows disagree,
    the first non-null value encountered is returned (this is sufficient for report header).
    """
    subset = get_subject_df(df, subject_id)
    cols = {c.lower(): c for c in subset.columns}

    def pick(*aliases: str):
        for a in aliases:
            if a in cols:
                s = subset[cols[a]].dropna()
                if not s.empty:
                    return s.iloc[0]
        return None

    meta = {
        "subject_id": _normalize_subject_id(subject_id),
        "age": pick("age", "subject_age"),
        "gender": pick("gender", "sex"),
        "bmi": pick("bmi", "body_mass_index"),
        "session_date": pick("session_date", "date", "recording_date"),
    }
    # sanitize gender capitalization
    if isinstance(meta["gender"], str):
        meta["gender"] = meta["gender"].strip().title()
    # coerce numeric fields
    for k in ("age", "bmi"):
        try:
            v = float(meta[k]) if meta[k] is not None else None
            if v is not None and (np.isnan(v) or np.isinf(v)):
                v = None
            meta[k] = v
        except Exception:
            meta[k] = None
    return meta


# ----------------------------
# Helpers
# ----------------------------

_SUBJECT_ALIASES = ("subject", "subject_id", "id",
                    "participant", "user", "pid", "sid")
_CONDITION_ALIASES = ("condition", "label", "state", "activity", "y")


def _find_subject_and_condition(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    cols = {c.lower(): c for c in df.columns}
    subj_col = next((cols[a] for a in _SUBJECT_ALIASES if a in cols), None)
    cond_col = next((cols[a] for a in _CONDITION_ALIASES if a in cols), None)
    return subj_col, cond_col


def _normalize_subject_id(x: Any) -> str:
    s = str(x).strip()
    if not s:
        return "S?"
    # If already like "S4"
    if s[0].upper() == "S":
        return f"S{s[1:]}"
    # If numeric like "4"
    if s.isdigit():
        return f"S{s}"
    # Fallback
    return f"S{s.upper()}"
