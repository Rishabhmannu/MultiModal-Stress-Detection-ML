"""
core/preprocessing.py (Enhanced with Debugging)
---------------------
Feature preparation for WESAD subject slices with extensive debugging.

This enhanced version adds comprehensive logging to identify exactly where
the feature extraction pipeline is failing and why we're getting 5 features
instead of the expected 78+ features.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import logging

# Configure detailed logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _apply_feature_selection(df: pd.DataFrame, feature_groups: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Apply the exact same feature selection used in the notebook.
    
    From the notebook, these 18 features were selected using ANOVA F-test + Variance threshold:
    - 7 chest features
    - 11 wrist features  
    - 0 demographic features
    
    This ensures the pipeline uses exactly the same features the models were trained on.
    """
    logger.info("🎯 Applying notebook-matching feature selection...")

    # The exact 18 features selected in the notebook
    SELECTED_CHEST_FEATURES = [
        'chest_acc_min',
        'chest_eda_scr_count',
        'chest_eda_std',
        'chest_emg_power_ratio',
        'chest_hr_max',
        'chest_hr_mean',
        'chest_hr_min'
    ]

    SELECTED_WRIST_FEATURES = [
        'wrist_acc_max',
        'wrist_acc_min',
        'wrist_acc_std',
        'wrist_acc_x_std',
        'wrist_acc_z_std',
        'wrist_bvp_hr_range',
        'wrist_bvp_hr_std',
        'wrist_bvp_max',
        'wrist_bvp_min',
        'wrist_bvp_std',
        'wrist_eda_std'
    ]

    # No demographic features were selected in the notebook
    SELECTED_DEMO_FEATURES = []

    logger.info(f"📋 Notebook selected features:")
    logger.info(f"   💓 Chest: {len(SELECTED_CHEST_FEATURES)} features")
    logger.info(f"   ⌚ Wrist: {len(SELECTED_WRIST_FEATURES)} features")
    logger.info(f"   👤 Demo: {len(SELECTED_DEMO_FEATURES)} features")
    logger.info(
        f"   🎯 Total: {len(SELECTED_CHEST_FEATURES) + len(SELECTED_WRIST_FEATURES) + len(SELECTED_DEMO_FEATURES)} features")

    # Filter each group to match notebook selection
    available_chest = feature_groups.get('chest', [])
    available_wrist = feature_groups.get('wrist', [])
    available_demo = feature_groups.get('demo', [])

    # Keep only the selected features that are available in the data
    selected_chest = [
        f for f in SELECTED_CHEST_FEATURES if f in available_chest]
    selected_wrist = [
        f for f in SELECTED_WRIST_FEATURES if f in available_wrist]
    selected_demo = []  # No demographic features selected

    # Log what we found vs expected
    logger.info(f"🔍 Feature selection matching:")
    logger.info(
        f"   💓 Chest: {len(selected_chest)}/{len(SELECTED_CHEST_FEATURES)} found")
    if len(selected_chest) != len(SELECTED_CHEST_FEATURES):
        missing_chest = set(SELECTED_CHEST_FEATURES) - set(selected_chest)
        logger.warning(f"      Missing chest: {list(missing_chest)}")

    logger.info(
        f"   ⌚ Wrist: {len(selected_wrist)}/{len(SELECTED_WRIST_FEATURES)} found")
    if len(selected_wrist) != len(SELECTED_WRIST_FEATURES):
        missing_wrist = set(SELECTED_WRIST_FEATURES) - set(selected_wrist)
        logger.warning(f"      Missing wrist: {list(missing_wrist)}")

    logger.info(
        f"   👤 Demo: {len(selected_demo)}/{len(SELECTED_DEMO_FEATURES)} found (none expected)")

    total_selected = len(selected_chest) + \
        len(selected_wrist) + len(selected_demo)
    logger.info(f"   🎯 Total selected: {total_selected}/18 features")

    if total_selected == 18:
        logger.info("✅ Perfect match! All 18 notebook features found")
    else:
        logger.warning(f"⚠️ Partial match: {total_selected}/18 features found")

    return {
        'chest': selected_chest,
        'wrist': selected_wrist,
        'demo': selected_demo
    }


# Modified prepare_subject_features function with feature selection
def prepare_subject_features(
    subject_df: pd.DataFrame,
    scalers: Dict[str, Any],
    class_names: List[str],
) -> Dict[str, Any]:
    """
    Build X matrices for chest, wrist, demographics and a concatenated combined view.
    
    ENHANCED VERSION with feature selection to match the notebook's 18 selected features.
    """
    logger.info("=" * 60)
    logger.info("🔍 DEBUGGING FEATURE EXTRACTION PIPELINE")
    logger.info("=" * 60)

    df = subject_df.copy()
    warnings: List[str] = []

    # STEP 1: Log input data shape and columns
    logger.info(f"📊 Input DataFrame shape: {df.shape}")
    logger.info(f"📋 Total columns available: {len(df.columns)}")
    logger.info(f"📝 Column names (first 10): {list(df.columns[:10])}")
    logger.info(f"🔢 Subject windows: {len(df)} rows")

    # STEP 2: Log available scalers
    logger.info(
        f"🧮 Available scalers: {list(scalers.keys()) if scalers else 'None'}")
    for scaler_name, scaler in scalers.items():
        if scaler is not None:
            logger.info(f"  ✅ {scaler_name}: {type(scaler).__name__}")
            # Try to get expected features
            expected = _expected_features_from_scaler(scaler)
            logger.info(f"     Expected features: {len(expected)} features")
            if len(expected) > 0 and len(expected) <= 10:
                logger.info(f"     Feature names: {expected}")
            elif len(expected) > 10:
                logger.info(f"     Feature names (first 5): {expected[:5]}...")
        else:
            logger.warning(f"  ❌ {scaler_name}: None (not loaded)")

    # 1) Harmonize condition labels & optionally build y_true
    cond_col = _find_condition_col(df)
    logger.info(f"🏷️ Condition column found: {cond_col}")

    y_true = None
    if cond_col is not None:
        y_true = _map_labels_to_indices(df[cond_col], class_names)
        unique_conditions = df[cond_col].value_counts()
        logger.info(f"📊 Condition distribution:\n{unique_conditions}")

    # 2) Identify feature groups - THIS IS CRITICAL
    logger.info("🔍 STEP 2: Identifying feature groups...")
    chest_cols, wrist_cols, demo_cols = _infer_feature_groups(df)

    logger.info(f"💓 CHEST features identified: {len(chest_cols)}")
    logger.info(f"⌚ WRIST features identified: {len(wrist_cols)}")
    logger.info(f"👤 DEMOGRAPHIC features identified: {len(demo_cols)}")

    # *** NEW STEP: Apply feature selection to match notebook ***
    logger.info("🎯 STEP 2.1: Applying feature selection to match notebook...")
    feature_groups = {
        'chest': chest_cols,
        'wrist': wrist_cols,
        'demo': demo_cols
    }

    selected_groups = _apply_feature_selection(df, feature_groups)
    chest_cols = selected_groups['chest']
    wrist_cols = selected_groups['wrist']
    demo_cols = selected_groups['demo']

    logger.info(f"💓 CHEST features after selection: {len(chest_cols)}")
    if len(chest_cols) > 0:
        logger.info(f"   Selected chest features: {chest_cols}")
    else:
        logger.warning("   ⚠️ NO CHEST FEATURES SELECTED!")

    logger.info(f"⌚ WRIST features after selection: {len(wrist_cols)}")
    if len(wrist_cols) > 0:
        logger.info(f"   Selected wrist features: {wrist_cols}")
    else:
        logger.warning("   ⚠️ NO WRIST FEATURES SELECTED!")

    logger.info(f"👤 DEMOGRAPHIC features after selection: {len(demo_cols)}")
    if len(demo_cols) > 0:
        logger.info(f"   Selected demo features: {demo_cols}")
    else:
        logger.info("   ℹ️ No demographic features selected (matches notebook)")

    # 3) Extract numeric features per group (SAME AS BEFORE)
    logger.info("🔍 STEP 3: Extracting numeric features...")

    Xc, chest_cols_clean, w1 = _extract_numeric(df, chest_cols, group="chest")
    warnings += w1
    logger.info(
        f"💓 CHEST numeric extraction: {Xc.shape if Xc is not None else 'None'}")

    Xw, wrist_cols_clean, w2 = _extract_numeric(df, wrist_cols, group="wrist")
    warnings += w2
    logger.info(
        f"⌚ WRIST numeric extraction: {Xw.shape if Xw is not None else 'None'}")

    Xd, demo_cols_clean, w3 = _prepare_demographics(
        df, demo_cols, scalers.get("demo"))
    warnings += w3
    logger.info(
        f"👤 DEMO preparation: {Xd.shape if Xd is not None else 'None'}")

    n_windows = len(df)

    # 4) Apply scalers (SAME AS BEFORE - but now with correct feature counts)
    logger.info("🔍 STEP 4: Applying scalers...")

    Xc, chest_cols_final, w4 = _apply_scaler_if_available(
        Xc, chest_cols_clean, scalers.get("chest"), group="chest"
    )
    warnings += w4
    logger.info(
        f"💓 CHEST after scaling: {Xc.shape if Xc is not None else 'None'}")

    Xw, wrist_cols_final, w5 = _apply_scaler_if_available(
        Xw, wrist_cols_clean, scalers.get("wrist"), group="wrist"
    )
    warnings += w5
    logger.info(
        f"⌚ WRIST after scaling: {Xw.shape if Xw is not None else 'None'}")

    # Skip demo scaler since no demo features were selected
    if len(demo_cols_clean) > 0:
        Xd, demo_cols_final, w6 = _apply_scaler_if_available(
            Xd, demo_cols_clean, scalers.get("demo"), group="demo"
        )
        warnings += w6
    else:
        demo_cols_final = []
    logger.info(
        f"👤 DEMO after scaling: {Xd.shape if Xd is not None else 'None'}")

    # 5) Build combined matrix - CRITICAL STEP
    logger.info("🔍 STEP 5: Building combined feature matrix...")

    blocks: List[np.ndarray] = []
    names: List[str] = []

    for group_name, X, cols in [("CHEST", Xc, chest_cols_final), ("WRIST", Xw, wrist_cols_final), ("DEMO", Xd, demo_cols_final)]:
        if X is not None and X.size > 0 and len(cols) > 0:
            X_array = np.asarray(X, dtype=float)
            blocks.append(X_array)
            names.extend(list(cols))
            logger.info(f"  ✅ {group_name}: Added {X_array.shape[1]} features")
        else:
            logger.warning(f"  ❌ {group_name}: Skipped (empty or None)")

    X_combined = np.concatenate(blocks, axis=1) if blocks else None

    if X_combined is not None:
        logger.info(f"🎯 COMBINED MATRIX SHAPE: {X_combined.shape}")
        logger.info(f"   Total features: {X_combined.shape[1]}")
        logger.info(f"   Windows: {X_combined.shape[0]}")

        # Check if we hit the expected 18 features
        if X_combined.shape[1] == 18:
            logger.info(
                "✅ PERFECT: Exactly 18 features as expected from notebook!")
        elif X_combined.shape[1] >= 15:
            logger.warning(
                f"⚠️ CLOSE: Got {X_combined.shape[1]} features (expected 18)")
        else:
            logger.error(
                f"❌ WRONG: Got {X_combined.shape[1]} features (expected 18)")
    else:
        logger.error("❌ CRITICAL FAILURE: No combined matrix created!")

    # 6) Final NaN checks (SAME AS BEFORE)
    for key, X in (("X_chest", Xc), ("X_wrist", Xw), ("X_demo", Xd), ("X_combined", X_combined)):
        if X is not None:
            mask = np.isnan(X)
            if mask.any():
                nan_count = mask.sum()
                logger.warning(
                    f"🚨 {key} has {nan_count} NaN values; imputing zeros.")
                X[mask] = 0.0

    # Final results
    result = {
        "X_chest": Xc,
        "X_wrist": Xw,
        "X_demo": Xd,
        "X_combined": X_combined,
        "feature_names": {
            "chest": chest_cols_final,
            "wrist": wrist_cols_final,
            "demo": demo_cols_final,
            "combined": names,
        },
        "y_true": y_true,
        "n_windows": n_windows,
        "warnings": tuple(warnings),
    }

    # Final summary
    logger.info("=" * 60)
    logger.info("📋 FINAL FEATURE EXTRACTION SUMMARY")
    logger.info("=" * 60)
    logger.info(
        f"💓 Chest features: {len(chest_cols_final) if chest_cols_final else 0}")
    logger.info(
        f"⌚ Wrist features: {len(wrist_cols_final) if wrist_cols_final else 0}")
    logger.info(
        f"👤 Demo features: {len(demo_cols_final) if demo_cols_final else 0}")
    logger.info(f"🎯 TOTAL COMBINED: {len(names) if names else 0} features")
    logger.info(f"⚠️ Warnings: {len(warnings)}")
    if warnings:
        for i, w in enumerate(warnings[:5]):  # Show first 5 warnings
            logger.warning(f"   {i+1}. {w}")

    # Critical success check
    if len(names) == 18:
        logger.info(
            "🎉 SUCCESS: Perfect match with notebook's 18 selected features!")
    else:
        logger.error(f"🚨 MISMATCH: Expected 18 features, got {len(names)}")

    logger.info("=" * 60)

    return result

# Keep all the existing helper functions unchanged
def _infer_feature_groups(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """
    Split columns into chest, wrist, demographics based on name patterns.
    Enhanced with debugging.
    """
    chest = []
    wrist = []
    demo = []

    logger.info(f"🔍 Analyzing {len(df.columns)} columns for feature groups...")

    for col in df.columns:
        col_lower = col.lower()

        # Skip non-feature columns
        if col_lower in ("subject_id", "window_id", "condition", "label", "start_time"):
            logger.debug(f"   Skipping metadata column: {col}")
            continue

        # Chest sensor patterns
        if any(pattern in col_lower for pattern in ("chest_", "chest", "hr_", "hrv_", "eda_", "temp_", "resp_", "emg_", "acc_x", "acc_y", "acc_z")):
            if "wrist" not in col_lower:  # Make sure it's not wrist sensor
                chest.append(col)
                continue

        # Wrist sensor patterns
        if any(pattern in col_lower for pattern in ("wrist_", "wrist", "bvp_")):
            wrist.append(col)
            continue

        # Demographics
        if any(pattern in col_lower for pattern in ("age", "bmi", "height", "weight", "gender", "sex")):
            demo.append(col)
            continue

        # If unclear, try to categorize by common WESAD patterns
        if any(pattern in col_lower for pattern in ("mean", "std", "min", "max", "rate")):
            if "wrist" in col_lower:
                wrist.append(col)
            else:
                chest.append(col)  # Default to chest for sensor data
        else:
            logger.debug(f"   Uncategorized column: {col}")

    logger.info(f"   💓 Chest candidates: {len(chest)}")
    logger.info(f"   ⌚ Wrist candidates: {len(wrist)}")
    logger.info(f"   👤 Demo candidates: {len(demo)}")

    return chest, wrist, demo


def _extract_numeric(df: pd.DataFrame, cols: List[str], group: str) -> Tuple[Optional[np.ndarray], List[str], List[str]]:
    """Extract numeric features from specified columns with enhanced debugging."""
    warnings: List[str] = []

    if not cols:
        logger.info(f"   No {group} columns to extract")
        return None, [], warnings

    logger.info(f"   Processing {len(cols)} {group} columns...")

    # Select and convert to numeric
    subset = df[cols].copy()

    # Track conversion issues
    numeric_cols = []
    for col in cols:
        try:
            subset[col] = pd.to_numeric(subset[col], errors='coerce')
            if subset[col].notna().any():
                numeric_cols.append(col)
            else:
                logger.warning(
                    f"   Column {col} is all NaN after numeric conversion")
        except Exception as e:
            logger.warning(f"   Failed to convert {col} to numeric: {e}")

    if not numeric_cols:
        logger.warning(f"   No valid numeric {group} columns found")
        return None, [], warnings

    # Keep only successfully converted columns
    subset = subset[numeric_cols]

    # Handle missing values
    if subset.isna().any().any():
        logger.info(f"   Filling missing values in {group} features...")
        subset = subset.fillna(subset.median(numeric_only=True))

    logger.info(
        f"   Successfully extracted {len(numeric_cols)} {group} features")
    return subset.to_numpy(dtype=float), numeric_cols, warnings


def _prepare_demographics(df: pd.DataFrame, demo_cols: List[str], demo_scaler) -> Tuple[Optional[np.ndarray], List[str], List[str]]:
    """Prepare demographic features with enhanced debugging."""
    warnings: List[str] = []

    if not demo_cols:
        logger.info("   No demographic columns to prepare")
        return None, [], warnings

    logger.info(f"   Processing {len(demo_cols)} demographic features...")

    # Get demographic data
    base = df[demo_cols].copy()

    # Convert to numeric and handle encoding
    for col in demo_cols:
        if col.lower() in ('gender', 'sex'):
            # Encode gender/sex
            base[col] = base[col].map(
                {'male': 1, 'female': 0, 'M': 1, 'F': 0, 1: 1, 0: 0}).fillna(0)
        else:
            base[col] = pd.to_numeric(base[col], errors='coerce')

    # Align to scaler expectations if available
    if demo_scaler is not None:
        expected = _expected_features_from_scaler(demo_scaler)
        if expected:
            logger.info(f"   Aligning to scaler expected features: {expected}")
            # Add missing columns as zeros
            for c in expected:
                if c not in base.columns:
                    base[c] = 0.0
            # Keep only expected columns
            base = base[expected]

    # Final cleanup
    if base.shape[1] == 0:
        return None, [], warnings

    base = base.fillna(base.median(numeric_only=True))
    logger.info(
        f"   Successfully prepared {base.shape[1]} demographic features")

    return base.to_numpy(dtype=float), list(base.columns), warnings


def _apply_scaler_if_available(X: Optional[np.ndarray], cols: List[str], scaler, group: str) -> Tuple[Optional[np.ndarray], List[str], List[str]]:
    """Apply scaler transformation with enhanced debugging."""
    warnings: List[str] = []

    if X is None or scaler is None:
        if scaler is None:
            logger.warning(
                f"   {group} scaler not available - using raw features")
        return X, cols, warnings

    logger.info(f"   Applying {group} scaler transformation...")

    expected = _expected_features_from_scaler(scaler)
    if expected:
        logger.info(f"   Scaler expects {len(expected)} features")
        logger.info(f"   We have {len(cols)} features")

        # Align features to scaler expectations
        Xdf = pd.DataFrame(X, columns=cols)

        # Add missing expected features
        missing = [c for c in expected if c not in Xdf.columns]
        if missing:
            logger.info(f"   Adding {len(missing)} missing features as zeros")
            for c in missing:
                Xdf[c] = 0.0

        # Remove unexpected features
        extras = [c for c in Xdf.columns if c not in expected]
        if extras:
            logger.warning(f"   Removing {len(extras)} unexpected features")
            warnings.append(
                f"{group}: dropping {len(extras)} extra features not in scaler")

        # Reorder to match scaler expectations
        Xdf = Xdf[expected]
        cols = expected
        X = Xdf.to_numpy(dtype=float)

    # Apply transformation
    try:
        X_scaled = scaler.transform(X)
        logger.info(f"   ✅ {group} scaler applied successfully")
    except Exception as e:
        logger.error(f"   ❌ {group} scaler failed: {e}")
        warnings.append(
            f"{group}: scaler.transform failed ({e}); using unscaled features.")
        X_scaled = X

    return X_scaled, cols, warnings


def _expected_features_from_scaler(scaler) -> List[str]:
    """Extract expected feature names from scaler."""
    if scaler is None:
        return []
    if hasattr(scaler, "feature_names_in_"):
        return list(getattr(scaler, "feature_names_in_"))
    if hasattr(scaler, "expected_features"):
        return list(getattr(scaler, "expected_features"))
    return []


def _find_condition_col(df: pd.DataFrame) -> Optional[str]:
    """Find condition/label column."""
    for name in ("condition", "condition_name", "label", "state", "activity", "y"):
        if name in (c.lower() for c in df.columns):
            for c in df.columns:
                if c.lower() == name:
                    return c
    return None


def _map_labels_to_indices(series: pd.Series, class_names: List[str]) -> np.ndarray:
    """Map textual/ordinal labels to indices."""
    name_to_idx = {n.lower(): i for i, n in enumerate(class_names)}
    vals = []
    for v in series:
        if pd.isna(v):
            vals.append(-1)
            continue
        if isinstance(v, (int, np.integer)) and 0 <= int(v) < len(class_names):
            vals.append(int(v))
            continue
        s = str(v).strip().lower()
        vals.append(name_to_idx.get(s, -1))
    return np.asarray(vals, dtype=int)
