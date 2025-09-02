"""
core/model_manager.py
---------------------
Model loading and inference utilities for the WESAD Streamlit app.

Responsibilities
- Discover and load available models (TabPFN, Attention, legacy sklearn models)
- Load scalers and label encoders
- Run per-window predictions and probabilities
- Build a unified "Prediction Adapter" for the report generator

Design notes
- We keep hard dependencies optional (torch, tabpfn). If not installed or file missing,
  we continue with whatever is available and surface diagnostics.
- For probabilities:
  * If a model exposes predict_proba, we use it.
  * Else if it has decision_function, we softmax the scores.
  * Else we fall back to one-hot of predicted labels (last resort).

Adapter schema (returned by predict_all):
{
  'class_names': [...],
  'primary_model': 'tabpfn' (or first available),
  'n_windows': int,
  'window_preds': {'tabpfn': np.ndarray[int], 'attention': ..., 'rf': ...},
  'window_probas': {'tabpfn': np.ndarray[float][n_windows, n_classes], ...},
  'ensemble': {'preds': np.ndarray[int], 'probas': np.ndarray[float][n_windows, n_classes]}  # optional
  'interpretability': {'attention_summary': [(feature_name, weight_norm), ...]}               # optional
  'meta': {'tabpfn_version': '...', 'attention_version': '...', 'loaded_models': [...]}
}
"""

from __future__ import annotations
import logging
logger = logging.getLogger(__name__)

import os
import json
import math
import warnings
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, List

import numpy as np

# Optional imports guarded at runtime
try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None

try:
    from importlib.metadata import version as _pkg_version  # py3.8+
except Exception:  # pragma: no cover
    def _pkg_version(pkg: str) -> str:
        return "unknown"

# Torch is optional (for Attention model)
try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None
    _TORCH_AVAILABLE = False

# TabPFN is optional
try:
    from tabpfn import TabPFNClassifier  # type: ignore
    _TABPFN_AVAILABLE = True
except Exception:  # pragma: no cover
    TabPFNClassifier = None
    _TABPFN_AVAILABLE = False


# ----------------------------
# Public API
# ----------------------------

def load_all_models(
    models_dir: str,
    scalers_dir: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Discover and load models & scalers from the given directories.

    Returns
    -------
    models : dict
        Keys may include: 'tabpfn', 'attention', 'rf', 'xgb', 'et', ...
    scalers : dict
        Keys may include: 'chest', 'wrist', 'demo', 'label_encoder'
    diagnostics : dict
        {'warnings': [...], 'errors': [...], 'missing': [...], 'versions': {...}}
    """
    models: Dict[str, Any] = {}
    scalers: Dict[str, Any] = {}
    diags = {"warnings": [], "errors": [], "missing": [], "versions": {}}

    # ---- Load scalers ----
    if joblib is None:
        diags["errors"].append(
            "joblib not available; cannot load .pkl scalers/models.")
    else:
        _load_scaler_if_exists(scalers_dir, "chest", scalers, diags)
        _load_scaler_if_exists(scalers_dir, "wrist", scalers, diags)
        _load_scaler_if_exists(scalers_dir, "demo", scalers, diags)
        _load_label_encoder_if_exists(scalers_dir, scalers, diags)

    # ---- Load models ----
    if not os.path.isdir(models_dir):
        diags["missing"].append(f"models_dir not found: {models_dir}")
        return models, scalers, diags

    files = {f.lower(): os.path.join(models_dir, f)
             for f in os.listdir(models_dir)}

    # TabPFN (preferred primary)
    tabpfn_path = _first_present(files, ("tabpfn_model.pkl", "tabpfn.pkl"))
    if tabpfn_path:
        if not _TABPFN_AVAILABLE:
            diags["warnings"].append(
                "TabPFN model present but 'tabpfn' package not installed.")
        elif joblib is None:
            diags["errors"].append("Cannot load TabPFN .pkl without joblib.")
        else:
            try:
                models["tabpfn"] = joblib.load(tabpfn_path)
                diags["versions"]["tabpfn"] = _pkg_version("tabpfn")
            except Exception as e:
                diags["errors"].append(f"Failed to load TabPFN: {e}")

    # Attention (PyTorch)
    attn_path = _first_present(
        files, ("attention_model.pth", "attention.pth", "attention_model.pt"))
    attn_cfg = _first_present(
        files, ("attention_model_config.json", "attention_config.json"))
    if attn_path:
        if not _TORCH_AVAILABLE:
            diags["warnings"].append(
                "Attention model present but 'torch' not installed.")
        else:
            try:
                models["attention"] = _load_attention_model(
                    attn_path, attn_cfg, diags)
                diags["versions"]["torch"] = _pkg_version("torch")
            except Exception as e:
                diags["errors"].append(f"Failed to load Attention model: {e}")

    # Legacy sklearn models (optional): random forest / extra trees / gradient boosting
    if joblib is not None:
        for key, names in {
            "rf": ("random_forest_baseline.pkl", "rf.pkl"),
            "et": ("extra_trees.pkl", "extratrees.pkl", "et.pkl"),
            "gb": ("gradient_boosting.pkl", "gb.pkl"),
            "svm": ("svm.pkl", "svc.pkl"),
            "logreg": ("logreg.pkl", "logistic_regression.pkl"),
        }.items():
            p = _first_present(files, names)
            if p:
                try:
                    models[key] = joblib.load(p)
                except Exception as e:
                    diags["warnings"].append(
                        f"Could not load legacy model '{key}': {e}")

    if not models:
        diags["missing"].append(
            "No models loaded. Place files in models/trained_models.")

    return models, scalers, diags


def predict_all(
    features: Dict[str, Any],
    models: Dict[str, Any],
    class_names: List[str],
    primary_model: str = "tabpfn",
    use_attention: bool = True,
    include_legacy: bool = False,
) -> Dict[str, Any]:
    """
    Run predictions across available models and return a model-agnostic adapter.
    """
    # Choose the feature block: prefer combined, then chest+wrist, then any available
    X = _select_feature_block(features)
    n_windows = int(features.get("n_windows") or (
        X.shape[0] if X is not None else 0))
    if X is None or n_windows == 0:
        raise ValueError(
            "No features available for prediction (X_combined/chest/wrist missing).")

    window_preds: Dict[str, np.ndarray] = {}
    window_probas: Dict[str, np.ndarray] = {}
    model_meta: Dict[str, Any] = {"loaded_models": list(models.keys())}

    # --- TabPFN ---
    


    logger = logging.getLogger(__name__)

    # --- TabPFN with Enhanced Debugging ---
    if "tabpfn" in models:
        logger.info(f"🧠 Attempting TabPFN prediction...")
        logger.info(f"   Input shape: {X.shape}")
        logger.info(f"   Expected classes: {len(class_names)}")
        logger.info(f"   Class names: {class_names}")
        logger.info(f"   TabPFN model type: {type(models['tabpfn'])}")

        # Check if TabPFN model has expected methods
        tabpfn_model = models["tabpfn"]
        logger.info(f"   Has predict: {hasattr(tabpfn_model, 'predict')}")
        logger.info(
            f"   Has predict_proba: {hasattr(tabpfn_model, 'predict_proba')}")

        # Check data characteristics
        logger.info(f"   Data min: {X.min():.6f}")
        logger.info(f"   Data max: {X.max():.6f}")
        logger.info(f"   Data mean: {X.mean():.6f}")
        logger.info(f"   Data std: {X.std():.6f}")
        logger.info(f"   Has NaN: {np.isnan(X).any()}")
        logger.info(f"   Has Inf: {np.isinf(X).any()}")

        try:
            logger.info("🔄 Calling _predict_sklearn_like...")
            preds, probas = _predict_sklearn_like(
                tabpfn_model, X, len(class_names))

            logger.info(f"✅ TabPFN prediction SUCCESS!")
            logger.info(f"   Predictions shape: {preds.shape}")
            logger.info(f"   Probabilities shape: {probas.shape}")
            logger.info(f"   Unique predictions: {np.unique(preds)}")
            logger.info(f"   Prediction distribution: {np.bincount(preds)}")

            window_preds["tabpfn"] = preds
            window_probas["tabpfn"] = probas

        except Exception as e:
            logger.error(f"❌ TabPFN prediction FAILED!")
            logger.error(f"   Error type: {type(e).__name__}")
            logger.error(f"   Error message: {str(e)}")
            logger.error(f"   Full traceback:")

            import traceback
            logger.error(traceback.format_exc())

            model_meta["tabpfn_error"] = str(e)

            # Try to give more specific debugging info
            try:
                logger.info("🔍 Attempting direct TabPFN.predict()...")
                direct_preds = tabpfn_model.predict(X)
                logger.info(f"   Direct predict worked: {direct_preds.shape}")
            except Exception as e2:
                logger.error(f"   Direct predict also failed: {e2}")

            try:
                logger.info("🔍 Attempting direct TabPFN.predict_proba()...")
                direct_probas = tabpfn_model.predict_proba(X)
                logger.info(
                    f"   Direct predict_proba worked: {direct_probas.shape}")
            except Exception as e3:
                logger.error(f"   Direct predict_proba also failed: {e3}")



    # --- Attention (torch) ---
    attn_summary: List[Tuple[str, float]] = []
    if use_attention and "attention" in models:
        try:
            ainfo = models["attention"]
            # Lazy build if needed
            if ainfo.get("deferred"):
                in_dim = X.shape[1]
                n_classes = int(ainfo.get("n_classes") or len(class_names))
                hidden = int(ainfo.get("hidden", 128))
                dropout = float(ainfo.get("dropout", 0.1))
                arch = str(ainfo.get("arch", "mlp_attention"))
                model = _build_simple_attention_model(
                    in_dim, n_classes, hidden, dropout, arch)
                model.load_state_dict(ainfo["state_dict"], strict=False)
                model.to(ainfo["device"]).eval()
                ainfo["model"] = model
                ainfo["deferred"] = False

            attn_model = ainfo["model"]
            attn_device = ainfo["device"]
            with torch.no_grad():  # type: ignore
                tX = torch.tensor(X, dtype=torch.float32,
                                  device=attn_device)  # type: ignore
                logits, attn_weights = attn_model(tX)
                probas = _softmax_np(logits.detach().cpu().numpy())
                preds = probas.argmax(axis=1).astype(int)
                window_preds["attention"] = preds
                if probas.shape[1] != len(class_names):
                    probas = _pad_or_trim_cols(probas, len(class_names))
                window_probas["attention"] = probas

                if attn_weights is not None:
                    W = attn_weights.detach().cpu().numpy()
                    feat_names = features.get(
                        "feature_names", {}).get("combined") or []
                    attn_summary = _summarize_attention(W, feat_names, topk=8)
        except Exception as e:
            model_meta["attention_error"] = str(e)

    # --- Legacy models (optional) ---
    if include_legacy:
        for key in [k for k in models.keys() if k not in ("tabpfn", "attention")]:
            try:
                preds, probas = _predict_sklearn_like(
                    models[key], X, len(class_names))
                window_preds[key] = preds
                window_probas[key] = probas
            except Exception as e:
                model_meta[f"{key}_error"] = str(e)

    # --- Pick primary model actually available ---
    prim = primary_model if primary_model in window_probas else _first_available(
        window_probas, fallback="tabpfn")

    # --- Simple probability ensemble (average of available model probabilities) ---
    ensemble = None
    if include_legacy:
        mats = [P for name, P in window_probas.items() if name in (
            "tabpfn", "attention") or name not in ()]
        if len(mats) >= 2:
            P_ens = np.mean(mats, axis=0)
            P_ens = _renormalize_rows(P_ens)
            y_ens = P_ens.argmax(axis=1).astype(int)
            ensemble = {"preds": y_ens, "probas": P_ens}

    adapter = {
        "class_names": class_names,
        "primary_model": prim,
        "n_windows": n_windows,
        "window_preds": window_preds,
        "window_probas": window_probas,
        "meta": model_meta,
    }

    if ensemble is not None:
        adapter["ensemble"] = ensemble

    # interpretability (attention)
    if attn_summary:
        adapter["interpretability"] = {"attention_summary": attn_summary}

    return adapter


# ----------------------------
# Load helpers
# ----------------------------

def _load_scaler_if_exists(scalers_dir: str, name: str, scalers: Dict[str, Any], diags: Dict[str, Any]) -> None:
    candidates = [f"{name}_scaler.pkl", f"{name}.pkl", f"scaler_{name}.pkl"]
    p = _first_present({f.lower(): os.path.join(scalers_dir, f) for f in os.listdir(
        scalers_dir)} if os.path.isdir(scalers_dir) else {}, candidates)
    if p and joblib is not None:
        try:
            scalers[name] = joblib.load(p)
        except Exception as e:
            diags["warnings"].append(f"Failed to load scaler '{name}': {e}")
    else:
        diags["missing"].append(f"scaler not found: {name}")


def _load_label_encoder_if_exists(scalers_dir: str, scalers: Dict[str, Any], diags: Dict[str, Any]) -> None:
    if not os.path.isdir(scalers_dir) or joblib is None:
        return
    files = {f.lower(): os.path.join(scalers_dir, f)
             for f in os.listdir(scalers_dir)}
    p = _first_present(files, ("label_encoder.pkl", "labels.pkl", "le.pkl"))
    if p:
        try:
            scalers["label_encoder"] = joblib.load(p)
        except Exception as e:
            diags["warnings"].append(f"Failed to load label encoder: {e}")


def _load_attention_model(attn_path: str, cfg_path: Optional[str], diags: Dict[str, Any]) -> Dict[str, Any]:
    assert _TORCH_AVAILABLE
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")  # type: ignore

    # Try TorchScript first (ready to run)
    try:
        scripted = torch.jit.load(
            attn_path, map_location=device)  # type: ignore
        scripted.eval()
        return {"model": _WrapScripted(scripted), "device": device, "n_classes": None, "deferred": False}
    except Exception:
        pass  # fall through to state_dict route

    # State_dict: may require knowing input_dim. If we cannot, return "deferred".
    if cfg_path and os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        n_classes = int(cfg.get("n_classes", 4))
        hidden = int(cfg.get("hidden_dim", 128))
        dropout = float(cfg.get("dropout", 0.1))
        arch = cfg.get("arch", "mlp_attention").lower()
        input_dim = int(cfg.get("input_dim", 0))

        sd = torch.load(attn_path, map_location=device)  # type: ignore
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]

        if input_dim <= 0:
            # Defer build until we know X.shape[1]
            return {
                "model": None,
                "device": device,
                "n_classes": n_classes,
                "hidden": hidden,
                "dropout": dropout,
                "arch": arch,
                "state_dict": sd,
                "deferred": True,
            }

        # If input_dim is known now, build immediately
        model = _build_simple_attention_model(
            input_dim, n_classes, hidden, dropout, arch)
        model.load_state_dict(sd, strict=False)
        model.to(device).eval()
        return {"model": model, "device": device, "n_classes": n_classes, "deferred": False}

    diags["warnings"].append(
        "Attention model could not be loaded (no TorchScript and no config for state_dict).")
    raise RuntimeError("Unsupported attention model format.")



# ----------------------------
# Predict helpers
# ----------------------------

def _predict_sklearn_like(model, X: np.ndarray, n_classes: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Enhanced _predict_sklearn_like with comprehensive debugging.
    """
    logger.info(f"🔍 _predict_sklearn_like called:")
    logger.info(f"   Model: {type(model)}")
    logger.info(f"   X shape: {X.shape}")
    logger.info(f"   n_classes: {n_classes}")

    try:
        # Try predict first
        logger.info("   📊 Attempting model.predict()...")
        preds = model.predict(X)
        logger.info(f"   ✅ Predict successful: {preds.shape}")
        logger.info(f"      Unique values: {np.unique(preds)}")

        # Convert predictions to integers if needed
        if preds.dtype != int:
            logger.info(
                f"   🔄 Converting predictions from {preds.dtype} to int")
            preds = preds.astype(int)

    except Exception as e:
        logger.error(f"   ❌ model.predict() failed: {e}")
        raise e

    try:
        # Try predict_proba
        logger.info("   📈 Attempting model.predict_proba()...")
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(X)
            logger.info(f"   ✅ Predict_proba successful: {probas.shape}")

            # Check probabilities are valid
            if np.isnan(probas).any():
                logger.warning("   ⚠️ Probabilities contain NaN values")
            if np.isinf(probas).any():
                logger.warning("   ⚠️ Probabilities contain Inf values")
            if (probas < 0).any():
                logger.warning("   ⚠️ Probabilities contain negative values")
            if not np.allclose(probas.sum(axis=1), 1.0, rtol=1e-5):
                logger.warning("   ⚠️ Probabilities don't sum to 1")

        else:
            logger.warning(
                "   ⚠️ Model doesn't have predict_proba, using fallback")
            # Fallback: one-hot encoding
            probas = np.zeros((len(preds), n_classes))
            for i, pred in enumerate(preds):
                if 0 <= pred < n_classes:
                    probas[i, pred] = 1.0
                else:
                    logger.warning(
                        f"   ⚠️ Invalid prediction value: {pred} (expected 0-{n_classes-1})")
                    probas[i, 0] = 1.0  # Default to first class
            logger.info(f"   ✅ Fallback probabilities created: {probas.shape}")

    except Exception as e:
        logger.error(f"   ❌ Probability extraction failed: {e}")
        raise e

    # Final validation
    if probas.shape[1] != n_classes:
        logger.warning(
            f"   ⚠️ Probability shape mismatch: {probas.shape[1]} != {n_classes}")
        if probas.shape[1] > n_classes:
            logger.info(f"   🔄 Trimming probabilities to {n_classes} classes")
            probas = probas[:, :n_classes]
        else:
            logger.info(f"   🔄 Padding probabilities to {n_classes} classes")
            padding = np.zeros((probas.shape[0], n_classes - probas.shape[1]))
            probas = np.hstack([probas, padding])

    logger.info(f"   🎯 Final output:")
    logger.info(f"      Predictions: {preds.shape} ({preds.dtype})")
    logger.info(f"      Probabilities: {probas.shape} ({probas.dtype})")

    return preds, probas

def _softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    ex = np.exp(x)
    s = ex.sum(axis=1, keepdims=True) + 1e-12
    return ex / s


def _pad_or_trim_cols(P: np.ndarray, n_classes: int) -> np.ndarray:
    if P.shape[1] == n_classes:
        return P
    if P.shape[1] > n_classes:
        return P[:, :n_classes]
    # pad missing columns with tiny values
    pad = np.full((P.shape[0], n_classes - P.shape[1]), 1e-9, dtype=float)
    P2 = np.concatenate([P, pad], axis=1)
    return _renormalize_rows(P2)


def _renormalize_rows(P: np.ndarray) -> np.ndarray:
    s = P.sum(axis=1, keepdims=True)
    s = np.where(s <= 0, 1.0, s)
    return P / s


def _select_feature_block(features: Dict[str, Any]) -> Optional[np.ndarray]:
    if features.get("X_combined") is not None:
        return features["X_combined"]
    blocks = [features.get("X_chest"), features.get(
        "X_wrist"), features.get("X_demo")]
    for X in blocks:
        if X is not None:
            return X
    return None


# ----------------------------
# Attention utilities
# ----------------------------

def _summarize_attention(W: np.ndarray, feature_names: List[str], topk: int = 8) -> List[Tuple[str, float]]:
    """
    Aggregate attention weights across windows and normalize to produce a top-k list.
    """
    if W.ndim != 2:
        return []
    w = W.mean(axis=0)
    w = np.clip(w, 0.0, None)
    if w.sum() <= 0:
        return []
    w = w / (w.sum() + 1e-12)
    idx = np.argsort(-w)[:topk]
    out = []
    for i in idx:
        name = feature_names[i] if 0 <= i < len(feature_names) else f"f{i}"
        out.append((name, float(w[i])))
    return out


# ----------------------------
# Torch model stubs / wrappers
# ----------------------------

class _WrapScripted(torch.nn.Module):  # type: ignore
    """
    Wrap a scripted module to present a uniform forward() -> (logits, attention) API.
    If the scripted model returns only logits, we pair it with None for attention.
    """

    def __init__(self, scripted):
        super().__init__()
        self.scripted = scripted

    def forward(self, x):
        out = self.scripted(x)
        if isinstance(out, (tuple, list)) and len(out) >= 2:
            return out[0], out[1]
        return out, None


def _build_simple_attention_model(input_dim: int, n_classes: int, hidden: int, dropout: float, arch: str):
    """
    Minimal MLP-with-attention block compatible with many simple training scripts.
    You can replace this with your exact architecture if needed.
    Expected to return (logits, attention_weights).
    """
    import torch.nn as nn  # type: ignore
    import torch.nn.functional as F  # type: ignore

    class AttnMLP(nn.Module):
        def __init__(self, d_in, d_hid, d_out, p):
            super().__init__()
            self.fc1 = nn.Linear(d_in, d_hid)
            self.fc2 = nn.Linear(d_hid, d_hid)
            self.dropout = nn.Dropout(p)
            # feature attention vector
            self.attn = nn.Parameter(torch.zeros(d_in))
            nn.init.normal_(self.attn, mean=0.0, std=0.02)
            self.out = nn.Linear(d_hid, d_out)

        def forward(self, x):
            # x: (N, F)
            # apply feature attention by reweighting inputs
            a = torch.sigmoid(self.attn)            # (F,)
            xw = x * a                               # (N, F)
            h = torch.relu(self.fc1(xw))
            h = self.dropout(torch.relu(self.fc2(h)))
            logits = self.out(h)                     # (N, C)
            # broadcast per-sample attention
            return logits, a.expand_as(x)

    return AttnMLP(input_dim, hidden, n_classes, dropout)


# ----------------------------
# Misc helpers
# ----------------------------

def _first_present(files_map: Dict[str, str], names: Tuple[str, ...]) -> Optional[str]:
    for n in names:
        p = files_map.get(n.lower())
        if p and os.path.exists(p):
            return p
    return None


def _first_available(d: Dict[str, Any], fallback: str) -> str:
    return next(iter(d.keys()), fallback)
