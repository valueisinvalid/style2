#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
logic/outfit_mapper.py (enriched version)

Adds automatic enrichment of minimal items (e.g., ML/out.json with only item_id/productImage)
from a catalogue file if available, and makes thermal filtering more forgiving when CLO
is missing or pools become empty.

Catalogue sources (first found wins):
- ML/item_catalogue.json  (list of dicts with item_id, role, clo, etc.)
- ML/features_v1.npz      (expects arrays: item_id, role, clo)
"""

from __future__ import annotations
import argparse, json, pickle
import re
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
import pandas as pd

# ==== Aesthetic scorer hook ==============================================
import os, pickle
import numpy as np
import itertools

# --- global vector stores to reuse inside AestheticScorer ---
STYLE_VECS: Dict[str, np.ndarray] = {}
COLOR_VECS: Dict[str, np.ndarray] = {}


class AestheticScorer:
    """
    Tries to load a trained model from ML/aesthetic_scorer_v1.pkl.
    Falls back to a lightweight heuristic if model or features are missing.
    Expects items to optionally carry:
      - item.get("style_vec") -> np.ndarray or list
      - item.get("color_vec") -> np.ndarray or list
    Feature vector for (top, bottom, outer?) is concatenation of:
      [top.style, bottom.style, outer.style?,
       top.color, bottom.color, outer.color?,
       [pairwise sims], [clo features]]
    """

    def __init__(self, model_path="ML/aesthetic_scorer_v1.pkl"):
        self.model = None
        self.model_path = model_path
        if os.path.exists(model_path):
            try:
                with open(model_path, "rb") as f:
                    self.model = pickle.load(f)
            except Exception as e:
                print(f"[warn] Could not load aesthetic model: {e}")

    @staticmethod
    def _to_vec(x):
        if x is None:
            return None
        arr = np.asarray(x, dtype=float)
        return arr if arr.ndim == 1 else arr.ravel()

    @staticmethod
    def _sim(a, b):
        if a is None or b is None:
            return 0.0
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _build_features(self, top, bottom, outer, clo_top, clo_bottom, clo_outer):
        # grab vectors
        sv_t = self._to_vec(top.get("style_vec"))
        sv_b = self._to_vec(bottom.get("style_vec"))
        sv_o = self._to_vec(outer.get("style_vec")) if outer else None

        cv_t = self._to_vec(top.get("color_vec"))
        cv_b = self._to_vec(bottom.get("color_vec"))
        cv_o = self._to_vec(outer.get("color_vec")) if outer else None

        # sims
        style_tb = self._sim(sv_t, sv_b)
        style_to = self._sim(sv_t, sv_o) if outer else 0.0
        style_bo = self._sim(sv_b, sv_o) if outer else 0.0

        color_tb = self._sim(cv_t, cv_b)
        color_to = self._sim(cv_t, cv_o) if outer else 0.0
        color_bo = self._sim(cv_b, cv_o) if outer else 0.0

        # concatenation (pad missing with zeros so dimension is fixed)
        def pad(v, d=64):
            if v is None:
                return np.zeros(d, dtype=float)
            return v

        d_style = max([len(v) for v in [sv_t, sv_b, sv_o] if v is not None] + [0]) or 1
        d_color = max([len(v) for v in [cv_t, cv_b, cv_o] if v is not None] + [0]) or 1

        feats = np.concatenate([
            pad(sv_t, d_style), pad(sv_b, d_style), pad(sv_o, d_style),
            pad(cv_t, d_color), pad(cv_b, d_color), pad(cv_o, d_color),
            np.array([
                style_tb, style_to, style_bo,
                color_tb, color_to, color_bo,
                clo_top, clo_bottom, (clo_outer or 0.0)
            ], dtype=float)
        ], axis=0)
        return feats

    def score(self, top, bottom, outer, clo_top, clo_bottom, clo_outer):
        """
        Use the same 771-dim feature pipeline as the trained LGBM model.
        Model was trained on *pairs* (top,bottom); 'outer' burada *skora*
        dahil edilmez (dahil etmek istiyorsan ayrı bir outer modeli gerekir).
        """
        # model yüklüyse ve global vektörler hazırsa → 771-dim feature üret
        if self.model is not None and STYLE_VECS and COLOR_VECS:
            try:
                tid = str(top.get("item_id"))
                bid = str(bottom.get("item_id"))
                feat = create_feature_vector(tid, bid, STYLE_VECS, COLOR_VECS)  # shape (1,771)
                X_df = _to_feature_frame(feat)
                y = self.model.predict(X_df)
                return float(y[0])
            except Exception as e:
                print(f"[warn] AestheticScorer: LGBM predict failed; using heuristic. err={e}")

        # --- Heuristic fallback (outer/top-bottom örtüşmeleri vs.) ---
        def get(it, key):
            return (it or {}).get(key, "")

        t_type = get(top, "type").lower()
        b_type = get(bottom, "type").lower()
        o_type = get(outer, "type").lower() if outer else ""

        base_bonus = 0.0
        if any(k in t_type for k in ["tee", "t-shirt", "vest", "camisole", "blouse"]):
            base_bonus += 0.1
        if any(k in b_type for k in ["short", "jean", "trouser", "skirt"]):
            base_bonus += 0.1
        if outer and any(k in o_type for k in ["jacket", "coat", "blazer"]):
            base_bonus += 0.1

        neutrals = ["black", "white", "beige", "cream", "grey", "navy", "charcoal"]
        col_t = get(top, "color").lower()
        col_b = get(bottom, "color").lower()
        col_o = get(outer, "color").lower() if outer else ""
        neutral_frac = sum(any(n in c for n in neutrals) for c in [col_t, col_b, col_o if outer else ""]) / (3 if outer else 2)
        harmony = 0.4 + 0.4 * neutral_frac

        clo_tot = (clo_top or 0) + (clo_bottom or 0) + (clo_outer or 0)
        clo_top_ratio = (clo_top or 0) / (clo_tot + 1e-6)
        clo_balance = 0.4 if clo_top_ratio > 0.75 else 0.6

        return float(5.0 * harmony + 2.0 * base_bonus + 1.0 * clo_balance)
# ========================================================================

# ---- Aesthetic scorer singleton and wrapper ---------------------------

_AESTHETIC_SCORER_SINGLETON: Optional[AestheticScorer] = None


def _get_scorer() -> AestheticScorer:
    global _AESTHETIC_SCORER_SINGLETON
    if _AESTHETIC_SCORER_SINGLETON is None:
        _AESTHETIC_SCORER_SINGLETON = AestheticScorer("ML/aesthetic_scorer_v1.pkl")
    return _AESTHETIC_SCORER_SINGLETON


def _aesthetic_score(top: Dict, bottom: Dict, outer: Optional[Dict],
                     clo_top: float, clo_bottom: float, clo_outer: float = 0.0,
                     scorer: Optional[AestheticScorer] = None) -> float:
    scorer = scorer or _get_scorer()
    return scorer.score(top, bottom, outer, clo_top, clo_bottom, clo_outer)

# -------------------- quick ranking helpers --------------------

def _score_key(c):
    """Utility for sorting: higher aesthetic, smaller gap."""
    return (c["score"] - c["gap"]) if c else -1e9


def _enumerate_with_outer(base_combo: List[Dict], target_clo: float, outers: List[Dict], topk_outer: int = 10):
    """Given a 2- or 3-piece combo list, try adding an outer layer to approach target CLO.
    Returns extended candidates sorted by score.
    """
    results: List[Dict] = []
    for outer in outers[:topk_outer]:
        clo_total = sum(p.get("clo", 0.0) for p in base_combo) + outer.get("clo", 0.0)
        top_piece = base_combo[1] if len(base_combo) >= 2 else None
        bottom_piece = next((p for p in base_combo if "bottom" in p.get("roles", []) or p.get("role") == "bottom"), None)
        base_piece = base_combo[0] if len(base_combo) == 3 else None

        score = _aesthetic_score(
            top_piece or base_piece,
            bottom_piece,
            outer,
            _clo(top_piece or base_piece),
            _clo(bottom_piece),
            _clo(outer),
        )
        gap = abs(clo_total - target_clo)
        results.append({"combo": base_combo + [outer], "clo": clo_total, "score": score, "gap": gap})

    results.sort(key=_score_key, reverse=True)
    return results

import numpy as np
try:
    import joblib
except Exception:
    joblib = None

try:
    from weather_scoring import get_target_clo  # type: ignore
except Exception:
    get_target_clo = None

# -----------------------------
# Data containers
# -----------------------------

# 384 diff + 384 prod + 1 color_dist + 2 pair_type flags = 771
FEATURE_NAMES = (
    [f"style_diff_{i}" for i in range(384)]
    + [f"style_prod_{i}" for i in range(384)]
    + ["color_dist", "is_bb", "is_bm"]
)

def _to_feature_frame(X: np.ndarray) -> pd.DataFrame:
    # X shape: (n, 771)
    if X.shape[1] != 771:
        raise ValueError(f"Expected 771 features, got {X.shape[1]}")
    return pd.DataFrame(X, columns=FEATURE_NAMES)

# ---------------------------------------------
# Helper: safely read clo values (with np.isnan guard)
# ---------------------------------------------
def _clo(item: Dict[str, Any], default: float = 0.3) -> float:
    try:
        v = float(item.get("clo", default))
        if np.isnan(v):
            return default
        return v
    except Exception:
        return default


# -------------------------------------------------
# Text & coverage helper utilities (weather-aware)
# -------------------------------------------------

def _txt_blob(it: Dict[str, Any]) -> str:
    return (
        (it.get("fabric_composition", "") or "") + " " +
        (it.get("type", "") or "") + " " +
        (it.get("name", "") or "") + " " +
        (it.get("description", "") or "") + " " +
        " ".join([f"{k}:{v}" for k, v in (it.get("key_value_description") or {}).items()])
    ).lower()

# keyword groups
_KNIT_KEYS = ["knit", "knitted", "sweater", "jumper", "cardigan", "fleece", "pullover"]
_WARM_MAT = [
    "wool", "cashmere", "mohair", "alpaca", "angora", "acrylic", "pile", "teddy", "sherpa", "shearling", "faux fur"
]
_COLD_HOSTILE_BOTTOM = ["skirt"]  # skater, pencil, mini, midi vb. alt tipleri kapsar


def _is_sleepless_hint(t: str) -> bool:
    return any(k in t for k in ["sleeveless", "camisole", "tank", "vest", "halter"])


def _coverage_top(it: Dict[str, Any]) -> float:
    """
    0.0–1.0 aralığında yaklaşık kol ve boyun kapatıcılığı ölçümü.
    """
    t = _txt_blob(it)
    cov = 0.5
    if "long sleeve" in t:
        cov = 1.0
    elif "3/4" in t:
        cov = 0.7
    elif "short sleeve" in t:
        cov = 0.4
    if "turtleneck" in t or "high neck" in t:
        cov += 0.1
    if "sleeveless" in t or _is_sleepless_hint(t):
        cov = 0.2
    return max(0.0, min(1.1, cov))


def _coverage_bottom(it: Dict[str, Any]) -> float:
    """
    0.0–1.0: bacak kapatıcılığı (pantolon=1.0; midi=0.6; mini=0.3).
    Astarlı/termal/wool katkısı +0.1
    """
    t = _txt_blob(it)
    # temel kapsama
    if any(k in t for k in ["trouser", "pant", "jean", "leggings", "jogger", "chino"]):
        cov = 1.0
    else:
        # etek uzunluğu tahmini
        if "ankle length" in t or "long" in t:
            cov = 0.8
        elif "midi" in t or "calf" in t:
            cov = 0.6
        elif "mini" in t or "above knee" in t:
            cov = 0.3
        else:
            cov = 0.5
    if any(k in t for k in ["thermal", "fleece", "wool", "lined"]):
        cov += 0.1
    return max(0.0, min(1.1, cov))


def _is_knit_warm_top(it: Dict[str, Any]) -> bool:
    t = _txt_blob(it)
    return any(k in t for k in _KNIT_KEYS) or any(k in t for k in _WARM_MAT)

# -------------------------------------------------
# Fabric breathability estimation (RET) helpers
# -------------------------------------------------

_FABRIC_RET = {
    # lower is better (more breathable)
    "linen": 5.0, "cotton": 7.0, "silk": 8.0,
    "viscose": 9.0, "rayon": 9.0, "lyocell": 8.5, "modal": 9.0,
    "polyamide": 16.0, "nylon": 16.0, "polyester": 16.0, "acrylic": 15.0,
    "spandex": 28.0, "elastane": 28.0, "wool": 20.0, "cashmere": 20.0,
}


def _parse_fabric_perc(s: str):
    """Parse fabric composition percentages from free text."""
    s = (s or "").lower()
    out = []  # list of (name, pct)
    for name, pct in re.findall(r"([a-z\s]+?)\s*(\d{1,3})\s*%", s):
        n = name.strip().split()[-1]  # take last token ("recycled polyester" -> "polyester")
        try:
            out.append((n, float(pct)))
        except Exception:
            pass
    return out


def estimate_ret(it: Dict[str, Any], default: float = 12.0) -> float:
    """Estimate RET (breathability resistance); lower is more breathable."""
    mix = _parse_fabric_perc(it.get("fabric_composition", ""))
    if not mix:
        txt = _txt_blob(it)
        for k in _FABRIC_RET:
            if k in txt:
                return _FABRIC_RET[k]
        return default
    total = 0.0
    for name, pct in mix:
        base = None
        for k in _FABRIC_RET:
            if k in name:
                base = _FABRIC_RET[k]
                break
        if base is None:
            base = default
        total += base * (pct / 100.0)
    return total or default


def has_lining(it: Dict[str, Any]) -> bool:
    """Detect if item is lined based on text hints."""
    t = _txt_blob(it)
    return any(k in t for k in ["lining:", "lined", "jersey lining", "fully lined"])

# -------------------------------------------------
# Hot-weather fabric & coverage bias helpers
# -------------------------------------------------

HOT_TEMP_C = 28.0

# Bonus/penalty keyword dictionaries (regex: bonus value)
FABRIC_HOT_BONUS_KEYWORDS = {
    r"\blinen\b": 0.45,
    r"\bhemp\b": 0.35,
    r"\bramie\b": 0.30,
    r"\b(cotton|% *cotton)\b": 0.25,
    r"\blyocell\b|\btencel\b": 0.30,
    r"\bmodal\b": 0.25,
    r"\bbamboo\b": 0.25,
    r"\bsilk\b": 0.20,
    r"\brayon\b|\bviscose\b": 0.15,
    r"\bpolyamide\b|\bnylon\b": 0.10,
    # weave/knit structure
    r"\bpoplin\b": 0.20,
    r"\bvoile\b": 0.25,
    r"\blawn\b": 0.25,
    r"\bgauze\b|\bmuslin\b": 0.25,
    r"\bseersucker\b": 0.30,
    r"\bchambray\b": 0.25,
    r"\boxford\b": 0.10,
    r"\bpiqu[ée]\b": 0.10,
    r"\bmesh\b|\bperforated\b": 0.35,
    r"\beyelet\b|\bbroderie\b|\blace\b": 0.20,
    r"\bcrochet\b|\bopen-?knit\b": 0.20,
    # coverage cues
    r"\b(sleeveless|tank|vest|cap sleeve)\b": 0.15,
    r"\bshorts?\b|\bmini\b|\bskort\b": 0.20,
}

HOT_TECH_BONUS_KEYWORDS = {
    r"\b(drymove|dri-?fit|coolmax|climalite|aero?ready|dry[- ]?ex|heatgear|airism|moisture[- ]?wicking|quick dry|breathable)\b": 0.35
}

HOT_PENALTY_KEYWORDS = {
    r"\bpolyester\b": -0.25,
    r"\bacrylic\b": -0.25,
    r"\bwool\b": -0.15,
    r"\bfleece\b|\bpolar\b": -0.60,
    r"\bleather\b|\bpu\b|\bcoated\b": -0.50,
    r"\bquilt(ed)?\b|\bpadded\b|\blined\b": -0.40,
    r"\bbrushed\b|\bnap(ped)?\b": -0.25,
}

HOT_MERINO_BONUS = { r"\bmerino\b": 0.10 }


def _text_blob_full(item: Dict[str, Any]) -> str:
    return " ".join([
        str(item.get("name", "")),
        str(item.get("type", "")),
        str(item.get("description", "")),
        " ".join([f"{k} {v}" for k, v in (item.get("key_value_description") or {}).items()]),
        str(item.get("fabric_composition", "")),
    ]).lower()


def is_denim(item: Dict[str, Any]) -> bool:
    txt = _text_blob_full(item)
    return ("denim" in txt) or ("jean" in txt)


def is_short_bottom(item: Dict[str, Any]) -> bool:
    blob = _text_blob_full(item)
    return any(k in blob for k in ["short", "mini", "skort"])  # covers shorts & short skirts

# Skirt length helpers
def _is_midi(blob: str) -> bool:
    return "midi" in blob


def _is_maxi(blob: str) -> bool:
    return any(k in blob for k in ["maxi", "ankle length", "floor length", "full length"])


def fabric_hot_bonus(item: Dict[str, Any], temp_c: float) -> float:
    """Return hot-weather fabric bonus/penalty value for a single item."""
    if temp_c < HOT_TEMP_C:
        return 0.0

    blob = _text_blob_full(item)

    # Denim whitelist bonus
    denim_bonus = 0.0
    if is_denim(item) and is_short_bottom(item):
        denim_bonus = 0.20

    bonus = 0.0
    for pat, val in {**FABRIC_HOT_BONUS_KEYWORDS, **HOT_TECH_BONUS_KEYWORDS, **HOT_MERINO_BONUS}.items():
        if re.search(pat, blob):
            bonus += val

    penalty = 0.0
    for pat, val in HOT_PENALTY_KEYWORDS.items():
        if re.search(pat, blob):
            penalty += val

    # Reduce penalty for tech fabrics
    if any(re.search(pat, blob) for pat in HOT_TECH_BONUS_KEYWORDS):
        penalty *= 0.3

    return bonus + penalty + denim_bonus


@dataclass(frozen=True)
class Outfit:
    top: Dict[str, Any]
    bottom: Dict[str, Any]
    outer: Optional[Dict[str, Any]]
    base_score: float
    outer_score: Optional[float]

    def to_jsonable(self) -> Dict[str, Any]:
        return {
            "top": self.top,
            "bottom": self.bottom,
            "outer": self.outer,
            "scores": {
                "base_pair_score": float(self.base_score),
                "outer_pair_score": (None if self.outer_score is None else float(self.outer_score)),
            },
        }

# -----------------------------
# Loading helpers
# -----------------------------

def _load_joblib(path: Path):
    if joblib is None:
        raise RuntimeError("joblib is required to load the trained model. Please `pip install joblib`.")
    return joblib.load(str(path))

def _load_pickle(path: Path) -> Dict[str, np.ndarray]:
    with path.open("rb") as f:
        data = pickle.load(f)
    fixed = {}
    for k, v in data.items():
        fixed[str(k)] = np.asarray(v, dtype=np.float32)
    return fixed

def _resolve_existing_path(candidates: List[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists():
            return p
    return None

def _fallback_target_clo(weather: Dict[str, Any]) -> float:
    def _safe_float(val, default):
        try:
            if val is None or val == "":
                return float(default)
            return float(val)
        except Exception:
            return float(default)

    t = _safe_float(weather.get("temp_c"), 18.0)
    rh = _safe_float(weather.get("rh", weather.get("humidity")), 50.0)
    wind = _safe_float(weather.get("wind", weather.get("wind_kmh")), 5.0)
    rain = _safe_float(weather.get("rain", weather.get("precip_mm")), 0.0)

    # Yağış ihtimali veya mevcut yağış bilgisi
    is_snow_or_rain: bool = bool(weather.get("is_rain", False)) or float(weather.get("precip_prob", 0.0)) >= 60.0 or rain >= 0.5

    # Sıcaklığa bağlı baz clo değeri
    if t <= -5:
        base = 1.9
    elif t <= 0:
        base = 1.6
    elif t <= 5:
        base = 1.3
    elif t <= 10:
        base = 1.1
    elif t <= 15:
        base = 0.9
    elif t <= 20:
        base = 0.7
    elif t <= 25:
        base = 0.5
    else:
        base = 0.3

    # Rüzgâr (wind chill) etkisi: 8 km/h üzeri her 10 km/h için +0.10
    if wind > 8:
        base += 0.10 * ((wind - 8.0) / 10.0)

    # Yağış/sulu kar etkisi
    if is_snow_or_rain:
        base += 0.25

    # Yüksek nem soğukta hissi artırır
    if t <= 5 and rh >= 80:
        base += 0.10

    # Kapak değerler
    return float(max(0.3, min(base, 2.4)))

def estimate_target_clo(weather: Dict[str, Any]) -> float:
    if get_target_clo is not None:
        try:
            return float(get_target_clo(weather))  # type: ignore
        except Exception:
            pass
    return _fallback_target_clo(weather)

# -----------------------------
# Feature Engineering
# -----------------------------

def create_feature_vector(item1_id: str, item2_id: str,
                          style_vectors: Dict[str, np.ndarray],
                          color_vectors: Dict[str, np.ndarray]) -> np.ndarray:
    sv1 = style_vectors[item1_id]; sv2 = style_vectors[item2_id]
    if sv1.shape[0] != 384 or sv2.shape[0] != 384:
        raise ValueError(f"Style vectors must be length 384. Got {sv1.shape}, {sv2.shape}")
    cv1 = color_vectors[item1_id]; cv2 = color_vectors[item2_id]
    if cv1.shape[0] != 3 or cv2.shape[0] != 3:
        raise ValueError(f"Color vectors must be length 3 (CIELAB). Got {cv1.shape}, {cv2.shape}")
    style_diff = (sv1 - sv2).astype(np.float32)
    style_prod = (sv1 * sv2).astype(np.float32)
    color_dist = np.linalg.norm(cv1.astype(np.float32) - cv2.astype(np.float32)).astype(np.float32)
    # Append dummy pair-type flags (base_bottom=1, base_mid=0 by default)
    pair_flags = np.array([1.0, 0.0], dtype=np.float32)  # is_bb, is_bm
    feat = np.concatenate([style_diff, style_prod, np.array([color_dist], dtype=np.float32), pair_flags], axis=0)
    if feat.shape[0] != 771:
        raise AssertionError(f"Feature length must be 771, got {feat.shape[0]}")
    return feat.reshape(1, -1)

# -----------------------------
# Catalogue enrichment
# -----------------------------

def _infer_role_from_type(t: str) -> Optional[str]:
    t = (t or "").lower()
    # Treat heavy hoodies with pile/fleece as outer layers
    

    if any(k in t for k in ["shirt","t-shirt","top","blouse","jumper","sweater","hoodie","cardigan","vest","camisole","tank"]):
        return "top"
    if any(k in t for k in ["jean","pant","trouser","skirt","short","culotte","legging"]):
        return "bottom"
    if any(k in t for k in ["coat","jacket","blazer","parka","anorak","overcoat","gilet","puffer","trench"]):
        return "outer"
    return None

def _load_catalogue() -> Dict[str, Dict[str, Any]]:
    """Load a catalogue mapping item_id -> {role, clo, ...} if available."""
    cat = {}
    # Try JSON catalogue
    p_json = _resolve_existing_path([Path("ML/item_catalogue.json"), Path("/mnt/data/ML/item_catalogue.json")])
    if p_json:
        try:
            data = json.loads(p_json.read_text())
            for it in data:
                iid = str(it.get("item_id"))
                if not iid: continue
                cat[iid] = it
        except Exception:
            pass
    # Try NPZ features
    if not cat:
        p_npz = _resolve_existing_path([Path("ML/features_v1.npz"), Path("/mnt/data/ML/features_v1.npz")])
        if p_npz:
            try:
                npz = np.load(p_npz, allow_pickle=True)
                ids = [str(x) for x in npz["item_id"]]
                roles = [str(x) for x in npz["role"]]
                clos = np.array(npz["clo"], dtype=float)
                for iid, r, c in zip(ids, roles, clos):
                    cat[iid] = {"item_id": iid, "role": r, "clo": float(c)}
            except Exception:
                pass
    return cat

    

# -----------------------------
# Role normalisation helper (used by OutfitMapper)
# -----------------------------
def normalize_role(item: Dict[str, Any]) -> Optional[str]:
    roles_explicit = {str(r).strip().lower() for r in (item.get("roles") or []) if r}
    if "outer" in roles_explicit:
        return "outer"
    if "mid-layer" in roles_explicit or "mid" in roles_explicit:
        return "mid"
    if "base" in roles_explicit or "top" in roles_explicit:
        return "base"
    if "bottom" in roles_explicit or "bottoms" in roles_explicit:
        return "bottom"

    if item.get("role"):
        return str(item["role"]).lower()

    return _infer_role_from_type(str(item.get("type", "")))


def enrich_items(items: List[Dict[str, Any]], verbose: bool=True) -> List[Dict[str, Any]]:
    """Ensure each item has item_id, role, clo. Attempt to fill from catalogue;
    infer role from type; default clo if missing.
    """
    catalogue = _load_catalogue()
    out = []
    filled_from_cat = 0; inferred_role = 0; defaulted_clo = 0
    for it in items:
        iid = str(it.get("item_id", "")).strip()
        if not iid:
            continue  # skip invalid
        base = dict(it)
        # Fill from catalogue
        if iid in catalogue:
            # Don't overwrite existing fields, just fill missing
            for k, v in catalogue[iid].items():
                if k not in base or base[k] in (None, "", 0):
                    base[k] = v
            filled_from_cat += 1
        # Role
        role = base.get("role")
        if not role:
            # Infer role using both 'type' and 'name' fields for better accuracy (e.g., jeans/shorts in name)
            combined_txt = f"{base.get('type','')} {base.get('name','')}"
            role = _infer_role_from_type(str(combined_txt))
            if role:
                base["role"] = role
                inferred_role += 1
        # ------------------------------------
        # CLO – heuristic defaulting when missing
        # ------------------------------------
        if "clo" not in base or base["clo"] in (None, "", 0):
            r = str(base.get("role", "")).lower()
            t = str(base.get("type", "")).lower()
            nm = str(base.get("name", "")).lower()
            desc = (
                str(base.get("description", "")).lower()
                + " "
                + " ".join([f"{k}:{v}" for k, v in (base.get("key_value_description") or {}).items()]).lower()
            )

            heur = 0.3  # generic light layer

            if r == "outer":
                if any(k in (t + nm + desc) for k in ["puffer", "parka", "down", "quilt", "insulated"]):
                    heur = 1.6  # heavy insulated outerwear
                elif any(k in (t + nm + desc) for k in ["coat", "overcoat", "trench", "wool"]):
                    heur = 1.2  # mid-weight coats
                elif any(k in (t + nm + desc) for k in ["blazer", "jacket"]):
                    heur = 0.7  # light jackets/blazers
                else:
                    heur = 0.6  # generic outer layer

            elif r == "top":
                blob = (str(base.get("type", "")) + " " + str(base.get("name", "")) + " " + str(base.get("description", ""))).lower()
                if any(k in blob for k in ["hoodie", "sweater", "jumper", "fleece", "cardigan", "knit"]):
                    heur = 0.7  # knit & similar considered warmer
                    if any(k in blob for k in ["wool", "cashmere", "alpaca", "mohair", "angora", "sherpa", "teddy", "pile"]):
                        heur = 0.8  # high warmth knit
                elif _is_sleeveless(base) or any(
                    k in (t + nm + desc) for k in ["t-shirt", "tee", "camisole", "tank", "vest"]
                ):
                    heur = 0.2
                else:
                    heur = 0.35

            elif r == "bottom":
                if any(k in (t + nm + desc) for k in ["wool", "fleece", "thermal", "lined"]):
                    heur = 0.5
                elif any(k in (t + nm + desc) for k in ["skirt", "short"]):
                    heur = 0.25
                else:
                    heur = 0.35

            base["clo"] = float(heur)
            defaulted_clo += 1

        out.append(base)

    if verbose:
        print(
            f"[enrich] total={len(items)} | from_catalogue={filled_from_cat} | "
            f"inferred_role={inferred_role} | defaulted_clo={defaulted_clo}"
        )
    return out

def _score_pair(self, top_id: str, bottom_id: str) -> float:
    
    feat = create_feature_vector(top_id, bottom_id, self.style_vectors, self.color_vectors)  # (1, 771)
    X_df = _to_feature_frame(feat)
    score = float(self.model.predict(X_df)[0])
    return score
# -----------------------------
# Core Mapper
# -----------------------------

def _is_sleeveless(item):
    t = str(item.get("type","")).lower()
    name = str(item.get("name","")).lower()
    desc = (str(item.get("description","")).lower() + " " +
            " ".join([f"{k}:{v}" for k,v in (item.get("key_value_description") or {}).items()]).lower())
    if "sleeveless" in desc:
        return True
    if "vest" in t or "vest" in name:
        return True
    if "camisole" in t or "tank" in t:
        return True
    return False

class OutfitMapper:
    def __init__(self, model_path: Path, style_vec_path: Path, color_vec_path: Path, verbose: bool=True) -> None:
        self.verbose = verbose
        self.model = _load_joblib(model_path)
        if self.verbose: print(f"[ok] Loaded model from: {model_path}")
        self.style_vectors = _load_pickle(style_vec_path)
        self.color_vectors = _load_pickle(color_vec_path)
        if self.verbose: print(f"[ok] Loaded {len(self.style_vectors)} style vectors and {len(self.color_vectors)} color vectors.")
        if hasattr(self.model, "n_features_in_"):
            nfi = int(getattr(self.model, "n_features_in_", -1))
            if nfi != 771 and self.verbose:
                print(f"[warn] Model expects {nfi} features; pipeline produces 771. Check training features.")

        # expose to AestheticScorer (771-dim pipeline)
        global STYLE_VECS, COLOR_VECS
        STYLE_VECS = self.style_vectors
        COLOR_VECS = self.color_vectors

    @staticmethod
    def split_pools(items: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        tops: List[Dict[str, Any]] = []
        bottoms: List[Dict[str, Any]] = []
        outers: List[Dict[str, Any]] = []

        for it in items:
            primary = normalize_role(it)
            if primary in {"base", "top"}:
                tops.append(it)
            elif primary == "mid":
                # mid-layers are not tops but may be used later
                pass
            elif primary == "bottom":
                bottoms.append(it)
            elif primary == "outer":
                outers.append(it)

        return tops, bottoms, outers

    @staticmethod
    def _within_clo(target: float, item_clo: float, tol: float = 0.5) -> bool:
        return (abs(item_clo - target) <= tol) or (item_clo <= target + tol)

    # ---- helper: aesthetic scoring wrapper (uses module-level scorer) ----
    def _aesthetic_score(self, top: Optional[Dict[str, Any]] = None,
                         bottom: Optional[Dict[str, Any]] = None,
                         outer: Optional[Dict[str, Any]] = None,
                         base: Optional[Dict[str, Any]] = None) -> float:
        """Score a (top, bottom[, outer]) combination.
        If a base layer is provided, treat it as the visible top when a mid/top
        layer is not explicitly given.
        """
        scoring_top = top if top is not None else base
        if scoring_top is None or bottom is None:
            return -1e9
        return _aesthetic_score(
            scoring_top,
            bottom,
            outer,
            _clo(scoring_top),
            _clo(bottom),
            _clo(outer),
        )

    # ---- helper: ranking key (yüksek estetik, düşük ısı boşluğu daha iyi) ----
    def _score_key(self, c: Optional[Dict[str, Any]]) -> float:
        return (c["score"] - c["gap"]) if c else -1e9

    # ---- helper: outer ekleyerek 4. parçayı dene (2'li veya 3'lü taban kombin üzerine) ----
    def _enumerate_with_outer(self, base_combo: List[Dict[str, Any]],
                              target_clo: float,
                              outers: List[Dict[str, Any]],
                              topk_outer: int = 15) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []

        # bottom'ı bul (roles listesi yoksa eski 'role' alanına da bak)
        bottom_in_combo: Optional[Dict[str, Any]] = None
        for p in base_combo:
            roles = p.get("roles", [])
            if not roles and p.get("role"):
                roles = [p["role"]]
            roles_lower = {str(r).lower() for r in roles}
            if "bottom" in roles_lower:
                bottom_in_combo = p
                break

        # top/base parçaları ayır
        top_piece: Optional[Dict[str, Any]] = None
        base_piece: Optional[Dict[str, Any]] = None
        if len(base_combo) == 3:
            # [base, mid, bottom]
            base_piece = base_combo[0]
            top_piece = base_combo[1]
        elif len(base_combo) == 2:
            # [top, bottom]
            top_piece = base_combo[0]

        for outer in outers[:topk_outer]:
            clo_total = sum(p.get("clo", 0.0) for p in base_combo) + outer.get("clo", 0.0)
            score = self._aesthetic_score(
                top=top_piece,
                bottom=bottom_in_combo,
                outer=outer,
                base=base_piece,
            )
            gap = abs(clo_total - target_clo)
            results.append({
                "combo": base_combo + [outer],
                "clo": clo_total,
                "score": score,
                "gap": gap,
            })
        results.sort(key=self._score_key, reverse=True)
        return results

    # ---- çekirdek: 2'li → 3'lü → (gerekirse) 4'lü arama stratejisi ----
    def _build_candidates(self, items: List[Dict[str, Any]], weather: Dict[str, Any],
                          topk: int = 5, pool_limit: int = 200) -> Tuple[List[Dict[str, Any]], float]:
        # target clo (module-level estimate handles external get_target_clo if present)
        target_clo = estimate_target_clo(weather)

        # roles guard: eski 'role' alanından fallback ile list oluştur
        def roles_of(i: Dict[str, Any]) -> List[str]:
            r = i.get("roles")
            if r:
                return [str(x).lower() for x in r]
            return [str(i["role"]).lower()] if i.get("role") else []

        # Role-based pools with hygiene rules:
        # - Do not treat items that are also 'outer' as bases.
        # - Do not use dresses as mid-layers; keep mids to layering pieces.
        bases_all = [i for i in items if "base" in roles_of(i)]
        bases = [i for i in bases_all if "outer" not in roles_of(i)] or bases_all

        def _is_dress(it: Dict[str, Any]) -> bool:
            txt = (str(it.get("type", "")) + " " + str(it.get("name", ""))).lower()
            return "dress" in txt

        mids_raw = [i for i in items if "mid-layer" in roles_of(i)]
        mids = [i for i in mids_raw if ("outer" not in roles_of(i)) and (not _is_dress(i))] or mids_raw

        outers  = [i for i in items if "outer"     in roles_of(i)]
        bottoms = [i for i in items if "bottom"    in roles_of(i)]

        candidates: List[Dict[str, Any]] = []

        # --- 2'li kombinler: yalnızca "base" olabilen üst + bottom (mid-layer tek başına üst değildir)
        for top in bases:
            for bottom in bottoms:
                clo = top.get("clo", 0.0) + bottom.get("clo", 0.0)
                score = self._aesthetic_score(top=top, bottom=bottom)
                gap = abs(clo - target_clo)
                candidates.append({"combo": [top, bottom], "clo": clo, "score": score, "gap": gap})

        candidates.sort(key=self._score_key, reverse=True)
        candidates = candidates[:pool_limit]  # outer denemeleri için geniş havuz

        # --- 3'lü kombinler: base + mid + bottom  (cardigan burada devreye girer)
        layered_candidates: List[Dict[str, Any]] = []
        base_topk   = sorted(bases,   key=lambda i: float(i.get("clo", 0.0)), reverse=True)[:20]
        mid_topk    = sorted(mids,    key=lambda i: float(i.get("clo", 0.0)), reverse=True)[:20]
        bottom_topk = sorted(bottoms, key=lambda i: float(i.get("clo", 0.0)), reverse=True)[:20]

        for base in base_topk:
            for mid in mid_topk:
                if base.get("item_id") == mid.get("item_id"):
                    continue  # aynı ürünü iki rolde kullanma
                for bottom in bottom_topk:
                    clo = base.get("clo", 0.0) + mid.get("clo", 0.0) + bottom.get("clo", 0.0)
                    if clo < target_clo * 0.6:
                        continue  # aşırı soğuk kombinleri ele
                    score = self._aesthetic_score(top=mid, bottom=bottom, base=base)
                    gap = abs(clo - target_clo)
                    layered_candidates.append({"combo": [base, mid, bottom], "clo": clo, "score": score, "gap": gap})

        layered_candidates.sort(key=self._score_key, reverse=True)

        # 2'li + 3'lü tek havuzda
        pool = (candidates + layered_candidates)[: max(pool_limit, topk * 10)]

        # --- Outer aşaması: ortam soğuksa veya kombin yetersizse 4'lü dene
        need_outer_env = target_clo >= 1.0  # basit eşik; istersen hava/yağmur/rüzgarla zenginleştir
        extended: List[Dict[str, Any]] = []
        for cand in pool[:50]:
            if need_outer_env or (cand["clo"] < target_clo * 0.95):
                extended += self._enumerate_with_outer(cand["combo"], target_clo, outers, topk_outer=15)

        all_cands = pool + extended
        all_cands.sort(key=self._score_key, reverse=True)
        return all_cands[:topk], target_clo

    def filter_by_weather(self, items: List[Dict[str, Any]], weather: Dict[str, Any], need_outer: Optional[bool]=None
                          ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], float, bool]:
        tops, bottoms, outers = self.split_pools(items)
        target = estimate_target_clo(weather)
        auto_outer_required = target >= 0.9
        outer_required = bool(need_outer) if need_outer is not None else auto_outer_required

        # primary filters
        tops_f = [t for t in tops if self._within_clo(target*0.5, float(t.get("clo", 0.3)), tol=0.6)]
        bottoms_f = [b for b in bottoms if self._within_clo(target*0.5, float(b.get("clo", 0.3)), tol=0.6)]
        outers_f = [o for o in outers if _clo(o) >= 0.2]

        

        # ---- Cold guards: zayıf kumaşları eleyip uzun kolu zorunlu kıl ----
        def _blob(it):
            return (
                (it.get("fabric_composition", "") or "") + " " +
                (it.get("type", "") or "") + " " +
                (it.get("name", "") or "") + " " +
                (it.get("description", "") or "") + " " +
                " ".join([f"{k}:{v}" for k, v in (it.get("key_value_description") or {}).items()])
            ).lower()

        try:
            temp_c = float(weather.get("temp_c", 18.0))
        except Exception:
            temp_c = 18.0

        cold = (temp_c <= 12.0) or (target >= 1.0)
        very_cold = (temp_c <= 0.0) or (target >= 1.6)
        wet = bool(weather.get("is_rain", False)) or float(weather.get("precip_prob", 0.0)) >= 60.0 or float(weather.get("rain", 0.0)) >= 0.5

        # ---- Hard guard: skirts in deep-cold need lining/warmth ----
        def _is_skirt(it):
            return "skirt" in _txt_blob(it)

        def _is_warm_bottom(it):
            t = _txt_blob(it)
            return any(k in t for k in ["wool", "fleece", "thermal", "lined"]) or "skort" in t  # skort built-in shorts

        if very_cold or (cold and wet):
            # deep cold + wet: skirt needs warmth; shorts zaten eleniyor
            bottoms_f = [b for b in bottoms_f if ((not _is_skirt(b)) or _is_warm_bottom(b))]

        if cold:
            weak_tops: List[Dict[str, Any]] = []
            strong_tops: List[Dict[str, Any]] = []
            for t in tops_f:
                txt = _blob(t)
                # zayıf kumaş: viscose/rayon/modal/lyocell/linen/silk veya 3/4 sleeve/short sleeve
                weak = any(k in txt for k in [
                    "viscose", "rayon", "modal", "lyocell", "linen", "silk", "3/4 sleeve", "short sleeve"
                ])
                # kolsuzlar zaten yukarıda eleniyor (gerekirse)
                if very_cold and weak:
                    weak_tops.append(t)
                else:
                    strong_tops.append(t)
            if strong_tops:
                tops_f = strong_tops  # boşaltmıyorsa zayıfları çıkar

            # Altlar: çok soğuk/ıslaksa denim ve bilek boyu/bol paça dezavantaj
            strong_bottoms: List[Dict[str, Any]] = []
            for b in bottoms_f:
                txt = _blob(b)
                is_jeans = ("jean" in txt or "denim" in txt)
                ankle = ("ankle length" in txt)
                wide = ("wide" in txt and "leg" in txt)
                lined = any(k in txt for k in ["wool", "fleece", "thermal", "lined"])
                if very_cold and wet:
                    # çok soğuk ve ıslakta: astarlı/termal olanları öne al
                    if lined:
                        strong_bottoms.append(b)
                    else:
                        # jeans/ankle/wide kötü durumda: dışarıda bırak
                        if not (is_jeans or ankle or wide):
                            strong_bottoms.append(b)
                else:
                    strong_bottoms.append(b)
            if strong_bottoms:
                bottoms_f = strong_bottoms

        # ---- Cold preference: knit & pants tercihleri ----
        if cold:
            # ÜST: knit/warm materyal tercih et (boşaltıyorsa geri dön).
            pre_tops = tops_f[:]
            pref_tops = [t for t in tops_f if _is_knit_warm_top(t)]
            if pref_tops:
                tops_f = pref_tops
            else:
                tops_f = pre_tops  # tamamen boşaltma

            # ALT: pantolon/leggings öncelik ver (boşaltıyorsa geri dön).
            pre_bottoms = bottoms_f[:]
            pref_bottoms = [b for b in bottoms_f if any(k in _txt_blob(b) for k in [
                "trouser", "pant", "jean", "leggings", "jogger", "chino"
            ])]
            if pref_bottoms:
                bottoms_f = pref_bottoms
            else:
                bottoms_f = pre_bottoms

        # ---------------------------------------------
        # Weather sanity: in chilly weather (≤12 °C) or
        # when the required target Clo is high (≥1.0),
        # avoid sleeveless/vest tops. If removal wipes
        # out the entire pool, revert to the original.
        # ---------------------------------------------
        try:
            temp_c = float(weather.get("temp_c", 18.0))
        except Exception:
            temp_c = 18.0

        if temp_c <= 12.0 or target >= 1.0:
            _pre = tops_f[:]
            tops_f = [t for t in tops_f if not _is_sleeveless(t)]
            if not tops_f:
                tops_f = _pre

        # ---------------------------------------------
        # Hot-weather tightening: in very warm/humid weather,
        # prefer genuinely light garments by capping CLO.
        # ---------------------------------------------
        rh = 0.0
        try:
            rh = float(weather.get("rh", 0.0))
        except Exception:
            rh = 0.0
        hot = (temp_c >= 27.0) or ((target <= 0.5) and (rh >= 70.0))
        if hot:
            tops_cap = max(0.40, target + 0.10)
            bottoms_cap = max(0.40, target + 0.10)
            _pre_t, _pre_b = tops_f[:], bottoms_f[:]

            # Prefer breathable (low RET) and unlined pieces
            def _breathable(pool):
                # Keep best-breathing half; avoid lined if possible
                scored = [(estimate_ret(x), has_lining(x), x) for x in pool]
                if not scored:
                    return pool
                scored.sort(key=lambda z: (z[0], z[1]))  # lower RET, then unlined
                keep = max(1, len(scored) // 2)
                return [z[2] for z in scored[:keep]]

            tops_f = [t for t in tops_f if float(t.get("clo", 0.3)) <= tops_cap]
            bottoms_f = [b for b in bottoms_f if float(b.get("clo", 0.3)) <= bottoms_cap]

            # Apply breathability pruning (fallback if empty)
            t2 = _breathable(tops_f)
            b2 = _breathable(bottoms_f)
            if t2:
                tops_f = t2
            if b2:
                bottoms_f = b2

            if not tops_f:
                tops_f = _pre_t
            if not bottoms_f:
                bottoms_f = _pre_b

            # --- Hot strict prefs: short sleeve/tanks; shorts/mini/knee ---
            def _is_long_sleeve(it):
                t = _txt_blob(it)
                return ("long sleeve" in t) or ("3/4" in t)

            def _is_short_sleeve_or_sleeveless(it):
                t = _txt_blob(it)
                return any(k in t for k in ["sleeveless", "tank", "camisole", "vest", "short sleeve", "cap sleeve"])

            def _is_shorts_or_skort(it):
                t = _txt_blob(it)
                return ("shorts" in t) or ("skort" in t)

            def _is_skirt_mini_or_knee(it):
                t = _txt_blob(it)
                return ("skirt" in t) and any(k in t for k in ["mini", "short", "knee"])

            def _is_denim(it):
                return ("denim" in _txt_blob(it)) or ("jean" in _txt_blob(it))

            def _is_ankle_or_long(it):
                t = _txt_blob(it)
                return ("ankle length" in t) or (("long" in t) and ("length" in t))

            # TOPS: Prefer sleeveless/short sleeves
            pre_t = tops_f[:]
            pref_t = [t for t in tops_f if _is_short_sleeve_or_sleeveless(t)]
            if pref_t:
                tops_f = pref_t
            else:
                # keep most breathable half among remaining longs
                tops_f = sorted(tops_f, key=lambda x: (estimate_ret(x), has_lining(x)))[: max(1, len(tops_f)//2)]
                if not tops_f:
                    tops_f = pre_t

            # BOTTOMS: shorts/skort or mini/knee skirt first
            pre_b = bottoms_f[:]
            pref_b = [b for b in bottoms_f if _is_shorts_or_skort(b) or _is_skirt_mini_or_knee(b)]
            if pref_b:
                # remove denim & lined if possible
                cand = [b for b in pref_b if not has_lining(b)]
                cand = [b for b in cand if not (_is_denim(b) and _is_ankle_or_long(b))]
                bottoms_f = cand if cand else pref_b
            else:
                # unavoidable pants: drop ankle/long, denim, lined
                nb = [b for b in bottoms_f if not _is_ankle_or_long(b)]
                nb = [b for b in nb if not (_is_denim(b) and not _is_shorts_or_skort(b) and not _is_skirt_mini_or_knee(b))]
                if nb:
                    bottoms_f = nb
                else:
                    scored = sorted(bottoms_f, key=lambda x: (estimate_ret(x), has_lining(x)))
                    bottoms_f = scored[: max(1, len(scored)//2)]
                    if not bottoms_f:
                        bottoms_f = pre_b

        # ---------------------------------------------
        # If target is high (cold/snowy), prefer heavier outers by
        # imposing a minimum outer CLO threshold. Fall back if empty.
        # ---------------------------------------------
        min_outer = 0.0
        if target >= 1.8:
            min_outer = 1.2  # very cold: puffer/parka tier
        elif target >= 1.4:
            min_outer = 1.0  # cold: heavy coat
        elif target >= 1.0:
            min_outer = 0.8  # cool: jacket/coat
        if min_outer > 0:
            _pre_o = outers_f[:]
            outers_f = [o for o in outers_f if _clo(o) >= min_outer]
            if not outers_f:
                outers_f = _pre_o  # don't empty the pool

        # if too restrictive, fall back to unfiltered pools instead of returning empty
        if not tops_f and tops: tops_f = tops
        if not bottoms_f and bottoms: bottoms_f = bottoms
        if not outers_f and outers: outers_f = outers

        if self.verbose:
            print(f"[thermals] target_clo={target:.2f} -> pools: tops={len(tops_f)}, bottoms={len(bottoms_f)}, outers={len(outers_f)} (need_outer={outer_required})")
        return tops_f, bottoms_f, outers_f, target, outer_required

    def _score_batch_pairs(self, pairs: List[Tuple[str, str]], batch_size: int=1024, pair_type_hint: str="base_bottom") -> List[float]:
        scores: List[float] = []
        i = 0
        while i < len(pairs):
            chunk = pairs[i:i+batch_size]
            feats = []
            valid_idx = []
            for j, (a, b) in enumerate(chunk):
                try:
                    feats.append(create_feature_vector(a, b, self.style_vectors, self.color_vectors))
                    valid_idx.append(j)
                except Exception as e:
                    if self.verbose: print(f"[skip] pair ({a},{b}) -> {e}")
            if feats:
                X = np.vstack(feats).astype(np.float32)

                # --- Ensure we have expected 771-dim features ---
                if X.shape[1] != 771:
                    if self.verbose:
                        print(f"[fix] Rebuilding features to 771 dims (had {X.shape[1]}) for {len(chunk)} pairs using style/color vectors...")

                    def _get_vec(src: Dict[str, np.ndarray], key: str):
                        v = src.get(key)
                        return np.asarray(v, dtype=np.float32) if v is not None else None

                    rebuilt = []
                    for (aid, bid) in chunk:
                        sv_a = _get_vec(self.style_vectors, aid)
                        sv_b = _get_vec(self.style_vectors, bid)
                        cv_a = _get_vec(self.color_vectors, aid)
                        cv_b = _get_vec(self.color_vectors, bid)

                        if (sv_a is None) or (sv_b is None):
                            sd = np.zeros(384, dtype=np.float32)
                            sp = np.zeros(384, dtype=np.float32)
                        else:
                            sd = (sv_a - sv_b).astype(np.float32)
                            sp = (sv_a * sv_b).astype(np.float32)

                        if (cv_a is None) or (cv_b is None):
                            cd = np.array([0.0], dtype=np.float32)
                        else:
                            cd = np.array([float(np.linalg.norm(cv_a - cv_b))], dtype=np.float32)

                        # pair type one-hot flags
                        if pair_type_hint == "base_mid":
                            pt = np.array([0.0, 1.0], dtype=np.float32)
                        else:
                            pt = np.array([1.0, 0.0], dtype=np.float32)

                        rebuilt.append(np.concatenate([sd, sp, cd, pt], axis=0))  # 771
                    X = np.vstack(rebuilt).astype(np.float32)

                # Predict (try DataFrame first for named columns)
                try:
                    X_df = _to_feature_frame(X)
                    preds = self.model.predict(X_df)
                except Exception:
                    preds = self.model.predict(X)

                out = [None]*len(chunk)
                k=0
                for j in valid_idx:
                    out[j] = float(preds[k]); k+=1
                scores.extend([(-1e9 if s is None else s) for s in out])
            else:
                scores.extend([-1e9]*len(chunk))
            i += batch_size
        return scores

    def suggest_outfit(self, items: List[Dict[str, Any]], weather: Dict[str, Any], need_outer: Optional[bool]=None, return_candidates: int=0) -> Dict[str, Any]:
        tops, bottoms, outers, target_clo, outer_required = self.filter_by_weather(items, weather, need_outer)
        if not tops or not bottoms:
            raise ValueError("No valid tops/bottoms after thermal filtering. Ensure items have roles (top/bottom).")

        base_pairs = [(t["item_id"], b["item_id"]) for t, b in product(tops, bottoms)]
        if self.verbose: print(f"[pairs] evaluating {len(base_pairs)} base pairs...")
        base_scores = self._score_batch_pairs(base_pairs)

        id_to_top = {t["item_id"]: t for t in tops}
        id_to_bottom = {b["item_id"]: b for b in bottoms}

        # -------------------------------------------------
        # Build scored_base with optional cold-weather penalty
        # -------------------------------------------------
        scored_base = []

        # Determine if the conditions are "cold" (≤12 °C) or target clo ≥1.0
        try:
            temp_c = float(weather.get("temp_c", 18.0))
        except Exception:
            temp_c = 18.0
        cold = (temp_c <= 12.0) or (estimate_target_clo(weather) >= 1.0)
        very_cold = (temp_c <= 0.0) or (estimate_target_clo(weather) >= 1.6)
        wet = bool(weather.get("is_rain", False)) or float(weather.get("precip_prob", 0.0)) >= 60.0 or float(weather.get("rain", 0.0)) >= 0.5

        for (tid, bid), s in zip(base_pairs, base_scores):
            if s <= -1e8:
                continue
            top_it = id_to_top[tid]
            bottom_it = id_to_bottom[bid]

            pen = 0.0
            # Penalise sleeveless tops in cold weather
            if cold and _is_sleeveless(top_it):
                pen -= 1.0

            # --- Cold fabric penalties ---
            t_txt = (str(top_it.get("fabric_composition", "")) + " " + str(top_it.get("description", ""))).lower()
            b_txt = (str(bottom_it.get("fabric_composition", "")) + " " + str(bottom_it.get("description", ""))).lower()

            if cold:
                # zayıf üst kumaş: viscose/rayon/modal/lyocell/linen/silk
                if any(k in t_txt for k in ["viscose", "rayon", "modal", "lyocell", "linen", "silk"]):
                    pen -= (0.6 if very_cold else 0.4)

                # alt: ıslakta denim/ankle/wide paça cezası
                if wet:
                    if ("denim" in b_txt or "jean" in b_txt):
                        pen -= (0.8 if very_cold else 0.5)
                    if "ankle length" in b_txt:
                        pen -= 0.3
                    if "wide leg" in b_txt:
                        pen -= 0.2

                # bonuslar: astarlı/termal/wool/fleece
                if any(k in b_txt for k in ["wool", "fleece", "thermal", "lined"]):
                    pen += 0.4

            # Hot-weather penalties/bonuses
            rh = 0.0
            is_rain = False
            try:
                rh = float(weather.get("rh", 0.0))
            except Exception:
                rh = 0.0
            is_rain = bool(weather.get("is_rain", False)) or float(weather.get("precip_prob", 0.0)) >= 60.0
            hot = (temp_c >= 27.0) or ((estimate_target_clo(weather) <= 0.5) and (rh >= 70.0))

            def _text_blob(item):
                return (str(item.get("type", "")) + " " + str(item.get("name", "")) + " " +
                        str(item.get("description", "")) + " " +
                        " ".join([f"{k}:{v}" for k, v in (item.get("key_value_description") or {}).items()])
                       ).lower()

            if hot:
                t_blob = _text_blob(top_it)
                b_blob = _text_blob(bottom_it)
                # Penalise heavy/stifling materials and long sleeves
                if any(k in t_blob for k in ["denim","wool","fleece","sweater","jacket","coat"]):
                    pen -= 0.6
                if any(k in b_blob for k in ["denim","wool","fleece"]):
                    pen -= 0.8
                # Long-sleeve (non-sleeveless) tops are less ideal in heat
                if ("long sleeve" in t_blob) and (not _is_sleeveless(top_it)):
                    pen -= 0.4
                # Breezy bonuses
                if any(k in t_blob for k in ["linen","camisole","tank","vest"]):
                    pen += 0.2
                if any(k in b_blob for k in ["skirt","short"]):
                    pen += 0.3
                # Humid/rain preferences: prefer quick-dry synthetics; de-prioritise heavy cotton bottoms
                if rh >= 70.0 or is_rain:
                    if "cotton" in str(bottom_it.get("fabric_composition","")) .lower() and "denim" in b_blob:
                        pen -= 0.2
                    if any(k in str(top_it.get("fabric_composition","")) .lower() for k in ["poly", "nylon"]):
                        pen += 0.1

                # Fabric breathability bonus/penalty
                t_ret = estimate_ret(top_it)
                b_ret = estimate_ret(bottom_it)

                # Big bonus for airy naturals
                if any(k in t_blob for k in ["linen", "cotton", "viscose", "rayon", "lyocell", "modal", "seersucker", "crinkle", "eyelet"]):
                    pen += 0.4
                if any(k in b_blob for k in ["linen", "cotton", "viscose", "rayon", "lyocell", "modal", "seersucker", "crinkle"]):
                    pen += 0.3

                # Penalize poly-heavy and lined in heat
                def _poly_heavy(it):
                    mix = _parse_fabric_perc(it.get("fabric_composition", ""))
                    return bool(mix) and sum(p for n, p in mix if ("polyester" in n or "polyamide" in n or "nylon" in n)) >= 60.0

                if _poly_heavy(top_it):
                    pen -= 0.4
                if _poly_heavy(bottom_it):
                    pen -= 0.3
                if has_lining(top_it):
                    pen -= 0.3
                if has_lining(bottom_it):
                    pen -= 0.3

                # RET-based shaping (lower is better)
                pen += (-0.02) * (t_ret - 10.0)
                pen += (-0.015) * (b_ret - 10.0)

                # Extra bump for sleeveless/cami/tank
                if any(k in t_blob for k in ["camisole", "tank", "vest", "sleeveless", "cap sleeve"]):
                    pen += 0.1

                # Fabric hot bonus integration
                pen += fabric_hot_bonus(top_it, temp_c)
                pen += fabric_hot_bonus(bottom_it, temp_c)

                # --- Hot strengthening ---
                is_long = ("long sleeve" in t_blob) or ("3/4" in t_blob)
                is_tiny_top = any(k in t_blob for k in ["sleeveless", "tank", "camisole", "vest", "cap sleeve"])

                is_shorts = ("shorts" in b_blob) or ("skort" in b_blob)
                is_mini_or_knee_skirt = ("skirt" in b_blob) and any(k in b_blob for k in ["mini", "short"])  # knee not included
                is_denim = ("denim" in b_blob) or ("jean" in b_blob)
                is_ankle_or_long = ("ankle length" in b_blob) or (("long" in b_blob) and ("length" in b_blob))
                is_midi = _is_midi(b_blob)
                is_maxi = _is_maxi(b_blob)

                if is_tiny_top:
                    pen += 0.4
                if is_long:
                    pen -= 0.7

                if is_shorts:
                    pen += 0.6
                elif is_mini_or_knee_skirt:
                    pen += 0.4
                elif is_midi:
                     pen += 0.1

                if is_denim:
                    if is_shorts or is_mini_or_knee_skirt:
                        pass  # whitelist denim shorts or mini skirts
                    else:
                        pen -= 0.5
                if is_ankle_or_long:
                    pen -= 0.4
                if is_maxi:
                     pen -= 0.25
                if has_lining(bottom_it):
                    pen -= 0.4

                # amplify RET impact
                pen += (-0.035) * (t_ret - 10.0)
                pen += (-0.025) * (b_ret - 10.0)

            # --- Coverage penalties/bonuses for cold weather ---
            t_cov = _coverage_top(top_it)
            b_cov = _coverage_bottom(bottom_it)

            if cold:
                very_cold = (temp_c <= 0.0) or (estimate_target_clo(weather) >= 1.6)
                # Üst coverage düşükse ceza
                if t_cov < (0.9 if very_cold else 0.8):
                    pen -= (0.6 if very_cold else 0.4)
                # Alt coverage düşükse ceza
                if b_cov < (0.9 if very_cold else 0.8):
                    pen -= (1.2 if very_cold else 0.8)

                # Knit/warm üst bonus
                if _is_knit_warm_top(top_it):
                    pen += (0.5 if very_cold else 0.3)

                # Wet denim ankle wide strengthened penalties
                if wet:
                    btxt = _txt_blob(bottom_it)
                    if ("denim" in btxt or "jean" in btxt):
                        pen -= (0.9 if very_cold else 0.6)
                    if "ankle length" in btxt:
                        pen -= 0.3
                    if "wide leg" in btxt:
                        pen -= 0.2

            # Stronger skirt penalty in cold; reduced if warm-lined
            if cold:
                btxt = _txt_blob(bottom_it)
                is_skirt = "skirt" in btxt
                warm_bottom = any(k in btxt for k in ["wool", "fleece", "thermal", "lined", "skort"])
                if is_skirt:
                    if very_cold:
                        pen -= (0.6 if warm_bottom else 1.2)
                    else:
                        pen -= (0.3 if warm_bottom else 0.8)

            # --- Outer warm-lining bonus (fur/sherpa/teddy/pile) ---
            if cold:
                ttxt = _txt_blob(top_it)
                if any(k in ttxt for k in ["pile", "teddy", "sherpa", "shearling", "faux fur"]):
                    pen += 0.2
 
            scored_base.append((float(s) + pen, top_it, bottom_it))

        if not scored_base:
            raise ValueError("All candidate pairs were invalid due to missing vectors or roles.")

        scored_base.sort(key=lambda x: x[0], reverse=True)
        best_score, best_top, best_bottom = scored_base[0]

        # -------------------------
        # (YENİ) Mid-layer seçimi
        # -------------------------
        def _roles_of(i: Dict[str, Any]) -> list:
            r = i.get("roles", [])
            if r:
                return [str(x).lower() for x in r]
            return [str(i.get("role")).lower()] if i.get("role") else []

        def _is_dress(it: Dict[str, Any]) -> bool:
            txt = (str(it.get("type", "")) + " " + str(it.get("name", ""))).lower()
            return "dress" in txt

        mids_all = [i for i in items if any(x in {"mid-layer", "mid"} for x in _roles_of(i))]
        # Exclude outers and dresses from mid-layer pool; apply light thermal threshold
        mids_all = [m for m in mids_all if ("outer" not in _roles_of(m)) and (not _is_dress(m)) and _clo(m) >= 0.2]

        mid_selected = None
        if mids_all:
            # 1) Şu anki 2'li kombin ile hedef CLO farkını (gap2) ölç
            gap2 = abs(estimate_target_clo(weather) - (_clo(best_top) + _clo(best_bottom)))

            # 2) Mid ekleyerek skor oluştur: mevcut _aesthetic_score'u kullan
            scored_mids = []
            for m in mids_all:
                if m.get("item_id") == best_top.get("item_id"):
                    continue  # aynı ürünü iki rolde kullanma
                sc_mid = self._aesthetic_score(top=m, bottom=id_to_bottom[best_bottom["item_id"]], base=best_top)
                clo_mid_total = _clo(best_top) + _clo(m) + _clo(best_bottom)
                gap3 = abs(estimate_target_clo(weather) - clo_mid_total)
                # mevcut sıralama anahtarına benzer: yüksek estetik, düşük gap
                scored_mids.append((sc_mid - gap3, sc_mid, gap3, m, clo_mid_total))

            if scored_mids:
                scored_mids.sort(key=lambda x: x[0], reverse=True)
                best_mid = scored_mids[0]
                # Yenisi: soğukta daha agresif mid seçimi
                # target clo yüksekse (>=1.0) veya iyileşme ufak da olsa (>=0.05) mid ekle
                tgt = estimate_target_clo(weather)
                if (tgt >= 1.0) or (gap2 - best_mid[2] >= 0.05):
                    mid_selected = best_mid[3]

        best_outer = None
        best_outer_score = None
        if outer_required and outers:
            outer_pairs = [(best_top["item_id"], o["item_id"]) for o in outers]
            base_outer_scores = self._score_batch_pairs(outer_pairs)

            # Add a temperature-aware bonus to favor warmer outers in the cold
            try:
                temp_c2 = float(weather.get("temp_c", 18.0))
            except Exception:
                temp_c2 = 18.0
            tgt = estimate_target_clo(weather)
            is_cold = (temp_c2 <= 12.0) or (tgt >= 1.0)
            bonus_alpha = 0.0
            if tgt >= 1.8:
                bonus_alpha = 0.8
            elif tgt >= 1.4:
                bonus_alpha = 0.6
            elif tgt >= 1.0:
                bonus_alpha = 0.4
            adjusted_scores = []
            for sc, o in zip(base_outer_scores, outers):
                if sc <= -1e8:
                    adjusted_scores.append(sc)
                    continue
                bonus = (bonus_alpha * _clo(o)) if is_cold else 0.0
                otxt = _txt_blob(o)
                if is_cold and any(k in otxt for k in ["pile", "teddy", "sherpa", "shearling", "faux fur"]):
                    bonus += 0.3
                adjusted_scores.append(float(sc) + float(bonus))

            best_idx = int(np.argmax(adjusted_scores))
            if adjusted_scores[best_idx] > -1e8:
                best_outer = outers[best_idx]
                best_outer_score = float(adjusted_scores[best_idx])

        # ------------------------------
        # Selected payload (2/3/4 parça)
        # ------------------------------
        visible_top = mid_selected if mid_selected is not None else best_top

        # toplam clo ve thermal gap
        sel_top_clo = _clo(visible_top)
        sel_base_clo = _clo(best_top)  # base/top (mid olsa da base var)
        sel_bottom_clo = _clo(best_bottom)
        sel_outer_clo = _clo(best_outer) if best_outer is not None else 0.0

        if mid_selected is not None:
            total_clo = sel_base_clo + _clo(mid_selected) + sel_bottom_clo + sel_outer_clo
        else:
            total_clo = sel_top_clo + sel_bottom_clo + sel_outer_clo

        gap = float(target_clo) - float(total_clo)

        selected_block = {
            "base": best_top,              # iç katman (base)   # ekranda görünen (mid varsa mid, yoksa top)             # şema uyumu için ayrıca base
            "mid": mid_selected,          # None ise 2 parça
            "bottom": best_bottom,
            "outer": best_outer,
            "scores": {
                "base_pair_score": float(best_score),
                "outer_pair_score": (None if best_outer is None else float(best_outer_score or 0.0)),
            },
        }

        payload = {
            "selected": selected_block,
            "meta": {
                "target_clo": float(target_clo),
                "outer_required": bool(outer_required),
                "evaluated_pairs": int(len(scored_base)),
                "total_clo": round(float(total_clo), 2),
                "thermal_gap": round(float(max(0.0, gap)), 2),
            },
        }

        # ------------------------------
        # Thermal validation & advice
        # ------------------------------
        # Use the already computed total_clo (which includes mid if present) and gap
        max_outer_available = max([_clo(o) for o in outers], default=0.0) if outers else 0.0
        payload["meta"].update({
            "total_clo": round(float(total_clo), 2),
            "thermal_gap": round(float(max(0.0, gap)), 2),
            "max_outer_clo_available": round(max_outer_available, 2),
        })

        # If we are under target, add advice field with actionable hints
        if gap > 0.2:
            advice = []
            # If there exists a heavier outer, suggest it
            if outers:
                heavier_candidates = sorted([o for o in outers if _clo(o) > sel_outer_clo], key=_clo, reverse=True)
                if heavier_candidates:
                    advice.append({
                        "action": "try_heavier_outer",
                        "suggested_item_id": heavier_candidates[0].get("item_id"),
                        "suggested_clo": _clo(heavier_candidates[0])
                    })
            # Generic suggestions independent of catalogue completeness
            advice.extend([
                {"action": "add_mid_layer_top", "examples": ["sweater", "fleece", "cardigan"]},
                {"action": "thermal_base_layer", "examples": ["thermal top", "thermal leggings"]},
                {"action": "warmer_bottom", "examples": ["wool-lined pants", "fleece-lined tights"]},
                {"action": "winter_accessories", "examples": ["beanie", "scarf", "gloves"]},
            ])
            payload["advice"] = advice
        # -------------------------------------------------------------
        # Optionally return the top-K base pairs (and outer, if needed)
        # -------------------------------------------------------------
        if return_candidates and return_candidates > 0:
            # --- Enforce diversity: avoid repeating the same top or bottom family ---
            def _bottom_family(b: Dict[str, Any]):
                """Rough grouping of bottoms by type + denim flag + length category."""
                blob = _txt_blob(b)
                is_denim = ("denim" in blob) or ("jean" in blob)

                # crude length classification
                if ("mini" in blob) or ("short" in blob):
                    length = "short"
                elif ("midi" in blob) or ("calf" in blob):
                    length = "midi"
                elif ("ankle length" in blob) or (("long" in blob) and ("length" in blob)):
                    length = "long"
                else:
                    length = "other"

                return (str(b.get("type", "")).lower(), is_denim, length)

            diverse: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
            seen_tops: Set[str] = set()
            seen_bottom_fams: Set[Tuple[str, bool, str]] = set()

            for (s, t, b) in scored_base:
                t_id = t.get("item_id")
                bfam = _bottom_family(b)

                if t_id in seen_tops:
                    continue
                if bfam in seen_bottom_fams:
                    continue

                diverse.append((s, t, b))
                seen_tops.add(t_id)
                seen_bottom_fams.add(bfam)

                if len(diverse) >= return_candidates:
                    break

            top_k = diverse

            cand_list = []
            for (s, t, b) in top_k:
                cand: Dict[str, Any] = {"top": t, "bottom": b, "score": float(s)}

                # When an outer layer is required, attach the best outer for
                # this top–bottom pair so that the caller can see it.
                if outer_required and outers:
                    outer_pairs = [(t["item_id"], o["item_id"]) for o in outers]
                    outer_scores = self._score_batch_pairs(outer_pairs)
                    best_idx = int(np.argmax(outer_scores))
                    if outer_scores[best_idx] > -1e8:
                        cand["outer"] = outers[best_idx]
                        cand["outer_pair_score"] = float(outer_scores[best_idx])
                    else:
                        cand["outer"] = None
                        cand["outer_pair_score"] = None
                cand_list.append(cand)
            payload["candidates"] = cand_list
        return payload

# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Taste-aware outfit mapper with automatic catalogue enrichment.")
    parser.add_argument("--items", type=str, required=True, help="Path to JSON list of items (can be minimal; must include item_id).")
    parser.add_argument("--weather", type=str, required=True, help="JSON string or path to JSON with at least temp_c.")
    parser.add_argument("--need_outer", type=str, default=None, help="true/false to force outer selection; omit for auto.")
    parser.add_argument("--model", type=str, default="ML/aesthetic_scorer_v1.pkl", help="Path to joblib model file.")
    parser.add_argument("--style", type=str, default="ML/style_vectors.pkl", help="Path to style vectors pickle.")
    parser.add_argument("--color", type=str, default="ML/color_vectors.pkl", help="Path to color vectors pickle.")
    parser.add_argument("--topk", type=int, default=0, help="Also return top-K base pairs.")
    parser.add_argument("--no_enrich", action="store_true", help="Disable catalogue enrichment.")
    args = parser.parse_args()

    def load_json_maybe_path(s: str) -> Dict[str, Any]:
        p = Path(s)
        if p.exists():
            return json.loads(p.read_text())
        else:
            return json.loads(s)

    items = json.loads(Path(args.items).read_text())
    # Accept dict with "items" key too
    if isinstance(items, dict) and "items" in items:
        items = items["items"]
    if not isinstance(items, list):
        raise ValueError("--items must be a JSON list of item dicts")

    if not args.no_enrich:
        items = enrich_items(items, verbose=True)

    weather = load_json_maybe_path(args.weather)

    # Resolve assets
    def resolve(path: str, fallbacks: List[str]) -> Path:
        cands = [Path(path)] + [Path(p) for p in fallbacks]
        for p in cands:
            if p.exists(): return p
        raise FileNotFoundError(f"Not found: {cands}")

    model_path = resolve(args.model, ["/mnt/data/aesthetic_scorer_v1.pkl", "ML/aesthetic_scorer_v1.pkl"])
    style_path = resolve(args.style, ["/mnt/data/ML/style_vectors.pkl", "ML/style_vectors.pkl"])
    color_path = resolve(args.color, ["/mnt/data/ML/color_vectors.pkl", "ML/color_vectors.pkl"])

    mapper = OutfitMapper(model_path, style_path, color_path, verbose=True)
    need_outer = None if args.need_outer is None else args.need_outer.strip().lower() in {"1","true","yes","y"}

    result = mapper.suggest_outfit(items, weather, need_outer, return_candidates=args.topk)
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

