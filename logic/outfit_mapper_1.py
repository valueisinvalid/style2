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
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd

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

# 384 diff + 384 prod + 1 color_dist = 769
FEATURE_NAMES = (
    [f"style_diff_{i}" for i in range(384)]
    + [f"style_prod_{i}" for i in range(384)]
    + ["color_dist"]
)

def _to_feature_frame(X: np.ndarray) -> pd.DataFrame:
    # X shape: (n, 769)
    if X.shape[1] != 769:
        raise ValueError(f"Expected 769 features, got {X.shape[1]}")
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
    t = float(weather.get("temp_c", 18.0))
    rh = float(weather.get("rh", weather.get("humidity", 50.0)))
    wind = float(weather.get("wind", weather.get("wind_kmh", 5.0)))
    rain = float(weather.get("rain", weather.get("precip_mm", 0.0)))

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
    feat = np.concatenate([style_diff, style_prod, np.array([color_dist], dtype=np.float32)], axis=0)
    if feat.shape[0] != 769:
        raise AssertionError(f"Feature length must be 769, got {feat.shape[0]}")
    return feat.reshape(1, -1)

# -----------------------------
# Catalogue enrichment
# -----------------------------

def _infer_role_from_type(t: str) -> Optional[str]:
    t = (t or "").lower()
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
            role = _infer_role_from_type(str(base.get("type","")))
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
    
    feat = create_feature_vector(top_id, bottom_id, self.style_vectors, self.color_vectors)  # (1, 769)
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
            if nfi != 769:
                print(f"[warn] Model expects {nfi} features; pipeline produces 769. Check training features.")

    @staticmethod
    def split_pools(items: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        tops = [it for it in items if str(it.get("role","")).lower() == "top"]
        bottoms = [it for it in items if str(it.get("role","")).lower() == "bottom"]
        outers = [it for it in items if str(it.get("role","")).lower() == "outer"]
        return tops, bottoms, outers

    @staticmethod
    def _within_clo(target: float, item_clo: float, tol: float = 0.5) -> bool:
        return (abs(item_clo - target) <= tol) or (item_clo <= target + tol)

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
            pre_b = bottoms_f[:]
            bottoms_f = [b for b in bottoms_f if (not _is_skirt(b)) or _is_warm_bottom(b)]
            if not bottoms_f:
                bottoms_f = pre_b  # don't empty the pool completely

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
            tops_f = [t for t in tops_f if float(t.get("clo", 0.3)) <= tops_cap]
            bottoms_f = [b for b in bottoms_f if float(b.get("clo", 0.3)) <= bottoms_cap]
            if not tops_f: tops_f = _pre_t
            if not bottoms_f: bottoms_f = _pre_b

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

    def _score_batch_pairs(self, pairs: List[Tuple[str, str]], batch_size: int=1024) -> List[float]:
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
                X = np.vstack(feats)
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

        payload = {
            "selected": Outfit(best_top, best_bottom, best_outer, float(best_score), (None if best_outer is None else float(best_outer_score or 0.0))).to_jsonable(),
            "meta": {
                "target_clo": float(target_clo),
                "outer_required": bool(outer_required),
                "evaluated_pairs": int(len(scored_base)),
            },
        }
        # ------------------------------
        # Thermal validation & advice
        # ------------------------------
        sel_top_clo = _clo(best_top)
        sel_bottom_clo = _clo(best_bottom)
        sel_outer_clo = _clo(best_outer) if best_outer is not None else 0.0
        total_clo = sel_top_clo + sel_bottom_clo + sel_outer_clo
        gap = float(target_clo) - float(total_clo)

        # Attach diagnostics to meta
        max_outer_available = max([_clo(o) for o in outers], default=0.0) if outers else 0.0
        payload["meta"].update({
            "total_clo": round(total_clo, 2),
            "thermal_gap": round(max(0.0, gap), 2),
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
            top_k = scored_base[:return_candidates]
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
