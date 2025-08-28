#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
StylePops – Weather→Clothing Mapper (rule-based v2)
- Weather → target Clo (temp, wind, RH, sun, rain)
- Always builds a full outfit: TOP + BOTTOM + (JACKET if needed)
- Uses your Clo/RET from weather_scoring
"""

import math, json, argparse, sys
from typing import Dict, Any, List, Tuple, Optional

# -------- Import weather_scoring (package OR same-folder fallback) --------
try:
    from logic.weather_scoring import calculate_item_properties, get_meteoblue_current
except Exception:
    import importlib.util
    from pathlib import Path
    this_dir = Path(__file__).resolve().parent
    ws_path = this_dir / "weather_scoring.py"
    spec = importlib.util.spec_from_file_location("weather_scoring", str(ws_path))
    if not spec or not spec.loader:
        raise ImportError("Cannot import weather_scoring from package or same folder.")
    ws = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ws)  # type: ignore
    calculate_item_properties = ws.calculate_item_properties
    get_meteoblue_current = ws.get_meteoblue_current

# --------------------------------------------
# Helpers: humidity, apparent temperature, dew point
# --------------------------------------------
def dew_point_c(t_c: float, rh_pct: float) -> float:
    """Magnus formula (°C)."""
    a, b = 17.27, 237.7
    alpha = ((a * t_c) / (b + t_c)) + math.log(max(1e-6, rh_pct / 100.0))
    return (b * alpha) / (a - alpha)

def apparent_temp_c(t_c: float, rh_pct: float, wind_mps: float) -> float:
    """Australian BoM 'Apparent Temperature' (°C)."""
    e = (rh_pct / 100.0) * 6.105 * math.exp(17.27 * t_c / (237.7 + t_c))
    return t_c + 0.33 * e - 0.70 * wind_mps - 4.0

# --------------------------------------------
# Target Clo mapping (piecewise) + modifiers
# --------------------------------------------
TARGET_CLO_POINTS = [
    # (apparent temp °C, target clo)  cold → hot
    (-20, 3.00), (-10, 2.30), (-5, 2.00), (0, 1.75),
    (3, 1.55), (6, 1.35), (9, 1.20), (12, 1.05),
    (15, 0.90), (18, 0.75), (21, 0.60), (24, 0.45),
    (27, 0.35), (30, 0.20), (35, 0.12), (40, 0.08),
]

def interp_target_clo(at_c: float) -> float:
    pts = sorted(TARGET_CLO_POINTS, key=lambda x: x[0])  # ASC
    if at_c <= pts[0][0]:
        return pts[0][1]
    if at_c >= pts[-1][0]:
        return pts[-1][1]
    for (t1, c1), (t2, c2) in zip(pts, pts[1:]):
        if t1 <= at_c <= t2:
            ratio = (at_c - t1) / max(1e-6, (t2 - t1))
            return c1 + (c2 - c1) * ratio
    return pts[-1][1]

def target_clo_from_weather(
    temp_c: float, rh_pct: float, wind_mps: float,
    sun: float = 0.5,  # 0=shade, 1=strong sun
    precip_mmph: float = 0.0,
    activity: str = "light",  # 'sedentary' | 'light' | 'active'
    wind_exposed: bool = True
) -> Tuple[float, Dict[str, float]]:
    at = apparent_temp_c(temp_c, rh_pct, wind_mps)
    base = interp_target_clo(at)
    act_delta = {"sedentary": 0.0, "light": -0.15, "active": -0.35}.get(activity, -0.15)
    sun_delta = -0.20 * max(0.0, min(1.0, sun))
    wind_delta = max(0.0, (wind_mps - 1.0)) * 0.025 if wind_exposed else 0.0
    rain_delta = 0.10 if precip_mmph >= 0.5 else (0.05 if precip_mmph >= 0.1 else 0.0)
    tgt = max(0.05, base + act_delta + sun_delta + wind_delta + rain_delta)
    breakdown = {
        "apparent_temp": at, "base": base,
        "act_delta": act_delta, "sun_delta": sun_delta,
        "wind_delta": wind_delta, "rain_delta": rain_delta
    }
    return round(tgt, 2), breakdown

# --------------------------------------------
# Type roles & torso/bottom requirement
# --------------------------------------------
OUTER_TYPES = {
    "Jacket", "Jackets", "Coat", "Coats",
    "Sweater", "Sweaters", "Cardigan", "Cardigans",
    "Hoodie", "Hoodies", "Outerwear",
    "Blazer", "Blazers", "Knitted tops", "Knitwear", "Jumpers",
}
ONE_PIECE_TYPES = {
    "Short Dresses","Midi Dresses","Maxi Dresses","A-line Dresses",
    "Bodycon Dresses","Denim Dresses","Dress","Dresses"
}
TOP_TYPES = {
    "Short Sleeve","Long Sleeve","Tops","Tops & T-Shirts",
    "Graphic & Printed Tees","Peplum Tops","Low Cut Tops",
    "Vests","Blouses","Shirts","Tank Tops","Camisoles"
}
BOTTOM_TYPES = {
    "Shorts","Skirts","Trousers","Jeans","Joggers","Leggings","Pants","Chinos",
    "Denim Shorts","Mini Skirts","Midi Skirts","Maxi Skirts"
}

def is_top(item_type: str) -> bool:
    return (item_type or "").strip() in TOP_TYPES

def is_bottom(item_type: str) -> bool:
    return (item_type or "").strip() in BOTTOM_TYPES

def layer_role(item_type: str) -> str:
    t = (item_type or "").strip()
    if t in ONE_PIECE_TYPES: return "one"
    if t in OUTER_TYPES: return "outer"
    if t in BOTTOM_TYPES: return "bottom"
    return "base"  # tees, shirts, blouses, generic tops

# --------------------------------------------
# Outfit selection: TOP + BOTTOM + (JACKET if needed)
# --------------------------------------------
def pick_outfit(
    items: List[Dict[str, Any]],
    target_clo: float,
    temp_c: float,
    rh_pct: float,
    wind_mps: float,
    precip_mmph: float,
    max_layers: int = 3,
    tol: float = 0.18,
    prefer_low_ret_when_humid: bool = True,
) -> Dict[str, Any]:
    """
    Always returns at least a TOP and a BOTTOM.
    Adds an OUTER (jacket) if Clo gap is meaningful OR rainy/windy/cool.
    """
    # Precompute properties (Clo/RET) if not present
    enriched = []
    for it in items:
        if not all(k in it for k in ("effective_resistance","base_breathability")):
            props = calculate_item_properties(it)
            it = {**it, **props}
        it["_role"] = layer_role(it.get("type"))
        enriched.append(it)

    # Pools
    top_pool    = [x for x in enriched if is_top(x.get("type"))]          # required
    bottom_pool = [x for x in enriched if is_bottom(x.get("type"))]       # required
    outer_pool  = [x for x in enriched if x["_role"] == "outer"]          # optional
    # Note: we intentionally ignore ONE_PIECE_TYPES for "full combo" modunda

    if not top_pool or not bottom_pool:
        return {"combo": [], "sum_clo": 0.0, "gap": target_clo, "notes": ["need top+bottom candidates"]}

    # RET thresholds
    ret_skin_max  = next_to_skin_ret_threshold(temp_c, rh_pct) if prefer_low_ret_when_humid else 30.0
    ret_outer_max = outer_layer_ret_limit(temp_c, rh_pct, precip_mmph)

    # ---------- 1) Pick TOP (next-to-skin) ----------
    top_sorted = sorted(
        top_pool,
        key=lambda x: (
            0 if x["base_breathability"] <= ret_skin_max else 1,   # pass inner RET
            abs(x["effective_resistance"] - min(target_clo, 0.9)), # closeness
            x["base_breathability"]
        )
    )
    top = top_sorted[0]
    outfit = [top]
    sum_clo = top["effective_resistance"]
    notes = [f"top='{top['name']}' (RET={top['base_breathability']})"]

    # ---------- 2) Pick BOTTOM ----------
    # For warm/humid, prefer breathable/lower Clo bottoms; else, aim to close the Clo gap.
    warm_humid = (temp_c >= 22 and rh_pct >= 60)
    def bottom_key(x):
        # Score by resulting gap; add small penalty for high RET in humid warmth
        gap = abs((sum_clo + x["effective_resistance"]) - target_clo)
        ret_pen = (x["base_breathability"] - 12.0) / 100.0 if warm_humid else 0.0
        return (gap + max(0.0, ret_pen), x["base_breathability"])
    bottom = sorted(bottom_pool, key=bottom_key)[0]
    outfit.append(bottom)
    sum_clo += bottom["effective_resistance"]
    notes.append(f"bottom='{bottom['name']}' (RET={bottom['base_breathability']})")

    # ---------- 3) Decide if a JACKET/OUTER is needed ----------
    need_outer = False
    # thermal need
    if (target_clo - sum_clo) > tol:
        need_outer = True
    # weather triggers
    if precip_mmph >= 0.5:
        need_outer = True
    if temp_c <= 16:
        need_outer = True
    if wind_mps >= 5.0 and temp_c <= 22:
        need_outer = True

    if need_outer and len(outfit) < max_layers and outer_pool:
        # filter by outer RET (unless quite cold)
        def outer_ok(x): return x["base_breathability"] <= ret_outer_max or target_clo >= 1.2
        outer_sorted = sorted(
            [x for x in outer_pool if outer_ok(x)],
            key=lambda x: abs((sum_clo + x["effective_resistance"]) - target_clo)
        )
        if outer_sorted:
            outer = outer_sorted[0]
            outfit.append(outer)
            sum_clo += outer["effective_resistance"]
            notes.append(f"outer='{outer['name']}' (RET={outer['base_breathability']})")

    return {
        "combo": outfit,
        "sum_clo": round(sum_clo, 3),
        "gap": round(target_clo - sum_clo, 3),
        "notes": notes
    }

# --------------------------------------------
# RET preferences (tune here)
# --------------------------------------------
def next_to_skin_ret_threshold(temp_c: float, rh_pct: float) -> float:
    """RET ceiling for the innermost layer."""
    dp = dew_point_c(temp_c, rh_pct)
    if temp_c >= 26 and (rh_pct >= 60 or dp >= 19):  # hot & humid
        return 9.0
    if temp_c >= 22 and (rh_pct >= 55 or dp >= 17):  # warm & humid
        return 12.0
    return 20.0

def outer_layer_ret_limit(temp_c: float, rh_pct: float, precip_mmph: float) -> float:
    """Outer layers can be less breathable in cool/cold."""
    if precip_mmph >= 0.5 and temp_c >= 18:
        return 25.0
    if temp_c <= 10:
        return 30.0
    return 28.0

# --------------------------------------------
# CLI
# --------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Map weather to outfit using Clo/RET (full outfit: top+bottom+optional jacket).")
    ap.add_argument("-i","--input", default="out.json", help="Items JSON (list of dicts).")
    ap.add_argument("--meteoblue", action="store_true", help="Fetch weather for central Istanbul.")
    ap.add_argument("--apikey", default=None, help="Meteoblue API key.")
    ap.add_argument("--temp", type=float, default=None, help="Air temperature °C.")
    ap.add_argument("--rh", type=float, default=None, help="Relative humidity %.")
    ap.add_argument("--wind", type=float, default=None, help="Wind speed m/s.")
    ap.add_argument("--sun", type=float, default=0.5, help="Sun factor 0..1 (0 shade, 1 strong sun).")
    ap.add_argument("--rain", type=float, default=0.0, help="Rain mm/h (instant).")
    ap.add_argument("--activity", default="light", choices=["sedentary","light","active"])
    ap.add_argument("--max_layers", type=int, default=3)
    args = ap.parse_args()

    # Load items
    with open(args.input, "r", encoding="utf-8") as f:
        items = json.load(f)

    # Weather
    if args.meteoblue:
        api_key = args.apikey or "2jPlBVkzULmX0agT"
        wx = get_meteoblue_current(api_key)
        if not wx:
            print("Could not fetch Meteoblue weather; use --temp/--rh/--wind."); sys.exit(1)
        try:
            temp_c = float(wx["data_1h"]["temperature"][0])
            rh = float(wx["data_1h"]["relativehumidity"][0])
            wind = float(wx["data_1h"]["windspeed"][0])
            rain = args.rain  # add package field if available
        except Exception:
            print("Unexpected Meteoblue JSON structure; provide --temp/--rh/--wind."); sys.exit(1)
    else:
        if args.temp is None or args.rh is None or args.wind is None:
            print("Provide --temp, --rh, --wind OR use --meteoblue."); sys.exit(1)
        temp_c, rh, wind, rain = args.temp, args.rh, args.wind, args.rain

    tgt_clo, brk = target_clo_from_weather(
        temp_c=temp_c, rh_pct=rh, wind_mps=wind, sun=args.sun,
        precip_mmph=rain, activity=args.activity, wind_exposed=True
    )

    suggestion = pick_outfit(
        items=items, target_clo=tgt_clo,
        temp_c=temp_c, rh_pct=rh, wind_mps=wind, precip_mmph=rain,
        max_layers=args.max_layers, tol=0.18, prefer_low_ret_when_humid=True
    )

    # ---- Print summary ----
    print("\n=== Weather → Target Clo ===")
    print(f"T={temp_c:.1f}°C, RH={rh:.0f}%, wind={wind:.1f} m/s, sun={args.sun:.2f}, rain={rain:.2f} mm/h")
    print(f"Apparent temp: {brk['apparent_temp']:.1f}°C | base={brk['base']:.2f} clo | "
          f"act={brk['act_delta']:+.2f} | sun={brk['sun_delta']:+.2f} | wind={brk['wind_delta']:+.2f} | rain={brk['rain_delta']:+.2f}")
    print(f"→ Target Clo: {tgt_clo:.2f}")

    combo = suggestion["combo"]
    if not combo:
        print("\nNo suitable combo found (check dataset/types)."); sys.exit(0)

    print("\n=== Suggested Outfit (TOP + BOTTOM [+ OUTER]) ===")
    sum_clo = suggestion["sum_clo"]
    for idx, it in enumerate(combo, 1):
        print(f"{idx}. {it.get('name')}  "
              f"[type={it.get('type')}]  Clo={it['effective_resistance']:.3f}  RET={it['base_breathability']:.1f}")
        print(f"   fabric: {it.get('fabric_composition')}")
    print(f"\nSum Clo = {sum_clo:.3f}  (target {tgt_clo:.2f}, gap {suggestion['gap']:+.3f})")
    print("Notes:", "; ".join(suggestion["notes"]))

if __name__ == "__main__":
    main()