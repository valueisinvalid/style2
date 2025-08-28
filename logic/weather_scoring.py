#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
StylePops – thermal properties from fabric composition (EN dataset).

- Parses 'fabric_composition' strings (incl. "Shell:", "Lining:", "Collar:").
- Computes:
    base_resistance (Clo per 1000 gsm)
    effective_resistance (Clo, coverage-adjusted)
    base_breathability (RET; lower = better)
- Sleeve-aware coverage nudge.

If you later have GSM, multiply base_resistance by (gsm/1000).
"""

import json, re, argparse
from typing import Dict, Any, List, Tuple, Optional

# -------------------------------
# 1) Fabric properties (priors)
# Clo per 1000 gsm; RET in m²·Pa/W (ISO 11092). Lower RET = more breathable.
# -------------------------------
FABRIC_PROPERTIES: Dict[str, Dict[str, float]] = {
    # Naturals
    "cotton":    {"clo_per_kg_m2": 0.55, "ret": 7.0},
    "linen":     {"clo_per_kg_m2": 0.45, "ret": 5.0},
    "wool":      {"clo_per_kg_m2": 0.85, "ret": 10.0},
    "silk":      {"clo_per_kg_m2": 0.50, "ret": 6.5},
    "cashmere":  {"clo_per_kg_m2": 0.95, "ret": 9.0},

    # Regenerated cellulosics
    "viscose":   {"clo_per_kg_m2": 0.52, "ret": 7.5},
    "rayon":     {"clo_per_kg_m2": 0.52, "ret": 7.5},
    "lyocell":   {"clo_per_kg_m2": 0.50, "ret": 6.5},
    "modal":     {"clo_per_kg_m2": 0.51, "ret": 7.0},

    # Synthetics
    "polyester": {"clo_per_kg_m2": 0.48, "ret": 16.0},
    "polyamide": {"clo_per_kg_m2": 0.46, "ret": 18.0},  # aka nylon
    "nylon":     {"clo_per_kg_m2": 0.46, "ret": 18.0},
    "acrylic":   {"clo_per_kg_m2": 0.60, "ret": 15.0},
    "elastane":  {"clo_per_kg_m2": 0.30, "ret": 28.0},
    "spandex":   {"clo_per_kg_m2": 0.30, "ret": 28.0},
}

FABRIC_PROPERTIES.update({
    # Stretch polyester family (Elastomultiester / EME / elasterell-P)
    "elastomultiester": {"clo_per_kg_m2": 0.47, "ret": 17.0},

    # Polyurethane (PU) — genelde kaplama/lamine; nefes alabilirliği düşürür
    "polyurethane": {"clo_per_kg_m2": 0.35, "ret": 40.0},

    # Cellulosics (asetat/tri-asetat) — viskoz’a benzer, nefes alabilirliği iyi
    "acetate": {"clo_per_kg_m2": 0.50, "ret": 8.5},
    "triacetate": {"clo_per_kg_m2": 0.51, "ret": 8.5},

    # Animal fibers
    "mohair": {"clo_per_kg_m2": 0.90, "ret": 10.0},  # yün benzeri, loft iyi

    # Leather / suede (gerçek deri benzeri; “suede” genelde split leather)
    "leather": {"clo_per_kg_m2": 0.80, "ret": 25.0},
    "leather_suede": {"clo_per_kg_m2": 0.82, "ret": 26.0},
})

# -------------------------------
# 2) Coverage multipliers (by type label)
# -------------------------------
COVERAGE_MULTIPLIERS: Dict[str, float] = {
    # Tops
    "Short Sleeve": 0.55,
    "Long Sleeve": 0.75,
    "Tops": 0.65,
    "HM.com": 0.65,  # generic tops bucket in your data
    "Tops & T-Shirts": 0.60,
    "Graphic & Printed Tees": 0.55,
    "Peplum Tops": 0.60,
    "Low Cut Tops": 0.55,
    "Vests": 0.45,
    "Waistcoats": 0.55,
    "Women's Basic Long Sleeve Tops | Cropped & T-Shirt ": 0.70,

    # Dresses
    "Short Dresses": 0.95,
    "Midi Dresses": 1.05,
    "Maxi Dresses": 1.10,
    "A-line Dresses": 0.95,
    "Bodycon Dresses": 0.95,
    "Denim Dresses": 1.00,

    # Fallbacks
    "Jacket": 1.00,
    "Sweater": 0.85,
    "Trousers": 0.80,
    "Shorts": 0.40,
}

SLEEVE_NUDGE = {
    "Sleeveless": -0.10,
    "Short sleeve": -0.05,
    "Elbow-length": -0.02,
    "Long sleeve": +0.05,
}

# -------------------------------
# Aliases / normalization (EN only)
# -------------------------------
FABRIC_ALIASES: Dict[str, str] = {
    "pa": "polyamide",
    "polyamid": "polyamide",
    "lycra": "elastane",
    "tencel": "lyocell",
    # Keep rayon separate from viscose (some vendors use rayon for modal too)
}
FABRIC_ALIASES.update({
    # Elastomultiester varyantları
    "eme": "elastomultiester",
    "elasterellp": "elastomultiester",
    "elasterell": "elastomultiester",

    # Polyurethane kısaltmaları
    "pu": "polyurethane",
    "polyurethan": "polyurethane",

    # Suede → deri benzeri
    "suede": "leather_suede",
    "splitleather": "leather_suede",
})

LAYER_WEIGHTS_DEFAULT = {
    "shell": 0.70,
    "lining": 0.30,
    "collar": 0.10,
    "pocketing": 0.10,
    # unknown → treated like 1.0 block then normalized across layers
}

def norm_fabric(name: str) -> str:
    """Normalize 'Polyamide', 'PA', 'Cotton ' → canonical lowercase keys."""
    n = (name or "").strip().lower()
    n = re.sub(r"[^a-z]", "", n)        # letters only
    return FABRIC_ALIASES.get(n, n)     # alias fallback

def parse_fabric_composition(s: str) -> List[Tuple[str, float, Optional[str]]]:
    """
    Returns list of (fabric, fraction_0_1, layer_label_or_None).
    Supports:
      "Cotton 60%, Polyester 35%, Elastane 5%"
      "95% Polyamide, 5% Elastane"
      "Shell: Polyester 93%, Elastane 7%, Lining: Polyester 100%"
    """
    if not s or not isinstance(s, str):
        return []

    # Split by layers if present
    parts = re.split(r"(?i)\b(shell|lining|collar|pocketing)\s*:", s)
    tokens: List[Tuple[Optional[str], str]] = []
    if len(parts) > 1:
        pre = parts[0].strip(", ;")
        if pre:
            tokens.append((None, pre))
        for i in range(1, len(parts), 2):
            label = parts[i].lower()
            seg = (parts[i+1] if i + 1 < len(parts) else "").strip(", ;")
            tokens.append((label, seg))
    else:
        tokens.append((None, s))

    out: List[Tuple[str, float, Optional[str]]] = []
    for label, seg in tokens:
        found: List[Tuple[str, str]] = []
        # "95% Polyamide"  ->  (pct, name)
        found += re.findall(r"(\d{1,3})\s*%\s*([A-Za-z\-\(\) ]+)", seg)
        # "Polyamide 95%"  ->  (pct, name)  **fixed**
        found += [(p, n) for (n, p) in re.findall(r"([A-Za-z\-\(\) ]+?)\s*(\d{1,3})\s*%", seg)]
        if not found:
            # relaxed per-chunk fallback
            for chunk in re.split(r"[,/;+]| and ", seg):
                m = re.search(r"(\d{1,3})\s*%?\s*([A-Za-z\-\(\) ]+)", chunk)
                if m:
                    found.append((m.group(1), m.group(2)))

        # Consolidate inside the segment  **robust to swapped pairs**
        seg_map: Dict[str, float] = {}
        for a, b in found:
            s1, s2 = str(a).strip(), str(b).strip()
            if re.fullmatch(r"\d{1,3}", s1):
                pct_f = float(s1); nm = s2
            elif re.fullmatch(r"\d{1,3}", s2):
                pct_f = float(s2); nm = s1
            else:
                continue
            cname = norm_fabric(nm)
            seg_map[cname] = seg_map.get(cname, 0.0) + pct_f

        total = sum(seg_map.values())
        if total > 0:
            for fab, pct in seg_map.items():
                out.append((fab, pct / total, label))
    return out

def weighted_mix_with_layers(comp: List[Tuple[str, float, Optional[str]]],
                             debug: bool = False) -> Tuple[float, float]:
    """
    Returns (base_clo_per_kg_m2, base_RET).
    Applies layer weights if labels are present; otherwise simple weighted average.
    """
    if not comp:
        return (0.0, 0.0)

    # group by layer
    layer_groups: Dict[str, List[Tuple[str, float]]] = {}
    for fab, frac, label in comp:
        lay = (label or "unknown").lower()
        layer_groups.setdefault(lay, []).append((fab, frac))

    base_clo = 0.0
    base_ret = 0.0
    total_w = 0.0

    for lay, items in layer_groups.items():
        w = LAYER_WEIGHTS_DEFAULT.get(lay, 1.0 if lay == "unknown" else 0.50)
        clo_l = 0.0
        ret_l = 0.0
        for fab, frac in items:
            props = FABRIC_PROPERTIES.get(fab) or FABRIC_PROPERTIES.get(norm_fabric(fab))
            if not props:
                if debug:
                    print("No props for:", repr(fab))
                continue
            clo_l += frac * props["clo_per_kg_m2"]
            ret_l += frac * props["ret"]
        if clo_l == 0 and ret_l == 0:
            continue
        base_clo += w * clo_l
        base_ret += w * ret_l
        total_w += w

    if total_w > 0:
        base_clo /= total_w
        base_ret /= total_w

    return (base_clo, base_ret)

def sleeve_nudge_from_meta(item: Dict[str, Any]) -> float:
    """Look into key_value_description for sleeve hints and nudge coverage."""
    kv = item.get("key_value_description") or {}
    text = " ".join([str(v) for v in kv.values()]) if isinstance(kv, dict) else str(kv)
    for k, delta in SLEEVE_NUDGE.items():
        if k.lower() in text.lower():
            return delta
    return 0.0

def coverage_for_item(item: Dict[str, Any]) -> float:
    """Pick base coverage by 'type' with sleeve-aware adjustment."""
    gtype = (item.get("type") or "").strip()
    cov = COVERAGE_MULTIPLIERS.get(gtype, 0.70 if "dress" not in gtype.lower() else 0.95)
    cov = max(0.30, min(1.15, cov + sleeve_nudge_from_meta(item)))
    return cov

def calculate_item_properties(item: Dict[str, Any], debug: bool = False) -> Dict[str, float]:
    """
    Returns:
      {
        'base_resistance': Clo per 1000 gsm,
        'effective_resistance': Clo (coverage-scaled),
        'base_breathability': RET
      }
    """
    comp = parse_fabric_composition(item.get("fabric_composition", ""))
    base_clo, base_ret = weighted_mix_with_layers(comp, debug=debug)
    cov = coverage_for_item(item)
    return {
        "base_resistance": round(base_clo, 3),
        "effective_resistance": round(base_clo * cov, 3),
        "base_breathability": round(base_ret, 1),
        "coverage_multiplier": round(cov, 2),
    }

# -------------------------------
# Optional: Meteoblue fetch for Istanbul center
# -------------------------------
def get_meteoblue_current(api_key: str, lat: float = 69.6496, lon: float = 18.9560) -> Optional[Dict[str, Any]]:
    try:
        import requests
        url = (
            "https://my.meteoblue.com/packages/basic-1h?"
            f"apikey={api_key}&lat={lat}&lon={lon}&format=json"
        )
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print("Meteoblue fetch failed:", e)
        return None

# -------------------------------
# CLI / batch
# -------------------------------
def main():
    p = argparse.ArgumentParser(description="Compute thermal properties for a garment dataset.")
    p.add_argument("-i", "--input", default="out.json", help="Path to JSON list of items.")
    p.add_argument("--show", type=int, default=5, help="Show first N results.")
    p.add_argument("--weather", action="store_true", help="Also fetch Meteoblue current (Istanbul).")
    p.add_argument("--apikey", default=None, help="Meteoblue API key (or set in code).")
    p.add_argument("--debug", action="store_true", help="Print missing fabric keys.")
    args = p.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    results: List[Dict[str, Any]] = []
    for it in data:
        props = calculate_item_properties(it, debug=args.debug)
        results.append({
            "item_id": it.get("item_id"),
            "name": it.get("name"),
            "type": it.get("type"),
            "fabric": it.get("fabric_composition"),
            **props
        })

    print(f"\nComputed {len(results)} items.\n")
    for row in results[:args.show]:
        print(f"- {row['name']} [{row['type']}]: "
              f"base_clo={row['base_resistance']}, "
              f"cov={row['coverage_multiplier']}, "
              f"effective_clo={row['effective_resistance']}, "
              f"RET={row['base_breathability']} | {row['fabric']}")

    ex = next((r for r in results if r["item_id"] == "1142908008"), None)
    if ex:
        print("\nExample – Fitted microfibre T-shirt (1142908008)")
        for k in ("base_resistance", "coverage_multiplier", "effective_resistance", "base_breathability"):
            print(f"  {k}: {ex[k]}")

    if args.weather:
        api_key = args.apikey or "2jPlBVkzULmX0agT"
        meteo = get_meteoblue_current(api_key)
        if meteo is not None:
            try:
                t = meteo["data_1h"]["temperature"][0]
                rh = meteo["data_1h"]["relativehumidity"][0]
                wind = meteo["data_1h"]["windspeed"][0]
                print(f"\nIstanbul now (Meteoblue): T={t}°C, RH={rh}%, wind={wind} m/s")
            except Exception:
                print("Meteoblue JSON structure not as expected; inspect 'meteo' variable.")

if __name__ == "__main__":
    main()