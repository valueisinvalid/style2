#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
StylePops – Batch Weather Test Harness

Runs your rule-based outfit mapper across multiple weather scenarios
and prints a compact report (optionally saves a CSV).

Usage:
  python -m logic.run_batch_tests -i out.json
  python -m logic.run_batch_tests -i out.json --save-csv batch_results.csv
  # also include the current Meteoblue reading for Istanbul:
  python -m logic.run_batch_tests -i out.json --meteoblue --apikey 2jPlBVkzULmX0agT
"""

import json, csv, argparse, sys
from typing import Dict, Any, List, Tuple, Optional

# --- Import outfit_mapper (package OR same-folder fallback) ---
try:
    from logic.outfit_mapper import (
        target_clo_from_weather, pick_outfit, get_meteoblue_current
    )
except Exception:
    import importlib.util
    from pathlib import Path
    this_dir = Path(__file__).resolve().parent
    om_path = this_dir / "outfit_mapper.py"
    spec = importlib.util.spec_from_file_location("outfit_mapper", str(om_path))
    if not spec or not spec.loader:
        raise ImportError("Cannot import outfit_mapper from package or same folder.")
    om = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(om)  # type: ignore
    target_clo_from_weather = om.target_clo_from_weather
    pick_outfit = om.pick_outfit
    get_meteoblue_current = om.get_meteoblue_current

# ------------------------
# Scenarios to exercise
# ------------------------
SCENARIOS = [
    # name, tempC, RH%, wind m/s, sun (0-1), rain mm/h, activity
    ("Heatwave dry noon",                 36.0, 25.0, 2.0, 1.0, 0.0, "light"),
    ("Hot & humid pm",                    30.0, 70.0, 2.0, 0.8, 0.0, "light"),
    ("Warm humid breezy",                 26.0, 65.0, 4.0, 0.7, 0.0, "light"),
    ("Mild cloudy, breezy",               18.0, 60.0, 5.0, 0.3, 0.0, "light"),
    ("Cool & windy",                      12.0, 55.0, 6.0, 0.4, 0.0, "light"),
    ("Cold, calm morning",                 5.0, 60.0, 1.0, 0.2, 0.0, "sedentary"),
    ("Chilly rain, moderate wind",        10.0, 85.0, 4.0, 0.2, 1.5, "light"),
    ("Warm shower (summer rain)",         24.0, 80.0, 2.0, 0.3, 0.8, "light"),
    ("Dry, sunny spring",                 20.0, 40.0, 2.0, 0.9, 0.0, "active"),
]

def format_combo(items: List[Dict[str, Any]]) -> str:
    parts = []
    for it in items:
        parts.append(f"{it.get('name')} [{it.get('type')}] Clo={it['effective_resistance']:.3f} RET={it['base_breathability']:.1f}")
    return " | ".join(parts)

def run_one(items, name, temp, rh, wind, sun, rain, activity):
    tgt, brk = target_clo_from_weather(
        temp_c=temp, rh_pct=rh, wind_mps=wind, sun=sun,
        precip_mmph=rain, activity=activity, wind_exposed=True
    )
    sug = pick_outfit(
        items=items, target_clo=tgt,
        temp_c=temp, rh_pct=rh, wind_mps=wind, precip_mmph=rain,
        max_layers=3, tol=0.18, prefer_low_ret_when_humid=True
    )
    return {
        "name": name,
        "temp": temp, "rh": rh, "wind": wind, "sun": sun, "rain": rain, "activity": activity,
        "apparent": brk["apparent_temp"], "base_clo": brk["base"],
        "act": brk["act_delta"], "sun_adj": brk["sun_delta"], "wind_adj": brk["wind_delta"], "rain_adj": brk["rain_delta"],
        "target_clo": tgt,
        "sum_clo": sug["sum_clo"], "gap": sug["gap"],
        "notes": "; ".join(sug["notes"]),
        "combo": sug["combo"],
    }

def main():
    ap = argparse.ArgumentParser(description="Batch test StylePops outfit mapper on multiple weather scenarios.")
    ap.add_argument("-i", "--input", default="out.json", help="Catalog JSON (list of items).")
    ap.add_argument("--meteoblue", action="store_true", help="Append current Istanbul weather as an extra scenario.")
    ap.add_argument("--apikey", default=None, help="Meteoblue API key.")
    ap.add_argument("--save-csv", default=None, help="Path to save CSV summary.")
    args = ap.parse_args()

    # Load catalog
    with open(args.input, "r", encoding="utf-8") as f:
        items = json.load(f)

    results = []

    # Run predefined scenarios
    for (name, t, rh, wind, sun, rain, activity) in SCENARIOS:
        res = run_one(items, name, t, rh, wind, sun, rain, activity)
        results.append(res)

    # Optional: append live Meteoblue Istanbul
    if args.meteoblue:
        api_key = args.apikey or "2jPlBVkzULmX0agT"
        wx = get_meteoblue_current(api_key)
        if wx:
            try:
                temp_c = float(wx["data_1h"]["temperature"][0])
                rh = float(wx["data_1h"]["relativehumidity"][0])
                wind = float(wx["data_1h"]["windspeed"][0])
                # If your package has rain/hour, map here; else pass 0.0
                live = run_one(items, "LIVE – Istanbul (Meteoblue)", temp_c, rh, wind, 0.5, 0.0, "light")
                results.append(live)
            except Exception:
                print("Meteoblue JSON shape unexpected; skipping live scenario.", file=sys.stderr)

    # ---- Print compact report
    print("\n=== Batch Weather Scenarios ===\n")
    for r in results:
        print(f"[{r['name']}]  T={r['temp']:.1f}°C RH={r['rh']:.0f}% wind={r['wind']:.1f} m/s sun={r['sun']:.2f} rain={r['rain']:.2f}  activity={r['activity']}")
        print(f"  apparent={r['apparent']:.1f}°C  base={r['base_clo']:.2f} clo  adj: act={r['act']:+.2f} sun={r['sun_adj']:+.2f} wind={r['wind_adj']:+.2f} rain={r['rain_adj']:+.2f}")
        print(f"  target={r['target_clo']:.2f}  sum={r['sum_clo']:.3f}  gap={r['gap']:+.3f}")
        print("  outfit:", format_combo(r["combo"]))
        print("  notes: ", r["notes"])
        print("-" * 90)

    # ---- Optional: save CSV
    if args.save_csv:
        fieldnames = [
            "name","temp","rh","wind","sun","rain","activity",
            "apparent","base_clo","act","sun_adj","wind_adj","rain_adj",
            "target_clo","sum_clo","gap","notes","items","types","cloes","rets"
        ]
        with open(args.save_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                items_str = " | ".join([i.get("name","") for i in r["combo"]])
                types_str = " | ".join([str(i.get("type","")) for i in r["combo"]])
                clo_str   = " | ".join([f"{i['effective_resistance']:.3f}" for i in r["combo"]])
                ret_str   = " | ".join([f"{i['base_breathability']:.1f}" for i in r["combo"]])
                writer.writerow({
                    "name": r["name"], "temp": r["temp"], "rh": r["rh"], "wind": r["wind"], "sun": r["sun"], "rain": r["rain"], "activity": r["activity"],
                    "apparent": r["apparent"], "base_clo": r["base_clo"], "act": r["act"], "sun_adj": r["sun_adj"], "wind_adj": r["wind_adj"], "rain_adj": r["rain_adj"],
                    "target_clo": r["target_clo"], "sum_clo": r["sum_clo"], "gap": r["gap"], "notes": r["notes"],
                    "items": items_str, "types": types_str, "cloes": clo_str, "rets": ret_str
                })
        print(f"\nSaved CSV → {args.save_csv}")

if __name__ == "__main__":
    main()