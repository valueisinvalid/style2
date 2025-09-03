# logic/fetch_weather.py
import json
import argparse
import sys
from typing import Any, Dict
from weather_scoring import get_meteoblue_current  # aynı klasördeyse bu import çalışır

def main():
    p = argparse.ArgumentParser(description="Fetch current weather via Meteoblue and save to JSON")
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--apikey", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    try:
        data = get_meteoblue_current(api_key=args.apikey, lat=args.lat, lon=args.lon)
        if not data:
            print("No data from Meteoblue", file=sys.stderr)
            sys.exit(2)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Map meteoblue fields to mapper schema ---
    if "data_1h" in data:
        # Use the first hour slot (index 0)
        d1h = data["data_1h"]
        idx = 0
        temp_c = d1h.get("temperature", [None])[idx]
        rh = d1h.get("relativehumidity", [None])[idx]
        wind_ms = d1h.get("windspeed", [None])[idx]
        wind_kmh = None if wind_ms is None else round(wind_ms * 3.6, 2)
        precip_prob = d1h.get("precipitation_probability", [0])[idx]
        precip_mm = d1h.get("precipitation", [0])[idx]
        is_rain = bool(precip_mm and precip_mm > 0)
    else:
        # Fallback for flat structures (unlikely with Meteoblue)
        temp_c = data.get("temperature", data.get("t2m"))
        rh = data.get("relative_humidity", data.get("rh"))
        wind_kmh = data.get("wind_speed_kmh", data.get("wind_speed"))
        precip_prob = data.get("precip_probability", data.get("pp", 0))
        is_rain = bool(data.get("is_rain", False) or (data.get("precipitation", 0) > 0))

    mapped: Dict[str, Any] = {
        "temp_c": temp_c,
        "rh": rh,
        "wind_kmh": wind_kmh,
        "precip_prob": precip_prob,
        "is_rain": is_rain,
        "source": "meteoblue",
    }

    # Write normalized JSON for the outfit mapper.
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(mapped, f, ensure_ascii=False, indent=2)
    print(f"[fetch_weather] wrote {args.out} (normalized)")
    return

    # Fallback: write full raw payload (unreached due to return above)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[fetch_weather] wrote {args.out} (raw)")

if __name__ == "__main__":
    main()