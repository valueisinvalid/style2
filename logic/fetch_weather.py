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

    # İstersen burada şemayı sadeleştir (outfit_mapper'ın beklediğine göre).
    # Şimdilik olduğu gibi yazıyoruz:
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[fetch_weather] wrote {args.out}")

if __name__ == "__main__":
    main()