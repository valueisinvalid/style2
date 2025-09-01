#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
add_item_types.py
-----------------
Quick utility to enrich a label CSV with the `type` (and optionally `role`) of each
item ID by looking them up in an items JSON catalogue (e.g., out.json).

Unlike `annotate_pair_type.py`, this script **never drops rows**. If an item's
metadata is missing in the catalogue, the new columns will just be left blank.

Usage
~~~~~
python dataset_ayk/add_item_types.py \
    --csv   ML/ground_truth_labels_cleaned.csv \
    --items out.json \
    --out   ML/ground_truth_with_types.csv \
    [--include-role]
"""
import argparse, csv, json, os, sys
from typing import Dict

# ------------------- Type -> Role classification -------------------
BOTTOM_TYPES = {
    "Skirt","Skirts","Denim Skirt","Denim Skirts",
    "Trousers","Pants","Jeans","Shorts","Leggings","Joggers",
    "Chinos","Culottes","Cargo Pants","Cargo Trousers","Flare & Bootcut",
    "Bootcut","Flare","Flared"
}
MID_TYPES = {
    "Jumpers","Sweaters","Cardigans","Hoodies","Sweatshirts",
    "Vest","Vests","Gilet","Knitted tops","Knitwear"
}
OUTER_TYPES = {
    "Coat","Coats","Jacket","Jackets","Trench","Parka","Puffer","Overcoat","Blazer","Windbreaker"
}
ONEPIECE_TYPES = {
    "Dress","Dresses","Short Dresses","Midi Dresses","Maxi Dresses",
    "Jumpsuit","Playsuit","Romper","Overall","Dungarees"
}
BASE_TYPES = {
    "Tops","Top","T-Shirts","T-Shirt","Tee","Tank","Camisole",
    "Shirts","Blouses","Long Sleeve","Short Sleeve","Crop Top","Bodysuit",
    "Vest Top","Vest Tops","vest top","vest tops"
}

# keyword lists for fallback substring search (lowercase)
BASE_KW = [
    "t-shirt","tee","tank","camisole","singlet","bodysuit","crop top",
    "long sleeve","short sleeve","knit top","crewneck","shirt","blouse","top","vest","vest top",
]
MID_KW  = ["cardigan","sweater","jumper","hoodie","pullover","sweatshirt","zip-up","zip up","gilet"]
BOTTOM_KW = [
    "jeans","trouser","trousers","pants","pant","chino","skirt","shorts",
    "leggings","legging","jogger","culotte","cargo",
]
ONEPIECE_KW = ["dress","jumpsuit","playsuit","overall","dungaree","romper"]
OUTER_KW = ["coat","jacket","trench","parka","puffer","overcoat","blazer","windbreaker","anorak","raincoat"]

def _role_from_type_str(t: str) -> str:
    t = (t or "").strip()
    if not t:
        return ""
    if t in BASE_TYPES:     return "base"
    if t in MID_TYPES:      return "mid"
    if t in BOTTOM_TYPES:   return "bottom"
    if t in OUTER_TYPES:    return "outer"
    if t in ONEPIECE_TYPES: return "one-piece"
    # try last token fallback
    parts = t.split()
    if parts and parts[-1] != t:
        guess = _role_from_type_str(parts[-1])
        if guess:
            return guess

    # --- keyword fallback (lowercase contains) ---
    low = t.lower()
    def _has_any(words):
        return any(w in low for w in words)

    if _has_any(ONEPIECE_KW): return "one-piece"
    if _has_any(OUTER_KW):    return "outer"
    if _has_any(MID_KW):      return "mid"
    if _has_any(BOTTOM_KW):   return "bottom"
    if _has_any(BASE_KW):     return "base"
    return ""

def load_items(path: str) -> Dict[str, dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {str(it.get("item_id", "")).strip(): it for it in data if it.get("item_id")}
    except Exception as e:
        print(f"[ERROR] Could not read items JSON: {e}")
        sys.exit(1)

def main():
    ap = argparse.ArgumentParser(description="Append type (and optionally role) columns to a CSV of item pairs.")
    ap.add_argument("--csv", required=True, help="Input CSV with columns containing two item IDs")
    ap.add_argument("--items", default="out.json", help="JSON catalogue with item metadata (default: out.json)")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--include-role", action="store_true", help="Also add role_1 and role_2 columns if available")
    args = ap.parse_args()

    items = load_items(args.items)

    # Open input
    with open(args.csv, "r", encoding="utf-8") as fi:
        reader = csv.DictReader(fi)
        cols = {c.lower().strip(): c for c in (reader.fieldnames or [])}
        def pick_col(cands):
            for c in cands:
                if c in cols:
                    return cols[c]
            raise KeyError(f"Required column not found (tried {', '.join(cands)}) in CSV")

        c_i1 = pick_col(["item_id_1", "id1", "item1", "a", "source_id"])
        c_i2 = pick_col(["item_id_2", "id2", "item2", "b", "target_id"])

        fieldnames = reader.fieldnames + ["type_1", "type_2"]
        if args.include_role:
            fieldnames += ["role_1", "role_2"]

        rows = list(reader)

    # Write output
    kept = 0; dropped = 0
    with open(args.out, "w", newline="", encoding="utf-8") as fo:
        writer = csv.DictWriter(fo, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            id1 = str(row.get(c_i1, "")).strip()
            id2 = str(row.get(c_i2, "")).strip()
            it1 = items.get(id1, {})
            it2 = items.get(id2, {})

            t1 = str(it1.get("type", "")).strip()
            t2 = str(it2.get("type", "")).strip()
            if t1.lower() == "hm.com" or t2.lower() == "hm.com":
                dropped += 1
                continue  # skip rows with ambiguous generic type

            row["type_1"] = t1
            row["type_2"] = t2

            if args.include_role:
                r1 = it1.get("role", "") or _role_from_type_str(t1)
                r2 = it2.get("role", "") or _role_from_type_str(t2)
                row["role_1"] = r1
                row["role_2"] = r2

            writer.writerow(row)
            kept += 1

    print(f"Enriched CSV written to: {args.out}  (kept: {kept}, dropped (HM.com): {dropped})")

if __name__ == "__main__":
    main()
