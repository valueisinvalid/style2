#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Annotate/repair pair_type in label CSV using roles from out.json.

- Reads items from out.json (expects items with "item_id" and either "role" OR enough metadata to infer a role)
- If "role" is missing, infers it from item "type"/"name"/"description"
- Reads CSV (flexible headers), appends or fills 'pair_type'
- Optionally reorders rows so base is first (left) and other is right
- Writes back inplace (creates a .bak backup) unless --out is specified
"""

import argparse, csv, json, os, shutil
from typing import Dict, Optional, Tuple

# --- Role inference helpers (from type/name/description) ---
BOTTOM_TYPES = {
    "Skirt","Skirts","Denim Skirt","Denim Skirts",
    "Trousers","Pants","Jeans","Shorts","Leggings","Joggers",
    "Chinos","Culottes","Cargo Pants","Cargo Trousers",
}
MID_TYPES = {
    "Jumpers","Sweaters","Cardigans","Hoodies","Sweatshirts","Gilet",
    "Knitted tops","Knitwear",
}
OUTER_TYPES = {
    "Coat","Coats","Jacket","Jackets","Trench","Parka","Puffer","Overcoat","Blazer","Windbreaker",
}
ONEPIECE_TYPES = {
    "Dress","Dresses","Short Dresses","Midi Dresses","Maxi Dresses",
    "Jumpsuit","Playsuit","Romper","Overall","Dungarees",
}
BASE_TYPES = {
    "Tops","Top","T-Shirts","T-Shirt","Tee","Tank","Camisole",
    "Shirts","Blouses","Long Sleeve","Short Sleeve","Crop Top","Bodysuit",
    "Vest","Vests","Vest Top","Low Cut Tops",
}

def role_from_type(itype: str) -> str | None:
    t = (itype or "").strip()
    low = t.lower()
    # Special rule: any 'vest' that is not 'gilet' is a base layer
    if "vest" in low and "gilet" not in low:
        return "base"
    if t in BOTTOM_TYPES:   return "bottom"
    if t in MID_TYPES:      return "mid"
    if t in OUTER_TYPES:    return "outer"
    if t in ONEPIECE_TYPES: return "one-piece"
    if t in BASE_TYPES:     return "base"
    # try last token fallback (e.g., "Denim Skirts" -> "Skirts")
    parts = t.split()
    if parts:
        last = parts[-1]
        if last == t:
            return None
        return role_from_type(last)
    return None

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
OUTER_KW = ["coat","jacket","trench","parka","puffer","overcoat","blazer","windbreaker","shell","anorak","raincoat"]

def _has_any(txt: str, words: list[str]) -> bool:
    return any(w in txt for w in words)

def keyword_role_infer(name: str, itype: str, desc: str) -> str | None:
    txt = " ".join([name or "", itype or "", desc or ""]).lower()
    if _has_any(txt, ONEPIECE_KW): return "one-piece"
    if _has_any(txt, OUTER_KW):    return "outer"
    if _has_any(txt, MID_KW):      return "mid"
    # prefer bottom when both bottom/base words appear
    if _has_any(txt, BOTTOM_KW) and not _has_any(txt, BASE_KW): return "bottom"
    if _has_any(txt, BASE_KW):     return "base"
    if _has_any(txt, BOTTOM_KW):   return "bottom"
    return None

def normalize_or_infer_role(item: dict) -> str | None:
    # normalize explicit role if present
    explicit = item.get("role")
    if explicit and str(explicit).strip():
        return normalize_role_value(str(explicit))
    # else infer from type/name/description
    r = role_from_type(item.get("type",""))
    if r:
        return r
    return keyword_role_infer(item.get("name",""), item.get("type",""), item.get("description",""))

def normalize_role_value(s: str) -> str:
    r = (s or "").strip().lower().replace("_","-")
    m = {
        "top":"base",
        "bottoms":"bottom",
        "mid-layer":"mid","midlayer":"mid","layer":"mid",
        "outerwear":"outer",
        "onepiece":"one-piece",
        "dress":"one-piece","jumpsuit":"one-piece","romper":"one-piece",
    }
    return m.get(r, r)

def load_roles(items_path: str) -> Dict[str,str]:
    with open(items_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    id2role: Dict[str,str] = {}
    for it in data:
        iid = str(it.get("item_id","")).strip()
        if not iid:
            continue
        role = normalize_or_infer_role(it) or ""
        role = normalize_role_value(role)
        if role:
            id2role[iid] = role
    return id2role

def pick_cols(fieldnames):
    cols = {c.lower().strip(): c for c in (fieldnames or [])}
    def get(cands, required=True, name_hint=""):
        for c in cands:
            if c in cols: return cols[c]
        if required:
            want = " | ".join(cands)
            raise SystemExit(f"[ERROR] Missing column ({name_hint or want}). Available: {list(cols.values())}")
        return None
    c_i1 = get(["item_id_1","id1","item1","a","source_id"], True, "item_id_1")
    c_i2 = get(["item_id_2","id2","item2","b","target_id"], True, "item_id_2")
    c_sc = get(["score","compatibility_score","rating","label","grade","value"], False, "score")
    c_pt = get(["pair_type","pairtype","ptype"], False, "pair_type")
    return c_i1, c_i2, c_sc, c_pt

def infer_pair_type(r1: str, r2: str) -> Optional[str]:
    if r1 == "base" and r2 == "bottom": return "base_bottom"
    if r1 == "base" and r2 == "mid":    return "base_mid"
    return None

def maybe_reorder(id1, id2, r1, r2, do_reorder: bool) -> Tuple[str,str,str,str]:
    if not do_reorder:
        return id1, id2, r1, r2
    if r2 == "base" and r1 in {"bottom","mid"}:
        return id2, id1, r2, r1
    return id1, id2, r1, r2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to labels CSV")
    ap.add_argument("--items", default="out.json", help="Path to items JSON (default: out.json)")
    ap.add_argument("--out", default=None, help="Output CSV path (default: inplace)")
    ap.add_argument("--no-reorder", action="store_true", help="Do not reorder so base is first")
    ap.add_argument("--add-roles", action="store_true", help="Add role_1 and role_2 columns")
    ap.add_argument("--write-types", action="store_true", help="Add type_1 and type_2 columns from items JSON")
    args = ap.parse_args()

    id2role = load_roles(args.items)
    if not id2role:
        print("[WARN] Could not infer any roles from items JSON; rows may be skipped if types are unknown.")

    # Also keep items by id for type/name access
    with open(args.items, "r", encoding="utf-8") as _f:
        _data = json.load(_f)
    items_by_id = {str(it.get("item_id","")).strip(): it for it in _data if it.get("item_id")}

    # Decide output path & backup
    out_path = args.out or args.csv
    tmp_path = out_path + ".tmp"

    # Read all rows
    with open(args.csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        c_i1, c_i2, c_sc, c_pt = pick_cols(reader.fieldnames)
        rows = list(reader)

    # Build header
    base_header = ["item_id_1","item_id_2"]
    if c_sc: base_header.append("score")  # we will normalize score name to 'score' if present
    header = base_header + ["pair_type"]
    if args.add_roles:
        header += ["role_1","role_2"]
    if args.write_types:
        header += ["type_1","type_2"]

    # Write processed rows
    kept = 0
    skipped = 0
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            id1 = (r.get(c_i1) or "").strip()
            id2 = (r.get(c_i2) or "").strip()
            if not id1 or not id2:
                skipped += 1
                continue
            r1 = normalize_role_value(id2role.get(id1, ""))
            r2 = normalize_role_value(id2role.get(id2, ""))

            # If any role missing, skip row (safer)
            if not r1 or not r2:
                skipped += 1
                continue

            id1o, id2o, ro1, ro2 = maybe_reorder(id1, id2, r1, r2, do_reorder=(not args.no_reorder))
            ptype = infer_pair_type(ro1, ro2)
            if not ptype:
                skipped += 1
                continue

            # score normalize (optional)
            score_val = None
            if c_sc:
                val = (r.get(c_sc) or "").strip()
                if val != "":
                    try:
                        score_val = int(round(float(val)))
                    except Exception:
                        pass

            out_row = [id1o, id2o]
            if c_sc:
                if score_val is None:
                    skipped += 1
                    continue
                out_row.append(score_val)
            out_row.append(ptype)
            if args.add_roles:
                out_row += [ro1, ro2]
            if args.write_types:
                t1 = (items_by_id.get(id1o, {}) or {}).get("type", "")
                t2 = (items_by_id.get(id2o, {}) or {}).get("type", "")
                out_row += [t1, t2]
            w.writerow(out_row)
            kept += 1

    # If writing inplace, back up original first time
    if args.out is None:
        bak = args.csv + ".bak"
        if not os.path.exists(bak):
            shutil.copy2(args.csv, bak)
        os.replace(tmp_path, args.csv)
        final = args.csv
    else:
        os.replace(tmp_path, out_path)
        final = out_path

    print(f"Annotated file written to: {final}")
    print(f"Kept rows: {kept} | Skipped rows: {skipped}")

if __name__ == "__main__":
    main()