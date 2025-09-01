#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make a clean 4-column labels file (item_id_1,item_id_2,score,pair_type)
from a CSV that already contains role/type columns.

Input CSV needs:
- item_id_1, item_id_2
- score (or compatibility_score/rating/label/grade/value)
- role_1, role_2 (if missing, type_1/type_2 is used for inference)

Output:
- Clean CSV with: item_id_1,item_id_2,score,pair_type
"""

import argparse, csv
from typing import Dict, Optional, Tuple

ALLOWED_PAIR_TYPES = {"base_bottom", "base_mid"}

BASE_KW = ["t-shirt","tee","tank","camisole","bodysuit","crop top","long sleeve","short sleeve","shirt","blouse","top","vest"]
MID_KW = ["cardigan","sweater","jumper","hoodie","pullover","sweatshirt","gilet"]
BOTTOM_KW = ["jeans","trouser","trousers","pants","skirt","shorts","leggings","jogger","culotte","cargo"]
ONEPIECE_KW = ["dress","jumpsuit","playsuit","overall","dungaree","romper"]
OUTER_KW = ["coat","jacket","trench","parka","puffer","overcoat","blazer","windbreaker"]

def norm(s: str) -> str:
    return (s or "").strip()

def norm_role(s: str) -> str:
    r = norm(s).lower().replace("_","-")
    mapping = {"top":"base","bottoms":"bottom","mid-layer":"mid","midlayer":"mid","outerwear":"outer","onepiece":"one-piece"}
    return mapping.get(r, r)

def has_any(txt: str, words) -> bool:
    return any(w in txt for w in words)

def infer_role_from_type(name: str, typ: str) -> Optional[str]:
    txt = f"{name} {typ}".lower()
    if has_any(txt, ONEPIECE_KW): return "one-piece"
    if has_any(txt, OUTER_KW):    return "outer"
    if has_any(txt, MID_KW):      return "mid"
    if has_any(txt, BOTTOM_KW) and not has_any(txt, BASE_KW): return "bottom"
    if has_any(txt, BASE_KW):     return "base"
    if has_any(txt, BOTTOM_KW):   return "bottom"
    return None

def pick_cols(fieldnames):
    cols = {c.lower().strip(): c for c in (fieldnames or [])}
    def get(cands, required=True, name_hint=""):
        for c in cands:
            if c in cols: return cols[c]
        if required:
            raise SystemExit(f"[ERROR] Missing column ({name_hint or cands}) in CSV. Found: {list(cols.values())}")
        return None
    c_i1 = get(["item_id_1","id1","item1","a"], True, "item_id_1")
    c_i2 = get(["item_id_2","id2","item2","b"], True, "item_id_2")
    c_sc = get(["score","compatibility_score","rating","label","grade","value"], True, "score")
    c_r1 = get(["role_1"], False)
    c_r2 = get(["role_2"], False)
    c_t1 = get(["type_1"], False)
    c_t2 = get(["type_2"], False)
    return c_i1, c_i2, c_sc, c_r1, c_r2, c_t1, c_t2

def infer_pair_type(r1: str, r2: str) -> Optional[str]:
    if r1 == "base" and r2 == "bottom": return "base_bottom"
    if r1 == "base" and r2 == "mid":    return "base_mid"
    return None

def maybe_reorder(id1, id2, r1, r2, reorder: bool):
    if not reorder: return id1,id2,r1,r2
    if r2 == "base" and r1 in {"bottom","mid"}:
        return id2,id1,r2,r1
    return id1,id2,r1,r2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Input CSV with roles/types")
    ap.add_argument("--out", required=True, help="Output cleaned CSV")
    ap.add_argument("--no-reorder", action="store_true", help="Do not reorder base to first")
    args = ap.parse_args()

    with open(args.csv,"r",encoding="utf-8") as f:
        reader = csv.DictReader(f)
        c_i1,c_i2,c_sc,c_r1,c_r2,c_t1,c_t2 = pick_cols(reader.fieldnames)
        rows = list(reader)

    seen = {}
    for r in rows:
        id1 = norm(r.get(c_i1))
        id2 = norm(r.get(c_i2))
        if not id1 or not id2: continue
        key = tuple(sorted((id1,id2)))
        seen[key] = r  # last wins

    kept, skipped = 0, 0
    with open(args.out,"w",newline="",encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id_1","item_id_2","score","pair_type"])
        for _, r in seen.items():
            id1,id2 = norm(r.get(c_i1)), norm(r.get(c_i2))
            if id1==id2: continue
            sc_raw = norm(r.get(c_sc))
            try: sc = int(round(float(sc_raw)))
            except: skipped+=1; continue
            if not (1<=sc<=5): skipped+=1; continue

            r1 = norm_role(r.get(c_r1) or "") if c_r1 else ""
            r2 = norm_role(r.get(c_r2) or "") if c_r2 else ""
            if not r1 and c_t1: r1 = infer_role_from_type("", r.get(c_t1) or "")
            if not r2 and c_t2: r2 = infer_role_from_type("", r.get(c_t2) or "")

            id1o,id2o,ro1,ro2 = maybe_reorder(id1,id2,r1,r2,not args.no_reorder)
            ptype = infer_pair_type(ro1,ro2)
            if not ptype: skipped+=1; continue

            w.writerow([id1o,id2o,sc,ptype])
            kept+=1

    print(f"Wrote {args.out} | Kept: {kept} | Skipped: {skipped}")

if __name__=="__main__":
    main()