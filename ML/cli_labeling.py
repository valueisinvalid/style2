#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI Labeling – Probabilistic Pair Generator (base+bottom) vs (base+mid)

- Input: out.json (varsayılan) – H&M benzeri item listesi
- Output: ground_truth_labels_graded.csv (varsayılan)
- Pair tipleri (olasılıklı):
    * %50: (base, bottom)
    * %50: (base, mid)
- Duplicate önleme: (id1,id2) ve (id2,id1) aynı kabul edilir, CSV’den yüklenir.
- Kaydetme: Her puan (1–5) girildiğinde CSV’ye anında ekler.
- Komutlar: [1–5] puan, (s)kip, (q)uit
- Ekler:
    * --limit N  : bu oturumda en fazla N çift puanla
    * --seed N   : RNG seed
    * --no-infer : rol çıkarımını kapat (yalnızca item["role"] kullan)
    * --stats-only : rol kovalarını yaz ve çık
"""

import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

# --------------------------
# Role normalization & inference
# --------------------------
# ---- Type → Role zorlayıcı eşleme (hard rules) ----
BOTTOM_TYPES = {
    "Skirt","Skirts","Denim Skirt","Denim Skirts",
    "Trousers","Pants","Jeans","Shorts","Leggings","Joggers",
    "Chinos","Culottes","Cargo Pants","Cargo Trousers",
    "Flare & Bootcut","Bootcut","Flare","Flared" 
}
MID_TYPES = {
    "Jumper","Sweater","Sweaters","Cardigan","Cardigans","Hoodies","Sweatshirt","Sweatshirts",
    "Vest","Vests","Gilet","Knitted top", "Knitted tops","Knitwear","Jumpers","Hoodie","hoodie","Hooded Jackets"
}
OUTER_TYPES = {
    "Coat","Coats","Jacket","Jackets","Trench","Parka","Puffer","Overcoat","Blazer","Windbreaker"
}
ONEPIECE_TYPES = {
    "Dress","Dresses","Short Dress","Midi Dress","Maxi Dress","Short dress","Midi dress","Maxi dress"
    "Jumpsuit","Playsuit","Romper","Overall","Dungarees","Short Dresses","Midi Dresses","Maxi Dresses"
}
BASE_TYPES = {
    "Tops","Top","T-Shirts","T-Shirts","T-shirt","Tee","Tank","Camisole",
    "Shirts","Blouses","Long Sleeve","Short Sleeve","Crop Top","Bodysuit",
    "vest top","vest tops","Vest Top","Vest Tops"
}

# Pre-compute lower-case lookup sets for case-insensitive match
_BOTTOM_TYPES_LC   = {s.lower() for s in BOTTOM_TYPES}
_MID_TYPES_LC      = {s.lower() for s in MID_TYPES}
_OUTER_TYPES_LC    = {s.lower() for s in OUTER_TYPES}
_ONEPIECE_TYPES_LC = {s.lower() for s in ONEPIECE_TYPES}
_BASE_TYPES_LC     = {s.lower() for s in BASE_TYPES}


def role_from_type(itype: str) -> str | None:
    t = (itype or "").strip()
    t_norm = t.lower()
    # Esnek karşılaştırma: son kelimeyi de dene (örn. "Denim Skirts")
    last = t.split()[-1] if t else ""
    last_norm = last.lower()

    def in_any_lc(x):
        return (x in _BOTTOM_TYPES_LC or x in _MID_TYPES_LC or x in _OUTER_TYPES_LC or 
                x in _ONEPIECE_TYPES_LC or x in _BASE_TYPES_LC)

    # Tam eşleşme (lowercase compare)
    if t_norm in _BOTTOM_TYPES_LC:   return "bottom"
    if t_norm in _MID_TYPES_LC:      return "mid"
    if t_norm in _OUTER_TYPES_LC:    return "outer"
    if t_norm in _ONEPIECE_TYPES_LC: return "one-piece"
    if t_norm in _BASE_TYPES_LC:     return "base"
    # Son kelime eşleşmesi
    if in_any_lc(last_norm):
        return role_from_type(last)  # recurse with original string to preserve special cases
    return None

def coerce_role_by_type(item: dict, prefer_type: bool = True) -> None:
    """type → role eşleşmesi baskın olsun (prefer_type=True iken)."""
    if not prefer_type:
        return
    itype = item.get("type", "")
    derived = role_from_type(itype)
    # Special case: H&M catalogs "vest top" items under generic "Vests" type.
    # If the name explicitly contains "vest top", treat it as a base layer.
    name_lower = str(item.get("name", "")).lower()
    itype_lower = itype.lower()
    if itype_lower in {"vest", "vests"} and "vest top" in name_lower:
        derived = "base"
    if derived and derived != item.get("role"):
        item["role"] = derived

def normalize_role_value(s: str) -> str:
    r = s.strip().lower().replace("_", "-")
    mapping = {
        "base": "base", "top": "base",
        "bottom": "bottom", "bottoms": "bottom",
        "mid": "mid", "mid-layer": "mid", "midlayer": "mid", "layer": "mid",
        "outer": "outer", "outerwear": "outer",
        "one-piece": "one-piece", "onepiece": "one-piece",
        "dress": "one-piece", "jumpsuit": "one-piece", "romper": "one-piece",
    }
    return mapping.get(r, r)

# Basit anahtar kelime listeleri (gerektiğinde genişletebilirsin)
ONEPIECE_KW = ["dress","jumpsuit","playsuit","overall","dungaree","romper"]
OUTER_KW    = ["coat","jacket","trench","parka","puffer","overcoat","anorak","raincoat","blazer","windbreaker","shell"]
MID_KW      = ["cardigan","sweater","jumper","hoodie","pullover","vest","gilet","sweatshirt","zip-up","zip up"]
BOTTOM_KW   = ["jeans","trouser","trousers","pants","pant","chino","skirt","shorts","leggings","legging","jogger","culotte","cargo"]
BASE_KW     = ["t-shirt","tee","tank","camisole","singlet","bodysuit","crop top","long sleeve","short sleeve","knit top","crewneck","shirt","blouse","top"]

def _has_any(txt: str, words: List[str]) -> bool:
    return any(w in txt for w in words)

def keyword_role_infer(name: str, itype: str, desc: str) -> Optional[str]:
    txt = " ".join([name or "", itype or "", desc or ""]).lower()
    # Öncelik: one-piece > outer > mid > (bottom vs base)
    if _has_any(txt, ONEPIECE_KW): return "one-piece"
    if _has_any(txt, OUTER_KW):    return "outer"
    if _has_any(txt, MID_KW):      return "mid"
    if _has_any(txt, BOTTOM_KW) and not _has_any(txt, BASE_KW): return "bottom"
    if _has_any(txt, BASE_KW):     return "base"
    if _has_any(txt, BOTTOM_KW):   return "bottom"
    return None

def normalize_or_infer_role(item: Dict, allow_infer: bool = True) -> Optional[str]:
    raw = item.get("role")
    if raw and str(raw).strip():
        return normalize_role_value(str(raw))
    if not allow_infer:
        return None
    return keyword_role_infer(item.get("name",""), item.get("type",""), item.get("description",""))

# --------------------------
# Data loading & bucketing
# --------------------------

def load_items(json_path: str, allow_infer: bool = True, prefer_type: bool = True) -> List[Dict]:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Input file '{json_path}' not found.")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of item dictionaries.")
    for item in data:
        item["role"] = normalize_or_infer_role(item, allow_infer=allow_infer)
        coerce_role_by_type(item, prefer_type=prefer_type)  # <-- kritik satır
    return data

def bucket_by_role(items: List[Dict]) -> Dict[str, List[Dict]]:
    buckets: Dict[str, List[Dict]] = defaultdict(list)
    for it in items:
        iid = it.get("item_id")
        if not iid:
            continue
        r = it.get("role")
        if r:
            buckets[r].append(it)
    return buckets

# --------------------------
# CSV state (resume/append)
# --------------------------

def ensure_csv_with_header(csv_path: str) -> None:
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["item_id_1","item_id_2","score"])

def load_seen_pairs(csv_path: str) -> Set[Tuple[str,str]]:
    seen: Set[Tuple[str,str]] = set()
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        with open(csv_path, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                id1 = (row.get("item_id_1") or "").strip()
                id2 = (row.get("item_id_2") or "").strip()
                if not id1 or not id2:
                    continue
                seen.add(tuple(sorted((id1,id2))))
    return seen

def append_label(csv_path: str, id1: str, id2: str, score: int) -> None:
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([id1, id2, score])

# --------------------------
# Pair sampling (50/50)
# --------------------------

def pick_pair(buckets: Dict[str, List[Dict]], seen: Set[Tuple[str,str]]) -> Optional[Tuple[Dict,Dict,Tuple[str,str]]]:
    """
    50% (base, bottom), 50% (base, mid)
    Gerekli kovalar yoksa diğer tipe geri düşer; yine olmuyorsa None.
    Duplicate olanları atlar (makul sayıda deneme yapar).
    """
    bases   = buckets.get("base", [])
    bottoms = buckets.get("bottom", [])
    mids    = buckets.get("mid", [])

    # tip seçimi
    types = [("base","bottom"), ("base","mid")]
    weights = [0.5, 0.5]
    desired = random.choices(types, weights=weights, k=1)[0]

    candidates: List[Tuple[str,str]] = [desired]
    # Fallback olarak diğer tipi de dene
    for t in types:
        if t != desired:
            candidates.append(t)

    for pt in candidates:
        a_pool = bases
        b_pool = bottoms if pt == ("base","bottom") else mids
        if not a_pool or not b_pool:
            continue
        # makul sayıda deneme (örn. 100) – duplicate olmayan bir eş bulmaya çalış
        for _ in range(100):
            a = random.choice(a_pool)
            b = random.choice(b_pool)
            if a["item_id"] == b["item_id"]:
                continue
            key = tuple(sorted((a["item_id"], b["item_id"])))
            if key in seen:
                continue
            return a, b, pt
    return None

# --------------------------
# UI helpers
# --------------------------

def _shorten(s: str, n: int = 300) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"

def format_item_block(title: str, item: Dict) -> str:
    name = item.get("name") or "(no name)"
    role = item.get("role") or "(no role)"
    itype = item.get("type") or ""
    desc = item.get("description") or ""
    img  = item.get("productImage") or ""
    iid  = item.get("item_id") or ""
    lines = [
        f"[{title}]",
        f"ID:      {iid}",
        f"Name:    {name}",
        f"Role:    {role}",
    ]
    if itype: lines.append(f"Type:    {itype}")
    if desc:  lines.append(f"Desc:    {_shorten(desc)}")
    if img:   lines.append(f"Image:   {img}")
    return "\n".join(lines)

def print_pair_ui(a: Dict, b: Dict, pair_type: Tuple[str,str]) -> None:
    if pair_type == ("base","bottom"):
        header = "-------------------- NEW PAIR (Base + Bottom) --------------------"
    else:
        header = "-------------------- NEW PAIR (Base + Mid-Layer) -----------------"
    print(header)
    print(format_item_block("ITEM 1 - Role: base", a))
    print()
    role_label = "mid-layer" if pair_type == ("base","mid") else "bottom"
    print(format_item_block(f"ITEM 2 - Role: {role_label}", b))
    print("---------------------------------------------------------------------")
    print("Rate Compatibility (1-5), or (s)kip, (q)uit > ", end="", flush=True)

# --------------------------
# Main loop
# --------------------------

def main(
    input_path: str = "out.json",
    output_csv: str = "ground_truth_labels_graded.csv",
    seed: Optional[int] = None,
    allow_infer: bool = True,
    stats_only: bool = False,
    session_limit: Optional[int] = None,
    prefer_type: bool = True,   # <--- eklendi
) -> None:

    if seed is not None:
        random.seed(seed)

    items = load_items(input_path, allow_infer=allow_infer, prefer_type=prefer_type)
    buckets = bucket_by_role(items)

    if stats_only:
        print("Role bucket counts:")
        for k in ["base","bottom","mid","outer","one-piece"]:
            print(f"  {k:10s}: {len(buckets.get(k, []))}")
        return

    ensure_csv_with_header(output_csv)
    seen = load_seen_pairs(output_csv)

    labeled_this_session = 0
    skipped_this_session = 0
    invalid_inputs = 0

    print("Interactive labeling started. Press 'q' to quit.\n")

    while True:
        if session_limit is not None and labeled_this_session >= session_limit:
            print(f"\nReached session limit ({session_limit}). Exiting.")
            break

        picked = pick_pair(buckets, seen)
        if not picked:
            print("No more unseen valid pairs could be generated. Exiting.")
            break
        a, b, pair_type = picked

        print_pair_ui(a, b, pair_type)
        user_in = input().strip().lower()

        if user_in == "q":
            print("Quitting—good job!")
            break

        if user_in == "s":
            # Skip: bu çifti bir daha sorma (oturum boyunca)
            skipped_this_session += 1
            seen.add(tuple(sorted((a["item_id"], b["item_id"]))))
            print("Skipped.\n")
            continue

        if user_in in {"1","2","3","4","5"}:
            score = int(user_in)
            id1, id2 = a["item_id"], b["item_id"]
            try:
                append_label(output_csv, id1, id2, score)
            except Exception as e:
                print(f"Error writing to CSV: {e}", file=sys.stderr)
                continue
            seen.add(tuple(sorted((id1,id2))))
            labeled_this_session += 1
            print(f"Saved: ({id1}, {id2}) -> {score}\n")
            continue

        invalid_inputs += 1
        print("Invalid input. Please enter 1, 2, 3, 4, 5, 's' to skip, or 'q' to quit.\n")

    print("\nSession summary:")
    print(f"  Labeled: {labeled_this_session}")
    print(f"  Skipped: {skipped_this_session}")
    print(f"  Invalid inputs: {invalid_inputs}")

# --------------------------
# CLI
# --------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI for graded fashion compatibility labeling.")
    parser.add_argument("-i","--input", default="out.json", help="Path to input JSON (default: out.json)")
    parser.add_argument("-o","--output", default="ground_truth_labels_graded.csv", help="Path to output CSV (default: ground_truth_labels_graded.csv)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--no-infer", action="store_true", help="Disable role inference (use only explicit 'role')")
    parser.add_argument("--stats-only", action="store_true", help="Print role bucket counts and exit")
    parser.add_argument("--limit", type=int, default=None, help="Stop after labeling N pairs this session (batching)")
    parser.add_argument("--no-type-coerce", action="store_true",
                        help="Disable coercing role from item 'type' (by default type wins).")
    args = parser.parse_args()

    main(
        input_path=args.input,
        output_csv=args.output,
        seed=args.seed,
        allow_infer=(not args.no_infer),
        stats_only=args.stats_only,
        session_limit=args.limit,
        prefer_type=(not args.no_type_coerce)   # <--- eklendi
    )