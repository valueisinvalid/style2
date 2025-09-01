#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
De-dupe & Validate label CSV for StylePops
- Input CSV columns (flexible):
    * item_id_1 | id1 | item1 | a | source_id
    * item_id_2 | id2 | item2 | b | target_id
    * score | rating | label | grade | value | compatibility_score   <-- seninki bu
- Items JSON: out.json (id -> role)  (base/mid/bottom)
- Output: ground_truth_labels_cleaned.csv with columns:
    item_id_1, item_id_2, score, pair_type   (score 1..5 int, base first)
"""

import argparse
import csv
import json
import os
from collections import defaultdict, Counter
from statistics import mean, median, mode
from typing import Dict, List, Optional, Tuple

ALLOWED_PAIR_TYPES_DEFAULT = {"base_bottom", "base_mid"}

def norm(s: Optional[str]) -> str:
    return (s or "").strip()

def normalize_role_value(s: str) -> str:
    r = s.strip().lower().replace("_","-")
    mapping = {
        "base":"base","top":"base",
        "bottom":"bottom","bottoms":"bottom",
        "mid":"mid","mid-layer":"mid","midlayer":"mid",
        "outer":"outer","outerwear":"outer",
        "one-piece":"one-piece","onepiece":"one-piece",
    }
    return mapping.get(r, r)

def load_items_roles(path: str) -> Dict[str, str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Items JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Items JSON must be a list of dicts.")
    id2role: Dict[str,str] = {}
    for it in data:
        iid = norm(str(it.get("item_id","")))
        if not iid:
            continue
        role = it.get("role")
        if role:
            id2role[iid] = normalize_role_value(str(role))
    return id2role

def parse_score(x: str) -> Optional[int]:
    xs = norm(x)
    if xs == "":
        return None
    try:
        v = int(round(float(xs)))
        if 1 <= v <= 5:
            return v
    except Exception:
        return None
    return None

def canonical_key(id1: str, id2: str) -> Tuple[str,str]:
    return tuple(sorted((id1,id2)))

def orient_base_first(id1: str, id2: str, r1: str, r2: str) -> Tuple[str,str,str,str]:
    if r1 == "base" and r2 in {"bottom","mid"}:
        return id1,id2,r1,r2
    if r2 == "base" and r1 in {"bottom","mid"}:
        return id2,id1,r2,r1
    return id1,id2,r1,r2

def pair_type_of(r1: str, r2: str) -> Optional[str]:
    if r1 == "base" and r2 == "bottom": return "base_bottom"
    if r1 == "base" and r2 == "mid":    return "base_mid"
    return None

def resolve(scores: List[int], policy: str) -> int:
    if not scores: raise ValueError("Empty scores")
    if policy == "last":   return scores[-1]
    if policy == "first":  return scores[0]
    if policy == "mean":   return int(round(mean(scores)))
    if policy == "median": return int(round(median(scores)))
    if policy == "mode":
        try: return mode(scores)
        except Exception: return int(round(median(scores)))
    raise ValueError(f"Unknown conflict policy: {policy}")

def main():
    ap = argparse.ArgumentParser(description="De-dupe & validate label CSVs (compatibility_score aware).")
    ap.add_argument("-l","--labels", nargs="+", required=True, help="Label CSV path(s)")
    ap.add_argument("-i","--items", default="out.json", help="Items JSON with roles (default: out.json)")
    ap.add_argument("-o","--output", default="ground_truth_labels_cleaned.csv", help="Output CSV")
    ap.add_argument("--allowed", default="base_bottom,base_mid",
                    help="Comma-separated allowed pair types")
    ap.add_argument("--conflict-policy", choices=["last","first","mean","median","mode"],
                    default="last")
    ap.add_argument("--keep-unknown", action="store_true",
                    help="Keep rows where item ids are missing in items JSON")
    ap.add_argument("--no-role-validate", action="store_true",
                    help="Skip pair-type validation (still dedupes & cleans).")
    args = ap.parse_args()

    allowed_types = {s.strip().lower() for s in args.allowed.split(",") if s.strip()}
    if not allowed_types:
        allowed_types = set(ALLOWED_PAIR_TYPES_DEFAULT)

    id2role = load_items_roles(args.items)

    total_in = 0
    self_pairs = 0
    bad_scores = 0
    unknown_ids = 0
    disallowed = 0

    bucket = defaultdict(list)  # canonical_key -> list of rows

    for csv_path in args.labels:
        if not os.path.exists(csv_path):
            print(f"[WARN] Label CSV not found, skipping: {csv_path}")
            continue
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            cols = {c.lower().strip(): c for c in (reader.fieldnames or [])}

            def pick_col(cands, required=True, err_name=""):
                for c in cands:
                    if c in cols:
                        return cols[c]
                if required:
                    want = " | ".join(cands)
                    raise SystemExit(f"[ERROR] Missing required column ({err_name or want}). "
                                     f"Available: {list(cols.values())}")
                return None

            # id columns (flex)
            col_i1 = pick_col(["item_id_1","id1","item1","item_a","a","source_id"], True, "item_id_1")
            col_i2 = pick_col(["item_id_2","id2","item2","item_b","b","target_id"], True, "item_id_2")

            # score column (accept compatibility_score)
            col_sc = pick_col(["score","rating","label","grade","value","compatibility_score"], True, "score/compatibility_score")

            for row in reader:
                total_in += 1
                id1 = norm(row.get(col_i1, ""))
                id2 = norm(row.get(col_i2, ""))
                sc  = parse_score(str(row.get(col_sc, "")))

                if not id1 or not id2:
                    unknown_ids += 1
                    continue
                if id1 == id2:
                    self_pairs += 1
                    continue
                if sc is None:
                    bad_scores += 1
                    continue

                r1 = normalize_role_value(id2role.get(id1, "")) if not args.keep_unknown else id2role.get(id1, "")
                r2 = normalize_role_value(id2role.get(id2, "")) if not args.keep_unknown else id2role.get(id2, "")

                if not args.keep_unknown and (not r1 or not r2):
                    unknown_ids += 1
                    continue

                id1o, id2o, ro1, ro2 = orient_base_first(id1, id2, r1 or "", r2 or "")
                ptype = pair_type_of(ro1 or "", ro2 or "")

                if not args.no_role_validate and not args.keep_unknown:
                    if ptype is None or ptype not in allowed_types:
                        disallowed += 1
                        continue

                key = canonical_key(id1o, id2o)
                bucket[key].append((id1o, id2o, sc, ro1, ro2, ptype))

    kept = []
    duplicate_pairs = 0
    conflict_pairs = 0
    per_ptype = Counter()

    for key, rows in bucket.items():
        if not rows:
            continue
        if len(rows) > 1:
            duplicate_pairs += len(rows) - 1
        scores = [r[2] for r in rows]
        chosen_score = resolve(scores, args.conflict_policy)

        base_first_rows = [r for r in rows if r[3] == "base" and r[4] in {"bottom","mid"}]
        chosen_row = (base_first_rows[-1] if base_first_rows else rows[-1])
        id1o, id2o, _, ro1, ro2, ptype = chosen_row

        if len(set(scores)) > 1:
            conflict_pairs += 1

        per_ptype[ptype or "unknown"] += 1
        kept.append((id1o, id2o, chosen_score, ptype or ""))

    out_path = args.output
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id_1","item_id_2","score","pair_type"])
        for id1o, id2o, sc, ptype in kept:
            w.writerow([id1o, id2o, sc, ptype])

    print("\nDe-dupe & Validate Summary")
    print("--------------------------")
    print(f"Input rows total     : {total_in}")
    print(f"Kept (unique pairs)  : {len(kept)}")
    print(f"Removed self-pairs   : {self_pairs}")
    print(f"Removed bad scores   : {bad_scores}")
    print(f"Removed unknown IDs  : {unknown_ids}" + ("" if not args.keep_unknown else " (kept unknown ids)"))
    print(f"Removed disallowed   : {disallowed} (pair types not in {sorted(allowed_types)})")
    print(f"Collapsed duplicates : {duplicate_pairs}")
    print(f"Score conflicts      : {conflict_pairs} (resolved by: {args.conflict_policy})")
    print("\nPair type counts:")
    for k,v in sorted(per_ptype.items()):
        print(f"  {k:12s} : {v}")
    print(f"\nClean file written to: {out_path}\n")

if __name__ == "__main__":
    main()