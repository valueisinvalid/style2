#!/usr/bin/env python3
"""
Advanced Python CLI Tool for Graded Fashion Compatibility Labeling (1–5)

- Reads items from a JSON file (default: out.json).
- Presents logical pairs based on role with weighted probabilities:
    * 50%: base + bottom
    * 30%: base + mid
    * 20%: one-piece + (mid or outer)
- Avoids relabeling the same pair (or its reverse).
- Appends each rating immediately to CSV (default: ground_truth_labels_graded.csv).
- Can be stopped anytime (q) and resumed later; previously labeled pairs are skipped.
- UI: shows IDs, names, roles, types, image URLs for both items.
- NEW: --limit option to cap labels per session (e.g., 500 for batching).

USAGE:
    python cli_labeling.py -i out.json -o ground_truth_labels_graded.csv --limit 500
OPTIONS:
    --no-infer     Disable role inference (use only explicit 'role' in JSON)
    --stats-only   Print role stats and exit
    --seed N       RNG seed
    --limit N      Stop after labeling N pairs this session
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

ONEPIECE_KW = ["dress","jumpsuit","playsuit","overall","dungaree","romper"]
OUTER_KW    = ["coat","jacket","trench","parka","puffer","overcoat","anorak","raincoat","blazer","windbreaker","shell"]
MID_KW      = ["cardigan","sweater","jumper","hoodie","pullover","vest","gilet","sweatshirt","zip-up","zip up"]
BOTTOM_KW   = ["jeans","trouser","trousers","pants","pant","chino","skirt","shorts","leggings","legging","jogger","culotte","cargo"]
BASE_KW     = ["t-shirt","tee","tank","camisole","singlet","bodysuit","shirt","blouse","polo","henley","tube top","crop top","long sleeve","short sleeve","knit top","crewneck"]

def _has_any(txt: str, words: List[str]) -> bool:
    return any(w in txt for w in words)

def keyword_role_infer(name: str, itype: str, desc: str) -> Optional[str]:
    txt = " ".join([name or "", itype or "", desc or ""]).lower()
    # Priority: one-piece > outer > mid > (bottom vs base)
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

def load_items(json_path: str, allow_infer: bool = True) -> List[Dict]:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Input file '{json_path}' not found.")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of item dictionaries.")
    for item in data:
        item["role"] = normalize_or_infer_role(item, allow_infer=allow_infer)
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
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(["item_id_1", "item_id_2", "compatibility_score"])

def load_labeled_pairs(csv_path: str) -> Set[Tuple[str, str]]:
    seen: Set[Tuple[str, str]] = set()
    if not os.path.exists(csv_path):
        return seen
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        first = next(reader, None)
        def is_header(row):
            if not row: return False
            j = ",".join(c.lower() for c in row)
            return "item_id_1" in j and "item_id_2" in j
        if first and not is_header(first):
            row = first
            if len(row) >= 2:
                a, b = row[0].strip(), row[1].strip()
                if a and b:
                    seen.add(tuple(sorted((a,b))))
        for row in reader:
            if len(row) < 2:
                continue
            a, b = row[0].strip(), row[1].strip()
            if a and b:
                seen.add(tuple(sorted((a,b))))
    return seen

def append_label(csv_path: str, id1: str, id2: str, score: int) -> None:
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow([id1, id2, score])

# --------------------------
# Pair generation (role-aware)
# --------------------------

class PairGenerator:
    def __init__(self, buckets: Dict[str, List[Dict]], seen: Set[Tuple[str, str]]):
        self.b = buckets
        self.seen = seen

    def feasible_types(self) -> List[Tuple[str, str]]:
        feas = []
        if self.b.get("base") and self.b.get("bottom"):
            feas.append(("base","bottom"))
        if self.b.get("base") and self.b.get("mid"):
            feas.append(("base","mid"))
        if self.b.get("one-piece") and self.b.get("mid"):
            feas.append(("one-piece","mid"))
        if self.b.get("one-piece") and self.b.get("outer"):
            feas.append(("one-piece","outer"))
        return feas

    def choose_pair_type(self) -> Optional[Tuple[str, str]]:
        feas = self.feasible_types()
        if not feas:
            return None
        desired = {
            ("base","bottom"): 0.50,
            ("base","mid"):    0.30,
            ("one-piece","mid"):  0.10,  # split of 0.20
            ("one-piece","outer"):0.10,
        }
        weights = [desired[t] for t in feas]
        s = sum(weights)
        if s <= 0:
            weights = [1.0]*len(feas)
            s = float(len(feas))
        weights = [w/s for w in weights]
        return random.choices(feas, weights=weights, k=1)[0]

    def next_unlabeled_pair(self, max_attempts: int = 500) -> Optional[Tuple[Dict, Dict, Tuple[str,str]]]:
        if not self.feasible_types():
            return None
        for _ in range(max_attempts):
            pt = self.choose_pair_type()
            if not pt:
                return None
            ra, rb = pt
            la, lb = self.b.get(ra, []), self.b.get(rb, [])
            if not la or not lb:
                continue
            a, b = random.choice(la), random.choice(lb)
            if a.get("item_id") == b.get("item_id"):
                continue
            id1, id2 = a["item_id"], b["item_id"]
            if tuple(sorted((id1,id2))) in self.seen:
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

def header_for_pair_type(pair_type: Tuple[str, str]) -> str:
    return f"-------------------- NEW PAIR ({pair_type[0].capitalize()} + {pair_type[1].capitalize()}) --------------------"

# --------------------------
# Main labeling loop (CLI) — supports session limit
# --------------------------

def labeling_loop(items: List[Dict], csv_path: str, seed: Optional[int] = None, session_limit: Optional[int] = None) -> None:
    if seed is not None:
        random.seed(seed)

    ensure_csv_with_header(csv_path)
    seen = load_labeled_pairs(csv_path)

    buckets = bucket_by_role(items)
    counts = {r: len(buckets.get(r, [])) for r in ["base","bottom","mid","outer","one-piece"]}
    total_with_role = sum(counts.values())
    print(f"Loaded {total_with_role} items with role (base:{counts['base']} bottom:{counts['bottom']} mid:{counts['mid']} outer:{counts['outer']} one-piece:{counts['one-piece']}).")
    print(f"Existing labeled pairs: {len(seen)}.")
    missing_roles = [it for it in items if it.get("item_id") and not it.get("role")]
    if missing_roles:
        print(f"Note: {len(missing_roles)} items still have unknown role (kept out of pairing).")

    if session_limit is not None and session_limit <= 0:
        print("Session limit is 0 — nothing to do.")
        return

    pairgen = PairGenerator(buckets, seen)

    labeled_this_session = 0
    skipped_this_session = 0
    invalid_inputs = 0

    print("\nInteractive Graded Labeling (1–5). Commands: 1/2/3/4/5 to rate, 's' to skip, 'q' to quit.\n")

    while True:
        if session_limit is not None and labeled_this_session >= session_limit:
            print(f"\nReached session limit of {session_limit} labels. Take a break! ☕")
            break

        res = pairgen.next_unlabeled_pair()
        if not res:
            print("\nNo new feasible pairs remain. You're all caught up 🎉")
            break

        a, b, pair_type = res
        print(header_for_pair_type(pair_type))
        print(format_item_block("ITEM 1", a))
        print()
        print(format_item_block("ITEM 2", b))
        print("------------------------------------------------------------------------------------------------")
        if session_limit:
            print(f"Progress this session: {labeled_this_session}/{session_limit}")

        try:
            user_in = input("Rate Compatibility (1-5), or (s)kip, (q)uit > ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            user_in = "q"

        if user_in == "q":
            print("\nSession summary:")
            print(f"  Labeled: {labeled_this_session}")
            print(f"  Skipped: {skipped_this_session}")
            print(f"  Invalid inputs: {invalid_inputs}")
            print("Goodbye!")
            return

        if user_in == "s":
            skipped_this_session += 1
            print("Skipped.\n")
            continue

        if user_in in {"1","2","3","4","5"}:
            score = int(user_in)
            id1, id2 = a["item_id"], b["item_id"]
            try:
                append_label(csv_path, id1, id2, score)
            except Exception as e:
                print(f"Error writing to CSV: {e}", file=sys.stderr)
                continue
            seen.add(tuple(sorted((id1,id2))))
            labeled_this_session += 1
            print(f"Saved: ({id1}, {id2}) -> {score}\n")
            continue

        # If pair is base+mid, add a random bottom for context
        context_bottom = None
        if pair_type == ("base", "mid") and buckets.get("bottom"):
            context_bottom = random.choice(buckets["bottom"])
            print(format_item_block("CONTEXT BOTTOM", context_bottom))
            print("------------------------------------------------------------------------------------------------")

        invalid_inputs += 1
        print("Invalid input. Please enter 1, 2, 3, 4, 5, 's' to skip, or 'q' to quit.")

    

    print("\nSession summary:")
    print(f"  Labeled: {labeled_this_session}")
    print(f"  Skipped: {skipped_this_session}")
    print(f"  Invalid inputs: {invalid_inputs}")
    print("Goodbye!")

# --------------------------
# Entrypoint
# --------------------------

def main(input_path: str = "out.json",
         output_csv: str = "ground_truth_labels_graded.csv",
         seed: Optional[int] = None,
         allow_infer: bool = True,
         stats_only: bool = False,
         session_limit: Optional[int] = None) -> None:
    items = load_items(input_path, allow_infer=allow_infer)
    if stats_only:
        buckets = bucket_by_role(items)
        total = sum(len(v) for v in buckets.values())
        c = {k: len(v) for k,v in buckets.items()}
        all_roles = ["base","bottom","mid","outer","one-piece"]
        print(f"Total items with item_id & role: {total}")
        for r in all_roles:
            print(f"{r}: {c.get(r,0)}")
        unknown = [x for x in items if x.get("item_id") and not x.get("role")]
        print(f"Unknown role after normalization/inference: {len(unknown)}")
        if unknown[:3]:
            print("Sample unknowns:", [u.get("name") for u in unknown[:3]])
        return

    labeling_loop(items, output_csv, seed=seed, session_limit=session_limit)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graded fashion compatibility labeling tool (1–5) with role-aware pairing.")
    parser.add_argument("-i","--input", default="out.json", help="Path to input JSON (default: out.json)")
    parser.add_argument("-o","--output", default="ground_truth_labels_graded.csv", help="Path to output CSV (default: ground_truth_labels_graded.csv)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--no-infer", action="store_true", help="Disable role inference (use only explicit 'role')")
    parser.add_argument("--stats-only", action="store_true", help="Print role bucket counts and exit")
    parser.add_argument("--limit", type=int, default=None, help="Stop after labeling N pairs this session (batching)")
    args = parser.parse_args()

    main(
        input_path=args.input,
        output_csv=args.output,
        seed=args.seed,
        allow_infer=(not args.no_infer),
        stats_only=args.stats_only,
        session_limit=args.limit
    )