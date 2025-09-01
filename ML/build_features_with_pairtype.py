#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build feature matrix (X, y) with pair_type info baked in.

Inputs
  - ground_truth_labels_cleaned.csv  (item_id_1,item_id_2,score,pair_type)
  - style_vectors.pkl  (dict id -> np.ndarray)
  - color_vectors.pkl  (dict id -> np.ndarray)

Outputs
  - training_data.npz containing:
      X : features
      y : targets
      pair_type_code : int array where 0=base_bottom, 1=base_mid
      ids : string array of "id1|id2" for debugging
"""

import argparse, csv, pickle, numpy as np

PAIR_MAP = {"base_bottom": 0, "base_mid": 1}

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="ground_truth_labels_cleaned.csv")
    ap.add_argument("--style", default="style_vectors.pkl")
    ap.add_argument("--color", default="color_vectors.pkl")
    ap.add_argument("--out",   default="training_data.npz")
    args = ap.parse_args()

    style_vecs = load_pickle(args.style)
    color_vecs = load_pickle(args.color)

    X_rows, y_rows, pt_rows, id_rows = [], [], [], []
    dropped = 0

    with open(args.labels, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        cols = {c.lower().strip(): c for c in (r.fieldnames or [])}
        def pick_col(cands, required=True):
            for c in cands:
                if c in cols: return cols[c]
            if required:
                raise KeyError(f"Missing required column, tried: {', '.join(cands)}")
            return None # Return None if not required

        c_i1 = pick_col(["item_id_1","id1","item1","a","source_id"])
        c_i2 = pick_col(["item_id_2","id2","item2","b","target_id"])
        c_sc = pick_col(["score","rating","label","grade","value","compatibility_score"])
        c_pt = pick_col(["pair_type","pairtype","ptype"], required=False)

        for row in r:
            id1, id2 = row[c_i1], row[c_i2]
            ptype = (row[c_pt] or "base_bottom").lower().strip() if c_pt else "base_bottom"
            if ptype not in PAIR_MAP:
                # unrecognized pair type, drop
                dropped += 1; continue
            try:
                y = int(round(float(row[c_sc])))
            except:
                dropped += 1; continue
            if id1 not in style_vecs or id2 not in style_vecs: dropped += 1; continue
            if id1 not in color_vecs or id2 not in color_vecs: dropped += 1; continue

            v1 = np.asarray(style_vecs[id1], dtype=float)
            v2 = np.asarray(style_vecs[id2], dtype=float)
            c1 = np.asarray(color_vecs[id1], dtype=float)
            c2 = np.asarray(color_vecs[id2], dtype=float)

            if v1.shape != (384,) or v2.shape != (384,) or c1.shape != (3,) or c2.shape != (3,):
                dropped += 1; continue

            style_diff = np.abs(v1 - v2)
            style_prod = v1 * v2
            color_dist = np.linalg.norm(c1 - c2)

            is_bb = 1 if ptype == "base_bottom" else 0
            is_bm = 1 if ptype == "base_mid" else 0
            feat = np.concatenate([style_diff, style_prod, [color_dist, is_bb, is_bm]])

            X_rows.append(feat)
            y_rows.append(y)
            pt_rows.append(PAIR_MAP[ptype])
            id_rows.append(f"{id1}|{id2}")

    if not X_rows:
        print("No valid rows; check your cleaned CSV & vector pickles.")
        return

    X = np.vstack(X_rows)
    y = np.array(y_rows, dtype=np.int16)
    pair_type_code = np.array(pt_rows, dtype=np.int8)
    ids = np.array(id_rows, dtype=object)

    np.savez_compressed(args.out, X=X, y=y, pair_type_code=pair_type_code, ids=ids)

    # quick summary
    n_bb = int((pair_type_code == 0).sum())
    n_bm = int((pair_type_code == 1).sum())
    print(f"Features built: X {X.shape}, y {y.shape} | base_bottom={n_bb}, base_mid={n_bm}, dropped={dropped}")

if __name__ == "__main__":
    main()