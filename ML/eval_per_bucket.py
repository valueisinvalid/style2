#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate LightGBM model per pair_type bucket (base_bottom vs base_mid).

Assumptions about feature layout (as built earlier):
  X = [ style_diff (384) | style_prod (384) | color_dist (1) | is_base_bottom (1) | is_base_mid (1) ]
So the last two columns are the one-hots used to identify buckets.

Usage:
  python ML/eval_per_bucket.py \
    --data training_data.npz \
    --model aesthetic_scorer_v1.pkl \
    --seed 42 --test-size 0.20
"""

import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib

def main():
    ap = argparse.ArgumentParser(description="Per-bucket MAE eval for StylePops model.")
    ap.add_argument("--data", default="training_data.npz", help="NPZ with X,y")
    ap.add_argument("--model", default="aesthetic_scorer_v1.pkl", help="Path to saved model")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (must match training)")
    ap.add_argument("--test-size", type=float, default=0.20, help="Validation split (must match training)")
    args = ap.parse_args()

    # Load data
    d = np.load(args.data)
    X, y = d["X"], d["y"]

    # Recreate the SAME split used in training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )

    # Load model
    model = joblib.load(args.model)

    # Predict (use best_iteration_ if present)
    try:
        preds = model.predict(X_test, num_iteration=getattr(model, "best_iteration_", None))
    except TypeError:
        preds = model.predict(X_test)

    # Overall MAE
    overall_mae = mean_absolute_error(y_test, preds)

    # Identify buckets from one-hots (by convention from our feature builder)
    is_base_bottom = (X_test[:, -2] > 0.5)
    is_base_mid    = (X_test[:, -1] > 0.5)

    # Compute MAE per bucket (guard if empty)
    results = [("overall", overall_mae, len(y_test))]

    if is_base_bottom.any():
        mae_bb = mean_absolute_error(y_test[is_base_bottom], preds[is_base_bottom])
        results.append(("base_bottom", mae_bb, int(is_base_bottom.sum())))
    else:
        results.append(("base_bottom", float("nan"), 0))

    if is_base_mid.any():
        mae_bm = mean_absolute_error(y_test[is_base_mid], preds[is_base_mid])
        results.append(("base_mid", mae_bm, int(is_base_mid.sum())))
    else:
        results.append(("base_mid", float("nan"), 0))

    # Pretty print
    print("\nPer-bucket evaluation (MAE)")
    print("---------------------------")
    for name, mae, n in results:
        if n == 0 or (mae != mae):  # NaN check
            print(f"{name:12s} : MAE =   n/a   (n={n})")
        else:
            print(f"{name:12s} : MAE = {mae:.3f} (n={n})")
    print("")

if __name__ == "__main__":
    main()