#!/usr/bin/env python3
"""
Train a simple aesthetic compatibility model from labeled pairs.

Inputs:
  - features_v1.npz   (created earlier: ids, lab, sty[L2-normalized])
  - ground_truth_labels.csv  (columns: item_id_1,item_id_2,is_compatible)

Output:
  - compat_model.joblib  (sklearn LogisticRegression + features for convenience)

Usage:
  python train_compat.py
  python train_compat.py --labels ground_truth_labels.csv --features features_v1.npz --balanced
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix
from joblib import dump

def deltaE76(a, b):
    d = a - b
    return np.sqrt((d**2).sum(axis=-1))

def build_pair_features(ids, lab, sty, labels_df):
    """Return X (N,2) and y (N,) using cosine(sty) and ΔE76(lab). Drops rows missing in features."""
    index = {i: k for k, i in enumerate(ids)}
    keep = labels_df.apply(lambda r: (r.item_id_1 in index) and (r.item_id_2 in index), axis=1)
    df = labels_df[keep].reset_index(drop=True)

    rows = []
    for r in df.itertuples(index=False):
        i, j = index[r.item_id_1], index[r.item_id_2]
        cos = float((sty[i] * sty[j]).sum())                 # cosine since sty is L2-normalized
        de  = float(deltaE76(lab[i], lab[j]))                # perceptual color distance
        rows.append([cos, de])

    X = np.array(rows, dtype=np.float32)
    y = df["is_compatible"].astype(int).values
    return X, y, len(labels_df) - len(df)  # dropped count

def train_and_eval(X, y, balanced: bool, seed: int = 42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=seed
    )
    kwargs = dict(max_iter=1000)
    if balanced:
        kwargs["class_weight"] = "balanced"

    clf = LogisticRegression(**kwargs)
    clf.fit(X_train, y_train)

    probs = clf.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    auc = roc_auc_score(y_test, probs)
    acc = accuracy_score(y_test, preds)
    cm  = confusion_matrix(y_test, preds)
    rep = classification_report(y_test, preds, digits=3)

    return clf, dict(AUC=auc, ACC=acc, CM=cm, REPORT=rep)

def save_model(clf, ids, lab, sty, out_path="compat_model.joblib"):
    dump({"model": clf, "ids": ids, "lab": lab, "sty": sty}, out_path)
    print(f"Saved model to {out_path}")

# -------- inference helper you can import elsewhere --------
def rank_neighbors(query_id: str, bundle, k: int = 12, pool_size: int = 200):
    """
    Re-rank top style neighbors with the learned model.
    Returns: (ids_top, scores) sorted by descending compatibility probability.
    """
    from sklearn.neighbors import NearestNeighbors

    clf = bundle["model"]; ids = bundle["ids"]; lab = bundle["lab"]; sty = bundle["sty"]
    index = {i: k for k, i in enumerate(ids)}
    if query_id not in index:
        raise ValueError(f"{query_id} not found in features")

    nn = NearestNeighbors(n_neighbors=min(pool_size, len(ids)), metric="cosine").fit(sty)
    qi = index[query_id]
    _, inds = nn.kneighbors(sty[[qi]], n_neighbors=min(pool_size, len(ids)))
    cand = inds[0]

    cos = (sty[qi] * sty[cand]).sum(axis=1)
    de  = np.sqrt(((lab[qi] - lab[cand])**2).sum(axis=1))
    Xq  = np.vstack([cos, de]).T.astype(np.float32)
    p   = clf.predict_proba(Xq)[:, 1]
    order = np.argsort(-p)
    top = cand[order][:k]
    return ids[top], p[order][:k]
# -----------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features_v1.npz", help="NPZ with ids, lab, sty")
    ap.add_argument("--labels",   default="ground_truth_labels.csv", help="CSV with labeled pairs")
    ap.add_argument("--balanced", action="store_true", help="Use class_weight='balanced' in LogisticRegression")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    # load features
    z = np.load(args.features)
    ids = z["ids"]
    lab = z["lab"].astype(np.float32)
    sty = z["sty"].astype(np.float32)

    # load labels
    df = pd.read_csv(args.labels)
    df = df.dropna(subset=["item_id_1", "item_id_2", "is_compatible"])
    df["item_id_1"] = df["item_id_1"].astype(str)
    df["item_id_2"] = df["item_id_2"].astype(str)
    df["is_compatible"] = df["is_compatible"].astype(int)

    # basic QA
    print(f"Labels loaded: {len(df)} rows")
    print("Label balance:\n", df["is_compatible"].value_counts())

    # build pair features
    X, y, dropped = build_pair_features(ids, lab, sty, df)
    if dropped:
        print(f"Dropped {dropped} rows whose IDs were missing from features.")
    print("Dataset X shape:", X.shape, " Pos rate:", float(y.mean()).__round__(3))

    # train + eval
    clf, metrics = train_and_eval(X, y, balanced=args.balanced, seed=args.seed)
    print(f"AUC: {metrics['AUC']:.3f}  ACC: {metrics['ACC']:.3f}")
    print("Confusion matrix:\n", metrics["CM"])
    print(metrics["REPORT"])

    # save model
    save_model(clf, ids, lab, sty, out_path="compat_model.joblib")

if __name__ == "__main__":
    main()