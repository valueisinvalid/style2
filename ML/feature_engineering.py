#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python Script for ML Feature Engineering from Fashion Data
Amaç:
- ground_truth_labels_cleaned.csv
- style_vectors.pkl (384-dim S-BERT)
- color_vectors.pkl (3-dim CIE Lab)
dosyalarını kullanarak X(769-dim) ve y vektörlerini üretip training_data.npz olarak kaydeder.

Çalıştırma:
    python feature_engineering.py \
        --labels ground_truth_labels_cleaned.csv \
        --style style_vectors.pkl \
        --color color_vectors.pkl \
        --out training_data.npz

Terminal çıktısı yalnızca iki satır doğrulama içindir:
- X şekli
- y şekli
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import json


def build_matrices(labels_csv: Path, style_pkl: Path, color_pkl: Path):
    # Yüklemeler
    df = pd.read_csv(labels_csv, dtype={"item_id_1": str, "item_id_2": str})
    with open(style_pkl, "rb") as f:
        style_vectors = pickle.load(f)
    with open(color_pkl, "rb") as f:
        color_vectors = pickle.load(f)

    # ---- Load item metadata for role inference ----
    # Expect a JSON list of dicts with at least item_id, type, name, and optionally role.
    # Falls back to simple keyword inference if role not provided.
    items_meta = {}
    try:
        with open(Path(style_pkl).parent / "your_data.json", "r", encoding="utf-8") as f_meta:
            data_meta = json.load(f_meta)
            for it in data_meta:
                iid = str(it.get("item_id", "").strip())
                if not iid:
                    continue
                items_meta[iid] = it
    except Exception:
        # Metadata not essential; proceed with empty dict (role inference will rely solely on keyword matching)
        items_meta = {}

    def _infer_role_from_type(t: str) -> str | None:  # noqa: E701, F821  (python >=3.9 for |)
        t = (t or "").lower()
        if any(k in t for k in ["shirt", "t-shirt", "top", "blouse", "jumper", "sweater", "hoodie", "cardigan", "vest", "camisole", "tank"]):
            return "base"  # treat as base/top
        if any(k in t for k in ["jean", "pant", "trouser", "skirt", "short", "culotte", "legging"]):
            return "bottom"
        if any(k in t for k in ["coat", "jacket", "blazer", "parka", "anorak", "overcoat", "gilet", "puffer", "trench"]):
            return "outer"
        return None

    X_list, y_list = [], []

    for _, row in df.iterrows():
        id1 = str(row["item_id_1"])
        id2 = str(row["item_id_2"])
        y_val = float(row["compatibility_score"])

        s1 = style_vectors.get(id1)
        s2 = style_vectors.get(id2)
        c1 = color_vectors.get(id1)
        c2 = color_vectors.get(id2)

        # Vektör kontrolü
        if s1 is None or s2 is None or c1 is None or c2 is None:
            continue
        s1 = np.asarray(s1, dtype=float)
        s2 = np.asarray(s2, dtype=float)
        c1 = np.asarray(c1, dtype=float)
        c2 = np.asarray(c2, dtype=float)
        if s1.shape != (384,) or s2.shape != (384,) or c1.shape != (3,) or c2.shape != (3,):
            continue

        # ---------------- Relational / interactional features ----------------
        style_diff = s1 - s2
        style_prod = s1 * s2  # element-wise
        color_dist = np.linalg.norm(c1 - c2)  # scalar

        # ---- Role-based pair type flags ----
        role1 = None; role2 = None
        if id1 in items_meta:
            role1 = (items_meta[id1].get("role") or _infer_role_from_type(
                f"{items_meta[id1].get('type','')} {items_meta[id1].get('name','')}"))
        if id2 in items_meta:
            role2 = (items_meta[id2].get("role") or _infer_role_from_type(
                f"{items_meta[id2].get('type','')} {items_meta[id2].get('name','')}"))

        # If still missing, infer directly from style/type keywords (best effort)
        role1 = role1 or _infer_role_from_type("") or "unknown"
        role2 = role2 or _infer_role_from_type("") or "unknown"

        pair_type = f"{role1}_{role2}".lower()
        is_base_mid = 1.0 if pair_type == "base_mid" else 0.0
        is_base_bottom = 1.0 if pair_type == "base_bottom" else 0.0

        final_feature_vector = np.concatenate([
            style_diff,
            style_prod,
            [color_dist, is_base_mid, is_base_bottom],
        ])

        if final_feature_vector.shape != (771,):
            continue

        X_list.append(final_feature_vector)
        y_list.append(y_val)

    X = np.array(X_list, dtype=float)
    y = np.array(y_list, dtype=float)
    return X, y


def main():
    parser = argparse.ArgumentParser(description="Build X (769-dim) and y from fashion pair data.")
    parser.add_argument("--labels", type=Path, default=Path("ground_truth_labels_cleaned.csv"))
    parser.add_argument("--style", type=Path, default=Path("style_vectors.pkl"))
    parser.add_argument("--color", type=Path, default=Path("color_vectors.pkl"))
    parser.add_argument("--out", type=Path, default=Path("training_data.npz"))
    args = parser.parse_args()

    X, y = build_matrices(args.labels, args.style, args.color)

    # Kaydet
    np.savez_compressed(args.out, X=X, y=y)

    # Sadece istenen doğrulama çıktılarını yazdır
    print(f"Özellik (X) matrisi oluşturuldu. Şekil: {X.shape}")
    print(f"Hedef (y) vektörü oluşturuldu. Şekil: {y.shape}")


if __name__ == "__main__":
    main()