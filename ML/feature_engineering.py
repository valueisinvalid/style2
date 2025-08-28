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


def build_matrices(labels_csv: Path, style_pkl: Path, color_pkl: Path):
    # Yüklemeler
    df = pd.read_csv(labels_csv, dtype={"item_id_1": str, "item_id_2": str})
    with open(style_pkl, "rb") as f:
        style_vectors = pickle.load(f)
    with open(color_pkl, "rb") as f:
        color_vectors = pickle.load(f)

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

        # İlişkisel özellikler
        style_diff = s1 - s2
        style_prod = s1 * s2  # element-wise
        color_dist = np.linalg.norm(c1 - c2)  # skaler

        final_feature_vector = np.concatenate([style_diff, style_prod, [color_dist]])
        if final_feature_vector.shape != (769,):
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