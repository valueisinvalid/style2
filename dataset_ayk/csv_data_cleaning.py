#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV Veri Temizleme Script'i (Kopya ve Ters Kopya Giderme)
--------------------------------------------------------
Gereksinimler:
- Yalnızca pandas kütüphanesini kullanır.

Girdi:
- ground_truth_labels_graded.csv
  Zorunlu sütunlar: item_id_1, item_id_2, compatibility_score

Çıktı:
- ground_truth_labels_cleaned.csv (index=False)

Çalıştırıldığında konsola şu raporları yazdırır:
- "İşlem öncesi toplam satır sayısı: X"
- "İşlem sonrası benzersiz satır sayısı: Y"
- "Toplam Z adet kopya (veya ters kopya) satır temizlendi."
"""
import pandas as pd

def main():
    input_file = "ground_truth_labels_graded.csv"
    output_file = "ground_truth_labels_cleaned.csv"

    # Veriyi yükle
    df = pd.read_csv(input_file, dtype={"item_id_1": str, "item_id_2": str})
    print(f"İşlem öncesi toplam satır sayısı: {len(df)}")

    # Kanonik anahtar: sıra önemsiz (A,B) == (B,A)
    df["canonical_pair"] = df.apply(
        lambda row: frozenset([row["item_id_1"], row["item_id_2"]]), axis=1
    )

    # Kopyaları (ve ters kopyaları) temizle
    cleaned = df.drop_duplicates(subset="canonical_pair", keep="first").copy()

    # Geçici sütunu kaldır
    cleaned.drop(columns=["canonical_pair"], inplace=True)

    # Rapor
    before = len(df)
    after = len(cleaned)
    removed = before - after
    print(f"İşlem sonrası benzersiz satır sayısı: {after}")
    print(f"Toplam {removed} adet kopya (veya ters kopya) satır temizlendi.")

    # Kaydet
    cleaned.to_csv(output_file, index=False)

if __name__ == "__main__":
    main()