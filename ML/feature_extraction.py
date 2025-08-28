#!/usr/bin/env python3
"""
Text-Based Fashion Feature Extraction (Style & Color)
- Input : a JSON list of items with the exact schema (default: out.json)
- Output:
    color_vectors.pkl  -> {item_id: [L, a, b]}
    style_vectors.pkl  -> {item_id: [384-dim vector]}
    features_v1.npz    -> ids, lab (Nx3), sty (Nx384 normalized)

Run:
    python feature_extraction.py
    # or with flags:
    python feature_extraction.py --input out.json --colors color_vectors.pkl --styles style_vectors.pkl --bundle features_v1.npz
"""

# --- keep Transformers from touching TF/Flax (prevents keras/tf_keras import errors) ---
import os as _os
_os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
_os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

import os
import re
import json
import pickle
import argparse
from typing import Dict, List, Any, Tuple

import numpy as np
from matplotlib import colors as mcolors
from skimage.color import rgb2lab
from sentence_transformers import SentenceTransformer


# =========================
# Atomic pickle save helper
# =========================
def atomic_pickle_dump(obj, path: str) -> None:
    """
    Write pickle atomically to avoid truncated/corrupted files on crashes.
    """
    import tempfile
    dir_ = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=dir_)
    os.close(fd)
    try:
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)  # atomic on macOS/Linux
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except:
                pass


# ===========================================
# Improved color parsing for fashion colorways
# ===========================================
DEFAULT_LAB = np.array([50.0, 0.0, 0.0], dtype=float)  # fallback gray

# Modifiers we discard
MODIFIER_WORDS = {
    "light","dark","medium","bright","pale","deep","soft","warm","cool",
    "vivid","muted","pastel","rich","neon","vibrant","dusty","off","faded",
    "washed","washed-out","washedout"
}
# Patterns/prints we ignore for base color extraction
PATTERN_WORDS = {
    "striped","stripe","stripes","checked","check","chequered","plaid",
    "floral","flowers","paisley","argyle","zebra","leopard","animal",
    "print","pattern","patterned","pinstriped","melange","marl","heather",
    "spotted","dots","polka","colour","color","colourblock","colorblock"
}
# Collection/location tokens we ignore
LOCATION_OR_COLLECTION_WORDS = {
    "milano","atelier","amore","amour","paris","london","tokyo","le","marais",
    "postcards","palm","trees"
}
# Fashion aliases → base colors or hexes
EXTRA_COLOR_ALIASES = {
    "greige": "#B7A99A",
    "denim": "navy",
    "khaki green": "#8A9A5B",
    "marl": "grey",
    "melange": "grey",
    "heather": "grey",
    "leopard": "brown",
    "leopard print": "brown",
    "washed out": "",  # ignore finish adjectives
    # tricky grey-green family
    "grey-green": "#7E8F7A",
    "gray-green": "#7E8F7A",
    "light grey-green": "#93A28D",
    "dark grey-green": "#5F6E5C",
}
# Whitelist to help pick the base color out of multi-token names
BASE_COLOR_WORDS = {
    "white","black","grey","gray","beige","cream","brown","navy","blue","light blue","dark blue",
    "pink","light pink","dark pink","red","burgundy","maroon",
    "orange","yellow","gold","mustard",
    "green","khaki","olive",
    "purple","lilac","violet",
    "teal","turquoise","aqua",
    "tan","camel","ivory","off white","off-white","ecru",
    "greige"
}

def _clean_separators(s: str) -> str:
    s = s.replace("/", " ").replace("&", " ").replace(",", " ").replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _strip_noise_tokens(tokens: List[str]) -> List[str]:
    out = []
    for t in tokens:
        if t in MODIFIER_WORDS or t in PATTERN_WORDS or t in LOCATION_OR_COLLECTION_WORDS:
            continue
        out.append(t)
    return out

def _apply_aliases(phrase: str) -> str:
    # replace longer keys first, then shorter
    for k in sorted(EXTRA_COLOR_ALIASES.keys(), key=len, reverse=True):
        repl = EXTRA_COLOR_ALIASES[k]
        phrase = re.sub(rf"\b{k}\b", repl if repl is not None else "", phrase)
    phrase = re.sub(r"\s+", " ", phrase).strip()
    return phrase

def _candidate_list(raw: str) -> List[str]:
    """
    Build a descending-likelihood list of color candidates from a fashion color name.
    Examples:
      "White/Blue striped" -> ["white", "white blue", "blue"]
      "Beige marl"         -> ["beige", "beige marl", "marl", "grey"]
      "Dark Grey Melange"  -> ["grey", "dark grey", "grey melange", "melange"]
    """
    s = _clean_separators(raw.lower())
    # stop at patterns/collections
    base_tokens: List[str] = []
    for tok in s.split():
        if tok in PATTERN_WORDS or tok in LOCATION_OR_COLLECTION_WORDS:
            break
        base_tokens.append(tok)
    if not base_tokens:
        base_tokens = s.split()

    tokens = _strip_noise_tokens(base_tokens)
    phrase = " ".join(tokens).strip()
    phrase = _apply_aliases(phrase)

    cands: List[str] = []
    if phrase:
        toks = phrase.split()
        # Prefer first recognized base color left-to-right
        inserted = False
        for i in range(len(toks)):
            two = " ".join(toks[i:i+2])
            if two in BASE_COLOR_WORDS:
                cands.append(two)
                inserted = True
                break
            if toks[i] in BASE_COLOR_WORDS:
                cands.append(toks[i])
                inserted = True
                break
        if not inserted:
            cands.append(phrase)

        # Try last two / last one (helps with "navy blue")
        if len(toks) >= 2:
            cands.append(" ".join(toks[-2:]))
        if len(toks) >= 1:
            cands.append(toks[-1])

        # Map any leftover alias tokens individually (e.g., 'melange' -> 'grey')
        for t in toks:
            if t in EXTRA_COLOR_ALIASES and EXTRA_COLOR_ALIASES[t]:
                cands.append(EXTRA_COLOR_ALIASES[t])

    # Dedupe, preserve order
    seen = set(); unique = []
    for c in cands:
        c = c.strip()
        if c and c not in seen:
            unique.append(c); seen.add(c)
    return unique or ([phrase] if phrase else [])

def _to_lab_from_name(name: str) -> np.ndarray | None:
    """Try CSS4/hex, then XKCD namespace."""
    try:
        rgb = mcolors.to_rgb(name)
        arr = np.array([[rgb]], dtype=float)
        return rgb2lab(arr)[0, 0, :].astype(float)
    except ValueError:
        pass
    try:
        rgb = mcolors.to_rgb(f"xkcd:{name}")
        arr = np.array([[rgb]], dtype=float)
        return rgb2lab(arr)[0, 0, :].astype(float)
    except ValueError:
        return None

def color_name_to_lab(color_name: str, item_id: str) -> np.ndarray:
    if not color_name or not str(color_name).strip():
        print(f"[WARN] Empty color for item {item_id}. Using default gray.")
        return DEFAULT_LAB.copy()
    for cand in _candidate_list(color_name):
        lab = _to_lab_from_name(cand)
        if lab is not None:
            return lab
    print(f"[WARN] Could not parse color '{color_name}' for item {item_id}. Using default gray.")
    return DEFAULT_LAB.copy()


# ==================================
# Style text consolidation + encode
# ==================================
def build_style_document(item: Dict[str, Any]) -> str:
    """
    Create a single 'style document' string:
      name -> type -> Fit -> Neckline -> Concept -> description
    """
    name = item.get("name", "") or ""
    type_ = item.get("type", "") or ""
    kv = item.get("key_value_description", {}) or {}
    fit = kv.get("Fit:", "") or ""
    neckline = kv.get("Neckline:", "") or ""
    concept = kv.get("Concept:", "") or ""
    desc = item.get("description", "") or ""
    parts = [name, type_, fit, neckline, concept]
    header = " ".join(p.strip() for p in parts if p and p.strip())
    doc = (header + ". " if header else "") + desc.strip()
    return doc.strip()

def encode_style_documents(docs: List[str]) -> np.ndarray:
    """
    Batch-encode all documents using SentenceTransformer('all-MiniLM-L6-v2').
    Returns float32 array (N, 384).
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = model.encode(
        docs,
        batch_size=max(1, min(1024, len(docs))),  # safe batch size
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return emb.astype(np.float32)


# ==========
# Main flow
# ==========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="out.json", help="Path to input JSON list (default: out.json)")
    ap.add_argument("--colors", default="color_vectors.pkl", help="Output pickle for LAB colors")
    ap.add_argument("--styles", default="style_vectors.pkl", help="Output pickle for style embeddings")
    ap.add_argument("--bundle", default="features_v1.npz", help="Combined NPZ (ids, lab, sty-normalized)")
    args = ap.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file '{input_path}' not found.")

    with open(input_path, "r", encoding="utf-8") as f:
        items: List[Dict[str, Any]] = json.load(f)
    if not isinstance(items, list):
        raise ValueError("The input JSON must be a list of item dictionaries.")

    # ---- Part 1: Color vectors (CIELAB) ----
    color_vectors: Dict[str, List[float]] = {}
    for item in items:
        item_id = str(item.get("item_id", "")).strip()
        if not item_id:
            print("[WARN] Found an item without 'item_id'; skipping.")
            continue
        color_name = item.get("color", "")
        lab = color_name_to_lab(color_name, item_id)
        color_vectors[item_id] = [float(lab[0]), float(lab[1]), float(lab[2])]

    # ---- Part 2: Style vectors (sentence embeddings) ----
    id_order: List[str] = []
    docs: List[str] = []
    for item in items:
        item_id = str(item.get("item_id", "")).strip()
        if not item_id:
            continue
        id_order.append(item_id)
        doc = build_style_document(item)
        docs.append(doc if doc else " ")

    style_matrix = encode_style_documents(docs) if docs else np.zeros((0, 384), dtype=np.float32)

    style_vectors: Dict[str, List[float]] = {
        iid: style_matrix[idx].astype(float).tolist() for idx, iid in enumerate(id_order)
    }

    # ---- Save outputs (atomic) ----
    atomic_pickle_dump(color_vectors, args.colors)
    atomic_pickle_dump(style_vectors, args.styles)

    # ---- Build normalized bundle (ids, LAB, normalized style) ----
    ids = np.array(id_order)
    lab = np.array([color_vectors[i] for i in id_order], dtype=np.float32)
    sty = style_matrix.astype(np.float32)
    sty = sty / (np.linalg.norm(sty, axis=1, keepdims=True) + 1e-12)
    np.savez_compressed(args.bundle, ids=ids, lab=lab, sty=sty)
    print(f"Saved {args.bundle}", lab.shape, sty.shape)

    # ---- Verification ----
    print("Processing complete.")
    print(f"Total items processed: {len(style_vectors)}")
    sample_id = "0963662139"
    if sample_id in color_vectors:
        L, a, b = color_vectors[sample_id]
        print(f"Sample color vector for item '{sample_id}': [{L:.2f}, {a:.2f}, {b:.2f}]")
    else:
        print(f"Sample color vector for item '{sample_id}': not found in dataset.")
    if sample_id in style_vectors:
        vec = np.array(style_vectors[sample_id], dtype=float)
        print(f"Shape of style vector for item '{sample_id}': {tuple(vec.shape)}")
    else:
        print(f"Shape of style vector for item '{sample_id}': not found in dataset.")


if __name__ == "__main__":
    main()