#!/usr/bin/env python3
"""
Refactor: support multi-role garments via a list-based `roles` field.

Key changes:
- Replace single `role` with `roles: List[str]`.
- Provide deterministic mapping from item metadata to one or more roles.

Examples enforced by spec:
- T-shirt, Vest Top, Blouse   -> roles = ["base"]
- Jumper, Sweater, Hoodie,
  Button-up Shirt             -> roles = ["base", "mid-layer"]
- Cardigan                    -> roles = ["mid-layer"]
- Coat, Jacket                -> roles = ["outer"]
- Trousers, Jeans             -> roles = ["bottom"]

Notes:
- We extend bottoms to include Shorts/Skirts/Skorts for completeness; if you want
  to restrict strictly to Trousers/Jeans, prune in the mapping below.
- Backwards-compat: if some caller still uses `role`, we emit `item["role_legacy"]`
  for inspection but do not rely on it anywhere else.
"""
from __future__ import annotations
from typing import Dict, List
import re

# Canonical role labels we use in the pipeline
ROLE_BASE = "base"
ROLE_MID = "mid-layer"
ROLE_OUTER = "outer"
ROLE_BOTTOM = "bottom"


# --- Public API -------------------------------------------------------------

def assign_roles(item: Dict) -> List[str]:
    """
    Determine all roles a garment can play based on its coarse type/name.

    The function is conservative by default and only returns roles explicitly
    enabled for the garment category.

    Args:
        item: Dict with at least one of {"type", "name", "category"}.

    Returns:
        List of role strings. Also mutates item["roles"] in-place for convenience.
    """
    raw_type = _str(item.get("type"))
    raw_name = _str(item.get("name"))
    raw_cat  = _str(item.get("category"))
    haystack = f"{raw_type} {raw_name} {raw_cat}".lower()

    # Normalize common families (kept intentionally simple & robust)
    is_tshirt     = any(s in haystack for s in ["t-shirt", "tshirt", "tee", "tops & t-shirts", "low cut tops"]) and "long sleeve" not in haystack
    is_vest       = any(s in haystack for s in ["vest", "tank", "camisole"]) and "vestidos" not in haystack
    is_blouse     = "blouse" in haystack
    is_button_shirt = ("button" in haystack or "button-up" in haystack or "button down" in haystack) and ("shirt" in haystack or "overshirt" in haystack)
    is_shirt      = "shirt" in haystack and not is_blouse  # general shirt
    is_jumper     = any(s in haystack for s in ["jumper", "sweater", "sweatshirt", "hoodie"]) and "cardigan" not in haystack
    is_cardigan   = "cardigan" in haystack
    is_coat       = "coat" in haystack
    is_jacket     = any(s in haystack for s in ["jacket", "blazer", "denim jacket"]) and not is_coat

    is_trousers   = any(s in haystack for s in ["trousers", "pants", "smart trousers", "high waisted trousers"]) and "short" not in haystack
    is_jeans      = "jeans" in haystack or "denim" in haystack and "skirt" not in haystack and "jacket" not in haystack
    is_shorts     = "shorts" in haystack or "skort" in haystack
    is_skirt      = "skirt" in haystack and "skort" not in haystack

    roles: List[str] = []

    # --- Tops ---
    if is_tshirt or is_vest or is_blouse:
        roles.extend([ROLE_BASE])
    elif is_jumper:
        roles.extend([ROLE_BASE, ROLE_MID])
    elif is_cardigan:
        roles.extend([ROLE_MID])
    elif is_button_shirt or is_shirt:
        roles.extend([ROLE_BASE, ROLE_MID])

    # --- Outerwear ---
    if is_coat or is_jacket:
        roles = _uniq(roles + [ROLE_OUTER])

    # --- Bottoms ---
    if is_trousers or is_jeans or is_shorts or is_skirt:
        roles = _uniq(roles + [ROLE_BOTTOM])

    # Fallbacks: if we couldn't classify at all, stay empty (caller may decide)

    item["roles"] = _uniq(roles)

    # Optional: keep the previous single role for compatibility/inspection only
    if "role" in item and item.get("role") and not item.get("role_legacy"):
        item["role_legacy"] = item["role"]  # do not rely on this downstream
        # but intentionally DO NOT overwrite item["role"], we move away from it

    return item["roles"]


# --- Helpers ----------------------------------------------------------------

def _str(x) -> str:
    return x if isinstance(x, str) else ("" if x is None else str(x))


def _uniq(xs: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out