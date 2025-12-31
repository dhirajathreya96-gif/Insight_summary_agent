# common/io.py
from __future__ import annotations

import re
from typing import Tuple, List, Dict, Optional
import pandas as pd


_CANON: Dict[str, List[str]] = {
    "sku": ["sku_name", "sku", "product_name", "product", "item_name", "title"],
    "brand": ["brand_name", "brand", "manufacturer"],
    "pincode": ["pincode", "pin_code", "pin", "postal_code", "postcode", "zip", "zip_code", "delivery_pincode"],
    "city": ["city", "town", "district", "locality"],
    "region": ["region", "state", "zone", "area", "cluster", "market"],
    "platform": ["ecommerce", "platform", "channel", "marketplace", "site"],
    "stock": ["stock", "availability", "in_stock", "stock_status", "availability_status", "is_in_stock", "is_oos", "oos"],
    "serviced": ["service_status", "servicestatus", "serviceable", "is_serviced", "deliverable"],
    "price": ["price", "selling_price", "sale_price", "current_price", "sp"],
    "mrp": ["mrp", "list_price", "regular_price", "original_price", "max_retail_price"],
    "disc_pct": ["discount_percentage", "discount_pct", "discount_percent", "discount_percentatge", "off_pct"],
    "disc_amt": ["discount_amount", "discount_amt", "discount_value", "off_amount", "savings"],
    "avg_rating": ["avg_rating", "rating", "avg_customer_rating", "average_rating", "stars"],
    "coupon": ["coupon", "coupon_text"],
    "offer": ["offer", "promotion", "promo", "deal", "deal_text", "offer_text"],
    "super_offer": ["super_offer", "superoffer"],
    "super_saver": ["super_saver", "supersaver"],
    "date": ["crawled_date", "crawl_date", "date", "crawledatetime", "scrape_date", "capture_date"],
}


def _norm_key(name: str) -> str:
    s = str(name).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _make_unique(cols: List[str]) -> Tuple[List[str], List[str]]:
    seen: Dict[str, int] = {}
    out: List[str] = []
    warnings: List[str] = []
    for c in cols:
        base = c
        if base not in seen:
            seen[base] = 0
            out.append(base)
        else:
            seen[base] += 1
            new_c = f"{base}.{seen[base]}"
            warnings.append(f"Duplicate column '{base}' detected; renamed one occurrence to '{new_c}'.")
            out.append(new_c)
    return out, warnings


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    norm_lookup = {_norm_key(c): c for c in df.columns}
    for cand in candidates:
        nk = _norm_key(cand)
        if nk in norm_lookup:
            return norm_lookup[nk]
    return None


def _norm_pincode_series(s: pd.Series) -> pd.Series:
    out = s.astype(str).str.strip()
    out = out.str.replace(r"\.0$", "", regex=True)
    out = out.replace({"nan": pd.NA, "None": pd.NA, "": pd.NA})
    out = out.str.replace(r"\D+", "", regex=True)
    out = out.apply(lambda x: x.zfill(6) if isinstance(x, str) and x.isdigit() and len(x) < 6 else x)
    return out


def normalize_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    if df is None:
        return pd.DataFrame(), {}, ["Input dataframe is None."]

    warnings: List[str] = []
    mapping: Dict[str, str] = {}

    df_norm = df.copy()
    raw_cols = [str(c) for c in df_norm.columns]
    stripped = [c.strip() for c in raw_cols]
    unique_cols, dup_warn = _make_unique(stripped)
    warnings.extend(dup_warn)
    df_norm.columns = unique_cols

    for key, cands in _CANON.items():
        col = _find_col(df_norm, cands)
        if col is not None:
            mapping[key] = col

    if "platform" in mapping and "ecommerce" not in mapping:
        mapping["ecommerce"] = mapping["platform"]

    for req in ["sku", "pincode", "stock"]:
        if req not in mapping:
            raise ValueError(
                f"Required column missing: {req}. "
                f"Available columns: {list(df_norm.columns)}"
            )

    stock_col = mapping["stock"]
    df_norm[stock_col] = df_norm[stock_col].astype(str).str.strip()

    pin_col = mapping["pincode"]
    df_norm[pin_col] = _norm_pincode_series(df_norm[pin_col])

    plat_col = mapping.get("platform")
    if plat_col:
        df_norm[plat_col] = df_norm[plat_col].astype(str).str.strip()

    return df_norm, mapping, warnings
