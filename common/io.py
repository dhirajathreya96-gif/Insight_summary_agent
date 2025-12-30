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
    "rating_count": ["rating_count", "review_count", "ratings", "reviews"],

    "speciality": ["speciality", "category", "product_category", "subcategory"],

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


def enrich_city_region_from_pincode(
    df: pd.DataFrame,
    pincode_col: str,
    map_df: pd.DataFrame,
    map_pin_col: str = "pincode",
    map_city_col: str = "city",
    map_region_col: str = "region",
    target_city_col: str = "city",
    target_region_col: str = "region",
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    df_out = df.copy()

    if target_city_col not in df_out.columns:
        df_out[target_city_col] = pd.NA
    if target_region_col not in df_out.columns:
        df_out[target_region_col] = pd.NA

    df_out[pincode_col] = _norm_pincode_series(df_out[pincode_col])

    pm = map_df.copy()
    pm.columns = [c.strip() for c in pm.columns]

    pm_cols_lower = {c.lower(): c for c in pm.columns}
    mp = pm_cols_lower.get(map_pin_col.lower())
    mc = pm_cols_lower.get(map_city_col.lower())
    mr = pm_cols_lower.get(map_region_col.lower())

    if mp is None:
        raise ValueError(f"Pincode map is missing '{map_pin_col}' column. Found: {list(pm.columns)}")

    pm[mp] = _norm_pincode_series(pm[mp])

    pin_to_city = {}
    pin_to_region = {}

    if mc is not None:
        pin_to_city = dict(zip(pm[mp], pm[mc].astype(str).str.strip()))
    if mr is not None:
        pin_to_region = dict(zip(pm[mp], pm[mr].astype(str).str.strip()))

    city_existing = df_out[target_city_col].astype(str).str.strip()
    city_missing = city_existing.eq("") | city_existing.str.lower().eq("nan")
    if mc is not None:
        df_out.loc[city_missing, target_city_col] = df_out.loc[city_missing, pincode_col].map(pin_to_city)

    region_existing = df_out[target_region_col].astype(str).str.strip()
    region_missing = region_existing.eq("") | region_existing.str.lower().eq("nan")
    if mr is not None:
        df_out.loc[region_missing, target_region_col] = df_out.loc[region_missing, pincode_col].map(pin_to_region)

    city_filled = df_out[target_city_col].astype(str).str.strip()
    region_filled = df_out[target_region_col].astype(str).str.strip()

    city_cov = (~city_filled.eq("") & ~city_filled.str.lower().eq("nan")).mean() * 100.0
    region_cov = (~region_filled.eq("") & ~region_filled.str.lower().eq("nan")).mean() * 100.0

    return df_out, {"city_coverage_pct": city_cov, "region_coverage_pct": region_cov}
