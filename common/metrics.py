# common/metrics.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, List
import pandas as pd
import numpy as np


# -----------------------------
# Column normalization (light)
# -----------------------------
def normalize_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str], list]:
    """
    Map raw CSV columns to internal canonical keys.

    Returns:
      df (unchanged)
      colmap: canonical_key -> actual_column_name
      warnings: list[str]
    """
    warnings: list[str] = []
    if df is None or len(df) == 0:
        return df, {}, ["Empty dataframe received."]

    lower_to_actual: Dict[str, str] = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl not in lower_to_actual:
            lower_to_actual[cl] = c

    def pick(*candidates: str) -> Optional[str]:
        for cand in candidates:
            key = cand.strip().lower()
            if key in lower_to_actual:
                return lower_to_actual[key]
        return None

    colmap: Dict[str, str] = {}

    colmap["sku"] = pick("sku_name", "sku", "product_sku", "navision_code", "product_name") or ""
    colmap["brand"] = pick("brand_name", "brand") or ""
    colmap["pincode"] = pick("pincode", "pin_code", "postal_code", "zip") or ""
    colmap["city"] = pick("city", "region", "area") or ""
    colmap["service_status"] = pick("service_status", "servicestatus", "serviceability", "serviceable") or ""

    colmap["platform"] = pick("ecommerce", "platform", "channel", "marketplace", "source") or ""
    colmap["stock"] = pick("stock", "availability", "in_stock", "stock_status") or ""

    # price / promo
    colmap["price"] = pick("price", "selling_price", "sale_price", "sp", "current_price") or ""
    colmap["mrp"] = pick("mrp", "list_price", "msrp", "regular_price") or ""
    colmap["disc_pct"] = pick("discount_percentage", "discount_pct", "disc_pct", "discount_percent") or ""
    colmap["disc_amt"] = pick("discount_amount", "discount_amt", "disc_amt") or ""
    colmap["coupon"] = pick("coupon", "coupon_text") or ""
    colmap["offer"] = pick("offer", "offer_text", "offers", "promotion", "promo") or ""
    colmap["super_offer"] = pick("super_offer", "superoffer") or ""
    colmap["super_saver"] = pick("super_saver", "supersaver") or ""

    colmap["avg_rating"] = pick("avg_rating", "average_rating", "rating") or ""
    colmap["crawled_date"] = pick("crawled_date", "crawl_date", "date", "timestamp", "crawledatetime") or ""

    colmap = {k: v for k, v in colmap.items() if v}

    if "sku" not in colmap:
        warnings.append("SKU column not detected (expected SKU_Name/sku). Some metrics may be skipped.")
    if "pincode" not in colmap:
        warnings.append("Pincode column not detected (expected Pincode/pin_code). Stockout metrics may be skipped.")
    if "stock" not in colmap:
        warnings.append("Stock column not detected (expected Stock/availability). Stockout metrics may be skipped.")
    if "platform" not in colmap:
        warnings.append("Platform column not detected (expected ecommerce/platform). Some metrics may be skipped.")

    return df, colmap, warnings


# -----------------------------
# Helpers
# -----------------------------
def _get_series(df: pd.DataFrame, colname: str) -> pd.Series:
    obj = df.loc[:, colname]
    if isinstance(obj, pd.DataFrame):
        return obj.iloc[:, 0]
    return obj


def _standardize_pincode(s: pd.Series) -> pd.Series:
    s2 = s.astype(str).str.strip()
    s2 = s2.str.replace(r"\.0$", "", regex=True)
    s2 = s2.apply(lambda x: x.zfill(6) if x.isdigit() and len(x) <= 6 else x)
    return s2


def _filter_serviced(df: pd.DataFrame, colmap: Dict[str, str], require_serviced: bool) -> pd.DataFrame:
    if not require_serviced:
        return df
    if "service_status" not in colmap:
        return df

    ss = _get_series(df, colmap["service_status"]).astype(str).str.strip().str.lower()
    keep = ss.isin({"serviced", "serviceable", "yes", "true", "1"})
    return df.loc[keep].copy()


def _infer_period(df: pd.DataFrame, colmap: Dict[str, str]) -> Optional[str]:
    if "crawled_date" not in colmap:
        return None
    dt = pd.to_datetime(_get_series(df, colmap["crawled_date"]), errors="coerce")
    if dt.notna().any():
        return dt.max().date().isoformat()
    return None


def _is_oos(stock_series: pd.Series) -> pd.Series:
    """
    Returns True if stock status represents Out Of Stock.
    Handles real-world variations safely.
    """
    s = stock_series.astype(str).str.strip().str.lower()

    # normalize spacing
    s = s.str.replace(r"\s+", " ", regex=True)

    OOS_VALUES = {
        "out of stock",
        "out-of-stock",
        "out_of_stock",
        "oos",
        "sold out",
        "not available",
        "0",
        "false"
    }

    return s.isin(OOS_VALUES)


def _build_pin_to_city(map_df: pd.DataFrame) -> Tuple[Dict[str, str], Optional[str], Optional[str]]:
    if map_df is None or len(map_df) == 0:
        return {}, None, None

    lower_to_actual = {str(c).strip().lower(): c for c in map_df.columns}

    def pick(*names: str) -> Optional[str]:
        for n in names:
            k = n.strip().lower()
            if k in lower_to_actual:
                return lower_to_actual[k]
        return None

    pin_col = pick("pincode", "pin", "pin_code", "postal_code", "zip", "postcode")
    city_col = pick("city", "region", "area", "town", "district")

    if not pin_col or not city_col:
        return {}, pin_col, city_col

    pin = map_df[pin_col].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    pin = pin.apply(lambda x: x.zfill(6) if x.isdigit() and len(x) <= 6 else x)
    city = map_df[city_col].astype(str).str.strip()

    good = (
        (pin != "")
        & (city != "")
        & (~pin.str.lower().isin({"nan", "none"}))
        & (~city.str.lower().isin({"nan", "none"}))
    )
    pin_to_city = dict(zip(pin[good], city[good]))
    return pin_to_city, pin_col, city_col


def _coerce_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _promo_flag(df: pd.DataFrame, colmap: Dict[str, str]) -> pd.Series:
    # A robust boolean: promo if any discount/coupon/offer/super flags present.
    promo = pd.Series([False] * len(df), index=df.index)

    if "disc_pct" in colmap:
        promo = promo | (_coerce_num(_get_series(df, colmap["disc_pct"])).fillna(0) > 0)
    if "disc_amt" in colmap:
        promo = promo | (_coerce_num(_get_series(df, colmap["disc_amt"])).fillna(0) > 0)

    if "coupon" in colmap:
        c = _get_series(df, colmap["coupon"]).astype(str).fillna("").str.strip()
        promo = promo | (c != "") & (~c.str.lower().isin({"nan", "none"}))

    if "offer" in colmap:
        o = _get_series(df, colmap["offer"]).astype(str).fillna("").str.strip()
        promo = promo | (o != "") & (~o.str.lower().isin({"nan", "none"}))

    for key in ["super_offer", "super_saver"]:
        if key in colmap:
            v = _get_series(df, colmap[key]).astype(str).fillna("").str.strip().str.lower()
            promo = promo | v.isin({"true", "1", "yes", "y"})
    return promo


# -----------------------------
# STOCKOUT METRICS
# -----------------------------
def compute_stockout_metrics(
    df: pd.DataFrame,
    require_serviced: bool = True,
    pincode_map_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    df, colmap, warnings = normalize_columns(df)
    df = _filter_serviced(df, colmap, require_serviced)

    out: Dict[str, Any] = {"warnings": warnings, "colmap": colmap}
    out["period"] = _infer_period(df, colmap)

    # always-present defaults
    out["sku_stockouts"] = pd.DataFrame(columns=["sku", "oos_pincodes", "total_pincodes", "oos_pct"])
    out["sku_platform_stockouts"] = pd.DataFrame(columns=["sku", "platform", "oos_pincodes", "total_pincodes", "oos_pct"])

    out["top_city_oos"] = None
    out["top_city_oos_skus"] = None

    # geo drilldown tables (SKU-based)
    out["pincode_oos_sku_table"] = pd.DataFrame(columns=["pincode", "city", "oos_skus", "total_skus", "oos_pct"])
    out["city_oos_sku_table"] = pd.DataFrame(columns=["city", "oos_skus", "total_skus", "oos_pct"])

    # platform drilldown tables
    out["pincode_platform_oos_sku_table"] = pd.DataFrame(
        columns=["pincode", "city", "platform", "oos_skus", "total_skus", "oos_pct"]
    )
    out["city_platform_oos_sku_table"] = pd.DataFrame(
        columns=["city", "platform", "oos_skus", "total_skus", "oos_pct"]
    )

    out["pincode_map_coverage_pct"] = None

    if df is None or len(df) == 0:
        out["warnings"].append("No rows available after filters (brand/serviced).")
        return out

    if not {"sku", "pincode", "stock"}.issubset(colmap.keys()):
        out["warnings"].append("Missing required columns for stockout metrics (sku/pincode/stock).")
        return out

    sku_col = colmap["sku"]
    pin_col = colmap["pincode"]
    stock_col = colmap["stock"]

    sku = _get_series(df, sku_col).astype(str).str.strip()
    pincode = _standardize_pincode(_get_series(df, pin_col))
    oos_mask = _is_oos(_get_series(df, stock_col))

    platform = None
    if "platform" in colmap:
        platform = _get_series(df, colmap["platform"]).astype(str).str.strip()
        platform = platform.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

    # City derivation
    city_series = None
    if "city" in colmap:
        city_series = _get_series(df, colmap["city"]).astype(str).str.strip()

    pin_to_city: Dict[str, str] = {}
    if pincode_map_df is not None and len(pincode_map_df):
        pin_to_city, _, _ = _build_pin_to_city(pincode_map_df)
        if pin_to_city:
            if city_series is None:
                city_series = pd.Series([pd.NA] * len(df), index=df.index, dtype="object")

            c0 = city_series.astype(str).str.strip()
            missing = c0.isna() | (c0 == "") | (c0.str.lower().isin({"nan", "none"}))
            filled = pincode.map(pin_to_city)
            city_series = c0.where(~missing, filled)

            cov = city_series.astype(str).str.strip()
            covered = (~cov.isna()) & (cov != "") & (~cov.str.lower().isin({"nan", "none"}))
            out["pincode_map_coverage_pct"] = float(covered.mean() * 100.0)
        else:
            out["warnings"].append("Pincode map provided but could not infer pincode/city columns.")

    base = pd.DataFrame({"sku": sku, "pincode": pincode, "is_oos": oos_mask})
    if platform is not None:
        base["platform"] = platform
    if city_series is not None:
        c = city_series.astype(str).str.strip()
        c = c.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "none": pd.NA})
        base["city"] = c

    base = base.replace({"": pd.NA}).dropna(subset=["sku", "pincode"])
    if len(base) == 0:
        out["warnings"].append("No usable rows after cleaning sku/pincode.")
        return out

    # -------------------------
    # SKU-level stockouts (by pincodes)
    # -------------------------
    total_pins = base.groupby("sku")["pincode"].nunique().rename("total_pincodes")
    oos_pins = base.loc[base["is_oos"]].groupby("sku")["pincode"].nunique().rename("oos_pincodes")

    t = pd.concat([oos_pins, total_pins], axis=1).fillna(0)
    t["oos_pincodes"] = t["oos_pincodes"].astype(int)
    t["total_pincodes"] = t["total_pincodes"].astype(int)
    t["oos_pct"] = t.apply(
        lambda r: (100.0 * r["oos_pincodes"] / r["total_pincodes"]) if r["total_pincodes"] > 0 else 0.0,
        axis=1,
    )
    out["sku_stockouts"] = t.reset_index().sort_values(["oos_pincodes", "oos_pct"], ascending=False)

    # -------------------------
    # SKU x PLATFORM stockouts (by pincodes)
    # -------------------------
    if "platform" in base.columns:
        bp = base.dropna(subset=["platform"]).copy()
        if len(bp):
            total_pp = bp.groupby(["sku", "platform"])["pincode"].nunique().rename("total_pincodes")
            oos_pp = bp.loc[bp["is_oos"]].groupby(["sku", "platform"])["pincode"].nunique().rename("oos_pincodes")

            tp = pd.concat([oos_pp, total_pp], axis=1).fillna(0)
            tp["oos_pincodes"] = tp["oos_pincodes"].astype(int)
            tp["total_pincodes"] = tp["total_pincodes"].astype(int)
            tp["oos_pct"] = tp.apply(
                lambda r: (100.0 * r["oos_pincodes"] / r["total_pincodes"]) if r["total_pincodes"] > 0 else 0.0,
                axis=1,
            )
            out["sku_platform_stockouts"] = tp.reset_index().sort_values(
                ["sku", "oos_pincodes", "oos_pct"],
                ascending=[True, False, False],
            )

    # -------------------------
    # Legacy top city outputs
    # -------------------------
    if "city" in base.columns and base["city"].notna().any():
        city_base = base.dropna(subset=["city", "pincode", "sku"]).copy()
        oos_city_pins = (
            city_base.loc[city_base["is_oos"]]
            .groupby("city")["pincode"]
            .nunique()
            .rename("oos_pincodes")
            .reset_index()
            .sort_values(["oos_pincodes", "city"], ascending=[False, True])
        )
        if len(oos_city_pins):
            row = oos_city_pins.iloc[0]
            out["top_city_oos"] = {"city": str(row["city"]), "oos_pincodes": int(row["oos_pincodes"])}

        oos_city_skus = (
            city_base.loc[city_base["is_oos"]]
            .groupby("city")["sku"]
            .nunique()
            .rename("oos_skus")
            .reset_index()
            .sort_values(["oos_skus", "city"], ascending=[False, True])
        )
        if len(oos_city_skus):
            row = oos_city_skus.iloc[0]
            out["top_city_oos_skus"] = {"city": str(row["city"]), "oos_skus": int(row["oos_skus"])}

    # -------------------------
    # Pincode ranking (OOS SKUs + OOS%)
    # -------------------------
    pin_total_skus = base.groupby("pincode")["sku"].nunique().rename("total_skus")
    pin_oos_skus = base.loc[base["is_oos"]].groupby("pincode")["sku"].nunique().rename("oos_skus")

    pin_tbl = pd.concat([pin_oos_skus, pin_total_skus], axis=1).fillna(0).reset_index()
    pin_tbl["oos_skus"] = pin_tbl["oos_skus"].astype(int)
    pin_tbl["total_skus"] = pin_tbl["total_skus"].astype(int)
    pin_tbl["oos_pct"] = pin_tbl.apply(
        lambda r: (100.0 * r["oos_skus"] / r["total_skus"]) if r["total_skus"] > 0 else 0.0,
        axis=1,
    )

    if pin_to_city:
        pin_tbl["city"] = pin_tbl["pincode"].astype(str).map(pin_to_city).fillna("")
    elif "city" in base.columns:
        tmp = base.dropna(subset=["city"]).copy()
        if len(tmp):
            mode_city = tmp.groupby("pincode")["city"].agg(lambda s: s.value_counts().index[0] if len(s.dropna()) else "")
            pin_tbl["city"] = pin_tbl["pincode"].map(mode_city).fillna("")
        else:
            pin_tbl["city"] = ""
    else:
        pin_tbl["city"] = ""

    pin_tbl = pin_tbl.sort_values(
        ["oos_skus", "oos_pct", "total_skus", "pincode"],
        ascending=[False, False, False, True],
    )
    out["pincode_oos_sku_table"] = pin_tbl[["pincode", "city", "oos_skus", "total_skus", "oos_pct"]]

    # -------------------------
    # City ranking (OOS SKUs + OOS%)
    # -------------------------
    if "city" in base.columns and base["city"].notna().any():
        cb = base.dropna(subset=["city"]).copy()

        city_total_skus = cb.groupby("city")["sku"].nunique().rename("total_skus")
        city_oos_skus = cb.loc[cb["is_oos"]].groupby("city")["sku"].nunique().rename("oos_skus")

        city_tbl = pd.concat([city_oos_skus, city_total_skus], axis=1).fillna(0).reset_index()
        city_tbl["oos_skus"] = city_tbl["oos_skus"].astype(int)
        city_tbl["total_skus"] = city_tbl["total_skus"].astype(int)
        city_tbl["oos_pct"] = city_tbl.apply(
            lambda r: (100.0 * r["oos_skus"] / r["total_skus"]) if r["total_skus"] > 0 else 0.0,
            axis=1,
        )

        city_tbl = city_tbl.sort_values(
            ["oos_skus", "oos_pct", "total_skus", "city"],
            ascending=[False, False, False, True],
        )
        out["city_oos_sku_table"] = city_tbl[["city", "oos_skus", "total_skus", "oos_pct"]]

    # -------------------------
    # Platform drilldown (Pincode × Platform, City × Platform)
    # -------------------------
    if "platform" in base.columns:
        bp = base.dropna(subset=["platform"]).copy()
        if len(bp):
            # pincode x platform
            pp_total = bp.groupby(["pincode", "platform"])["sku"].nunique().rename("total_skus")
            pp_oos = bp.loc[bp["is_oos"]].groupby(["pincode", "platform"])["sku"].nunique().rename("oos_skus")

            pp_tbl = pd.concat([pp_oos, pp_total], axis=1).fillna(0).reset_index()
            pp_tbl["oos_skus"] = pp_tbl["oos_skus"].astype(int)
            pp_tbl["total_skus"] = pp_tbl["total_skus"].astype(int)
            pp_tbl["oos_pct"] = pp_tbl.apply(
                lambda r: (100.0 * r["oos_skus"] / r["total_skus"]) if r["total_skus"] > 0 else 0.0,
                axis=1,
            )

            # attach city
            if pin_to_city:
                pp_tbl["city"] = pp_tbl["pincode"].astype(str).map(pin_to_city).fillna("")
            elif "city" in bp.columns:
                tmp = bp.dropna(subset=["city"]).copy()
                if len(tmp):
                    mode_city = tmp.groupby("pincode")["city"].agg(lambda s: s.value_counts().index[0] if len(s.dropna()) else "")
                    pp_tbl["city"] = pp_tbl["pincode"].map(mode_city).fillna("")
                else:
                    pp_tbl["city"] = ""
            else:
                pp_tbl["city"] = ""

            pp_tbl = pp_tbl.sort_values(
                ["pincode", "oos_skus", "oos_pct", "total_skus", "platform"],
                ascending=[True, False, False, False, True],
            )
            out["pincode_platform_oos_sku_table"] = pp_tbl[["pincode", "city", "platform", "oos_skus", "total_skus", "oos_pct"]]

            # city x platform
            if "city" in bp.columns and bp["city"].notna().any():
                bpc = bp.dropna(subset=["city"]).copy()

                cp_total = bpc.groupby(["city", "platform"])["sku"].nunique().rename("total_skus")
                cp_oos = bpc.loc[bpc["is_oos"]].groupby(["city", "platform"])["sku"].nunique().rename("oos_skus")

                cp_tbl = pd.concat([cp_oos, cp_total], axis=1).fillna(0).reset_index()
                cp_tbl["oos_skus"] = cp_tbl["oos_skus"].astype(int)
                cp_tbl["total_skus"] = cp_tbl["total_skus"].astype(int)
                cp_tbl["oos_pct"] = cp_tbl.apply(
                    lambda r: (100.0 * r["oos_skus"] / r["total_skus"]) if r["total_skus"] > 0 else 0.0,
                    axis=1,
                )

                cp_tbl = cp_tbl.sort_values(
                    ["city", "oos_skus", "oos_pct", "total_skus", "platform"],
                    ascending=[True, False, False, False, True],
                )
                out["city_platform_oos_sku_table"] = cp_tbl[["city", "platform", "oos_skus", "total_skus", "oos_pct"]]

    return out


# -----------------------------
# PRICE / PROMO / REVIEW METRICS
# -----------------------------
def compute_price_promo_review_metrics(df: pd.DataFrame, require_serviced: bool = True) -> Dict[str, Any]:
    df, colmap, warnings = normalize_columns(df)
    df = _filter_serviced(df, colmap, require_serviced)

    out: Dict[str, Any] = {"warnings": warnings, "colmap": colmap}

    if df is None or len(df) == 0:
        out["warnings"].append("No rows available after filters (brand/serviced).")
        out["sku_price_promo"] = pd.DataFrame(columns=["sku", "avg_price", "avg_mrp", "avg_disc_pct", "avg_disc_amt"])
        out["promo_signature"] = pd.DataFrame(columns=["sku", "promo_sig"])
        return out

    if "sku" not in colmap:
        out["warnings"].append("SKU column not detected; price/promo metrics skipped.")
        out["promo_signature"] = pd.DataFrame(columns=["sku", "promo_sig"])
        return out

    sku_col = colmap["sku"]
    sku = _get_series(df, sku_col).astype(str).str.strip()

    def _num(colkey: str) -> Optional[pd.Series]:
        if colkey not in colmap:
            return None
        return pd.to_numeric(_get_series(df, colmap[colkey]), errors="coerce")

    price = _num("price")
    mrp = _num("mrp")
    disc_pct = _num("disc_pct")
    disc_amt = _num("disc_amt")

    agg = {}
    if price is not None:
        agg["avg_price"] = price.groupby(sku).mean()
    if mrp is not None:
        agg["avg_mrp"] = mrp.groupby(sku).mean()
    if disc_pct is not None:
        agg["avg_disc_pct"] = disc_pct.groupby(sku).mean()
    if disc_amt is not None:
        agg["avg_disc_amt"] = disc_amt.groupby(sku).mean()

    if agg:
        t = pd.DataFrame(agg)
        t.index.name = "sku"
        out["sku_price_promo"] = t.reset_index()
    else:
        out["sku_price_promo"] = pd.DataFrame(columns=["sku", "avg_price", "avg_mrp", "avg_disc_pct", "avg_disc_amt"])

    promo_keys = ["coupon", "offer", "super_offer", "super_saver"]
    promo_cols = [colmap[k] for k in promo_keys if k in colmap]

    if promo_cols:
        tmp_parts = []
        tmp_names = []
        for c in promo_cols:
            s = _get_series(df, c).astype(str).fillna("").str.strip()
            tmp_parts.append(s)
            tmp_names.append(c)

        tmp = pd.concat(tmp_parts, axis=1)
        tmp.columns = tmp_names
        promo_sig = tmp.astype(str).fillna("").agg("|".join, axis=1)

        out["promo_signature"] = (
            pd.DataFrame({"sku": sku.astype(str), "promo_sig": promo_sig})
            .groupby("sku", as_index=False)["promo_sig"]
            .agg(lambda s: s.dropna().iloc[0] if len(s.dropna()) else "")
        )
    else:
        out.setdefault("warnings", []).append(
            "Promotion columns not found (coupon/offer/super_offer/super_saver); promo signature skipped."
        )
        out["promo_signature"] = pd.DataFrame(columns=["sku", "promo_sig"])

    if "avg_rating" in colmap:
        ar = pd.to_numeric(_get_series(df, colmap["avg_rating"]), errors="coerce")
        out["avg_rating_by_sku"] = ar.groupby(sku).mean().reset_index(name="avg_rating")
        out["overall_avg_rating"] = float(ar.mean()) if ar.notna().any() else None

    return out


# -----------------------------
# PERIOD COMPARISON
# -----------------------------
def compare_periods(
    current: Dict[str, Any],
    previous: Dict[str, Any],
    top_n: int = 5,
    price_threshold_pct: float = 1.0,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"comparisons": {}}

    cur = current.get("sku_stockouts")
    prev = previous.get("sku_stockouts")
    if not isinstance(cur, pd.DataFrame) or not isinstance(prev, pd.DataFrame):
        return out
    if len(cur) == 0 or len(prev) == 0:
        return out

    m = cur.merge(prev[["sku", "oos_pct"]], on="sku", how="left", suffixes=("", "_prev"))
    m["delta_oos_pct"] = m["oos_pct"] - m["oos_pct_prev"]
    out["comparisons"]["stockouts_by_sku"] = m.sort_values("oos_pincodes", ascending=False)

    def _overall(df_stock: pd.DataFrame) -> float:
        denom = df_stock["total_pincodes"].sum()
        num = df_stock["oos_pincodes"].sum()
        return (100.0 * num / denom) if denom > 0 else 0.0

    overall_cur = _overall(cur)
    overall_prev = _overall(prev)
    out["comparisons"]["overall_stockout_delta_pp"] = overall_cur - overall_prev

    crit = m.copy()
    crit["is_critical"] = (crit["oos_pct"] >= 90.0) | (crit["delta_oos_pct"] >= 5.0)
    out["comparisons"]["critical_skus"] = crit.loc[crit["is_critical"]].sort_values(
        ["oos_pct", "delta_oos_pct"], ascending=False
    ).head(max(top_n, 10))

    return out


# -----------------------------
# COMPETITOR PRICE / PROMO CHANGES (ADD-ON)
# -----------------------------
def compute_competitor_price_promo_changes(
    current_df: pd.DataFrame,
    previous_df: Optional[pd.DataFrame],
    require_serviced: bool,
    price_change_threshold_pct: float,
    promotion_change_threshold_pct: float,
) -> Dict[str, Any]:
    """
    Returns:
      - changed_rows: DataFrame of competitor SKU x platform with change flags
      - summary: counts & averages (used when list is large)
    """
    cur, ccol, cwarn = normalize_columns(current_df)
    cur = _filter_serviced(cur, ccol, require_serviced)

    out: Dict[str, Any] = {"warnings": cwarn}

    if cur is None or len(cur) == 0:
        out["changed_rows"] = pd.DataFrame()
        out["summary"] = {}
        out["warnings"].append("Competitor price/promo: no rows available in current after filters.")
        return out

    if not {"sku", "platform"}.issubset(set(ccol.keys())):
        out["changed_rows"] = pd.DataFrame()
        out["summary"] = {}
        out["warnings"].append("Competitor price/promo: missing sku/platform columns.")
        return out

    # current aggregates
    cur_tbl = pd.DataFrame({
        "sku": _get_series(cur, ccol["sku"]).astype(str).str.strip(),
        "platform": _get_series(cur, ccol["platform"]).astype(str).str.strip(),
    })

    if "price" in ccol:
        cur_tbl["price"] = _coerce_num(_get_series(cur, ccol["price"]))
    else:
        cur_tbl["price"] = np.nan

    if "disc_pct" in ccol:
        cur_tbl["disc_pct"] = _coerce_num(_get_series(cur, ccol["disc_pct"]))
    else:
        cur_tbl["disc_pct"] = 0.0

    cur_tbl["promo_flag"] = _promo_flag(cur, ccol)

    cur_agg = (
        cur_tbl.groupby(["platform", "sku"], as_index=False)
        .agg(
            price=("price", "median"),
            disc_pct=("disc_pct", "max"),
            promo_flag=("promo_flag", "max"),
        )
    )

    if previous_df is None or len(previous_df) == 0:
        # If no previous, treat "new promo" as promo_flag True and price_change NA
        cur_agg["prev_price"] = np.nan
        cur_agg["prev_disc_pct"] = 0.0
        cur_agg["prev_promo_flag"] = False
    else:
        prev, pcol, pwarn = normalize_columns(previous_df)
        prev = _filter_serviced(prev, pcol, require_serviced)

        out["warnings"] += pwarn

        if prev is None or len(prev) == 0 or not {"sku", "platform"}.issubset(set(pcol.keys())):
            cur_agg["prev_price"] = np.nan
            cur_agg["prev_disc_pct"] = 0.0
            cur_agg["prev_promo_flag"] = False
        else:
            prev_tbl = pd.DataFrame({
                "sku": _get_series(prev, pcol["sku"]).astype(str).str.strip(),
                "platform": _get_series(prev, pcol["platform"]).astype(str).str.strip(),
            })

            if "price" in pcol:
                prev_tbl["price"] = _coerce_num(_get_series(prev, pcol["price"]))
            else:
                prev_tbl["price"] = np.nan

            if "disc_pct" in pcol:
                prev_tbl["disc_pct"] = _coerce_num(_get_series(prev, pcol["disc_pct"]))
            else:
                prev_tbl["disc_pct"] = 0.0

            prev_tbl["promo_flag"] = _promo_flag(prev, pcol)

            prev_agg = (
                prev_tbl.groupby(["platform", "sku"], as_index=False)
                .agg(
                    prev_price=("price", "median"),
                    prev_disc_pct=("disc_pct", "max"),
                    prev_promo_flag=("promo_flag", "max"),
                )
            )

            cur_agg = cur_agg.merge(prev_agg, on=["platform", "sku"], how="left")
            cur_agg["prev_price"] = cur_agg["prev_price"].astype(float)
            cur_agg["prev_disc_pct"] = cur_agg["prev_disc_pct"].fillna(0.0).astype(float)
            cur_agg["prev_promo_flag"] = cur_agg["prev_promo_flag"].fillna(False).astype(bool)

    # deltas
    cur_agg["price_change_pct"] = np.where(
        (cur_agg["prev_price"].notna()) & (cur_agg["prev_price"] > 0) & (cur_agg["price"].notna()),
        (cur_agg["price"] - cur_agg["prev_price"]) / cur_agg["prev_price"] * 100.0,
        np.nan,
    )
    cur_agg["disc_change_pp"] = cur_agg["disc_pct"].fillna(0.0) - cur_agg["prev_disc_pct"].fillna(0.0)

    # flags
    cur_agg["is_price_change"] = cur_agg["price_change_pct"].abs() >= float(price_change_threshold_pct)
    cur_agg["is_new_promo"] = (cur_agg["promo_flag"] == True) & (cur_agg["prev_promo_flag"] == False)
    cur_agg["is_promo_increase"] = cur_agg["disc_change_pp"] >= float(promotion_change_threshold_pct)

    changed = cur_agg[cur_agg["is_price_change"] | cur_agg["is_new_promo"] | cur_agg["is_promo_increase"]].copy()

    # summary stats (for big lists)
    summary: Dict[str, Any] = {}
    if len(changed):
        price_increases = changed[changed["price_change_pct"].notna() & (changed["price_change_pct"] > 0)]
        promo_skus = changed[changed["promo_flag"] == True]

        summary = {
            "impacted_sku_platform_rows": int(len(changed)),
            "unique_skus": int(changed["sku"].nunique()),
            "unique_platforms": int(changed["platform"].nunique()),
            "avg_price_increase_pct": float(price_increases["price_change_pct"].mean()) if len(price_increases) else None,
            "avg_discount_pct": float(promo_skus["disc_pct"].mean()) if len(promo_skus) else None,
        }

    # order for readability
    changed = changed.sort_values(
        ["platform", "is_price_change", "price_change_pct", "disc_change_pp", "sku"],
        ascending=[True, False, False, False, True],
    )

    out["changed_rows"] = changed
    out["summary"] = summary
    return out

# -----------------------------
# HISTORY BUNDLE (LAST N INSTANCES)
# -----------------------------
def _period_from_df(df: pd.DataFrame, colmap: Dict[str, str]) -> str:
    p = _infer_period(df, colmap)
    if p:
        return str(p)
    # fallback: stable label if date not present
    return "unknown_period"


def build_history_bundle(
    history_dfs: Optional[List[pd.DataFrame]] = None,
    *,
    previous_dfs: Optional[List[pd.DataFrame]] = None,          # alias
    current_df: Optional[pd.DataFrame] = None,                  # ✅ included
    pincode_map_df: Optional[pd.DataFrame] = None,              # ✅ included (accepted even if unused)
    require_serviced: bool = True,
    n_last: int = 5,
    **kwargs: Any,                                              # ✅ future-proof: ignore unexpected kwargs
) -> Dict[str, Any]:
    """
    Builds compact history features used by the agent/render layer:
      - sku_oos_last_n: top SKUs that are OOS in the last n instances (count + pct)
      - oos_timeline: overall OOS% by period (for chart)
      - sku_sparklines: per SKU OOS% series (for sparkline rendering)

    Accepts flexible inputs:
      - history_dfs: explicit ordered list of dfs
      - previous_dfs + current_df: preferred wiring from agent/backend
      - pincode_map_df: accepted for API compatibility (unused here)
    """

    # ---------- Resolve the input frames list ----------
    if history_dfs is None:
        history_dfs = []

        if previous_dfs:
            history_dfs.extend(list(previous_dfs))

        if current_df is not None:
            history_dfs.append(current_df)

    out: Dict[str, Any] = {
        "warnings": [],
        "n_last": int(n_last),
        "available_instances": int(len(history_dfs or [])),
        "sku_oos_last_n": pd.DataFrame(columns=["sku", "oos_instances", "n_instances", "oos_instance_pct"]),
        "oos_timeline": pd.DataFrame(columns=["period", "overall_oos_pct"]),
        "sku_sparklines": {},  # sku -> list[float] aligned to oos_timeline periods
        "periods": [],
    }

    if not history_dfs:
        out["warnings"].append("No historical files provided; history bundle skipped.")
        return out

    # consider only last n_last inputs (caller should pass ordered, but we are defensive)
    frames = history_dfs[-int(n_last):] if len(history_dfs) > int(n_last) else history_dfs

    per_instance_rows: List[pd.DataFrame] = []
    overall_series: List[Dict[str, Any]] = []
    sku_period_rows: List[pd.DataFrame] = []

    for raw in frames:
        df, colmap, warn = normalize_columns(raw)
        df = _filter_serviced(df, colmap, require_serviced)
        out["warnings"] += list(warn or [])

        if df is None or len(df) == 0:
            continue
        if not {"sku", "pincode", "stock"}.issubset(set(colmap.keys())):
            out["warnings"].append("History instance missing required columns (sku/pincode/stock); skipped.")
            continue

        sku = _get_series(df, colmap["sku"]).astype(str).str.strip()
        pin = _standardize_pincode(_get_series(df, colmap["pincode"]))
        oos = _is_oos(_get_series(df, colmap["stock"]))
        period = _period_from_df(df, colmap)

        base = pd.DataFrame({"sku": sku, "pincode": pin, "is_oos": oos})
        base = base.replace({"": pd.NA}).dropna(subset=["sku", "pincode"])
        if base.empty:
            continue

        # overall OOS% for the instance (unique pincodes)
        total = base["pincode"].nunique()
        oos_pins = base.loc[base["is_oos"], "pincode"].nunique()
        overall_pct = (100.0 * oos_pins / total) if total else 0.0
        overall_series.append({"period": period, "overall_oos_pct": float(overall_pct)})

        # per-SKU OOS% for sparkline (OOS pincodes / total pincodes per SKU)
        total_p = base.groupby("sku")["pincode"].nunique().rename("total_pincodes")
        oos_p = base.loc[base["is_oos"]].groupby("sku")["pincode"].nunique().rename("oos_pincodes")
        t = pd.concat([oos_p, total_p], axis=1).fillna(0)
        t["oos_pincodes"] = t["oos_pincodes"].astype(int)
        t["total_pincodes"] = t["total_pincodes"].astype(int)
        t["oos_pct"] = t.apply(
            lambda r: (100.0 * r["oos_pincodes"] / r["total_pincodes"]) if r["total_pincodes"] > 0 else 0.0,
            axis=1,
        )
        t = t.reset_index()
        t["period"] = period
        sku_period_rows.append(t[["period", "sku", "oos_pct"]])

        # for last-N OOS instances: mark sku as OOS in this instance if any pin OOS
        sku_any_oos = base.groupby("sku")["is_oos"].any().reset_index()
        sku_any_oos["period"] = period
        per_instance_rows.append(sku_any_oos)

    if not overall_series:
        out["warnings"].append("No usable historical rows after cleaning; history bundle empty.")
        return out

    timeline = (
        pd.DataFrame(overall_series)
        .drop_duplicates(subset=["period"])
        .sort_values("period")
        .reset_index(drop=True)
    )
    out["oos_timeline"] = timeline
    out["periods"] = timeline["period"].astype(str).tolist()

    # Build sku_oos_last_n
    if per_instance_rows:
        inst = pd.concat(per_instance_rows, ignore_index=True)
        kept_periods = set(out["periods"])
        inst = inst[inst["period"].astype(str).isin(kept_periods)]

        agg = (
            inst.groupby("sku")["is_oos"]
            .agg(oos_instances=lambda s: int(s.sum()), n_instances="count")
            .reset_index()
        )
        agg["oos_instance_pct"] = agg.apply(
            lambda r: (100.0 * r["oos_instances"] / r["n_instances"]) if r["n_instances"] else 0.0,
            axis=1,
        )
        agg = agg.sort_values(["oos_instances", "oos_instance_pct", "sku"], ascending=[False, False, True])
        out["sku_oos_last_n"] = agg.reset_index(drop=True)

    # Build sku_sparklines aligned to timeline periods
    if sku_period_rows:
        sp = pd.concat(sku_period_rows, ignore_index=True)
        piv = sp.pivot_table(index="sku", columns="period", values="oos_pct", aggfunc="first").fillna(0.0)

        cols = out["periods"]
        for c in cols:
            if c not in piv.columns:
                piv[c] = 0.0
        piv = piv[cols]

        out["sku_sparklines"] = {
            str(s): [float(x) for x in row.values.tolist()]
            for s, row in piv.iterrows()
        }

    return out
