# common/metrics.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import pandas as pd


# -----------------------------
# Column normalization
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

    colmap["price"] = pick("price", "selling_price", "sale_price") or ""
    colmap["mrp"] = pick("mrp", "list_price", "msrp") or ""
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
    s = stock_series.astype(str).str.strip().str.lower()
    return s.str.contains("out", na=False) | s.isin({"oos", "out_of_stock", "outofstock", "0", "false"})


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

    # ✅ NEW: platform drilldown tables
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
    # ✅ NEW: Platform drilldown (Pincode × Platform, City × Platform)
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

            # attach city for pincode drilldown
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
