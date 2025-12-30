# common/agent.py
from __future__ import annotations

from typing import Optional, Dict, Any, List, Tuple
import pandas as pd

from .config import InsightAgentConfig
from .metrics import (
    compute_stockout_metrics,
    compute_price_promo_review_metrics,
    compare_periods,
)
from .io import normalize_columns


def serialize(obj):
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [serialize(x) for x in obj]
    return obj


def _norm_set(items: List[str]) -> set:
    return {str(x).strip().lower() for x in items if str(x).strip()}


def _brand_filter_not_own(df: pd.DataFrame, brand_col: str, own_brands: List[str]) -> pd.DataFrame:
    own = _norm_set(own_brands)
    if not own:
        return df
    return df[~df[brand_col].astype(str).str.strip().str.lower().isin(own)].copy()


def _build_pincode_city_map(pincode_map_df: Optional[pd.DataFrame]) -> Dict[str, str]:
    if pincode_map_df is None or len(pincode_map_df) == 0:
        return {}

    cols = {str(c).lower().strip(): c for c in pincode_map_df.columns}
    pin_key = None
    city_key = None

    for k in ["pincode", "pin", "pin_code", "postal_code", "postcode", "zip"]:
        if k in cols:
            pin_key = cols[k]
            break

    for k in ["city", "region", "town", "district", "area"]:
        if k in cols:
            city_key = cols[k]
            break

    if not pin_key or not city_key:
        return {}

    tmp = pincode_map_df[[pin_key, city_key]].copy()
    tmp[pin_key] = tmp[pin_key].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    tmp[pin_key] = tmp[pin_key].apply(lambda x: x.zfill(6) if x.isdigit() and len(x) <= 6 else x)
    tmp[city_key] = tmp[city_key].astype(str).str.strip()
    tmp = tmp[(tmp[pin_key] != "") & (tmp[city_key] != "")]
    return dict(zip(tmp[pin_key], tmp[city_key]))


def _ensure_city_from_pincode(
    df: pd.DataFrame,
    colmap: Dict[str, str],
    pincode_to_city: Dict[str, str]
) -> Tuple[pd.DataFrame, List[str]]:
    warnings: List[str] = []
    if not pincode_to_city:
        return df, warnings

    if "pincode" not in colmap:
        warnings.append("Pincode→City mapping provided but pincode column not detected; city derivation skipped.")
        return df, warnings

    pin_col = colmap["pincode"]
    city_col = colmap.get("city")

    df = df.copy()

    if city_col is None:
        df["City_Derived"] = df[pin_col].astype(str).str.strip().map(pincode_to_city).fillna("")
        colmap["city"] = "City_Derived"
        return df, warnings

    city_series = df[city_col].astype(str).fillna("").str.strip()
    need_fill = city_series.eq("") | city_series.str.lower().isin({"nan", "none"})
    if need_fill.any():
        df.loc[need_fill, city_col] = (
            df.loc[need_fill, pin_col].astype(str).str.strip().map(pincode_to_city).fillna("")
        )

    return df, warnings


def _own_brand_mask(df: pd.DataFrame, colmap: Dict[str, str], own_brands: List[str]) -> pd.Series:
    own_norm = [str(b).strip().lower() for b in (own_brands or []) if str(b).strip()]
    if not own_norm:
        return pd.Series([True] * len(df), index=df.index)

    mask = pd.Series([False] * len(df), index=df.index)

    if "brand" in colmap:
        b = df[colmap["brand"]].astype(str).str.strip().str.lower().fillna("")
        mask = mask | b.isin(set(own_norm))

    if "sku" in colmap:
        s = df[colmap["sku"]].astype(str).str.lower().fillna("")
        for k in own_norm:
            mask = mask | s.str.contains(k, na=False)

    return mask


def _safe_topn_rows(df: Any, n: int) -> List[Dict[str, Any]]:
    if isinstance(df, pd.DataFrame) and len(df):
        return df.head(n).to_dict(orient="records")
    return []


def run_insight_agent(
    current_df: pd.DataFrame,
    previous_df: Optional[pd.DataFrame],
    config: InsightAgentConfig,
    pincode_map_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:

    df_norm, colmap, warnings_norm = normalize_columns(current_df)

    pincode_to_city = _build_pincode_city_map(pincode_map_df)
    df_norm, warnings_city = _ensure_city_from_pincode(df_norm, colmap, pincode_to_city)

    own_df = df_norm
    if config.own_brands:
        mask = _own_brand_mask(df_norm, colmap, config.own_brands)
        own_df = df_norm.loc[mask].copy()

    cur_stock = compute_stockout_metrics(
        own_df,
        require_serviced=config.require_serviced,
        pincode_map_df=pincode_map_df,
    )
    cur_other = compute_price_promo_review_metrics(own_df, require_serviced=config.require_serviced)

    cur_stock.setdefault("warnings", [])
    cur_other.setdefault("warnings", [])
    for w in (warnings_norm or []):
        if w:
            cur_stock["warnings"].append(w)
    for w in (warnings_city or []):
        if w:
            cur_stock["warnings"].append(w)

    payload: Dict[str, Any] = {"current": {"stockouts": cur_stock, "other": cur_other}}

    base_summary: Dict[str, Any] = {
        "top_oos_pincodes": _safe_topn_rows(cur_stock.get("pincode_oos_sku_table"), n=3),
        "top_oos_cities": _safe_topn_rows(cur_stock.get("city_oos_sku_table"), n=3),
    }
    payload["summary"] = dict(base_summary)

    # Competitive (unchanged)
    competitive: Dict[str, Any] = {}
    try:
        if "brand" in colmap and config.own_brands:
            brand_col = colmap["brand"]
            comp_df = _brand_filter_not_own(df_norm, brand_col, config.own_brands)

            if len(comp_df):
                comp_stock = compute_stockout_metrics(comp_df, require_serviced=config.require_serviced)
                top = comp_stock["sku_stockouts"].head(config.top_n)

                own_list = [b for b in config.own_brands if str(b).strip()]
                own_list = list(dict.fromkeys(own_list))
                competitive["definition"] = f"Competitors = all brands not in {own_list}"

                competitive["top_competitor_stockouts"] = [
                    f"{r['sku']}: {int(r['oos_pincodes'])} OOS pincodes ({r['oos_pct']:.1f}%)"
                    for _, r in top.iterrows()
                ]

                plat_tbl = comp_stock.get("sku_platform_stockouts")
                detailed: List[Dict[str, Any]] = []
                if isinstance(plat_tbl, pd.DataFrame) and len(plat_tbl):
                    for _, r in top.iterrows():
                        sku_name = str(r["sku"])
                        sku_plat = plat_tbl[plat_tbl["sku"].astype(str) == sku_name].copy()
                        if len(sku_plat):
                            sku_plat = sku_plat.sort_values(["oos_pct", "oos_pincodes"], ascending=False).head(6)
                            platforms = sku_plat.to_dict(orient="records")
                        else:
                            platforms = []
                        detailed.append({
                            "sku": sku_name,
                            "oos_pincodes": int(r["oos_pincodes"]),
                            "total_pincodes": int(r["total_pincodes"]),
                            "oos_pct": float(r["oos_pct"]),
                            "platforms": platforms,
                        })
                else:
                    for _, r in top.iterrows():
                        detailed.append({
                            "sku": str(r["sku"]),
                            "oos_pincodes": int(r["oos_pincodes"]),
                            "total_pincodes": int(r["total_pincodes"]),
                            "oos_pct": float(r["oos_pct"]),
                            "platforms": [],
                        })

                competitive["top_competitor_stockouts_detail"] = detailed
            else:
                competitive["warning"] = "No competitor rows found after brand split."
        else:
            competitive["warning"] = "Brand column not detected; competitor split skipped."
    except Exception as e:
        competitive["error"] = f"Competitive block skipped: {e}"

    payload["current"]["competitive"] = competitive

    # Previous comparison
    if previous_df is not None:
        prev_norm, prev_colmap, prev_warn = normalize_columns(previous_df)
        prev_norm, prev_city_warn = _ensure_city_from_pincode(prev_norm, prev_colmap, pincode_to_city)

        prev_own_df = prev_norm
        if config.own_brands:
            mask_prev = _own_brand_mask(prev_norm, prev_colmap, config.own_brands)
            prev_own_df = prev_norm.loc[mask_prev].copy()

        prev_stock = compute_stockout_metrics(
            prev_own_df,
            require_serviced=config.require_serviced,
            pincode_map_df=pincode_map_df,
        )
        prev_other = compute_price_promo_review_metrics(prev_own_df, require_serviced=config.require_serviced)

        prev_stock.setdefault("warnings", [])
        prev_other.setdefault("warnings", [])
        for w in (prev_warn or []):
            if w:
                prev_stock["warnings"].append(w)
        for w in (prev_city_warn or []):
            if w:
                prev_stock["warnings"].append(w)

        payload["previous"] = {"stockouts": prev_stock, "other": prev_other}

        payload["comparison"] = compare_periods(
            current=cur_stock,
            previous=prev_stock,
            top_n=config.top_n,
            price_threshold_pct=config.price_change_threshold_pct,
        )

        comps = payload.get("comparison", {}).get("comparisons", {}) or {}
        overall_delta = comps.get("overall_stockout_delta_pp")

        critical_list: List[str] = []
        critical_df = comps.get("critical_skus")
        if isinstance(critical_df, pd.DataFrame) and len(critical_df) and "sku" in critical_df.columns:
            critical_list = critical_df["sku"].astype(str).tolist()

        trend = "stable"
        if overall_delta is not None:
            try:
                d = float(overall_delta)
                if d >= 0.5:
                    trend = "worsening"
                elif d <= -0.5:
                    trend = "improving"
            except Exception:
                pass

        payload["summary"] = {
            **base_summary,  # ✅ keep geo drilldowns
            "trend": trend,
            "overall_oos_delta_pp": round(float(overall_delta), 2) if overall_delta is not None else None,
            "critical_sku_count": len(critical_list),
            "critical_skus": critical_list[: config.top_n],
            "rules": {
                "critical_oos_pct_threshold": 90.0,
                "critical_delta_threshold_pp": 5.0,
            },
        }

    return payload
