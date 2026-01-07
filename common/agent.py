# common/agent.py  (corrected - no logic changes, only compatibility fixes + CSV export helper)
from __future__ import annotations

from typing import Optional, Dict, Any, List, Tuple
import pandas as pd

from .config import InsightAgentConfig
from .metrics import (
    compute_stockout_metrics,
    compute_price_promo_review_metrics,
    compare_periods,
    build_history_bundle,
    compute_competitor_price_promo_changes,
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
        df[city_col] = df[city_col].astype("object")
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


def run_insight_agent(
    current_df: pd.DataFrame,
    previous_dfs: Optional[List[pd.DataFrame]],
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

    hist_bundle = build_history_bundle(
        current_df=own_df,
        previous_dfs=(previous_dfs or []),
        require_serviced=config.require_serviced,
        pincode_map_df=pincode_map_df,
        n_last=5,
        top_n=config.top_n,
    )

    payload["history"] = hist_bundle

    payload["current"]["stockouts"]["recent5_top_skus"] = hist_bundle.get("sku_oos_last_n", pd.DataFrame())
    payload["current"]["stockouts"]["recent5_instances_used"] = hist_bundle.get("periods", [])
    payload["current"]["stockouts"]["oos_timeline"] = hist_bundle.get("oos_timeline", pd.DataFrame())

    prev_latest_df = None
    if previous_dfs:
        prev_latest_df = previous_dfs[-1]

        competitive: Dict[str, Any] = {}
        try:
            if "brand" in colmap and config.own_brands:
                brand_col = colmap["brand"]
                comp_df = _brand_filter_not_own(df_norm, brand_col, config.own_brands)

                if len(comp_df):
                    comp_stock = compute_stockout_metrics(comp_df, require_serviced=config.require_serviced)
                    top = comp_stock["sku_stockouts"].head(config.top_n)

                    # ✅ brand map per competitor sku (mode brand)
                    comp_tmp = comp_df.copy()
                    comp_tmp["_sku_"] = comp_tmp[colmap["sku"]].astype(str).str.strip()
                    comp_tmp["_brand_"] = comp_tmp[brand_col].astype(str).str.strip()
                    comp_tmp = comp_tmp[(comp_tmp["_sku_"] != "") & (comp_tmp["_brand_"] != "")]
                    if len(comp_tmp):
                        brand_mode = (
                            comp_tmp.groupby("_sku_")["_brand_"]
                            .agg(lambda s: s.value_counts().index[0] if len(s.value_counts()) else "")
                            .to_dict()
                        )
                    else:
                        brand_mode = {}

                    own_list = [b for b in config.own_brands if str(b).strip()]
                    own_list = list(dict.fromkeys(own_list))
                    competitive["definition"] = f"Competitors = all brands not in {own_list}"

                    # ✅ include brand in competitor stockout bullets
                    competitive["top_competitor_stockouts"] = [
                        f"{r['sku']} [{brand_mode.get(str(r['sku']).strip(), '')}]: {int(r['oos_pincodes'])} OOS pincodes ({r['oos_pct']:.1f}%)"
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
                                "brand": brand_mode.get(sku_name, ""),
                                "oos_pincodes": int(r["oos_pincodes"]),
                                "total_pincodes": int(r["total_pincodes"]),
                                "oos_pct": float(r["oos_pct"]),
                                "platforms": platforms,
                            })
                    else:
                        for _, r in top.iterrows():
                            sku_name = str(r["sku"])
                            detailed.append({
                                "sku": sku_name,
                                "brand": brand_mode.get(sku_name, ""),
                                "oos_pincodes": int(r["oos_pincodes"]),
                                "total_pincodes": int(r["total_pincodes"]),
                                "oos_pct": float(r["oos_pct"]),
                                "platforms": [],
                            })

                    competitive["top_competitor_stockouts_detail"] = detailed

                    prev_comp_df = None
                    if prev_latest_df is not None:
                        prev_norm, prev_colmap, _ = normalize_columns(prev_latest_df)
                        if "brand" in prev_colmap:
                            prev_comp_df = _brand_filter_not_own(prev_norm, prev_colmap["brand"], config.own_brands)

                    comp_changes = compute_competitor_price_promo_changes(
                        current_df=comp_df,
                        previous_df=prev_comp_df,
                        require_serviced=config.require_serviced,
                        price_change_threshold_pct=config.price_change_threshold_pct,
                        promotion_change_threshold_pct=getattr(config, "promotion_change_threshold_pct", 1.0),
                    )

                    competitive["competitor_price_promo_changes"] = comp_changes

                else:
                    competitive["warning"] = "No competitor rows found after brand split."
            else:
                competitive["warning"] = "Brand column not detected; competitor split skipped."
        except Exception as e:
            competitive["error"] = f"Competitive block skipped: {e}"

        payload["current"]["competitive"] = competitive



    if prev_latest_df is not None and len(prev_latest_df):
        prev_norm, prev_colmap, prev_warn = normalize_columns(prev_latest_df)
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

    return payload


# ============================================================
# NEW: Export Platform × Pincode insights as CSV-ready DataFrame
# (Uses common/io.normalize_columns canonical keys)
# ============================================================
def export_platform_pincode_insights(
    current_df: pd.DataFrame,
    previous_dfs: Optional[List[pd.DataFrame]],
    config: InsightAgentConfig,
    pincode_map_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Export platform/pincode/SKU stock state for own brand(s), serviced-filtered.
    Includes optional previous snapshot (latest previous file) to compute delta flag.

    Output columns:
      platform, pincode, city, sku, is_oos, previous_is_oos, delta_oos_flag, service_status
    """
    df_norm, colmap, _ = normalize_columns(current_df)

    pincode_to_city = _build_pincode_city_map(pincode_map_df)
    df_norm, _ = _ensure_city_from_pincode(df_norm, colmap, pincode_to_city)

    # Own brand filter (same as Stage 1)
    own_df = df_norm
    if config.own_brands:
        mask = _own_brand_mask(df_norm, colmap, config.own_brands)
        own_df = df_norm.loc[mask].copy()

    sku_col = colmap.get("sku")
    platform_col = colmap.get("platform")
    pincode_col = colmap.get("pincode")
    stock_col = colmap.get("stock")
    serviced_col = colmap.get("serviced")
    city_col = colmap.get("city")

    if not sku_col or not platform_col or not pincode_col or not stock_col:
        raise ValueError("Missing required columns after normalization: sku/platform/pincode/stock")

    df = own_df.copy()

    # Serviced filter (io.py uses canonical key 'serviced')
    if config.require_serviced and serviced_col and serviced_col in df.columns:
        ss = df[serviced_col].astype(str).str.strip().str.lower()
        df = df[ss.isin({"serviced", "serviceable", "yes", "true", "1"})].copy()

    # OOS detection consistent with metrics.py
    s = df[stock_col].astype(str).str.strip().str.lower()
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
    is_oos = s.isin(OOS_VALUES).astype(int)

    out = pd.DataFrame({
        "platform": df[platform_col].astype(str).str.strip(),
        "pincode": df[pincode_col].astype(str).str.strip(),
        "sku": df[sku_col].astype(str).str.strip(),
        "is_oos": is_oos,
    })

    if city_col and city_col in df.columns:
        out["city"] = df[city_col].astype(str).str.strip()
    else:
        out["city"] = ""

    out["service_status"] = "serviced" if config.require_serviced else ""

    # Add previous snapshot (latest previous file)
    prev_latest_df = (previous_dfs or [])[-1] if previous_dfs else None
    if prev_latest_df is not None and len(prev_latest_df):
        prev_norm, prev_colmap, _ = normalize_columns(prev_latest_df)
        prev_norm, _ = _ensure_city_from_pincode(prev_norm, prev_colmap, pincode_to_city)

        prev_own = prev_norm
        if config.own_brands:
            mask_prev = _own_brand_mask(prev_norm, prev_colmap, config.own_brands)
            prev_own = prev_norm.loc[mask_prev].copy()

        ps = prev_colmap.get("sku")
        pp = prev_colmap.get("platform")
        pn = prev_colmap.get("pincode")
        pstock = prev_colmap.get("stock")
        pserv = prev_colmap.get("serviced")

        if ps and pp and pn and pstock:
            p = prev_own.copy()

            if config.require_serviced and pserv and pserv in p.columns:
                ss = p[pserv].astype(str).str.strip().str.lower()
                p = p[ss.isin({"serviced", "serviceable", "yes", "true", "1"})].copy()

            psr = p[pstock].astype(str).str.strip().str.lower()
            psr = psr.str.replace(r"\s+", " ", regex=True)
            prev_is_oos = psr.isin(OOS_VALUES).astype(int)

            prev_out = pd.DataFrame({
                "platform": p[pp].astype(str).str.strip(),
                "pincode": p[pn].astype(str).str.strip(),
                "sku": p[ps].astype(str).str.strip(),
                "previous_is_oos": prev_is_oos,
            })

            out = out.merge(prev_out, on=["platform", "pincode", "sku"], how="left")
            out["previous_is_oos"] = out["previous_is_oos"].fillna(0).astype(int)
            out["delta_oos_flag"] = (out["is_oos"] - out["previous_is_oos"]).astype(int)
        else:
            out["previous_is_oos"] = 0
            out["delta_oos_flag"] = 0
    else:
        out["previous_is_oos"] = 0
        out["delta_oos_flag"] = 0

    out = out.sort_values(
        ["platform", "pincode", "is_oos", "sku"],
        ascending=[True, True, False, True]
    ).reset_index(drop=True)

    return out
