# common/render.py
from __future__ import annotations

from typing import Tuple, List, Dict, Optional, Any, Set
import pandas as pd
import numpy as np


def _fmt_pct(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NA"
    return f"{x:.1f}%"


def _fmt_delta(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NA"
    sign = "+" if x > 0 else ""
    return f"{sign}{x:.1f} pp"


def _critical_names(crit_list: List[Any], top_n: int) -> List[str]:
    names: List[str] = []
    for item in (crit_list or [])[:top_n]:
        if isinstance(item, dict):
            sku = str(item.get("sku", "")).strip()
        else:
            sku = str(item).strip()
        if sku:
            names.append(sku)
    return names


def _critical_set(crit_list: List[Any]) -> Set[str]:
    return set(_critical_names(crit_list, top_n=10_000))


def _to_df(obj: Any) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, list) and obj:
        try:
            return pd.DataFrame(obj)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def _delta_lookup(comp_stockouts_by_sku: Any, sku: str) -> Optional[float]:
    if not sku:
        return None
    df = _to_df(comp_stockouts_by_sku)
    if df.empty or "sku" not in df.columns or "delta_oos_pct" not in df.columns:
        return None
    m = df[df["sku"].astype(str).str.strip() == sku]
    if m.empty:
        return None
    try:
        v = m.iloc[0].get("delta_oos_pct")
        return float(v) if pd.notna(v) else None
    except Exception:
        return None


def _platform_split_lines(stock: Dict[str, Any], sku: str, max_platforms: int = 6) -> List[str]:
    pt = stock.get("sku_platform_stockouts")
    df = _to_df(pt)
    if df.empty:
        return []
    for c in ["sku", "platform", "oos_pincodes", "total_pincodes"]:
        if c not in df.columns:
            return []

    dsku = df[df["sku"].astype(str).str.strip() == sku]
    if dsku.empty:
        return []

    if "oos_pct" in dsku.columns:
        dsku = dsku.sort_values(["oos_pincodes", "oos_pct"], ascending=False)
    else:
        dsku = dsku.sort_values(["oos_pincodes"], ascending=False)

    lines: List[str] = []
    shown = 0
    for _, r in dsku.iterrows():
        if shown >= max_platforms:
            break
        platform = str(r.get("platform", "")).strip()
        if not platform:
            continue
        try:
            oos = int(float(r.get("oos_pincodes")))
            tot = int(float(r.get("total_pincodes")))
        except Exception:
            continue
        try:
            pct = float(r.get("oos_pct")) if "oos_pct" in dsku.columns else (100.0 * oos / tot if tot else None)
        except Exception:
            pct = None
        lines.append(f"  • {platform}: {oos}/{tot} ({_fmt_pct(pct)})")
        shown += 1
    return lines


def _format_top_pincode_rows(items: Any, top_n: int = 3) -> List[Dict[str, Any]]:
    if isinstance(items, list):
        return items[:top_n]
    if isinstance(items, pd.DataFrame):
        return items.head(top_n).to_dict(orient="records")
    return []


def _format_top_city_rows(items: Any, top_n: int = 3) -> List[Dict[str, Any]]:
    if isinstance(items, list):
        return items[:top_n]
    if isinstance(items, pd.DataFrame):
        return items.head(top_n).to_dict(orient="records")
    return []


def _pincode_platform_lines(stock: Dict[str, Any], pincode: str, max_platforms: int = 6) -> List[str]:
    df = _to_df(stock.get("pincode_platform_oos_sku_table"))
    if df.empty:
        return []

    need = {"pincode", "platform", "oos_skus", "total_skus", "oos_pct"}
    if not need.issubset(set(df.columns)):
        return []

    d = df[df["pincode"].astype(str).str.strip() == str(pincode).strip()].copy()
    if d.empty:
        return []

    d = d.sort_values(
        ["oos_skus", "oos_pct", "total_skus", "platform"],
        ascending=[False, False, False, True],
    ).head(max_platforms)

    lines: List[str] = []
    for _, r in d.iterrows():
        plat = str(r.get("platform", "")).strip()
        if not plat:
            continue
        try:
            oos = int(float(r.get("oos_skus")))
            tot = int(float(r.get("total_skus")))
            pct = float(r.get("oos_pct"))
        except Exception:
            continue
        lines.append(f"    - {plat}: {oos}/{tot} SKUs OOS ({_fmt_pct(pct)})")
    return lines


def _city_platform_lines(stock: Dict[str, Any], city: str, max_platforms: int = 6) -> List[str]:
    df = _to_df(stock.get("city_platform_oos_sku_table"))
    if df.empty:
        return []

    need = {"city", "platform", "oos_skus", "total_skus", "oos_pct"}
    if not need.issubset(set(df.columns)):
        return []

    d = df[df["city"].astype(str).str.strip() == str(city).strip()].copy()
    if d.empty:
        return []

    d = d.sort_values(
        ["oos_skus", "oos_pct", "total_skus", "platform"],
        ascending=[False, False, False, True],
    ).head(max_platforms)

    lines: List[str] = []
    for _, r in d.iterrows():
        plat = str(r.get("platform", "")).strip()
        if not plat:
            continue
        try:
            oos = int(float(r.get("oos_skus")))
            tot = int(float(r.get("total_skus")))
            pct = float(r.get("oos_pct"))
        except Exception:
            continue
        lines.append(f"    - {plat}: {oos}/{tot} SKUs OOS ({_fmt_pct(pct)})")
    return lines


def render_bullets(payload: Dict[str, Any], top_n: int = 5) -> str:
    cur = payload.get("current", {}) or {}
    prev = payload.get("previous")
    comp = payload.get("comparison", {}).get("comparisons", {}) or {}
    summary = payload.get("summary", {}) or {}

    stock = (cur.get("stockouts", {}) or {})
    other = (cur.get("other", {}) or {})

    lines: List[str] = []
    lines.append("INSIGHT SUMMARY (Accuracy-first)")

    period = stock.get("period", "NA")
    lines.append(f"Reporting period: {period}")
    if prev and isinstance(prev, dict):
        prev_period = (prev.get("stockouts", {}) or {}).get("period")
        if prev_period:
            lines.append(f"Compared to: {prev_period}")
    lines.append("")

    # -------------------------
    # EXECUTIVE SUMMARY
    # -------------------------
    overall_delta = summary.get("overall_oos_delta_pp")
    crit_count = summary.get("critical_sku_count")
    crit_list = summary.get("critical_skus", []) or []

    top_pins_src = summary.get("top_oos_pincodes") or stock.get("pincode_oos_sku_table")
    top_cities_src = summary.get("top_oos_cities") or stock.get("city_oos_sku_table")

    lines.append("EXECUTIVE SUMMARY")

    if overall_delta is not None:
        trend = summary.get("trend", "stable")
        direction_word = "remained stable"
        if trend == "worsening":
            direction_word = "worsened"
        elif trend == "improving":
            direction_word = "improved"
        lines.append(
            f"- Overall stock-out situation has {direction_word} ({float(overall_delta):+.1f} pp vs previous period)."
        )

    cc = int(crit_count or 0)
    if cc > 0:
        rules = summary.get("rules", {}) or {}
        pct_thr = float(rules.get("critical_oos_pct_threshold", 90.0))
        d_thr = float(rules.get("critical_delta_threshold_pp", 5.0))
        lines.append(f"- Critical SKUs: {cc} (criteria: OOS% ≥ {pct_thr:.0f}% OR ΔOOS ≥ +{d_thr:.0f} pp).")
    else:
        lines.append("- Critical SKUs: 0 (no SKU crossed critical thresholds).")

    crit_names = _critical_names(crit_list, top_n=top_n)
    if crit_names:
        lines.append(f"- Most critical: {', '.join(crit_names)}")

    # Top pincodes + platform drilldown
    pin_rows = _format_top_pincode_rows(top_pins_src, top_n=3)
    if pin_rows:
        lines.append("- Top OOS pincodes (by OOS SKUs):")
        for r in pin_rows:
            pin = str(r.get("pincode", "")).strip()
            if not pin:
                continue
            city = str(r.get("city", "")).strip()
            try:
                oos = int(float(r.get("oos_skus")))
                tot = int(float(r.get("total_skus")))
                pct = float(r.get("oos_pct"))
            except Exception:
                oos, tot, pct = None, None, None

            city_txt = f" ({city})" if city else ""
            count_txt = f"{oos}/{tot} SKUs OOS" if (oos is not None and tot is not None) else "NA"
            lines.append(f"  • {pin}{city_txt}: {count_txt} ({_fmt_pct(pct)})")

            plat_lines = _pincode_platform_lines(stock, pin, max_platforms=6)
            if plat_lines:
                lines.extend(plat_lines)

    # Top cities + platform drilldown
    city_rows = _format_top_city_rows(top_cities_src, top_n=3)
    if city_rows:
        lines.append("- Top OOS cities (by OOS SKUs):")
        for r in city_rows:
            city = str(r.get("city", "")).strip()
            if not city:
                continue
            try:
                oos = int(float(r.get("oos_skus")))
                tot = int(float(r.get("total_skus")))
                pct = float(r.get("oos_pct"))
            except Exception:
                oos, tot, pct = None, None, None

            count_txt = f"{oos}/{tot} SKUs OOS" if (oos is not None and tot is not None) else "NA"
            lines.append(f"  • {city}: {count_txt} ({_fmt_pct(pct)})")

            plat_lines = _city_platform_lines(stock, city, max_platforms=6)
            if plat_lines:
                lines.extend(plat_lines)

    lines.append("")

    # -------------------------
    # DATA QUALITY NOTES
    # -------------------------
    warnings = []
    warnings += stock.get("warnings", []) or []
    warnings += other.get("warnings", []) or []
    warnings = [w for w in warnings if w]

    if warnings:
        lines.append("DATA QUALITY NOTES")
        for w in warnings[:8]:
            lines.append(f"- {w}")
        lines.append("")

    # -------------------------
    # STOCK-OUT INSIGHTS
    # -------------------------
    lines.append("STOCK-OUT INSIGHTS")

    sku_df = _to_df(stock.get("sku_stockouts"))
    sku_table = sku_df.head(top_n) if not sku_df.empty else pd.DataFrame()

    critical_set = _critical_set(crit_list)
    comp_tbl = comp.get("stockouts_by_sku")

    if sku_table.empty:
        lines.append("- No SKU stock-out rows available after filters.")
    else:
        for _, r in sku_table.iterrows():
            sku = str(r.get("sku", "")).strip()
            if not sku:
                continue

            try:
                cur_pct = float(r.get("oos_pct"))
            except Exception:
                cur_pct = None

            try:
                cur_oos = int(float(r.get("oos_pincodes")))
            except Exception:
                cur_oos = None

            try:
                total = int(float(r.get("total_pincodes")))
            except Exception:
                total = None

            delta = _delta_lookup(comp_tbl, sku)
            delta_str = f" ({_fmt_delta(delta)} vs prev)" if delta is not None else ""

            marker = "🚨" if sku in critical_set else "•"
            oos_part = "NA"
            if cur_oos is not None and total is not None:
                oos_part = f"{cur_oos}/{total} pincodes OOS"

            pct_part = _fmt_pct(cur_pct) if cur_pct is not None else "NA"
            lines.append(f"{marker} {sku}: {oos_part} ({pct_part} of pincodes affected){delta_str}")

            plat_lines = _platform_split_lines(stock, sku, max_platforms=6)
            if plat_lines:
                lines.extend(plat_lines)

    top_city = stock.get("top_city_oos")
    if isinstance(top_city, dict) and top_city.get("city"):
        lines.append(
            f"- City/Region with highest stock-out pincodes: {top_city['city']} ({top_city.get('oos_pincodes', 'NA')} pincodes)"
        )

    lines.append("")

    # -------------------------
    # ✅ COMPETITIVE INTELLIGENCE (RESTORED — platform-level like earlier)
    # -------------------------
    ci = cur.get("competitive") or {}
    if isinstance(ci, dict) and ci:
        lines.append("COMPETITIVE INTELLIGENCE")

        if ci.get("definition"):
            lines.append(f"- Definition: {ci['definition']}")

        detail = ci.get("top_competitor_stockouts_detail")

        # Preferred: detailed rendering with platform splits
        if isinstance(detail, list) and detail:
            lines.append("- Competitor stock-outs (Top):")
            for item in detail[:top_n]:
                sku = str(item.get("sku", "")).strip()
                if not sku:
                    continue

                oos = item.get("oos_pincodes")
                pct = item.get("oos_pct")
                try:
                    oos_txt = f"{int(oos)}"
                except Exception:
                    oos_txt = "NA"
                try:
                    pct_txt = f"{float(pct):.1f}%"
                except Exception:
                    pct_txt = "NA"

                lines.append(f"  • {sku}: {oos_txt} OOS pincodes ({pct_txt})")

                platforms = item.get("platforms") or []
                for p in platforms[:6]:
                    plat = str(p.get("platform", "")).strip()
                    if not plat:
                        continue
                    try:
                        poos = int(float(p.get("oos_pincodes")))
                        ptot = int(float(p.get("total_pincodes")))
                        ppct = float(p.get("oos_pct"))
                        lines.append(f"     - {plat}: {poos}/{ptot} pincodes OOS ({ppct:.1f}%)")
                    except Exception:
                        lines.append(f"     - {plat}: (platform split unavailable)")

        # Fallback: old string list
        elif ci.get("top_competitor_stockouts"):
            lines.append("- Competitor stock-outs (Top):")
            for item in ci["top_competitor_stockouts"]:
                lines.append(f"  • {item}")

        if ci.get("price_change_summary"):
            lines.append(f"- Price changes: {ci['price_change_summary']}")
        if ci.get("promo_change_summary"):
            lines.append(f"- Promotion changes: {ci['promo_change_summary']}")

        lines.append("")

    lines.append("— Generated automatically by Insight Summary Agent")
    return "\n".join(lines)


def render_html_from_text(text: str) -> str:
    escaped = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    escaped = escaped.replace("\n", "<br>")
    return f"<div style='font-family: Arial, sans-serif; font-size: 14px; line-height: 1.4'>{escaped}</div>"
   