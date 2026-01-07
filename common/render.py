# common/render.py
from __future__ import annotations

from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
import re


def _fmt_pct(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NA"
    return f"{x:.1f}%"


def _fmt_delta_pp(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NA"
    sign = "+" if x > 0 else ""
    return f"{sign}{x:.1f} pp"


def _fmt_delta_pct(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NA"
    sign = "+" if x > 0 else ""
    return f"{sign}{x:.1f}%"


def _to_df(obj: Any) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, list) and obj:
        try:
            return pd.DataFrame(obj)
        except Exception:
            return pd.DataFrame()
    if isinstance(obj, dict) and "changed_rows" in obj:
        try:
            return pd.DataFrame(obj["changed_rows"])
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def _sparkline(values: List[Optional[float]]) -> str:
    blocks = "▁▂▃▄▅▆▇█"
    vals = []
    for v in values:
        if v is None:
            vals.append(None)
            continue
        try:
            fv = float(v)
            if np.isnan(fv):
                vals.append(None)
            else:
                vals.append(fv)
        except Exception:
            vals.append(None)

    good = [v for v in vals if v is not None]
    if not good:
        return ""
    mn, mx = min(good), max(good)
    if mx - mn < 1e-9:
        return blocks[0] * len(vals)

    out = []
    for v in vals:
        if v is None:
            out.append("·")
        else:
            idx = int(round((v - mn) / (mx - mn) * (len(blocks) - 1)))
            idx = max(0, min(len(blocks) - 1, idx))
            out.append(blocks[idx])
    return "".join(out)


def _norm_sku_key(x: str) -> str:
    s = str(x or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.replace(" g", "g").replace(" gm", "gm")
    return s


def _platforms_in_stock(stock: Dict[str, Any]) -> List[str]:
    df1 = _to_df(stock.get("pincode_platform_oos_sku_table"))
    df2 = _to_df(stock.get("sku_platform_stockouts"))
    plats = set()
    if not df1.empty and "platform" in df1.columns:
        plats |= set([p for p in df1["platform"].dropna().astype(str).str.strip().unique() if p])
    if not df2.empty and "platform" in df2.columns:
        plats |= set([p for p in df2["platform"].dropna().astype(str).str.strip().unique() if p])
    return sorted(plats)


def _top_pincodes_by_platform(stock: Dict[str, Any], platform: str, k: int = 3) -> pd.DataFrame:
    df = _to_df(stock.get("pincode_platform_oos_sku_table"))
    if df.empty:
        return pd.DataFrame(columns=["pincode", "oos_pct"])
    need = {"platform", "pincode", "oos_pct", "oos_skus", "total_skus"}
    if not need.issubset(set(df.columns)):
        return pd.DataFrame(columns=["pincode", "oos_pct"])
    d = df[df["platform"].astype(str).str.strip() == str(platform).strip()].copy()
    if d.empty:
        return pd.DataFrame(columns=["pincode", "oos_pct"])
    d["oos_pct"] = pd.to_numeric(d["oos_pct"], errors="coerce").fillna(0.0)
    d = d[d["oos_pct"] > 0]
    if d.empty:
        return pd.DataFrame(columns=["pincode", "oos_pct"])
    g = (
        d.groupby("pincode", as_index=False)
        .agg(oos_pct=("oos_pct", "max"), oos_skus=("oos_skus", "max"), total_skus=("total_skus", "max"))
        .sort_values(["oos_pct", "oos_skus", "total_skus", "pincode"], ascending=[False, False, False, True])
        .head(k)
    )
    return g


def _top_skus_by_platform(stock: Dict[str, Any], platform: str, top_n: int = 5) -> pd.DataFrame:
    df = _to_df(stock.get("sku_platform_stockouts"))
    if df.empty:
        return pd.DataFrame(columns=["sku", "oos_pct", "oos_pincodes", "total_pincodes"])
    need = {"platform", "sku", "oos_pct", "oos_pincodes", "total_pincodes"}
    if not need.issubset(set(df.columns)):
        return pd.DataFrame(columns=["sku", "oos_pct", "oos_pincodes", "total_pincodes"])
    d = df[df["platform"].astype(str).str.strip() == str(platform).strip()].copy()
    if d.empty:
        return pd.DataFrame(columns=["sku", "oos_pct", "oos_pincodes", "total_pincodes"])
    d["oos_pct"] = pd.to_numeric(d["oos_pct"], errors="coerce").fillna(0.0)
    d = d[d["oos_pct"] > 0]
    if d.empty:
        return pd.DataFrame(columns=["sku", "oos_pct", "oos_pincodes", "total_pincodes"])
    d = d.sort_values(
        ["oos_pincodes", "oos_pct", "total_pincodes", "sku"],
        ascending=[False, False, False, True],
    ).head(top_n)
    return d


def _delta_lookup_platform(comp_by_sku_platform: Any, platform: str, sku: str) -> Optional[float]:
    df = _to_df(comp_by_sku_platform)
    if df.empty:
        return None
    need = {"platform", "sku", "delta_oos_pct"}
    if not need.issubset(set(df.columns)):
        return None
    m = df[
        (df["platform"].astype(str).str.strip() == str(platform).strip())
        & (df["sku"].astype(str).str.strip() == str(sku).strip())
    ]
    if m.empty:
        return None
    try:
        v = m.iloc[0].get("delta_oos_pct")
        return float(v) if pd.notna(v) else None
    except Exception:
        return None


def render_bullets(payload: Dict[str, Any], top_n: int = 5) -> str:
    cur = payload.get("current", {}) or {}
    prev = payload.get("previous")
    comp = payload.get("comparison", {}).get("comparisons", {}) or {}

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

    kt = payload.get("key_takeaways")
    if isinstance(kt, str) and kt.strip():
        lines.append("KEY TAKEAWAYS")
        lines.append(f"- {kt.strip()}")
        lines.append("")

    lines.append("EXECUTIVE SUMMARY")
    lines.append("- Metric definition: Overall stock-out delta is the change in overall OOS% (unique OOS pincodes ÷ total pincodes) vs previous period.")
    overall_delta = comp.get("overall_stockout_delta_pp") if isinstance(comp, dict) else None
    if overall_delta is not None:
        lines.append(f"- Overall stock-out situation changed by {_fmt_delta_pp(float(overall_delta))} vs previous period.")

    crit_df = _to_df(comp.get("critical_skus"))
    lines.append("- Metric definition: High-priority SKUs are those with OOS% ≥ 90% OR OOS% increased by ≥ 5 pp vs previous period.")
    lines.append(f"- High-priority SKUs: {int(len(crit_df)) if not crit_df.empty else 0}")
    if not crit_df.empty and "sku" in crit_df.columns:
        names = crit_df["sku"].astype(str).head(top_n).tolist()
        if names:
            lines.append(f"- Highest priority: {', '.join(names)}")
    lines.append("")

    platforms = _platforms_in_stock(stock)
    lines.append("TOP OOS PINCODES (Platform-wise)")
    lines.append("- Metric definition: OOS% by pincode = OOS SKUs ÷ total SKUs within that pincode on the given platform.")
    if not platforms:
        lines.append("- NA (no platform information available)")
    else:
        for p in platforms:
            lines.append(f"- {p}:")
            tp = _top_pincodes_by_platform(stock, p, k=3)
            if tp.empty:
                lines.append("  • NA")
            else:
                for _, r in tp.iterrows():
                    lines.append(f"  • {str(r.get('pincode','')).strip()}: {_fmt_pct(float(r.get('oos_pct') or 0.0))}")
    lines.append("")

    lines.append("STOCK-OUT INSIGHTS (Platform-wise)")
    lines.append("- Metric definition: SKU OOS% = OOS pincodes ÷ total pincodes for that SKU on the given platform.")
    comp_plat = comp.get("stockouts_by_sku_platform")
    sku_plat_series = stock.get("sku_platform_oos_series") or {}

    if not platforms:
        lines.append("- NA (no platform information available)")
    else:
        for p in platforms:
            lines.append(f"- {p}:")
            ts = _top_skus_by_platform(stock, p, top_n=top_n)
            if ts.empty:
                lines.append("  • NA")
            else:
                for _, r in ts.iterrows():
                    sku = str(r.get("sku", "")).strip()
                    oos_pct = float(r.get("oos_pct") or 0.0)

                    try:
                        oos_pins = int(float(r.get("oos_pincodes") or 0))
                    except Exception:
                        oos_pins = 0
                    try:
                        tot_pins = int(float(r.get("total_pincodes") or 0))
                    except Exception:
                        tot_pins = 0
                    abs_txt = f" - {oos_pins}/{tot_pins} pincodes affected" if tot_pins > 0 else ""

                    delta = _delta_lookup_platform(comp_plat, p, sku)
                    delta_txt = f" ({_fmt_delta_pp(delta)} vs prev)" if delta is not None else ""

                    series = None
                    try:
                        series = sku_plat_series.get(p, {}).get(sku)
                    except Exception:
                        series = None
                    spark = _sparkline(series or [])
                    spark_txt = f"  {spark}" if spark else ""

                    lines.append(f"  • {sku}: {_fmt_pct(oos_pct)} of pincodes affected{abs_txt}{delta_txt}{spark_txt}")
    lines.append("")

    # Historical last-N (platform-level)
    history = payload.get("history") or {}

    lines.append("TOP SKUs OOS IN LAST 5 SCRAPING INSTANCES (Historical)")
    lines.append("- Metric definition: A SKU counts as 'OOS in an instance' if it is OOS in ≥1 pincode for that platform in that scrape instance.")

    hist_periods = history.get("periods") or []
    if hist_periods:
        lines.append(f"- Instances used: {', '.join([str(x) for x in hist_periods])}")

    hist_df = _to_df(history.get("sku_platform_oos_last_n"))
    hist_sparks = history.get("sku_platform_sparklines") or {}

    if hist_df.empty or ("platform" not in hist_df.columns) or ("sku" not in hist_df.columns):
        lines.append("- NA (upload previous files with platform + crawled_date to enable platform-level history)")
    else:
        hist_df = hist_df.copy()
        hist_df["platform"] = hist_df["platform"].astype(str).str.strip()
        hist_df["sku"] = hist_df["sku"].astype(str).str.strip()

        plats = sorted([p for p in hist_df["platform"].dropna().unique().tolist() if p])
        if not plats:
            lines.append("- NA (no platforms found in history)")
        else:
            per_platform_k = max(2, int(top_n))

            for plat in plats:
                lines.append(f"- {plat}:")
                dp = hist_df[hist_df["platform"] == plat].copy()

                if dp.empty:
                    lines.append("  • NA")
                    continue

                for c in ["oos_instances", "n_instances", "oos_instance_pct"]:
                    if c in dp.columns:
                        dp[c] = pd.to_numeric(dp[c], errors="coerce")

                dp = dp.sort_values(
                    ["oos_instances", "oos_instance_pct", "sku"],
                    ascending=[False, False, True],
                    kind="mergesort",
                ).head(per_platform_k)

                for _, rr in dp.iterrows():
                    sku = str(rr.get("sku", "")).strip()
                    try:
                        oi = int(float(rr.get("oos_instances") or 0))
                    except Exception:
                        oi = 0
                    try:
                        ni = int(float(rr.get("n_instances") or 0))
                    except Exception:
                        ni = 0

                    series_vals = []
                    try:
                        series_vals = (hist_sparks.get(plat, {}) or {}).get(sku, []) or []
                    except Exception:
                        series_vals = []
                    spark = _sparkline(series_vals) if series_vals else ""
                    spark_txt = f"  {spark}" if spark else ""

                    lines.append(f"  • {sku}: OOS in {oi}/{ni} instances{spark_txt}")
    lines.append("")

    # Competitive intelligence (unchanged)
    ci = cur.get("competitive") or {}
    if isinstance(ci, dict) and ci:
        lines.append("COMPETITIVE INTELLIGENCE")
        lines.append("- Metric definition: Competitor block uses all brands NOT in your own brand list (brand split).")

        if ci.get("definition"):
            lines.append(f"- Definition: {ci['definition']}")

        detail = ci.get("top_competitor_stockouts_detail")
        if isinstance(detail, list) and detail:
            lines.append("- Competitor stock-outs (Top):")
            lines.append("  Metric definition: Competitor SKU OOS% = OOS pincodes ÷ total pincodes (aggregated across serviced pincodes).")
            for item in detail[:top_n]:
                if not isinstance(item, dict):
                    continue
                sku = str(item.get("sku", "")).strip()
                brand = str(item.get("brand", "")).strip()
                brand_txt = f" [{brand}]" if brand else ""



                if not sku:
                    continue
                try:
                    oos_txt = str(int(float(item.get("oos_pincodes"))))
                except Exception:
                    oos_txt = "NA"
                try:
                    pct_txt = _fmt_pct(float(item.get("oos_pct")))
                except Exception:
                    pct_txt = "NA"
                lines.append(f"  • {sku}{brand_txt}: {oos_txt} OOS pincodes ({pct_txt})")

                for pdet in (item.get("platforms") or [])[:6]:
                    plat = str(pdet.get("platform", "")).strip()
                    if not plat:
                        continue
                    try:
                        poos = int(float(pdet.get("oos_pincodes")))
                        ptot = int(float(pdet.get("total_pincodes")))
                        ppct = float(pdet.get("oos_pct"))
                        lines.append(f"     - {plat}: {poos}/{ptot} pincodes OOS ({_fmt_pct(ppct)})")
                    except Exception:
                        lines.append(f"     - {plat}: (platform split unavailable)")

        cchg = (ci.get("competitor_price_promo_changes") or {})
        chg_df = _to_df(cchg.get("changed_rows"))

        lines.append("- Competitor price / promotion changes (platform-wise movers):")
        lines.append("  Metric definition: Price change% = (current median price - previous median price) ÷ previous median price × 100; Discount Δ pp = current max discount% - previous max discount%.")

        if chg_df is None or chg_df.empty:
            lines.append("  • NA (no significant price/promo changes detected)")
        else:
            d = chg_df.copy()

            for col in ["price_change_pct", "disc_pct", "disc_change_pp", "price", "prev_price", "prev_disc_pct"]:
                if col in d.columns:
                    d[col] = pd.to_numeric(d[col], errors="coerce")

            for col in ["is_price_change", "is_new_promo", "is_promo_increase", "promo_flag"]:
                if col in d.columns:
                    try:
                        d[col] = d[col].astype(bool)
                    except Exception:
                        pass

            if "platform" not in d.columns or "sku" not in d.columns:
                lines.append("  • NA (missing platform/sku columns in competitor change output)")
            else:
                d["platform"] = d["platform"].astype(str).str.strip()
                d["sku"] = d["sku"].astype(str).str.strip()

                def _topk(df: pd.DataFrame, k: int, sort_cols: List[str], asc: List[bool]) -> pd.DataFrame:
                    if df is None or df.empty:
                        return df
                    out2 = df.sort_values(sort_cols, ascending=asc, kind="mergesort")
                    return out2.head(k)

                def _fmt_price_move(r: pd.Series) -> str:
                    pct = r.get("price_change_pct")
                    cur_p = r.get("price")
                    prev_p = r.get("prev_price")
                    pct_txt = _fmt_delta_pct(float(pct)) if pd.notna(pct) else "NA"
                    if pd.notna(cur_p) and pd.notna(prev_p):
                        try:
                            return f"{pct_txt} (₹{float(prev_p):.0f} → ₹{float(cur_p):.0f})"
                        except Exception:
                            return pct_txt
                    return pct_txt

                def _fmt_disc(r: pd.Series) -> str:
                    disc = r.get("disc_pct")
                    dpp = r.get("disc_change_pp")
                    disc_txt = _fmt_pct(float(disc)) if pd.notna(disc) else "NA"
                    if pd.notna(dpp):
                        disc_txt += f" (Δ {_fmt_delta_pp(float(dpp))})"
                    return disc_txt

                plats = sorted([p for p in d["platform"].dropna().unique() if str(p).strip()])
                if not plats:
                    lines.append("  • NA (no platforms found)")
                else:
                    per_platform_k = max(2, int(top_n))

                    for plat in plats:
                        lines.append(f"  • {plat}:")

                        dp = d[d["platform"] == plat].copy()
                        if dp.empty:
                            lines.append("    - NA")
                            continue

                        if "price_change_pct" in dp.columns:
                            inc = dp[dp["price_change_pct"].notna() & (dp["price_change_pct"] > 0)].copy()
                            dec = dp[dp["price_change_pct"].notna() & (dp["price_change_pct"] < 0)].copy()

                            inc = _topk(inc, per_platform_k, ["price_change_pct", "disc_pct", "sku"], [False, False, True])
                            dec = _topk(dec, per_platform_k, ["price_change_pct", "disc_pct", "sku"], [True, False, True])

                            if inc is not None and not inc.empty:
                                lines.append("    - Max price increases:")
                                for _, r in inc.iterrows():
                                    sku = str(r.get("sku", "")).strip()
                                    brand = str(r.get("brand", "")).strip()
                                    brand_txt = f" [{brand}]" if brand else ""
                                    lines.append(f"      • {sku}{brand_txt}: {_fmt_price_move(r)}")
                            else:
                                lines.append("    - Max price increases: NA")

                            if dec is not None and not dec.empty:
                                lines.append("    - Max price decreases:")
                                for _, r in dec.iterrows():
                                    sku = str(r.get("sku", "")).strip()
                                    brand = str(r.get("brand", "")).strip()
                                    brand_txt = f" [{brand}]" if brand else ""
                                    lines.append(f"      • {sku}{brand_txt}: {_fmt_price_move(r)}")
                            else:
                                lines.append("    - Max price decreases: NA")
                        else:
                            lines.append("    - Price movers: NA (price_change_pct missing)")

                        if "disc_pct" in dp.columns:
                            disc_rank = dp[dp["disc_pct"].notna() & (dp["disc_pct"] > 0)].copy()
                            disc_rank = _topk(disc_rank, per_platform_k, ["disc_pct", "disc_change_pp", "sku"], [False, False, True])

                            if disc_rank is not None and not disc_rank.empty:
                                lines.append("    - Max discounts (current):")
                                for _, r in disc_rank.iterrows():
                                    sku = str(r.get("sku", "")).strip()
                                    brand = str(r.get("brand", "")).strip()
                                    brand_txt = f" [{brand}]" if brand else ""
                                    lines.append(f"      • {sku}{brand_txt}: {_fmt_disc(r)}")
                            else:
                                lines.append("    - Max discounts (current): NA")
                        else:
                            lines.append("    - Max discounts (current): NA (disc_pct missing)")

                        if "disc_change_pp" in dp.columns:
                            disc_up = dp[dp["disc_change_pp"].notna() & (dp["disc_change_pp"] > 0)].copy()
                            disc_up = _topk(disc_up, per_platform_k, ["disc_change_pp", "disc_pct", "sku"], [False, False, True])

                            if disc_up is not None and not disc_up.empty:
                                lines.append("    - Biggest discount increases (Δ pp):")
                                for _, r in disc_up.iterrows():
                                    sku = str(r.get("sku", "")).strip()
                                    brand = str(r.get("brand", "")).strip()
                                    brand_txt = f" [{brand}]" if brand else ""
                                    dpp = r.get("disc_change_pp")
                                    discv = r.get("disc_pct")
                                    dpp_txt = _fmt_delta_pp(float(dpp)) if pd.notna(dpp) else "NA"
                                    disc_txt = _fmt_pct(float(discv)) if pd.notna(discv) else "NA"
                                    lines.append(f"      • {sku}{brand_txt}: Δ {dpp_txt} (now {disc_txt})")
                            else:
                                lines.append("    - Biggest discount increases (Δ pp): NA")
                        else:
                            lines.append("    - Biggest discount increases (Δ pp): NA (disc_change_pp missing)")

                        if "is_new_promo" in dp.columns:
                            newp = dp[dp["is_new_promo"] == True].copy()
                            if "disc_pct" in newp.columns:
                                newp = newp.sort_values(["disc_pct", "sku"], ascending=[False, True], kind="mergesort")
                            else:
                                newp = newp.sort_values(["sku"], ascending=[True], kind="mergesort")
                            newp = newp.head(per_platform_k)

                            if newp is not None and not newp.empty:
                                lines.append("    - New promos:")
                                for _, r in newp.iterrows():
                                    sku = str(r.get("sku", "")).strip()
                                    disc_txt = ""
                                    if "disc_pct" in newp.columns and pd.notna(r.get("disc_pct")):
                                        disc_txt = f" (disc {_fmt_pct(float(r.get('disc_pct')))})"
                                    lines.append(f"      • {sku}{disc_txt}")
                            else:
                                lines.append("    - New promos: NA")
                        else:
                            lines.append("    - New promos: NA (is_new_promo missing)")

        if ci.get("warning"):
            lines.append(f"- Note: {ci['warning']}")
        if ci.get("error"):
            lines.append(f"- Error: {ci['error']}")
        lines.append("")

    warnings = []
    warnings += stock.get("warnings", []) or []
    warnings += other.get("warnings", []) or []
    warnings = [w for w in warnings if w]
    if warnings:
        lines.append("DATA QUALITY NOTES")
        lines.append("- Metric definition: These are issues detected during column normalization / missing required fields / mapping coverage.")
        for w in warnings[:8]:
            lines.append(f"- {w}")
        lines.append("")

    lines.append("— Generated automatically by Insight Summary Agent")
    return "\n".join(lines)


# -----------------------------
# ✅ FIXED: accept Any (not only str) so dict won't crash
# -----------------------------
def render_html_from_text(text: Any) -> str:
    """
    Convert plain text into safe HTML for email/preview.
    Defensive: if caller passes dict/list accidentally, it won't crash.
    """
    if text is None:
        text = ""
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            text = ""

    escaped = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    escaped = escaped.replace("\n", "<br>")
    return f"<div style='font-family: Arial, sans-serif; font-size: 14px; line-height: 1.4'>{escaped}</div>"


# -----------------------------
# ✅ ADDED: missing function used by backend/main.py import
# -----------------------------
def render_action_email_html(action_payload_or_text: Any) -> str:
    """
    Backend imports this: from common.render import render_action_email_html

    Accepts:
      - action_payload dict (expected key: 'action_text' OR 'text'), OR
      - raw action text string

    Returns:
      - HTML string
    """
    action_text = ""

    if isinstance(action_payload_or_text, dict):
        action_text = (
            action_payload_or_text.get("action_text")
            or action_payload_or_text.get("text")
            or action_payload_or_text.get("summary")
            or ""
        )
    elif isinstance(action_payload_or_text, str):
        action_text = action_payload_or_text
    else:
        action_text = str(action_payload_or_text) if action_payload_or_text is not None else ""

    return render_html_from_text(action_text)
