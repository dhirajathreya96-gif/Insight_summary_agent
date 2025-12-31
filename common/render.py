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


# -----------------------------
# Canonical keying for history lookups
# -----------------------------
def _norm_sku_key(x: str) -> str:
    s = str(x or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    # light normalization for common weight tokens
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
    overall_delta = comp.get("overall_stockout_delta_pp") if isinstance(comp, dict) else None
    if overall_delta is not None:
        lines.append(f"- Overall stock-out situation changed by {_fmt_delta_pp(float(overall_delta))} vs previous period.")

    crit_df = _to_df(comp.get("critical_skus"))
    lines.append(f"- Critical SKUs: {int(len(crit_df)) if not crit_df.empty else 0}")
    if not crit_df.empty and "sku" in crit_df.columns:
        names = crit_df["sku"].astype(str).head(top_n).tolist()
        if names:
            lines.append(f"- Most critical: {', '.join(names)}")
    lines.append("")

    platforms = _platforms_in_stock(stock)
    lines.append("TOP OOS PINCODES (Platform-wise)")
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

                    # ✅ Restore absolute counts: oos/total pincodes
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

    # -------------------------
    # Historical last-N section (FIXED: supports both old+new schema, and SKU keying)
    # -------------------------
    recent5_df = _to_df(stock.get("recent5_top_skus"))
    instances = stock.get("recent5_instances_used") or []

    # Optional: per-SKU sparklines bundle (new schema)
    hist_sparklines = payload.get("history", {}).get("sku_sparklines") if isinstance(payload.get("history"), dict) else None
    hist_periods = payload.get("history", {}).get("periods") if isinstance(payload.get("history"), dict) else None

    lines.append("TOP SKUs OOS IN LAST 5 SCRAPING INSTANCES (Historical)")
    if instances:
        lines.append(f"- Instances used: {', '.join([str(x) for x in instances])}")

    if recent5_df.empty:
        lines.append("- NA (upload previous files with crawled_date to enable this section)")
    else:
        # normalize column names between implementations:
        # new build_history_bundle => ["sku","oos_instances","n_instances","oos_instance_pct"]
        # old agent bundle => ["sku","instances_oos","max_instances","oos_pct_series"]
        colset = set([c for c in recent5_df.columns])

        def _get_int(row, keys, default=0):
            for k in keys:
                if k in row and pd.notna(row.get(k)):
                    try:
                        return int(float(row.get(k)))
                    except Exception:
                        pass
            return default

        def _get_series_vals(row):
            # prefer explicit series field if present
            if "oos_pct_series" in row and isinstance(row.get("oos_pct_series"), list):
                return row.get("oos_pct_series") or []
            # else try history bundle (by normalized sku key)
            if isinstance(hist_sparklines, dict):
                sku_raw = str(row.get("sku", "")).strip()
                key = _norm_sku_key(sku_raw)
                vals = hist_sparklines.get(key) or hist_sparklines.get(sku_raw)
                if isinstance(vals, list):
                    return vals
            return []

        for _, rr in recent5_df.iterrows():
            sku = str(rr.get("sku", "")).strip()

            # counts
            inst_oos = _get_int(rr, ["oos_instances", "instances_oos"], default=0)
            max_inst = _get_int(rr, ["n_instances", "max_instances"], default=5)

            series_vals = _get_series_vals(rr)
            spark = _sparkline(series_vals)
            spark_txt = f"  {spark}" if spark else ""

            lines.append(f"  • {sku}: OOS in {inst_oos}/{max_inst} instances{spark_txt}")
    lines.append("")

    # -------------------------
    # COMPETITIVE INTELLIGENCE (✅ extended)
    # -------------------------
    ci = cur.get("competitive") or {}
    if isinstance(ci, dict) and ci:
        lines.append("COMPETITIVE INTELLIGENCE")
        if ci.get("definition"):
            lines.append(f"- Definition: {ci['definition']}")

        detail = ci.get("top_competitor_stockouts_detail")
        if isinstance(detail, list) and detail:
            lines.append("- Competitor stock-outs (Top):")
            for item in detail[:top_n]:
                if not isinstance(item, dict):
                    continue
                sku = str(item.get("sku", "")).strip()
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
                lines.append(f"  • {sku}: {oos_txt} OOS pincodes ({pct_txt})")
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

        # ✅ competitor price/promo changes (platform-wise max movers, not rollups)
        cchg = (ci.get("competitor_price_promo_changes") or {})
        chg_df = _to_df(cchg.get("changed_rows"))

        lines.append("- Competitor price / promotion changes (platform-wise movers):")
        if chg_df is None or chg_df.empty:
            lines.append("  • NA (no significant price/promo changes detected)")
        else:
            d = chg_df.copy()

            # ---- normalize / coerce ----
            # numeric fields
            for col in ["price_change_pct", "disc_pct", "disc_change_pp", "price", "prev_price", "prev_disc_pct"]:
                if col in d.columns:
                    d[col] = pd.to_numeric(d[col], errors="coerce")

            # flag fields (may not exist in all schemas)
            for col in ["is_price_change", "is_new_promo", "is_promo_increase", "promo_flag"]:
                if col in d.columns:
                    try:
                        d[col] = d[col].astype(bool)
                    except Exception:
                        pass

            # required columns guard
            if "platform" not in d.columns or "sku" not in d.columns:
                lines.append("  • NA (missing platform/sku columns in competitor change output)")
            else:
                d["platform"] = d["platform"].astype(str).str.strip()
                d["sku"] = d["sku"].astype(str).str.strip()

                # helpers
                def _topk(df: pd.DataFrame, k: int, sort_cols: List[str], asc: List[bool]) -> pd.DataFrame:
                    if df is None or df.empty:
                        return df
                    out = df.sort_values(sort_cols, ascending=asc, kind="mergesort")
                    return out.head(k)

                def _fmt_price_move(r: pd.Series) -> str:
                    # e.g. "+9.3% (₹120 → ₹131)" if prices exist
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
                    # e.g. "42.0% (Δ +12.0 pp)"
                    disc = r.get("disc_pct")
                    dpp = r.get("disc_change_pp")
                    disc_txt = _fmt_pct(float(disc)) if pd.notna(disc) else "NA"
                    if pd.notna(dpp):
                        disc_txt += f" (Δ {_fmt_delta_pp(float(dpp))})"
                    return disc_txt

                # per platform sections
                plats = sorted([p for p in d["platform"].dropna().unique() if str(p).strip()])
                if not plats:
                    lines.append("  • NA (no platforms found)")
                else:
                    per_platform_k = max(2, int(top_n))  # uses render_bullets(top_n=...) as the list size

                    for plat in plats:
                        lines.append(f"  • {plat}:")

                        dp = d[d["platform"] == plat].copy()
                        if dp.empty:
                            lines.append("    - NA")
                            continue

                        # ----- PRICE MOVERS -----
                        if "price_change_pct" in dp.columns:
                            inc = dp[dp["price_change_pct"].notna() & (dp["price_change_pct"] > 0)].copy()
                            dec = dp[dp["price_change_pct"].notna() & (dp["price_change_pct"] < 0)].copy()

                            inc = _topk(inc, per_platform_k, ["price_change_pct", "disc_pct", "sku"], [False, False, True])
                            dec = _topk(dec, per_platform_k, ["price_change_pct", "disc_pct", "sku"], [True, False, True])

                            if inc is not None and not inc.empty:
                                lines.append("    - Max price increases:")
                                for _, r in inc.iterrows():
                                    sku = str(r.get("sku", "")).strip()
                                    lines.append(f"      • {sku}: {_fmt_price_move(r)}")
                            else:
                                lines.append("    - Max price increases: NA")

                            if dec is not None and not dec.empty:
                                lines.append("    - Max price decreases:")
                                for _, r in dec.iterrows():
                                    sku = str(r.get("sku", "")).strip()
                                    lines.append(f"      • {sku}: {_fmt_price_move(r)}")
                            else:
                                lines.append("    - Max price decreases: NA")
                        else:
                            lines.append("    - Price movers: NA (price_change_pct missing)")

                        # ----- PROMOS / DISCOUNTS -----
                        # 1) Maximum current discount%
                        if "disc_pct" in dp.columns:
                            disc_rank = dp[dp["disc_pct"].notna() & (dp["disc_pct"] > 0)].copy()
                            disc_rank = _topk(disc_rank, per_platform_k, ["disc_pct", "disc_change_pp", "sku"], [False, False, True])

                            if disc_rank is not None and not disc_rank.empty:
                                lines.append("    - Max discounts (current):")
                                for _, r in disc_rank.iterrows():
                                    sku = str(r.get("sku", "")).strip()
                                    lines.append(f"      • {sku}: {_fmt_disc(r)}")
                            else:
                                lines.append("    - Max discounts (current): NA")
                        else:
                            lines.append("    - Max discounts (current): NA (disc_pct missing)")

                        # 2) Biggest discount increases (Δ pp)
                        if "disc_change_pp" in dp.columns:
                            disc_up = dp[dp["disc_change_pp"].notna() & (dp["disc_change_pp"] > 0)].copy()
                            disc_up = _topk(disc_up, per_platform_k, ["disc_change_pp", "disc_pct", "sku"], [False, False, True])

                            if disc_up is not None and not disc_up.empty:
                                lines.append("    - Biggest discount increases (Δ pp):")
                                for _, r in disc_up.iterrows():
                                    sku = str(r.get("sku", "")).strip()
                                    dpp = r.get("disc_change_pp")
                                    disc = r.get("disc_pct")
                                    dpp_txt = _fmt_delta_pp(float(dpp)) if pd.notna(dpp) else "NA"
                                    disc_txt = _fmt_pct(float(disc)) if pd.notna(disc) else "NA"
                                    lines.append(f"      • {sku}: Δ {dpp_txt} (now {disc_txt})")
                            else:
                                lines.append("    - Biggest discount increases (Δ pp): NA")
                        else:
                            lines.append("    - Biggest discount increases (Δ pp): NA (disc_change_pp missing)")

                        # 3) New promos
                        if "is_new_promo" in dp.columns:
                            newp = dp[dp["is_new_promo"] == True].copy()
                            # rank new promos by discount pct if available, else keep stable order
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
        for w in warnings[:8]:
            lines.append(f"- {w}")
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
