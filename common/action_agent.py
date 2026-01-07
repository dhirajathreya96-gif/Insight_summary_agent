from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel

import json
import datetime
import random
import time

from .action_config import ActionAgentConfig


# ============================================================
# Output schema (STRICT)
# ============================================================
class ActionInsight(BaseModel):
    platform: str
    # "title" will be used as the INSIGHT line in the Stage2 render
    title: str
    severity: str
    trend: str
    urgency: str
    owner: str
    recommended_action: str
    evidence: List[str]
    drilldown_suggestion: Optional[str] = None


class BatchActionInsights(BaseModel):
    insights: List[ActionInsight]


# ============================================================
# Metric definitions (kept for LLM context; not printed)
# ============================================================
def _metric_definitions() -> List[str]:
    return [
        "OOS% (SKU): % of pincodes where the SKU is out of stock.",
        "Δ OOS (pp): Change in OOS% vs previous period.",
        "Price change%: (current median price - previous median price) ÷ previous median price × 100.",
        "Discount Δ (pp): current max discount% - previous max discount%.",
        "New promo: promo present now but not in previous period (when available).",
    ]


# ============================================================
# SYSTEM PROMPT (batch)
# ============================================================
BATCH_ACTION_SYSTEM_PROMPT = """
You are an Insight-to-Action Analyst for e-commerce performance data.

You will be given:
- a list of breached platforms
- evidence per platform (SKU-level metrics; mixed alert types: stockouts, price_promo)

STRICT RULES:
1) Output MUST be valid JSON that matches this schema:
   { "insights": [ ActionInsight, ... ] }
2) Return AT MOST ONE ActionInsight per platform.
3) Only include platforms present in breached_platforms.
4) Every insight MUST include:
   - platform (exact platform name from breached_platforms)
   - title: a one-line INSIGHT summary for that platform (what happened + why it matters)
   - recommended_action (platform-specific)
   - evidence: 2-6 bullet strings with SKU-level numbers (OOS%, price change%, discount%, etc)
5) Do NOT invent data. Use only the provided evidence.
6) Do NOT output anything outside JSON.
"""


# ============================================================
# Helpers
# ============================================================
def _to_float(x: Any) -> Optional[float]:
    """Robust numeric coercion (handles '100.0%', commas, blanks, None)."""
    try:
        if x is None or isinstance(x, bool):
            return None
        if isinstance(x, str):
            s = x.strip()
            if not s:
                return None
            s = s.replace(",", "").replace("%", "").strip()
            if not s or s.lower() == "na":
                return None
            x = s
        v = float(x)
        if v != v:  # NaN
            return None
        return v
    except Exception:
        return None


def _as_str(x: Any) -> str:
    return "" if x is None else str(x)


def _norm_platform(x: Any) -> str:
    return _as_str(x).strip()


def _norm_sku(x: Any) -> str:
    return _as_str(x).strip()


def _get_preferred_alert_types(cfg: ActionAgentConfig) -> List[str]:
    v = getattr(cfg, "preferred_alert_types", None)
    if isinstance(v, list) and v:
        return [str(x).strip() for x in v if str(x).strip()]
    return ["stockouts", "price_promo", "competition"]


# ============================================================
# Evidence builder: STOCKOUTS (PLATFORM-SCOPED)
# ============================================================
def _build_platform_stockout_evidence(
    stage1_payload: Dict[str, Any], cfg: ActionAgentConfig
) -> Dict[str, List[Dict[str, Any]]]:
    cur = stage1_payload.get("current", {}) or {}
    stock = cur.get("stockouts", {}) or {}
    rows = stock.get("sku_platform_stockouts", []) or []

    platforms: Dict[str, List[Dict[str, Any]]] = {}
    min_oos = float(getattr(cfg, "min_oos_pct_for_reporting", 50.0))

    for r in rows:
        if not isinstance(r, dict):
            continue

        plat = _norm_platform(r.get("platform"))
        if not plat:
            continue

        oos_pct = _to_float(r.get("oos_pct"))

        # ✅ FILTER: include ONLY OOS% > min_oos (e.g., >50)
        if oos_pct is None or oos_pct <= min_oos:
            continue

        platforms.setdefault(plat, []).append(
            {
                "type": "stockouts",
                "sku": r.get("sku"),
                "oos_pct": r.get("oos_pct"),
                "delta_oos_pct": r.get("delta_oos_pct"),
                "delta_oos_pp": r.get("delta_oos_pp"),
                "oos_pincodes": r.get("oos_pincodes"),
                "total_pincodes": r.get("total_pincodes"),
            }
        )

    # ✅ IMPORTANT: do NOT truncate stockouts to top_n; user wants ALL >50% visible
    for p in list(platforms.keys()):
        platforms[p] = sorted(
            platforms[p],
            key=lambda x: (_to_float(x.get("oos_pct")) or 0.0, _to_float(x.get("oos_pincodes")) or 0.0),
            reverse=True,
        )

    return platforms


# ============================================================
# Evidence builder: PRICE/PROMO (deep scan)
# ============================================================
def _iter_dicts_deep(obj: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if isinstance(obj, dict):
        out.append(obj)
        for v in obj.values():
            out.extend(_iter_dicts_deep(v))
    elif isinstance(obj, list):
        for it in obj:
            out.extend(_iter_dicts_deep(it))
    return out


def _find_candidate_price_promo_rows(stage1_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for d in _iter_dicts_deep(stage1_payload):
        if not isinstance(d, dict):
            continue
        if "platform" not in d or "sku" not in d:
            continue
        signal_keys = {
            "price_change_pct", "disc_change_pp", "disc_pct",
            "promo_flag", "is_new_promo", "is_price_change",
            "prev_price", "price", "prev_disc_pct",
            "brand",
        }
        if set(d.keys()).intersection(signal_keys):
            candidates.append(d)

    # de-dupe
    seen = set()
    uniq: List[Dict[str, Any]] = []
    for r in candidates:
        plat = _norm_platform(r.get("platform")).lower()
        sku = _norm_sku(r.get("sku")).lower()
        k = (
            plat,
            sku,
            _as_str(r.get("price_change_pct")),
            _as_str(r.get("disc_change_pp")),
            _as_str(r.get("disc_pct")),
            _as_str(r.get("is_new_promo")),
            _as_str(r.get("brand")),
        )
        if k in seen:
            continue
        seen.add(k)
        uniq.append(r)
    return uniq


def _build_platform_price_promo_evidence(
    stage1_payload: Dict[str, Any], cfg: ActionAgentConfig
) -> Dict[str, List[Dict[str, Any]]]:
    rows = _find_candidate_price_promo_rows(stage1_payload)
    platforms: Dict[str, List[Dict[str, Any]]] = {}

    for r in rows:
        plat = _norm_platform(r.get("platform"))
        sku = _norm_sku(r.get("sku"))
        if not plat or not sku:
            continue

        platforms.setdefault(plat, []).append(
            {
                "type": "price_promo",
                "sku": sku,
                "brand": r.get("brand"),
                "price_change_pct": r.get("price_change_pct"),
                "price": r.get("price"),
                "prev_price": r.get("prev_price"),
                "disc_pct": r.get("disc_pct"),
                "prev_disc_pct": r.get("prev_disc_pct"),
                "disc_change_pp": r.get("disc_change_pp"),
                "is_new_promo": r.get("is_new_promo"),
                "is_price_change": r.get("is_price_change"),
                "promo_flag": r.get("promo_flag"),
            }
        )

    # price/promo can stay top-N for readability
    top_n = int(getattr(cfg, "top_n", 5))

    def score(x: Dict[str, Any]) -> float:
        pc = abs(_to_float(x.get("price_change_pct")) or 0.0)
        dc = abs(_to_float(x.get("disc_change_pp")) or 0.0)
        dcur = abs(_to_float(x.get("disc_pct")) or 0.0)
        return pc * 1.0 + dc * 1.2 + dcur * 0.2

    for p in list(platforms.keys()):
        platforms[p] = sorted(platforms[p], key=score, reverse=True)[:top_n]

    return platforms


# ============================================================
# Merge evidence per platform
# ============================================================
def _build_platform_evidence(stage1_payload: Dict[str, Any], cfg: ActionAgentConfig) -> Dict[str, List[Dict[str, Any]]]:
    preferred = set(_get_preferred_alert_types(cfg))
    merged: Dict[str, List[Dict[str, Any]]] = {}

    if "stockouts" in preferred:
        s = _build_platform_stockout_evidence(stage1_payload, cfg)
        for plat, rows in s.items():
            merged.setdefault(plat, []).extend(rows)

    if "price_promo" in preferred:
        pp = _build_platform_price_promo_evidence(stage1_payload, cfg)
        for plat, rows in pp.items():
            merged.setdefault(plat, []).extend(rows)

    return merged


# ============================================================
# Action builder (platform-specific) based on evidence types
# ============================================================
def derive_platform_action(evidence: List[Dict[str, Any]], cfg: ActionAgentConfig) -> str:
    min_oos = float(getattr(cfg, "min_oos_pct_for_reporting", 50.0))
    price_thr = float(getattr(cfg, "price_change_threshold_pct", 1.0))
    disc_thr = float(getattr(cfg, "disc_change_threshold_pp", 2.0))

    has_stockouts = any(
        (_to_float(e.get("oos_pct")) is not None and (_to_float(e.get("oos_pct")) or 0.0) > min_oos)
        for e in evidence
    )

    has_price_change = any(
        (_to_float(e.get("price_change_pct")) is not None and abs(_to_float(e.get("price_change_pct")) or 0.0) >= price_thr)
        for e in evidence
    )

    has_discount_change = any(
        (_to_float(e.get("disc_change_pp")) is not None and abs(_to_float(e.get("disc_change_pp")) or 0.0) >= disc_thr)
        or bool(e.get("is_new_promo"))
        for e in evidence
    )

    has_new_promo = any(bool(e.get("is_new_promo")) for e in evidence)

    actions: List[str] = []

    if has_stockouts:
        actions.append("Restore availability for impacted SKUs in highest-OOS pincodes; validate inventory feed and listing/catalogue mapping")

    if has_price_change:
        actions.append("Audit price movements vs last period; verify MRP/pack-size mapping and correct unintended hikes/drops")

    if has_discount_change:
        if has_new_promo:
            actions.append("Validate new promo setup (funding, eligibility, coupon stacking) and ensure margin guardrails")
        else:
            actions.append("Review discount changes and promo funding; confirm intent and margins")

    if not actions:
        actions.append("Review platform signals; validate data freshness and investigate top impacted SKUs")

    return " | ".join(actions)


# ============================================================
# Threshold gating (what counts as breached)
# ============================================================
def _breach_thresholds_for_platform(evidence: List[Dict[str, Any]], cfg: ActionAgentConfig) -> bool:
    oos_high = float(getattr(cfg, "alert_stockout_oos_pct_high", 90.0))
    delta_high = float(getattr(cfg, "alert_delta_oos_pp_high", 5.0))

    price_change_high = float(getattr(cfg, "alert_price_change_pct_high", 5.0))
    disc_change_high = float(getattr(cfg, "alert_disc_change_pp_high", 5.0))
    disc_current_high = float(getattr(cfg, "alert_disc_pct_high", 30.0))

    for e in evidence:
        et = _as_str(e.get("type")).strip()

        if not et:
            if "oos_pct" in e or "oos_pincodes" in e:
                et = "stockouts"
            elif "price_change_pct" in e or "disc_change_pp" in e or "disc_pct" in e:
                et = "price_promo"

        if et == "stockouts":
            oos = _to_float(e.get("oos_pct")) or 0.0

            dlt = _to_float(e.get("delta_oos_pct"))
            if dlt is None:
                dlt = _to_float(e.get("delta_oos_pp"))
            dlt = dlt or 0.0

            if oos >= oos_high or dlt >= delta_high:
                return True

        elif et == "price_promo":
            pc = _to_float(e.get("price_change_pct"))
            dpp = _to_float(e.get("disc_change_pp"))
            disc = _to_float(e.get("disc_pct"))

            if pc is not None and abs(pc) >= price_change_high:
                return True
            if dpp is not None and abs(dpp) >= disc_change_high:
                return True
            if disc is not None and disc >= disc_current_high:
                return True

            if bool(e.get("is_new_promo")):
                return True

    return False


# ============================================================
# Formatting helpers
# ============================================================
def _fmt_pct(x: Any) -> str:
    v = _to_float(x)
    return "NA" if v is None else f"{v:.1f}%"


def _fmt_pp(x: Any) -> str:
    v = _to_float(x)
    if v is None:
        return "NA"
    sign = "+" if v > 0 else ""
    return f"{sign}{v:.1f} pp"


# ============================================================
# Deterministic fallback: Insight + Action + Evidence
# ============================================================
def _derive_platform_insight(platform: str, evidence: List[Dict[str, Any]], cfg: ActionAgentConfig) -> str:
    min_oos = float(getattr(cfg, "min_oos_pct_for_reporting", 50.0))

    stock = [
        e for e in evidence
        if (_as_str(e.get("type")) == "stockouts" or "oos_pct" in e)
        and (_to_float(e.get("oos_pct")) is not None)
        and (_to_float(e.get("oos_pct")) > min_oos)
    ]

    price_promo = [
        e for e in evidence
        if (_as_str(e.get("type")) == "price_promo"
        or "price_change_pct" in e
        or "disc_pct" in e
        or "disc_change_pp" in e)
    ]

    if stock and price_promo:
        top = max((_to_float(e.get("oos_pct")) or 0.0) for e in stock)
        return f"Mixed issue: high stockouts (up to {top:.0f}% OOS) along with notable price/discount movement vs last period."

    if stock:
        top = max((_to_float(e.get("oos_pct")) or 0.0) for e in stock)
        return f"High stockouts observed: multiple SKUs exceed 50% OOS (up to {top:.0f}% of pincodes)."

    if price_promo:
        return "Notable price/discount movement vs last period."

    return "Platform breached alert thresholds and requires review."


def _fallback_insight_for_platform(platform: str, evidence: List[Dict[str, Any]], cfg: ActionAgentConfig) -> ActionInsight:
    bullets: List[str] = []
    min_oos = float(getattr(cfg, "min_oos_pct_for_reporting", 50.0))

    stock_e = [
        e for e in evidence
        if (_as_str(e.get("type")) == "stockouts" or "oos_pct" in e)
        and (_to_float(e.get("oos_pct")) is not None)
        and ((_to_float(e.get("oos_pct")) or 0.0) > min_oos)
    ]
    stock_e = sorted(stock_e, key=lambda x: (_to_float(x.get("oos_pct")) or 0.0), reverse=True)

    pp_e = [
        e for e in evidence
        if (_as_str(e.get("type")) == "price_promo" or ("price_change_pct" in e or "disc_pct" in e or "disc_change_pp" in e))
    ]

    # ✅ include ALL stockouts >50% (user requirement)
    for e in stock_e:
        sku = _as_str(e.get("sku")).strip()
        if not sku:
            continue

        oos = _fmt_pct(e.get("oos_pct"))
        pins = _to_float(e.get("oos_pincodes")) or 0
        tot = _to_float(e.get("total_pincodes")) or 0
        denom = f"{int(pins)}/{int(tot)} pincodes" if tot else "pincodes NA"

        # ✅ delta only if numeric (avoid "NA vs prev")
        dlt_raw = e.get("delta_oos_pct") if e.get("delta_oos_pct") is not None else e.get("delta_oos_pp")
        dlt_val = _to_float(dlt_raw)

        if dlt_val is None:
            bullets.append(f"{sku}: OOS {oos} ({denom})")
        else:
            bullets.append(f"{sku}: OOS {oos} ({denom}), Δ OOS {_fmt_pp(dlt_val)} vs prev")

    # add a few price/promo lines after (keep it readable)
    max_pp_lines = int(getattr(cfg, "max_price_promo_lines_in_action_report", 4))
    added_pp = 0
    for e in pp_e:
        if added_pp >= max_pp_lines:
            break

        sku = _as_str(e.get("sku")).strip()
        if not sku:
            continue

        brand = _as_str(e.get("brand")).strip()
        brand_txt = f" [{brand}]" if brand else ""

        pc = _to_float(e.get("price_change_pct"))
        dpp = _to_float(e.get("disc_change_pp"))
        disc = _to_float(e.get("disc_pct"))

        parts = []
        if pc is not None:
            sign = "+" if pc > 0 else ""
            parts.append(f"Price {sign}{pc:.1f}%")
        if disc is not None:
            parts.append(f"Disc {disc:.1f}%")
        if dpp is not None:
            parts.append(f"Δ Disc {_fmt_pp(dpp)}")
        if bool(e.get("is_new_promo")):
            parts.append("New promo")

        if parts:
            bullets.append(f"{sku}{brand_txt}: " + ", ".join(parts))
            added_pp += 1

    # ✅ Never add "evidence available" placeholders
    bullets = [b for b in bullets if b][:50]  # allow many stockout lines; still safe cap

    if not bullets:
        bullets = ["No qualifying evidence after filters (check Stage 1 payload / thresholds)."]

    insight_line = _derive_platform_insight(platform, evidence, cfg)
    action = derive_platform_action(evidence, cfg)

    return ActionInsight(
        platform=platform,
        title=insight_line,  # ✅ this becomes "Insight:"
        severity="high",
        trend="worsening",
        urgency="today",
        owner="E-Commerce Ops",
        recommended_action=action,
        evidence=bullets,
        drilldown_suggestion="Drill down by pincode → SKU to identify concentrated OOS pockets; prioritize top revenue SKUs first.",
    )


# ============================================================
# LLM call (ONE request total)
# ============================================================
def _call_openai_batch(
    breached_platforms: List[str],
    evidence_by_platform: Dict[str, List[Dict[str, Any]]],
    cfg: ActionAgentConfig,
) -> List[ActionInsight]:
    from openai import OpenAI
    from openai import RateLimitError, APIError, APITimeoutError

    client = OpenAI()

    payload = {
        "breached_platforms": breached_platforms,
        "thresholds": {
            "stockouts": {
                "oos_high": float(getattr(cfg, "alert_stockout_oos_pct_high", 90.0)),
                "delta_pp_high": float(getattr(cfg, "alert_delta_oos_pp_high", 5.0)),
            },
            "price_promo": {
                "price_change_pct_high": float(getattr(cfg, "alert_price_change_pct_high", 5.0)),
                "disc_change_pp_high": float(getattr(cfg, "alert_disc_change_pp_high", 5.0)),
                "disc_pct_high": float(getattr(cfg, "alert_disc_pct_high", 30.0)),
            },
        },
        "preferred_alert_types": _get_preferred_alert_types(cfg),
        "metric_definitions": _metric_definitions(),
        "evidence_by_platform": evidence_by_platform,
        "output_rules": {
            "one_insight_per_platform_max": True,
            "evidence_bullets_min": 2,
            "evidence_bullets_max": 6,
        },
    }

    max_retries = 5
    base_sleep = 2.0
    last_err: Exception | None = None

    for attempt in range(max_retries):
        try:
            resp = client.responses.parse(
                model=getattr(cfg, "llm_model", "gpt-4o-mini"),
                input=[
                    {"role": "system", "content": BATCH_ACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                text_format=BatchActionInsights,
                timeout=60,
            )

            parsed: BatchActionInsights = resp.output_parsed  # type: ignore
            insights = parsed.insights or []

            allowed = {p.lower() for p in breached_platforms}
            out: List[ActionInsight] = []
            seen = set()

            for i in insights:
                p = (i.platform or "").strip()
                if not p or p.lower() not in allowed:
                    continue
                if p.lower() in seen:
                    continue
                if not (i.evidence or []):
                    continue
                # require title (insight)
                if not (i.title or "").strip():
                    continue
                seen.add(p.lower())
                out.append(i)

            return out

        except (RateLimitError, APITimeoutError, APIError) as e:
            last_err = e
            sleep_s = base_sleep * (2 ** attempt) + random.random()
            time.sleep(sleep_s)
            continue
        except Exception as e:
            last_err = e
            break

    if last_err:
        raise last_err
    return []


# ============================================================
# Rendering (Insight -> Action -> Evidence)
# ============================================================
def _render_action_text(insights: List[Dict[str, Any]]) -> str:
    """
    Stage 2 output format (FINAL):
    PLATFORM
    Insight
    Data points
    """

    if not insights:
        return "No platform breached alert thresholds."

    lines: List[str] = []

    for it in insights:
        platform = (it.get("platform") or "").strip()
        insight = (it.get("title") or "").strip()
        evidence = it.get("evidence") or []

        if platform:
            lines.append(f"PLATFORM: {platform}")

        if insight:
            lines.append(f"Insight: {insight}")

        lines.append("Data points:")

        for e in evidence:
            if isinstance(e, str):
                # safety cleanup
                cleaned = (
                    e.replace("Δ OOS NA vs prev", "")
                     .replace(", Δ OOS NA vs prev", "")
                     .strip()
                )
                if cleaned:
                    lines.append(f"- {cleaned}")
            else:
                lines.append(f"- {e}")

        lines.append("")

    return "\n".join(lines).strip()


# ============================================================
# Public entrypoint
# ============================================================
def generate_action_report(stage1_payload: Dict[str, Any], cfg: ActionAgentConfig) -> Dict[str, Any]:
    generated_at = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    scope = f"Stage 2 — Platform-Specific Actions (top_n={cfg.top_n})"

    platform_evidence = _build_platform_evidence(stage1_payload, cfg)

    breached = [p for p, ev in platform_evidence.items() if _breach_thresholds_for_platform(ev, cfg)]
    breached = sorted([p for p in breached if p])

    insights: List[ActionInsight] = []
    llm_error: Optional[str] = None

    if breached and getattr(cfg, "use_llm", True):
        try:
            evidence_subset = {p: platform_evidence[p] for p in breached}
            insights = _call_openai_batch(breached, evidence_subset, cfg)

            allowed = {p.lower() for p in breached}
            insights = [
                i for i in insights
                if i.platform and i.platform.lower() in allowed
                and (i.evidence or [])
                and (i.title or "").strip()
            ]
        except Exception as e:
            llm_error = f"{type(e).__name__}: {e}"
            insights = []

    # ✅ fallback if LLM disabled or empty
    if breached and not insights:
        insights = [_fallback_insight_for_platform(p, platform_evidence.get(p, []), cfg) for p in breached]

    action_text = _render_action_text(
        insights=[i.model_dump() for i in insights],
    )

    return {
        "generated_at": generated_at,
        "scope": scope,
        "metric_definitions": _metric_definitions(),
        "insights": [i.model_dump() for i in insights],
        "action_text": action_text,
        "debug": {
            "breached_platforms": breached,
            "platform_evidence_counts": {p: len(v) for p, v in platform_evidence.items()},
            "llm_error": llm_error,
            "use_llm": bool(getattr(cfg, "use_llm", True)),
            "model": getattr(cfg, "llm_model", None),
        },
    }
