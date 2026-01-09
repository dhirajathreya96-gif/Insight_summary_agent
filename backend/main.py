from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
import pandas as pd
import io
import json
import re

from common.config import InsightAgentConfig
from common.agent import run_insight_agent, serialize, export_platform_pincode_insights
from common.render import render_bullets, render_html_from_text
from backend.utils.emailer import send_email

# ✅ Stage 2
from common.action_config import ActionAgentConfig
from common.action_agent import generate_action_report
from common.render import render_action_email_html


app = FastAPI(title="Insight Summary Agent API", version="1.2.0")


class GenerateResponse(BaseModel):
    text_summary: str
    html_summary: str
    payload: Dict[str, Any]


class GenerateActionResponse(BaseModel):
    action_text: str
    action_html: str
    action_payload: Dict[str, Any]
    stage1_payload: Dict[str, Any]


class SendEmailRequest(BaseModel):
    subject: str
    to_emails: List[str]
    html_body: str
    text_body: Optional[str] = None


@app.get("/health")
def health():
    return {"ok": True}


def _sanitize_no_critical(s: str) -> str:
    """
    Ensure the word 'critical' never appears in user-facing outputs.
    """
    if not s:
        return s
    return re.sub(r"\bcritical\b", "high-priority", s, flags=re.IGNORECASE)


def _parse_bool(v: Union[str, bool, int, None], default: bool = True) -> bool:
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return bool(v)
    s = str(v).strip().lower()
    if s in {"true", "1", "yes", "y", "on"}:
        return True
    if s in {"false", "0", "no", "n", "off"}:
        return False
    return default


def _parse_list(val: Optional[str]) -> List[Any]:
    if not val:
        return []
    raw = val.strip()
    if not raw:
        return []

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, str) and parsed.strip():
            return [parsed.strip()]
    except Exception:
        pass

    items: List[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",") if p.strip()]
        items.extend(parts)

    return items


# ============================================================
# STAGE 1 (unchanged logic; only sanitization applied)
# ============================================================
@app.post("/generate", response_model=GenerateResponse)
async def generate(
    current_file: UploadFile = File(...),
    previous_files: List[UploadFile] = File(default=[]),
    pincode_map_file: Optional[UploadFile] = File(None),

    top_n: int = Form(5),
    require_serviced: Union[bool, str] = Form(True),
    price_change_threshold_pct: float = Form(1.0),
    promotion_change_threshold_pct: float = Form(1.0),

    competitor_brands_json: str = Form("[]"),
    priority_categories_json: str = Form("[]"),
    priority_skus_json: str = Form("[]"),
    own_brands_json: str = Form("[]"),
):
    # Read current
    try:
        cur_bytes = await current_file.read()
        cur_df = pd.read_csv(io.BytesIO(cur_bytes), low_memory=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read current CSV: {e}")

    # Read previous list
    prev_dfs: List[pd.DataFrame] = []
    for pf in (previous_files or []):
        try:
            b = await pf.read()
            df = pd.read_csv(io.BytesIO(b), low_memory=False)
            prev_dfs.append(df)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to read previous CSV '{getattr(pf, 'filename', 'file')}': {e}",
            )

    # Read pincode map
    pincode_map_df = None
    if pincode_map_file is not None:
        try:
            pm_bytes = await pincode_map_file.read()
            pincode_map_df = pd.read_csv(io.BytesIO(pm_bytes), dtype=str, low_memory=False)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read pincode mapping CSV: {e}")

    # Parse config
    competitor_brands = _parse_list(competitor_brands_json)
    priority_categories = _parse_list(priority_categories_json)
    priority_skus = _parse_list(priority_skus_json)
    own_brands = _parse_list(own_brands_json)

    cfg = InsightAgentConfig(
        top_n=int(top_n),
        priority_categories=priority_categories,
        priority_skus=priority_skus,
        own_brands=own_brands,
        competitor_brands=competitor_brands,
        price_change_threshold_pct=float(price_change_threshold_pct),
        promotion_change_threshold_pct=float(promotion_change_threshold_pct),
        require_serviced=_parse_bool(require_serviced, default=True),
    )

    # Run agent
    try:
        payload = run_insight_agent(
            current_df=cur_df,
            previous_dfs=prev_dfs,
            config=cfg,
            pincode_map_df=pincode_map_df,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent failed: {e}")

    # Render (sanitize user-facing text)
    text_summary = _sanitize_no_critical(render_bullets(payload, top_n=cfg.top_n))
    html_summary = _sanitize_no_critical(render_html_from_text(text_summary))

    safe_payload = serialize(payload)

    return GenerateResponse(
        text_summary=text_summary,
        html_summary=html_summary,
        payload=safe_payload,
    )


# ============================================================
# STAGE 2: Actionable insights (platform-specific; sanitize outputs)
# ✅ ONLY CHANGE: build a robust pincode_city_map list for action agent
# ============================================================
@app.post("/generate-actions", response_model=GenerateActionResponse)
async def generate_actions(
    current_file: UploadFile = File(...),
    previous_files: List[UploadFile] = File(default=[]),
    pincode_map_file: Optional[UploadFile] = File(None),

    # Stage 1 pass-through
    top_n: int = Form(5),
    require_serviced: Union[bool, str] = Form(True),
    price_change_threshold_pct: float = Form(1.0),
    promotion_change_threshold_pct: float = Form(1.0),

    competitor_brands_json: str = Form("[]"),
    priority_categories_json: str = Form("[]"),
    priority_skus_json: str = Form("[]"),
    own_brands_json: str = Form("[]"),

    # Stage 2 config
    alert_stockout_oos_pct_high: float = Form(90.0),
    alert_stockout_oos_pct_medium: float = Form(70.0),
    alert_delta_oos_pp_high: float = Form(5.0),
    alert_delta_oos_pp_medium: float = Form(2.0),

    preferred_alert_types_json: str = Form('["stockouts","price_promo","competition"]'),
    suppression_rules_json: str = Form("[]"),

    role_views_json: str = Form("[]"),
    use_llm: Union[bool, str] = Form(True),
    llm_model: str = Form("gpt-4o-mini"),
):
    # Read current
    try:
        cur_bytes = await current_file.read()
        cur_df = pd.read_csv(io.BytesIO(cur_bytes), low_memory=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read current CSV: {e}")

    # Read previous list
    prev_dfs: List[pd.DataFrame] = []
    for pf in (previous_files or []):
        try:
            b = await pf.read()
            df = pd.read_csv(io.BytesIO(b), low_memory=False)
            prev_dfs.append(df)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to read previous CSV '{getattr(pf, 'filename', 'file')}': {e}",
            )

    # Read pincode map (Stage1 passthrough dataframe + Stage2 pincode_city_map list)
    pincode_map_df = None
    pincode_city_map: Optional[List[Dict[str, Any]]] = None

    if pincode_map_file is not None:
        try:
            pm_bytes = await pincode_map_file.read()
            pincode_map_df = pd.read_csv(io.BytesIO(pm_bytes), dtype=str, low_memory=False)

            # ✅ Create list[dict] for Stage2 action agent (expects pincode, city)
            # Robust to column naming variants: pincode/pin/pin_code/postal_code/zip and city/region/district/etc.
            tmp = pincode_map_df.copy()
            cols = {str(c).strip().lower(): c for c in tmp.columns}

            pin_col = None
            city_col = None

            for k in ["pincode", "pin", "pin_code", "postal_code", "postcode", "zip"]:
                if k in cols:
                    pin_col = cols[k]
                    break

            for k in ["city", "region", "town", "district", "area"]:
                if k in cols:
                    city_col = cols[k]
                    break

            if pin_col and city_col:
                out = tmp[[pin_col, city_col]].copy()
                out.columns = ["pincode", "city"]

                out["pincode"] = (
                    out["pincode"]
                    .astype(str)
                    .str.replace(r"\.0$", "", regex=True)
                    .str.strip()
                )
                # keep leading zeros; ensure 6-digit for numeric pins when <=6
                out["pincode"] = out["pincode"].apply(
                    lambda x: x.zfill(6) if x.isdigit() and len(x) <= 6 else x
                )

                out["city"] = out["city"].astype(str).str.strip()

                out = out[(out["pincode"].str.len() > 0) & (out["city"].str.len() > 0)]
                pincode_city_map = out[["pincode", "city"]].to_dict(orient="records")
            else:
                # keep Stage1 df usage as-is; Stage2 hotspot will just not run
                pincode_city_map = None

        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read pincode mapping CSV: {e}")

    # Parse Stage 1 config
    competitor_brands = _parse_list(competitor_brands_json)
    priority_categories = _parse_list(priority_categories_json)
    priority_skus = _parse_list(priority_skus_json)
    own_brands = _parse_list(own_brands_json)

    cfg = InsightAgentConfig(
        top_n=int(top_n),
        priority_categories=priority_categories,
        priority_skus=priority_skus,
        own_brands=own_brands,
        competitor_brands=competitor_brands,
        price_change_threshold_pct=float(price_change_threshold_pct),
        promotion_change_threshold_pct=float(promotion_change_threshold_pct),
        require_serviced=_parse_bool(require_serviced, default=True),
    )

    # Run Stage 1 agent (evidence)
    try:
        payload = run_insight_agent(
            current_df=cur_df,
            previous_dfs=prev_dfs,
            config=cfg,
            pincode_map_df=pincode_map_df,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent failed: {e}")

    stage1_safe = serialize(payload)

    # Parse Stage 2 config
    preferred_alert_types = _parse_list(preferred_alert_types_json)
    suppression_rules = _parse_list(suppression_rules_json)
    role_views = _parse_list(role_views_json)

    action_cfg = ActionAgentConfig(
        top_n=int(top_n),
        alert_stockout_oos_pct_high=float(alert_stockout_oos_pct_high),
        alert_stockout_oos_pct_medium=float(alert_stockout_oos_pct_medium),
        alert_delta_oos_pp_high=float(alert_delta_oos_pp_high),
        alert_delta_oos_pp_medium=float(alert_delta_oos_pp_medium),
        preferred_alert_types=[str(x) for x in preferred_alert_types],
        suppression_rules=[str(x) for x in suppression_rules],
        role_views=[str(x) for x in role_views],
        use_llm=_parse_bool(use_llm, default=True),
        llm_model=str(llm_model).strip() if str(llm_model).strip() else "gpt-4o-mini",
    )

    # Generate Action Report (✅ pass pincode_city_map)
    try:
        action_payload = generate_action_report(
            stage1_safe,
            action_cfg,
            pincode_city_map=pincode_city_map,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Action agent failed: {e}")

    # Render Action Output (sanitize user-facing text)
    action_text = _sanitize_no_critical(action_payload.get("action_text", ""))
    action_html = _sanitize_no_critical(render_action_email_html(action_payload))

    return GenerateActionResponse(
        action_text=action_text,
        action_html=action_html,
        action_payload=action_payload,
        stage1_payload=stage1_safe,
    )


# ============================================================
# CSV Export: Platform × Pincode insights
# ============================================================
@app.post("/export-platform-pincode-insights")
async def export_platform_pincode_insights_endpoint(
    current_file: UploadFile = File(...),
    previous_files: List[UploadFile] = File(default=[]),
    pincode_map_file: Optional[UploadFile] = File(None),

    top_n: int = Form(5),
    require_serviced: Union[bool, str] = Form(True),
    price_change_threshold_pct: float = Form(1.0),
    promotion_change_threshold_pct: float = Form(1.0),

    competitor_brands_json: str = Form("[]"),
    priority_categories_json: str = Form("[]"),
    priority_skus_json: str = Form("[]"),
    own_brands_json: str = Form("[]"),
):
    # Read current
    try:
        cur_bytes = await current_file.read()
        cur_df = pd.read_csv(io.BytesIO(cur_bytes), low_memory=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read current CSV: {e}")

    # Read previous list
    prev_dfs: List[pd.DataFrame] = []
    for pf in (previous_files or []):
        try:
            b = await pf.read()
            df = pd.read_csv(io.BytesIO(b), low_memory=False)
            prev_dfs.append(df)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to read previous CSV '{getattr(pf, 'filename', 'file')}': {e}",
            )

    # Read pincode map
    pincode_map_df = None
    if pincode_map_file is not None:
        try:
            pm_bytes = await pincode_map_file.read()
            pincode_map_df = pd.read_csv(io.BytesIO(pm_bytes), dtype=str, low_memory=False)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read pincode mapping CSV: {e}")

    # Parse config
    competitor_brands = _parse_list(competitor_brands_json)
    priority_categories = _parse_list(priority_categories_json)
    priority_skus = _parse_list(priority_skus_json)
    own_brands = _parse_list(own_brands_json)

    cfg = InsightAgentConfig(
        top_n=int(top_n),
        priority_categories=priority_categories,
        priority_skus=priority_skus,
        own_brands=own_brands,
        competitor_brands=competitor_brands,
        price_change_threshold_pct=float(price_change_threshold_pct),
        promotion_change_threshold_pct=float(promotion_change_threshold_pct),
        require_serviced=_parse_bool(require_serviced, default=True),
    )

    try:
        out_df = export_platform_pincode_insights(
            current_df=cur_df,
            previous_dfs=prev_dfs,
            config=cfg,
            pincode_map_df=pincode_map_df,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {e}")

    buf = io.StringIO()
    out_df.to_csv(buf, index=False)
    buf.seek(0)

    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="platform_pincode_insights.csv"'},
    )


@app.post("/send-email")
def send_email_endpoint(req: SendEmailRequest):
    send_email(
        subject=req.subject,
        html_body=req.html_body,
        to_emails=req.to_emails,
        text_body=req.text_body,
    )
    return {"sent": True}
