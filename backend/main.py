# backend/main.py
from __future__ import annotations

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
import pandas as pd
import io
import json

from common.config import InsightAgentConfig
from common.agent import run_insight_agent, serialize
from common.render import render_bullets, render_html_from_text
from backend.utils.emailer import send_email

app = FastAPI(title="Insight Summary Agent API", version="1.1.0")


class GenerateResponse(BaseModel):
    text_summary: str
    html_summary: str
    payload: Dict[str, Any]


class SendEmailRequest(BaseModel):
    subject: str
    to_emails: List[str]
    html_body: str
    text_body: Optional[str] = None


@app.get("/health")
def health():
    return {"ok": True}


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


@app.post("/generate", response_model=GenerateResponse)
async def generate(
    current_file: UploadFile = File(...),

    # ✅ multi previous files under the same field name
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
    # -----------------------------
    # Read current
    # -----------------------------
    try:
        cur_bytes = await current_file.read()
        cur_df = pd.read_csv(io.BytesIO(cur_bytes), low_memory=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read current CSV: {e}")

    # -----------------------------
    # Read previous list
    # -----------------------------
    prev_dfs: List[pd.DataFrame] = []
    for pf in (previous_files or []):
        try:
            b = await pf.read()
            df = pd.read_csv(io.BytesIO(b), low_memory=False)
            prev_dfs.append(df)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read previous CSV '{getattr(pf, 'filename', 'file')}': {e}")

    # -----------------------------
    # Read pincode map
    # -----------------------------
    pincode_map_df = None
    if pincode_map_file is not None:
        try:
            pm_bytes = await pincode_map_file.read()
            pincode_map_df = pd.read_csv(io.BytesIO(pm_bytes), dtype=str, low_memory=False)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read pincode mapping CSV: {e}")

    # -----------------------------
    # Parse config
    # -----------------------------
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

    # -----------------------------
    # Run agent
    # -----------------------------
    try:
        payload = run_insight_agent(
            current_df=cur_df,
            previous_dfs=prev_dfs,   # ✅ list
            config=cfg,
            pincode_map_df=pincode_map_df,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent failed: {e}")

    # -----------------------------
    # Render
    # -----------------------------
    text_summary = render_bullets(payload, top_n=cfg.top_n)
    html_summary = render_html_from_text(text_summary)

    safe_payload = serialize(payload)

    return GenerateResponse(
        text_summary=text_summary,
        html_summary=html_summary,
        payload=safe_payload,
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
