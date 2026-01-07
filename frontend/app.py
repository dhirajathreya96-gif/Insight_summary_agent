import streamlit as st
import requests
import json
import html
import pandas as pd
import re


def html_to_text(html_str: str) -> str:
    """
    Convert HTML email content into readable plain text for UI previews / copy-paste.
    Lightweight on purpose (no external deps).
    """
    if not html_str:
        return ""

    s = str(html_str)

    s = re.sub(r"(?is)<(script|style).*?>.*?</\1>", "", s)

    s = re.sub(r"(?i)<br\s*/?>", "\n", s)
    s = re.sub(r"(?i)</p\s*>", "\n\n", s)
    s = re.sub(r"(?i)</div\s*>", "\n", s)
    s = re.sub(r"(?i)</li\s*>", "\n", s)
    s = re.sub(r"(?i)</tr\s*>", "\n", s)

    s = re.sub(r"(?s)<[^>]+>", "", s)

    s = html.unescape(s)

    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)

    return s.strip()


st.set_page_config(page_title="Insight Summary Agent", layout="wide")

st.title("Insight Summary Agent (Spoggle)")
st.caption("Accuracy-first insight summaries from Spoggle E-Commerce Analytics CSV exports.")

API_URL = st.sidebar.text_input("Backend API URL", value="http://localhost:8000")

# ============================================================
# Shared upload controls
# ============================================================
st.sidebar.header("Configuration (Stage 1)")
top_n = st.sidebar.slider("Top N items", min_value=3, max_value=10, value=5, step=1)
require_serviced = st.sidebar.checkbox("Only count Service_Status = serviced", value=True)
price_thr = st.sidebar.number_input("Price change threshold (%)", value=1.0, min_value=0.0, step=0.5)

st.sidebar.subheader("Brands")
own_brands_raw = st.sidebar.text_area("Own brand(s) (one per line)", value="Oetker\nDr. Oetker")
own_brands = [b.strip() for b in own_brands_raw.splitlines() if b.strip()]
st.sidebar.caption("Competitors are automatically treated as: all brands NOT in your own brand list.")

# ============================================================
# Stage 2 controls
# ============================================================
st.sidebar.header("Configuration (Stage 2: Action Agent)")

use_llm = st.sidebar.checkbox("Use OpenAI (LLM) for action intelligence", value=True)
llm_model = st.sidebar.text_input("OpenAI model (backend)", value="gpt-4o-mini", help="Must be supported by your OpenAI project/account.")

st.sidebar.subheader("Alert thresholds (defaults are safe)")
alert_stockout_oos_pct_high = st.sidebar.number_input("High severity: SKU OOS% >=", value=90.0, min_value=0.0, max_value=100.0, step=1.0)
alert_stockout_oos_pct_medium = st.sidebar.number_input("Medium severity: SKU OOS% >=", value=70.0, min_value=0.0, max_value=100.0, step=1.0)
alert_delta_oos_pp_high = st.sidebar.number_input("High severity: Δ OOS (pp) >=", value=5.0, min_value=0.0, step=0.5)
alert_delta_oos_pp_medium = st.sidebar.number_input("Medium severity: Δ OOS (pp) >=", value=2.0, min_value=0.0, step=0.5)

st.sidebar.subheader("Preferences (optional)")
preferred_alert_types_raw = st.sidebar.text_input(
    "Preferred alert types (comma-separated)",
    value="stockouts, price_promo, competition",
)
preferred_alert_types = [x.strip() for x in preferred_alert_types_raw.split(",") if x.strip()]

suppression_rules_raw = st.sidebar.text_area(
    "Suppression rules (one per line, optional)",
    value="",
    help="Example: 'ignore city=NA' or 'ignore sku contains sample'",
)
suppression_rules = [x.strip() for x in suppression_rules_raw.splitlines() if x.strip()]

role_views_raw = st.sidebar.text_input(
    "Role views (comma-separated, optional)",
    value="Supply Chain, E-Commerce Ops, Sales",
)
role_views = [x.strip() for x in role_views_raw.split(",") if x.strip()]


# ============================================================
# Upload section
# ============================================================
st.header("1) Upload input files")
c1, c2, c3 = st.columns(3)
with c1:
    current_file = st.file_uploader("Current period CSV", type=["csv"], key="cur")

with c2:
    previous_files = st.file_uploader(
        "Previous period CSVs (optional) — upload multiple (older → newer)",
        type=["csv"],
        accept_multiple_files=True,
        key="prev_multi",
    )
    previous_files = previous_files or []

with c3:
    pincode_map = st.file_uploader("Pincode → City mapping CSV (optional but recommended)", type=["csv"], key="pinmap")


def build_client_email_html(text_summary: str, key_takeaways: str = "", title: str = "Spoggle Insight Summary") -> str:
    if not text_summary:
        return ""

    lines = text_summary.splitlines()
    esc_lines = [html.escape(l) for l in lines]

    header_set = {
        "INSIGHT SUMMARY (Accuracy-first)",
        "EXECUTIVE SUMMARY",
        "TOP OOS PINCODES (Platform-wise)",
        "STOCK-OUT INSIGHTS (Platform-wise)",
        "TOP SKUs OOS IN LAST 5 SCRAPING INSTANCES (Historical)",
        "COMPETITIVE INTELLIGENCE",
        "DATA QUALITY NOTES",
        "— Generated automatically by Insight Summary Agent",
    }

    formatted = []
    for l in esc_lines:
        raw = html.unescape(l).strip()

        if raw in header_set and not raw.startswith("—"):
            formatted.append(
                f"<div style='margin-top:18px; margin-bottom:8px; font-weight:700; font-size:14px;'>{l}</div>"
            )
            continue

        if raw == "":
            formatted.append("<div style='height:10px;'></div>")
        elif raw.startswith("- "):
            formatted.append(f"<div style='margin:6px 0 0 0;'>{l}</div>")
        elif raw.startswith("  •"):
            formatted.append(f"<div style='margin:3px 0 0 16px;'>{l}</div>")
        elif raw.startswith("    "):
            formatted.append(f"<div style='margin:2px 0 0 28px; color:#333;'>{l}</div>")
        else:
            formatted.append(f"<div style='margin:2px 0 0 0;'>{l}</div>")

    body = "\n".join(formatted)
    kt = html.escape(key_takeaways.strip()) if key_takeaways else ""

    kt_block = ""
    if kt:
        kt_block = f"""
        <div style="border:1px solid #e8e8e8; border-radius:10px; padding:12px 14px; background:#fafafa; margin-bottom:12px;">
          <div style="font-weight:700; margin-bottom:6px;">Key takeaways</div>
          <div style="color:#222;">{kt}</div>
        </div>
        """

    wrapper = f"""
    <div style="font-family: Arial, sans-serif; font-size: 13.5px; line-height: 1.45; color: #111;">
      <div style="font-size:16px; font-weight:700; margin-bottom:6px;">{html.escape(title)}</div>
      <div style="color:#555; margin-bottom:14px;">
        Summary generated from Spoggle E-Commerce Analytics exports.
      </div>
      {kt_block}
      <div style="border:1px solid #e6e6e6; border-radius:10px; padding:14px;">
        {body}
      </div>
      <div style="margin-top:10px; color:#666; font-size:12px;">
        Tip: reply with any platform/SKU you want a deeper drill-down on.
      </div>
    </div>
    """
    return wrapper


tab1, tab2 = st.tabs(["Stage 1: Insight Summary", "Stage 2: Actionable Insights"])


# ----------------------------
# TAB 1: Stage 1
# ----------------------------
with tab1:
    st.header("2) Generate summary")
    if st.button("Generate Insight Summary", type="primary", disabled=current_file is None, key="btn_stage1"):
        files = []
        files.append(("current_file", ("current.csv", current_file.getvalue(), "text/csv")))

        for f in previous_files:
            files.append(("previous_files", (f.name, f.getvalue(), "text/csv")))

        if pincode_map is not None:
            files.append(("pincode_map_file", ("pincode_map.csv", pincode_map.getvalue(), "text/csv")))

        data = {
            "top_n": str(top_n),
            "require_serviced": str(require_serviced),
            "price_change_threshold_pct": str(price_thr),
            "own_brands_json": json.dumps(own_brands),
        }

        try:
            resp = requests.post(f"{API_URL}/generate", files=files, data=data, timeout=180)
        except Exception as e:
            st.error(f"Failed to call backend: {e}")
            st.stop()

        if resp.status_code != 200:
            st.error(resp.text)
            st.stop()

        st.session_state["generated_stage1"] = resp.json()

    if "generated_stage1" in st.session_state:
        out = st.session_state["generated_stage1"]

        st.subheader("Preview (Text)")
        st.code(out.get("text_summary", ""))

        payload = out.get("payload") or {}
        history = payload.get("history") or {}
        inst = history.get("instances") or []
        overall = history.get("overall_oos_pct") or []

        st.subheader("Timeline: Overall OOS%")
        if inst and overall and len(inst) == len(overall):
            df = pd.DataFrame({"instance": inst, "overall_oos_pct": overall})
            st.line_chart(df.set_index("instance")["overall_oos_pct"])
        else:
            st.info("Timeline unavailable (upload at least 1 previous file with crawled_date).")

        key_takeaways = payload.get("key_takeaways") or ""
        email_html = build_client_email_html(out.get("text_summary", ""), key_takeaways=key_takeaways)

        st.subheader("Preview (Client-ready Email HTML)")
        st.components.v1.html(email_html, height=520, scrolling=True)

        st.markdown("### Copy-paste options")
        st.text_area("Plain text (copy into email)", value=out.get("text_summary", ""), height=220)

        # ✅ NEW: CSV export
        st.subheader("Download: Platform × Pincode Insights (CSV)")
        if st.button("Generate & Download CSV", key="btn_csv_export", disabled=current_file is None):
            files = []
            files.append(("current_file", ("current.csv", current_file.getvalue(), "text/csv")))

            for f in previous_files:
                files.append(("previous_files", (f.name, f.getvalue(), "text/csv")))

            if pincode_map is not None:
                files.append(("pincode_map_file", ("pincode_map.csv", pincode_map.getvalue(), "text/csv")))

            data = {
                "top_n": str(top_n),
                "require_serviced": str(require_serviced),
                "price_change_threshold_pct": str(price_thr),
                "promotion_change_threshold_pct": "1.0",
                "own_brands_json": json.dumps(own_brands),
            }

            try:
                r = requests.post(f"{API_URL}/export-platform-pincode-insights", files=files, data=data, timeout=240)
            except Exception as e:
                st.error(f"Failed to call backend: {e}")
                r = None

            if r is None:
                st.stop()

            if r.status_code != 200:
                st.error(r.text)
            else:
                st.download_button(
                    label="Download platform_pincode_insights.csv",
                    data=r.content,
                    file_name="platform_pincode_insights.csv",
                    mime="text/csv",
                )

        st.header("3) Deliver")
        st.subheader("Email")
        to_emails = st.text_input("Recipients (comma-separated)", value="", key="to_emails_stage1")
        subject = st.text_input("Subject", value="[Spoggle] Insight Summary", key="subject_stage1")

        if st.button("Send Email", disabled=not to_emails.strip(), key="btn_send_stage1"):
            payload_req = {
                "subject": subject,
                "to_emails": [x.strip() for x in to_emails.split(",") if x.strip()],
                "html_body": email_html,
                "text_body": out.get("text_summary", ""),
            }
            r = requests.post(f"{API_URL}/send-email", json=payload_req, timeout=60)
            if r.status_code == 200:
                st.success("Email sent ✅")
            else:
                st.error(r.text)


# ----------------------------
# TAB 2: Stage 2
# ----------------------------
with tab2:
    st.header("2) Generate actionable insights (Stage 2)")

    st.caption(
        "This tab keeps Stage 1 unchanged and generates a separate action-oriented report. "
        "It can use OpenAI (server-side) via OPENAI_API_KEY."
    )

    if st.button("Generate Actionable Insights", type="primary", disabled=current_file is None, key="btn_stage2"):
        files = []
        files.append(("current_file", ("current.csv", current_file.getvalue(), "text/csv")))

        for f in previous_files:
            files.append(("previous_files", (f.name, f.getvalue(), "text/csv")))

        if pincode_map is not None:
            files.append(("pincode_map_file", ("pincode_map.csv", pincode_map.getvalue(), "text/csv")))

        data = {
            "top_n": str(top_n),
            "require_serviced": str(require_serviced),
            "price_change_threshold_pct": str(price_thr),
            "promotion_change_threshold_pct": "1.0",
            "own_brands_json": json.dumps(own_brands),

            "alert_stockout_oos_pct_high": str(alert_stockout_oos_pct_high),
            "alert_stockout_oos_pct_medium": str(alert_stockout_oos_pct_medium),
            "alert_delta_oos_pp_high": str(alert_delta_oos_pp_high),
            "alert_delta_oos_pp_medium": str(alert_delta_oos_pp_medium),

            "preferred_alert_types_json": json.dumps(preferred_alert_types),
            "suppression_rules_json": json.dumps(suppression_rules),
            "role_views_json": json.dumps(role_views),

            "use_llm": str(use_llm),
            "llm_model": llm_model.strip() or "gpt-4o-mini",
        }

        try:
            resp = requests.post(f"{API_URL}/generate-actions", files=files, data=data, timeout=240)
        except Exception as e:
            st.error(f"Failed to call backend: {e}")
            st.stop()

        if resp.status_code != 200:
            st.error(resp.text)
            st.stop()

        st.session_state["generated_stage2"] = resp.json()

    if "generated_stage2" in st.session_state:
        out2 = st.session_state["generated_stage2"]
        action_text = out2.get("action_text", "")
        action_html = out2.get("action_html", "")
        action_payload = out2.get("action_payload") or {}

        st.subheader("Preview (Action Report - Text)")
        st.code(action_text)

        preview_text = html_to_text(action_html)
        st.subheader("Preview (Text)")
        st.code(preview_text)

        st.subheader("Preview (Email HTML)")
        st.components.v1.html(action_html, height=520, scrolling=True)

        st.subheader("Structured Insights")
        insights = action_payload.get("insights") or []
        if insights:
            st.dataframe(pd.DataFrame(insights))
        else:
            st.info("No actionable insights returned (try increasing Top N or relaxing thresholds).")

        st.header("3) Deliver (Email)")
        to_emails2 = st.text_input("Recipients (comma-separated)", value="", key="to_emails_stage2")
        subject2 = st.text_input("Subject", value="[Spoggle] Actionable Insights", key="subject_stage2")

        if st.button("Send Action Email", disabled=not to_emails2.strip(), key="btn_send_stage2"):
            payload_req = {
                "subject": subject2,
                "to_emails": [x.strip() for x in to_emails2.split(",") if x.strip()],
                "html_body": action_html,
                "text_body": action_text,
            }
            r = requests.post(f"{API_URL}/send-email", json=payload_req, timeout=60)
            if r.status_code == 200:
                st.success("Action email sent ✅")
            else:
                st.error(r.text)
