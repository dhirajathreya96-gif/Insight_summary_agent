# frontend/app.py
import streamlit as st
import requests
import json
import html
import pandas as pd

st.set_page_config(page_title="Insight Summary Agent", layout="wide")

st.title("Insight Summary Agent (Spoggle)")
st.caption("Accuracy-first insight summaries from Spoggle E-Commerce Analytics CSV exports.")

API_URL = st.sidebar.text_input("Backend API URL", value="http://localhost:8000")

st.sidebar.header("Configuration")
top_n = st.sidebar.slider("Top N items", min_value=3, max_value=10, value=5, step=1)
require_serviced = st.sidebar.checkbox("Only count Service_Status = serviced", value=True)
price_thr = st.sidebar.number_input("Price change threshold (%)", value=1.0, min_value=0.0, step=0.5)

st.sidebar.subheader("Brands")
own_brands_raw = st.sidebar.text_area("Own brand(s) (one per line)", value="Oetker\nDr. Oetker")
own_brands = [b.strip() for b in own_brands_raw.splitlines() if b.strip()]
st.sidebar.caption("Competitors are automatically treated as: all brands NOT in your own brand list.")

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
    """
    Client-friendly email HTML that is safe in Gmail/Outlook.
    Uses the exact text output (no content changes), just better structure.
    """
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


st.header("2) Generate summary")
if st.button("Generate Insight Summary", type="primary", disabled=current_file is None):
    files = []
    files.append(("current_file", ("current.csv", current_file.getvalue(), "text/csv")))

    # ✅ multi previous files
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

    st.session_state["generated"] = resp.json()

if "generated" in st.session_state:
    out = st.session_state["generated"]

    st.subheader("Preview (Text)")
    st.code(out.get("text_summary", ""))

    # Timeline chart (overall OOS%) from payload
    payload = out.get("payload") or {}
    history = payload.get("history") or {}
    inst = history.get("instances") or []
    overall = history.get("overall_oos_pct") or []

    st.subheader("Timeline: Overall OOS%")
    if inst and overall and len(inst) == len(overall):
        df = pd.DataFrame({"instance": inst, "overall_oos_pct": overall})
        # Streamlit likes datetime index for line charts; we keep string-friendly too
        st.line_chart(df.set_index("instance")["overall_oos_pct"])
    else:
        st.info("Timeline unavailable (upload at least 1 previous file with crawled_date).")

    # Client-ready email HTML generated from text + key takeaways from backend
    key_takeaways = payload.get("key_takeaways") or ""
    email_html = build_client_email_html(out.get("text_summary", ""), key_takeaways=key_takeaways)

    st.subheader("Preview (Client-ready Email HTML)")
    st.components.v1.html(email_html, height=520, scrolling=True)

    st.markdown("### Copy-paste options")
    st.text_area("Plain text (copy into email)", value=out.get("text_summary", ""), height=220)

    st.header("3) Deliver")
    st.subheader("Email")
    to_emails = st.text_input("Recipients (comma-separated)", value="")
    subject = st.text_input("Subject", value="[Spoggle] Insight Summary")

    if st.button("Send Email", disabled=not to_emails.strip()):
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
