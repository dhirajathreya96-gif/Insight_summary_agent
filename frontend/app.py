
import streamlit as st
import requests
import json

st.set_page_config(page_title="Insight Summary Agent", layout="wide")

st.title("Insight Summary Agent (Spoggle)")
st.caption("Accuracy-first insight summaries from Spoggle E‑Commerce Analytics CSV exports.")

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
    previous_file = st.file_uploader("Previous period CSV (optional)", type=["csv"], key="prev")
with c3:
    pincode_map = st.file_uploader("Pincode → City mapping CSV (optional but recommended)", type=["csv"])


st.header("2) Generate summary")
if st.button("Generate Insight Summary", type="primary", disabled=current_file is None):
    files = {"current_file": ("current.csv", current_file.getvalue(), "text/csv")}
    if previous_file is not None:
        files["previous_file"] = ("previous.csv", previous_file.getvalue(), "text/csv")

    data = {
        "top_n": str(top_n),
        "require_serviced": str(require_serviced),
        "price_change_threshold_pct": str(price_thr),
        "own_brands_json": json.dumps(own_brands),
    }

    try:
        resp = requests.post(f"{API_URL}/generate", files=files, data=data, timeout=120)
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
    st.code(out["text_summary"])

    st.subheader("Preview (HTML)")
    st.components.v1.html(out["html_summary"], height=420, scrolling=True)

    st.header("3) Deliver")
    st.subheader("Email")
    to_emails = st.text_input("Recipients (comma-separated)", value="")
    subject = st.text_input("Subject", value="[Spoggle] Insight Summary")
    if st.button("Send Email", disabled=not to_emails.strip()):
        payload = {
            "subject": subject,
            "to_emails": [x.strip() for x in to_emails.split(",") if x.strip()],
            "html_body": out["html_summary"],
            "text_body": out["text_summary"],
        }
        r = requests.post(f"{API_URL}/send-email", json=payload, timeout=60)
        if r.status_code == 200:
            st.success("Email sent ✅")
        else:
            st.error(r.text)
