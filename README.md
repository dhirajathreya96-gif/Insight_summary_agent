# Insight Summary Agent (Spoggle E‑Commerce Analytics)

An **accuracy-first** Insight Summary Agent that:
- Ingests the latest Spoggle E‑Commerce CSV exports (current + optional previous period)
- Computes **predefined** insight KPIs (no free-form interpretation)
- Produces a **standardized bullet summary**
- Delivers via **Email** (SMTP) and optionally **WhatsApp** (Twilio)

## Architecture

- **Frontend:** Streamlit (`frontend/app.py`)  
  Upload files → set Top N / brands / recipients → preview summary → (optional) send.

- **Backend:** FastAPI (`backend/main.py`)  
  `/generate` builds the summary, `/send-email` sends the rendered email.

- **Core logic (deterministic):** `common/`  
  Metrics + comparisons + rendering are template-driven for consistency.

## Quick start (local)

### 1) Create & activate env
```bash
python -m venv .venv
source .venv/bin/activate   # mac/linux
# .venv\Scripts\activate  # windows
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Run backend
```bash
uvicorn backend.main:app --reload --port 8000
```

### 4) Run frontend
```bash
streamlit run frontend/app.py
```

Open Streamlit, upload your `current.csv` and (optionally) `previous.csv`.

## Email sending

Set environment variables (example with Gmail SMTP):
```bash
export SMTP_HOST="smtp.gmail.com"
export SMTP_PORT="587"
export SMTP_USER="you@gmail.com"
export SMTP_PASS="app_password"
export SMTP_FROM="Spoggle Insights <you@gmail.com>"
```

Then use the Streamlit **Send Email** button or call backend `/send-email`.

## WhatsApp (optional)

Uses Twilio WhatsApp. Set:
```bash
export TWILIO_ACCOUNT_SID="..."
export TWILIO_AUTH_TOKEN="..."
export TWILIO_WHATSAPP_FROM="whatsapp:+14155238886"
```

## Data expectations

CSV must include (case-insensitive match supported):
- SKU identifier: `SKU_Name` (fallback: `Product_Name`)
- Pincode: `Pincode`
- City/Region: `City`
- Stock status: `Stock` ("Instock" / "Out of Stock")
- Price fields: `Price`, `MRP`, `Discount_Percentage`, `Discount_Amount`
- Review fields: `Avg_Rating`, `Review_Count` or `Rating_Count`

Other fields are optional.

## Notes

- This system is **trust-first**: it will **surface data quality warnings** (missing columns, low serviced coverage, etc.)
- You can extend the renderer to produce HTML/PDF later.


## Competitor definition
- If you set `own_brands` (e.g., `Oetker`), the agent treats **competitors as all brands NOT in `own_brands`**.
