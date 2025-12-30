
from __future__ import annotations
import os
from typing import List
from twilio.rest import Client

def send_whatsapp(message: str, to_numbers: List[str]) -> None:
    sid = os.getenv("TWILIO_ACCOUNT_SID")
    token = os.getenv("TWILIO_AUTH_TOKEN")
    from_whatsapp = os.getenv("TWILIO_WHATSAPP_FROM")  # e.g., "whatsapp:+14155238886"
    if not sid or not token or not from_whatsapp:
        raise RuntimeError("Missing Twilio env vars. Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_FROM.")
    client = Client(sid, token)
    for n in to_numbers:
        client.messages.create(from_=from_whatsapp, to=f"whatsapp:{n}" if not n.startswith("whatsapp:") else n, body=message)
