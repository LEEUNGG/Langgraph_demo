# chat_summary_api.py
"""
Usage:
    from chat_summary_api import generate_chat_summary_markdown
    md = generate_chat_summary_markdown(chat_list)
    print(md)

Set these constants below to your environment's values before using.
""" 

import os
import json
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(dotenv_path)


# -----------------------------
# behavior config
# -----------------------------
ASSUME_TEXT_FROM = "fan"                # default owner for ambiguous types; "fan" or "creator"

# -----------------------------
# Event type sets (as requested)
# -----------------------------
AMBIGUOUS_TYPES = {"text", "audio", "image"}
CREATOR_TYPES = {"video", "sell"}
FAN_TYPES = {"buy", "tip", "gift", "like", "comment", "subscribe"}

# -----------------------------
# Helpers: datetime parsing/format
# -----------------------------
ISO_INPUT_FORMATS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S.%f",
]


def parse_dt(s: str) -> datetime:
    if s is None:
        return None
    for fmt in ISO_INPUT_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass
    # fallback to fromisoformat (may raise)
    try:
        return datetime.fromisoformat(s)
    except Exception:
        # last resort: ignore timezone and try substring
        return datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S")


def fmt_header_dt(dt: Optional[datetime], tz_label: str = "") -> str:
    if dt is None:
        return "N/A"
    # produce "September 8, 2025, 2:15 PM" in a portable way
    month = dt.strftime("%B")
    day = str(dt.day)
    year = str(dt.year)
    hour = dt.strftime("%I").lstrip("0") or "0"
    minute = dt.strftime("%M")
    ampm = dt.strftime("%p")
    return f"{month} {day}, {year}, {hour}:{minute} {ampm}" + (" " + tz_label if tz_label else "")


# -----------------------------
# Flatten + classify events
# -----------------------------
def flatten_events(input_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for rec in input_list:
        dt_raw = rec.get("datetime")
        try:
            dt = parse_dt(dt_raw) if dt_raw else None
        except Exception:
            dt = None
        name = rec.get("name", "Unknown")
        for ev in rec.get("events", []) or []:
            e = ev.copy()
            e["datetime_raw"] = dt_raw
            e["datetime"] = dt
            e["name"] = name
            out.append(e)
    out.sort(key=lambda x: x.get("datetime") or datetime.min)
    return out


def classify_event(ev: Dict[str, Any], assume_text_from: str = ASSUME_TEXT_FROM) -> str:
    t = (ev.get("type") or "").lower()
    if t in CREATOR_TYPES:
        return "creator"
    if t in FAN_TYPES:
        return "fan"
    if t in AMBIGUOUS_TYPES:
        return assume_text_from
    # default fallback
    return assume_text_from


def safe_get_first_name(events: List[Dict[str, Any]]) -> str:
    for rec in events:
        if rec.get("name"):
            return rec["name"]
    return "Unknown Fan"


def compute_aggregates(flat_events: List[Dict[str, Any]], assume_text_from: str = ASSUME_TEXT_FROM) -> Dict[str, Any]:
    if not flat_events:
        return {
            "fan_name": "Unknown",
            "first_dt": None,
            "last_dt": None,
            "session_span": None,
            "total_messages": 0,
            "fan_messages": 0,
            "creator_messages": 0,
            "total_creator_media_sent": 0,
            "total_creator_ppv_value_sent": 0.0,
            "monetized_total": 0.0,
            "monetized_counts": {},
        }

    dts = [e.get("datetime") for e in flat_events if e.get("datetime") is not None]
    first_dt = min(dts) if dts else None
    last_dt = max(dts) if dts else None

    total_messages = fan_messages = creator_messages = 0
    total_creator_media_sent = 0
    total_creator_ppv_value_sent = 0.0
    monetized_total = 0.0
    monetized_counts = {"tips": 0, "paid_unlocks": 0, "buys": 0, "gifts": 0, "subscribe": 0}

    for ev in flat_events:
        t = (ev.get("type") or "").lower()
        actor = classify_event(ev, assume_text_from=assume_text_from)
        if t in {"text", "audio", "image", "video", "sell", "comment", "like"}:
            total_messages += 1
            if actor == "fan":
                fan_messages += 1
            else:
                creator_messages += 1

        if actor == "creator" and t in {"audio", "image", "video"}:
            total_creator_media_sent += 1
            price = ev.get("price")
            if price:
                try:
                    total_creator_ppv_value_sent += float(price)
                except Exception:
                    pass

        if t == "sell":
            price = ev.get("price") or ev.get("paid")
            if price:
                try:
                    total_creator_ppv_value_sent += float(price)
                except Exception:
                    pass

        if t == "tip" and ev.get("price"):
            monetized_counts["tips"] += 1
            try:
                monetized_total += float(ev.get("price"))
            except Exception:
                pass
        if t == "buy":
            monetized_counts["buys"] += 1
        if t == "gift" and ev.get("price"):
            monetized_counts["gifts"] += 1
            try:
                monetized_total += float(ev.get("price"))
            except Exception:
                pass
        if t == "subscribe" and ev.get("price"):
            monetized_counts["subscribe"] += 1
            try:
                monetized_total += float(ev.get("price"))
            except Exception:
                pass

    return {
        "fan_name": safe_get_first_name(flat_events),
        "first_dt": first_dt,
        "last_dt": last_dt,
        "session_span": (first_dt, last_dt),
        "total_messages": total_messages,
        "fan_messages": fan_messages,
        "creator_messages": creator_messages,
        "total_creator_media_sent": total_creator_media_sent,
        "total_creator_ppv_value_sent": total_creator_ppv_value_sent,
        "monetized_total": monetized_total,
        "monetized_counts": monetized_counts,
    }


# -----------------------------
# Public: single-arg function
# -----------------------------
def generate_chat_summary_markdown(chat_list: List[Dict[str, Any]]) -> str:
    """
    Accepts a single chat_list (the events array you provided).
    Returns a markdown summary string.
    Uses SERVICE_ACCOUNT_PATH, GCP_PROJECT, GCP_LOCATION, MODEL_ID constants above.
    """
    flat = flatten_events(chat_list)
    
    MAX_EVENTS = 100
    if len(flat) > MAX_EVENTS:
        flat = flat[-MAX_EVENTS:]
    
    agg = compute_aggregates(flat, assume_text_from=ASSUME_TEXT_FROM)

    fan_name = agg["fan_name"]
    first_dt = agg["first_dt"]
    last_dt = agg["last_dt"]
    session_span = agg["session_span"]
    total_messages = agg["total_messages"]
    fan_messages = agg["fan_messages"]
    creator_messages = agg["creator_messages"]
    total_creator_media_sent = agg["total_creator_media_sent"]
    total_creator_ppv_value_sent = agg["total_creator_ppv_value_sent"]
    monetized_total = agg["monetized_total"]
    monetized_counts = agg["monetized_counts"]

    header_md = f"""# 🧠 FAN INTERACTION BRIEF

**Fan Name:** {fan_name}
**Last Interaction Date:** {fmt_header_dt(last_dt)}
"""
    
    if session_span and session_span[0] and session_span[1]:
        session_span_str = f"{fmt_header_dt(session_span[0])} → {fmt_header_dt(session_span[1])}"
    else:
        session_span_str = "N/A"
    
    header_md += f"""
**Session Span:** {session_span_str}
**Total Messages:** {total_messages}
**Fan Messages:** {fan_messages}
**Creator Messages:** {creator_messages}
**Total Creator Media Sent:** {total_creator_media_sent}
**Total Creator PPV Value Sent:** ${total_creator_ppv_value_sent:.2f}
**Monetized Events:** ${monetized_total:.2f} across {monetized_counts.get('tips',0)} tips, {monetized_counts.get('buys',0)} paid unlocks.

"""

    short_events = []
    for e in flat:
        ev = {
            "datetime": e.get("datetime_raw") or (e.get("datetime").isoformat() if e.get("datetime") else None),
            "type": e.get("type"),
            "actor": classify_event(e, assume_text_from=ASSUME_TEXT_FROM),
        }
        for k in ("content", "price", "description", "paid"):
            if e.get(k) is not None:
                ev[k] = e.get(k)
        short_events.append(ev)
  
    if len(short_events) > MAX_EVENTS:
        short_events = short_events[-MAX_EVENTS:]

    prompt = f"""
    You are an assistant that writes concise, actionable chat summaries for a creator, that is on a content creator platform, similar to Onlyfans.
    We will provide deterministic header fields and the raw, parsed event timeline below (JSON).
    Your task is to generate the following sections in standard Markdown format (use exactly the headings shown):
    ## 1. 🔑 KEY CONTEXT & HISTORY
    ## 2. 🧲 MONETIZATION OPPORTUNITIES
    ## 3. 🗣️ KEY FAN STATEMENTS
    ## 4. 📸 CONTENT DISCUSSED OR SENT
    ## 5. ⏳ TIMELINE OF EVENTS
    ## 6. 🔁 SUGGESTED NEXT MOVES

    Important rules:
    - DO NOT re-generate the header fields (Fan Name, Last Interaction Date, totals) — they are computed by code and will be prepended outside your reply.
    - Use the events JSON below to draft the 6 sections. Keep each section short and actionable (2-6 sentences or bullet points).
    - For MONETIZATION OPPORTUNITIES and SUGGESTED NEXT MOVES give 3 concise, prioritized recommendations each (one-line each).
    - For TIMELINE, produce 3-8 bullet points summarizing the main chronological actions and any purchases/price info.
    - If the fan has not replied, explicitly state "No fan response during this period" in KEY CONTEXT & HISTORY and in KEY FAN STATEMENTS use "No fan statements available".
    - Return ONLY the six sections in Markdown (no additional commentary, no JSON).
    - Money values should use $ and two decimals where applicable.
    - Tone: helpful and slightly sales-oriented.
    - Use **strong emphasis** for important terms, not for entire sentences.
    - Make sure each section heading is on its own line and uses the exact format specified (starting with ## 1., ## 2., etc.).
    - Ensure proper spacing between sections and content.

    Deterministic header (for your reference):
    {json.dumps({'fan_name': fan_name, 'last_interaction': fmt_header_dt(last_dt), 'session_span': session_span_str, 'total_messages': total_messages, 'fan_messages': fan_messages, 'creator_messages': creator_messages, 'total_creator_media_sent': total_creator_media_sent, 'total_creator_ppv_value_sent': f"${total_creator_ppv_value_sent:.2f}", 'monetized_events': f"${monetized_total:.2f}"}, ensure_ascii=False, indent=2)}

    Parsed events (short form):
    {json.dumps(short_events, ensure_ascii=False, indent=2)}

    Now generate the 6 sections in standard Markdown format. Start with "## 1. 🔑 KEY CONTEXT & HISTORY" heading.
    """

    try:
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    except Exception as e:
        return f'{{"error": "Failed to initialize model: {str(e)}"}}'

    try:
        content_for_model = [prompt]
        response = model.generate_content(content_for_model)
        
        if hasattr(response, 'text'):
            response_text = response.text
        elif hasattr(response, 'candidates') and response.candidates:
            response_text = response.candidates[0].content.parts[0].text
        else:
            response_text = str(response)

    except Exception as e:
        return f'{{"error": "Failed to generate content: {str(e)}"}}'

    final_md = header_md + response_text.strip()
    return final_md


# -----------------------------
# Example (only if run as script)
# -----------------------------
if __name__ == "__main__":
    # example input: replace with real chat list
    sample_chat = [
        {"datetime": "2025-09-08 14:06:00", "name": "Sparkling Grouse",
         "events": [{"type": "text", "content": "Hello"}]},
        {"datetime": "2025-09-08 14:06:30", "name": "Sparkling Grouse",
         "events": [{"type": "audio", "description": "", "price": "5"}]},
        {"datetime": "2025-09-08 14:15:00", "name": "Sparkling Grouse",
         "events": [{"type": "audio", "description": "", "price": "10"}, {"type": "image", "description": ""}]},
    ]

    md = generate_chat_summary_markdown(sample_chat)
    
    # print(md)
    
    # with open('chat_summary.md', 'w', encoding='utf-8') as f:
    #     f.write(md)