from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional

try:
    import google.generativeai as genai  # type: ignore
except Exception:  # pragma: no cover
    genai = None  # type: ignore


SYSTEM_PROMPT = (
    "You are an AI assistant specialized in analyzing Austin rideshare data for Fetii. "
    "You help users understand transportation patterns, popular locations, and rider demographics.\n\n"
    "Available data includes: trip locations/times, rider demographics (age), and group composition.\n"
    "Respond conversationally and include specific data insights to support your answers."
)

_model = None


def configure_gemini(api_key: Optional[str]) -> None:
    global _model
    if not api_key or genai is None:
        _model = None
        return
    try:
        genai.configure(api_key=api_key)
        _model = genai.GenerativeModel("gemini-1.5-flash")
    except Exception:
        _model = None


def classify_intent(query: str) -> str:
    q = (query or "").lower()
    if any(k in q for k in ["where", "pickup", "dropoff", "downtown", "moody", "domain", "airport", "map", "location"]):
        return "location"
    if any(k in q for k in ["when", "hour", "day", "time", "peak", "trend", "monthly", "daily", "weekly"]):
        return "temporal"
    if any(k in q for k in ["age", "year-old", "years old", "demographic", "teen", "adult"]):
        return "demographic"
    if any(k in q for k in ["group", "rider", "party", "size", "capacity", "large"]):
        return "operational"
    return "general"


def _extract_age(query: str) -> Dict[str, Any]:
    q = query
    # 18-24, 18–24, 21+, under 18
    m = re.search(r"(\d{1,2})\s*[–-]\s*(\d{1,2})", q)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return {"age_min": min(a, b), "age_max": max(a, b)}
    m = re.search(r"(\d{1,2})\s*\+", q)
    if m:
        return {"age_min": int(m.group(1))}
    m = re.search(r"under\s*(\d{1,2})", q.lower())
    if m:
        return {"age_max": int(m.group(1)) - 1}
    return {}


def _extract_place(query: str) -> Optional[str]:
    q = query.lower()
    places = ["moody center", "downtown", "domain", "airport", "ut", "sixth street", "6th street"]
    for p in places:
        if p in q:
            return p
    return None


def _extract_time(query: str) -> Dict[str, Any]:
    q = query.lower()
    ent: Dict[str, Any] = {}
    # Relative time
    if "last month" in q:
        ent["relative_time"] = "last_month"
    if "this month" in q:
        ent["relative_time"] = "this_month"
    if "last week" in q:
        ent["relative_time"] = "last_week"
    if "weekend" in q or "weekends" in q:
        ent.setdefault("days", []).extend(["Friday", "Saturday", "Sunday"])
    # Specific days
    for day in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]:
        if day in q:
            ent.setdefault("days", []).append(day.capitalize())
    # Night hours
    if "night" in q or "late" in q:
        ent["hour_range"] = (20, 3)
    return ent


def _extract_group_size(query: str) -> Dict[str, Any]:
    q = query.lower()
    m = re.search(r"(\d+)\s*\+\s*(riders|people|passengers|group)?", q)
    if m:
        return {"group_size_min": int(m.group(1))}
    m = re.search(r"(\d+)\s*(or more|and up)", q)
    if m:
        return {"group_size_min": int(m.group(1))}
    m = re.search(r"group[s]?\s*(of|size)\s*(\d+)", q)
    if m:
        return {"group_size_eq": int(m.group(2))}
    return {}


def extract_entities(query: str) -> Dict[str, Any]:
    ent: Dict[str, Any] = {}
    ent.update(_extract_age(query))
    place = _extract_place(query)
    if place:
        ent["place"] = place
    ent.update(_extract_time(query))
    ent.update(_extract_group_size(query))
    return ent


def generate_response(query: str, intent: str, entities: Dict[str, Any], context: Dict[str, Any]) -> str:
    """Use Gemini if available, otherwise return a templated summary."""
    summary = context.get("summary") or "I analyzed the data and prepared insights below."
    bullets = context.get("highlights") or []
    stats = context.get("stats") or {}

    if _model is None:
        # Simple fallback
        lines = [summary]
        if bullets:
            lines.append("")
            for b in bullets:
                lines.append(f"- {b}")
        if stats:
            lines.append("")
            for k, v in stats.items():
                lines.append(f"• {k}: {v}")
        return "\n".join(lines)

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"User question: {query}\n\n"
        f"Detected intent: {intent}\n"
        f"Entities: {entities}\n\n"
        f"Data summary/context: {context}\n\n"
        "Write a concise, conversational answer. Start with the headline insight, then 2-4 bullets."
    )

    try:
        resp = _model.generate_content(prompt)
        text = getattr(resp, "text", None)
        if not text:
            return "".join([summary] + [f"- {b}" for b in bullets])
        return text
    except Exception:
        return "".join([summary] + [f"- {b}" for b in bullets])
