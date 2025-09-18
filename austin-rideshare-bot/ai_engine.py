from __future__ import annotations

import os
import re
import json
from typing import Any, Dict, Optional, Tuple

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
    if any(k in q for k in ["where", "pickup", "dropoff", "downtown", "moody", "domain", "airport", "map", "location", "near", "around"]):
        return "location"
    if any(k in q for k in ["when", "hour", "day", "time", "peak", "trend", "monthly", "daily", "weekly", "weekend", "over time"]):
        return "temporal"
    if any(k in q for k in ["age", "year-old", "years old", "demographic", "teen", "adult"]):
        return "demographic"
    if any(k in q for k in ["group", "rider", "party", "size", "capacity", "large", "6+", "8+", "10+"]):
        return "operational"
    return "general"


def _extract_age(query: str) -> Dict[str, Any]:
    q = query
    ql = q.lower()
    # Ranges like 18-24
    m = re.search(r"(\d{1,2})\s*[–-]\s*(\d{1,2})", ql)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return {"age_min": min(a, b), "age_max": max(a, b)}
    # Plus pattern only if 'year' context is present
    if re.search(r"\b(year|years|yr|yrs|yo)\b", ql):
        m = re.search(r"(\d{1,2})\s*\+", ql)
        if m:
            return {"age_min": int(m.group(1))}
    # Under pattern
    if re.search(r"\bunder\b", ql) and re.search(r"\b(year|years|yr|yrs|yo)\b", ql):
        m = re.search(r"under\s*(\d{1,2})", ql)
        if m:
            return {"age_max": int(m.group(1)) - 1}
    return {}


def _extract_place(query: str) -> Optional[str]:
    q = query.lower()
    places = [
        "moody center",
        "downtown",
        "domain",
        "airport",
        "ut",
        "sixth street",
        "6th street",
        "south congress",
        "rainey",
        "rainey street",
        "east austin",
    ]
    for p in places:
        if p in q:
            return p
    return None


def _extract_time(query: str) -> Dict[str, Any]:
    q = query.lower()
    ent: Dict[str, Any] = {}
    if "last month" in q:
        ent["relative_time"] = "last_month"
    if "this month" in q:
        ent["relative_time"] = "this_month"
    if "last week" in q:
        ent["relative_time"] = "last_week"
    if "weekend" in q or "weekends" in q:
        ent.setdefault("days", []).extend(["Friday", "Saturday", "Sunday"])
    for day in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]:
        if day in q:
            ent.setdefault("days", []).append(day.capitalize())
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


def _normalize_entities(entities: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in (entities or {}).items():
        if k in {"age_min", "age_max", "group_size_min", "group_size_eq"}:
            try:
                out[k] = int(v)
            except Exception:
                continue
        elif k == "hour_range" and isinstance(v, (list, tuple)) and len(v) == 2:
            try:
                out[k] = (int(v[0]) % 24, int(v[1]) % 24)
            except Exception:
                pass
        elif k == "days" and isinstance(v, list):
            out[k] = [str(d).capitalize() for d in v]
        elif isinstance(v, str):
            out[k] = v.strip()
        else:
            out[k] = v
    # If group size is present, drop accidental age bounds inferred from patterns like '6+'
    if ("group_size_min" in out or "group_size_eq" in out):
        out.pop("age_min", None)
        out.pop("age_max", None)
    return out


def _parse_with_gemini(query: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    if _model is None:
        return None
    prompt = (
        "Extract the user's intent and entities from the question for Austin rideshare analytics.\n"
        "Return strictly JSON with keys: intent (one of: location, temporal, demographic, operational, general) and entities (object).\n"
        "Recognize optional entities: place, age_min, age_max, relative_time (last_month|this_month|last_week), days (list of weekday names), hour_range ([start,end]), group_size_min, group_size_eq.\n"
        "Do not add any text outside JSON.\n\n"
        f"Question: {query}\n\n"
        "Example JSON: {\"intent\": \"location\", \"entities\": {\"place\": \"downtown\"}}"
    )
    try:
        resp = _model.generate_content(prompt)
        text = getattr(resp, "text", None) or ""
        m = re.search(r"\{[\s\S]*\}$", text.strip())
        raw = m.group(0) if m else text.strip()
        data = json.loads(raw)
        intent = str(data.get("intent") or "").strip().lower() or "general"
        entities = _normalize_entities(data.get("entities") or {})
        return intent, entities
    except Exception:
        return None


def parse_query(query: str) -> Tuple[str, Dict[str, Any]]:
    """
    Parse the user query to determine intent and entities.
    This function uses a hybrid approach:
    1. A fast, local keyword-based classifier (`classify_intent`) is tried first.
    2. If the local classifier returns a specific intent, it's used immediately.
    3. If the local classifier returns "general", a more powerful Gemini model is used.
    """
    # First, use the reliable local classifier
    intent_h = classify_intent(query)
    ents_h = extract_entities(query)

    # If the local classifier finds a specific intent, trust it and we're done.
    if intent_h != "general":
        return intent_h, _normalize_entities(ents_h)

    # If local classification is "general", try the more powerful Gemini parser
    parsed = _parse_with_gemini(query)
    if parsed:
        intent_g, ents_g = parsed
        # Merge entities, giving Gemini's results priority for richness
        merged = {**ents_h, **ents_g}
        # Trust Gemini's intent if it found something specific
        final_intent = intent_g if intent_g != "general" else intent_h
        return final_intent, _normalize_entities(merged)

    # Fallback to local if Gemini fails completely
    return intent_h, _normalize_entities(ents_h)


def generate_response(query: str, intent: str, entities: Dict[str, Any], context: Dict[str, Any]) -> str:
    summary = context.get("summary") or "I analyzed the data and prepared insights below."
    bullets = context.get("highlights") or []
    stats = context.get("stats") or {}

    if _model is None:
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
