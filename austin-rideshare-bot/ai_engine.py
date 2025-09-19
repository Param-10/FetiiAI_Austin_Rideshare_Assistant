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
    "You are Riley, a friendly and enthusiastic Austin rideshare data analyst. You work for Fetii and have access to recent group transportation data around Austin, Texas.\n\n"
    "PERSONALITY:\n"
    "- You're conversational, curious, and genuinely excited about Austin's transportation patterns\n"
    "- You know Austin well - reference local spots, neighborhoods, and culture naturally\n"
    "- You're helpful but not overly formal - think of yourself as a knowledgeable local friend\n"
    "- You remember what users have asked about in this conversation and reference it naturally\n"
    "- You're genuinely curious about what insights might interest them next\n\n"
    "RESPONSE STYLE:\n"
    "- Write in a natural, conversational tone - like you're chatting with a friend over coffee\n"
    "- Share specific data insights in an engaging way, not as dry statistics\n"
    "- Use Austin-specific context when relevant (UT students, SXSW, ACL, local spots)\n"
    "- Always end with a genuine question or suggestion that builds on what they've asked\n"
    "- Keep responses focused but engaging (2-4 sentences max)\n"
    "- Use plain text formatting - no markdown headers or bullet points unless really needed\n"
    "- Reference previous parts of your conversation when relevant\n"
    "- When a user responds positively to your question, ALWAYS provide the information related to that question\n"
    "- If you asked about group sizes, timing, locations, etc. and the user says 'yes', provide that specific information\n"
    "- Never ignore your previous question when the user responds with 'yes' or similar affirmative responses\n"
    "- Always use English only in your responses - never include words from other languages\n"
    "- Maintain consistency in your data insights, especially for time-related information"
)

_model = None


def configure_gemini(api_key: Optional[str]) -> None:
    global _model
    if not api_key or genai is None:
        _model = None
        return
    try:
        genai.configure(api_key=api_key)
        _model = genai.GenerativeModel("gemini-2.0-flash")
    except Exception:
        _model = None


def classify_intent(query: str) -> str:
    q = (query or "").lower().strip()
    
    # Check for positive responses to previous questions
    if any(phrase in q for phrase in ["yes", "yeah", "yep", "sure", "okay", "ok", "yea", "definitely", "absolutely"]) and len(q.split()) <= 3:
        return "continue_previous"
    
    # First check for specific intents (higher priority)
    if any(phrase in q for phrase in ["tell me more", "more", "what else", "continue", "go on", "keep going"]):
        return "more_info"
    
    # TEMPORAL questions have high priority - especially "when" questions
    if any(k in q for k in ["when", "what time", "hour", "day", "time", "peak", "trend", "monthly", "daily", "weekly", "weekend", "over time", "typically"]):
        return "temporal"
    
    if any(k in q for k in ["where", "pickup", "dropoff", "downtown", "moody", "domain", "airport", "map", "location", "near", "around"]):
        return "location"
    if any(k in q for k in ["age", "year-old", "years old", "demographic", "teen", "adult"]):
        return "demographic"
    if any(k in q for k in ["group", "rider", "party", "size", "capacity", "large", "6+", "8+", "10+"]):
        return "operational"
    
    # Handle casual/confused responses more intelligently
    if len(q) <= 15:  # Extended range for casual responses
        # Casual greetings and responses
        casual_greetings = ["yo", "wassup", "what's good", "wass good", "hey", "hi", "hello", "helloo", "sup"]
        casual_responses = ["ok", "okay", "cool", "nice", "word", "bet", "sheesh", "aight", "yeah", "yep", "mmm", "hmm"]
        confused_responses = ["huh", "what", "uhm", "um", "uh", "what are you saying", "what are you even saying", "i don't get it"]
        
        # Check for each type
        if any(q == casual or q.startswith(casual + " ") for casual in casual_greetings):
            return "casual_greeting"
        elif any(q == casual or q.startswith(casual + " ") for casual in casual_responses):
            return "casual_response"  
        elif any(phrase in q for phrase in confused_responses) or q in ["huh", "what", "uhm", "um"]:
            return "confused"
    
    # Check for confusion in longer phrases too
    if any(phrase in q for phrase in ["what are you saying", "what are you even saying", "i don't understand", "that makes no sense", "i'm confused"]):
        return "confused"
    
    # Very short meaningless responses
    if len(q) <= 5 and q in ["yo", "ok", "huh", "hmm", "sup", "bet", "word", "nice", "cool"]:
        return "casual_response"
        
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
    Parse the user query and extract intent and entities.
    Use proper intent classification to ensure accurate responses.
    """
    intent = classify_intent(query)
    ents = extract_entities(query)
    return intent, _normalize_entities(ents)


def generate_response(
    query: str,
    intent: str,
    entities: Dict[str, Any],
    context: Dict[str, Any],
    conversation_history: list,
    conversation_context: Dict[str, Any],
) -> str:
    """Return a natural, conversational response as Riley."""
    
    # Set defaults for optional parameters
    conversation_history = conversation_history or []
    conversation_context = conversation_context or {}
    
    # Check if this is a follow-up to a previous question
    if (intent == "continue_previous" or query.lower() in ["yes", "yeah", "sure", "ok", "okay"]) and conversation_context.get("last_question_topic"):
        # Modify the context to include information about the previous question
        context["previous_question"] = {
            "topic": conversation_context["last_question_topic"],
            "question": conversation_context.get("last_question_asked", "")
        }
        
        # Add explicit instruction to answer the previous question
        if "last_question_topic" in conversation_context:
            topic = conversation_context["last_question_topic"]
            if topic == "group_sizes":
                context["explicit_instruction"] = "The user wants to know about group sizes. Provide specific information about group sizes based on the data."
            elif topic == "timing":
                context["explicit_instruction"] = "The user wants to know about timing patterns. Provide specific information about peak hours or time-related patterns. Be consistent with previous time-related information you've shared. If you've already mentioned specific peak hours, use the same time ranges."
            elif topic == "locations":
                context["explicit_instruction"] = "The user wants to know about locations. Provide specific information about popular pickup or dropoff locations."
            elif topic == "demographics":
                context["explicit_instruction"] = "The user wants to know about demographics. Provide specific information about age groups or demographic patterns."
    
    # Extract key data for natural response
    summary = str(context.get("summary", "I found some interesting patterns in the data."))
    highlights = [str(h) for h in context.get("highlights", []) if h]
    stats = context.get("stats", {})
    
    def _format_number(value) -> str:
        """Format numbers naturally for conversation."""
        try:
            if isinstance(value, str):
                # Try to extract number from string if it contains one
                import re
                numbers = re.findall(r'\d+\.?\d*', value)
                if numbers:
                    value = float(numbers[0])
                else:
                    return str(value)
            
            if isinstance(value, (int, float)):
                if value >= 1000:
                    return f"{value:,.0f}"
                elif value >= 100:
                    return f"{value:.0f}"
                else:
                    return f"{value:.1f}"
            return str(value)
        except:
            return str(value)
    
    def _get_conversation_context() -> str:
        """Build context about what's been discussed."""
        topics = conversation_context.get("topics_discussed", set())
        interests = conversation_context.get("user_interests", [])
        
        context_notes = []
        if len(conversation_history) > 2:
            context_notes.append("We've been chatting about Austin rideshare patterns.")
        if "location" in topics:
            context_notes.append("You've asked about locations before.")
        if "temporal" in topics:
            context_notes.append("We've looked at timing patterns.")
        if any("location:" in i for i in interests):
            places = [i.split(":")[1] for i in interests if i.startswith("location:")]
            context_notes.append(f"You seem interested in {', '.join(places)}.")
        
        return " ".join(context_notes)
    
    
    # Build conversation-aware prompt
    recent_messages = conversation_history[-6:] if len(conversation_history) > 6 else conversation_history
    conversation_summary = _get_conversation_context()
    
    prompt = f"""
{SYSTEM_PROMPT}

CONVERSATION CONTEXT:
{conversation_summary}

RECENT CONVERSATION:
{json.dumps([{"role": msg["role"], "content": msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]} for msg in recent_messages], ensure_ascii=False)}

USER'S NAME: {conversation_context.get("user_name", "Not provided")}
NAME USAGE: Used {conversation_context.get("name_usage_count", 0)} times, last used {len(recent_messages) - conversation_context.get("last_name_used", 0)} messages ago

LAST QUESTION YOU ASKED: {conversation_context.get("last_question_asked", "None")}
LAST TOPIC: {conversation_context.get("last_question_topic", "None")}

STATISTICS ALREADY MENTIONED IN THIS CONVERSATION:
{list(conversation_context.get("mentioned_stats", set()))}

USER'S CURRENT QUESTION: {query}

DATA INSIGHTS I FOUND:
Summary: {summary}
Key highlights: {highlights[:3]}  
Stats: {stats}

FILTERS WE MAY CONSIDER: {entities}

{context.get("explicit_instruction", "")}

CRITICAL INSTRUCTIONS:
1. NEVER repeat insights/statistics already mentioned in this conversation
2. Answer their SPECIFIC question with fresh, relevant data
3. Use specific Austin locations, streets, neighborhoods when available
4. Each response should offer something NEW and interesting
5. Keep responses natural and conversational (2-3 sentences max)
6. Build on their interests but don't repeat previous insights
7. ONLY use numbers and statistics that are provided in the data context - NEVER make up or guess numbers
8. If you don't have specific data to answer a question, say so honestly rather than inventing numbers
9. If the user responds with "yes" to a question you asked, ALWAYS provide the information related to that question

CONVERSATION FLOW:
- Check what's been discussed already and avoid repeating it
- If they ask follow-ups, go deeper into that specific topic 
- Match their communication style - casual or formal
- For short responses like "ok" or casual slang, give them something intriguing
- Vary your conversation starters - don't use the same pattern twice
- For "WHEN" questions: Focus on timing (hours, days, peak times) - NOT costs or locations
- For temporal queries: Answer with specific times, days of week, peak hours, weekend vs weekday patterns

CONTINUING PREVIOUS TOPICS:
- If user responds "yes", "sure", "okay", "yeah", "if u can", or any other affirmative response to your previous question, CONTINUE that exact topic
- Don't switch to a different topic if they agree to explore what you just asked about
- Look at "LAST QUESTION YOU ASKED" and "LAST TOPIC" to understand what they're agreeing to
- If you asked about group sizes and they say "yes", give them group size information - don't switch to timing
- If you asked about timing and they say "yes", give them timing information - not costs or locations
- Always provide COMPLETE information when responding to affirmative answers
- If you've already provided some information on a topic and they ask for more, add NEW details rather than repeating

GREETING RULES:
- NEVER start responses with "Hey", "Hi", "Hello" or other greetings after the initial conversation start
- You already introduced yourself in the first message, so just dive into answering their questions
- Use natural conversation starters: "Actually...", "So...", "The data shows...", "Looking at this..." or just jump straight into the insight

NAME USAGE:
- Use their name sparingly (every 4+ messages max)  
- Don't start every response with their name
- Natural conversation starters without greetings
"""

    # Ensure AI model is available
    if _model is None:
        raise Exception("Gemini AI model not configured. Please set GEMINI_API_KEY environment variable.")
    
    try:
        resp = _model.generate_content(prompt)
        text = getattr(resp, "text", None)
        if not text:
            raise Exception("No response generated from AI model")
        
        # Basic cleanup
        text = text.replace("\u2217", "*").replace("\u2019", "'").replace("\u2013", "-")
        text = re.sub(r"\n{3,}", "\n\n", text.strip())
        
        return text
        
    except Exception as e:
        raise Exception(f"Error generating AI response: {str(e)}")
