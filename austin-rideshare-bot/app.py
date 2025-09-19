import os
import random
import time
from typing import Any, Dict

import streamlit as st
import streamlit.components.v1
import plotly.graph_objects as go
from dotenv import load_dotenv

from ai_engine import configure_gemini, generate_response, parse_query
from analytics import answer_query
from data_processor import load_datasets

# --- Page Configuration ---
st.set_page_config(
    page_title="FetiiAI | Austin Rideshare Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --- Theme Definitions ---
THEMES = {
    "dark": {
        "bg_color": "#0E1117", "text_color": "#FAFAFA", "primary_color": "#3b82f6",
        "user_msg_bg": "#253341", "assistant_msg_bg": "#192734",
        "card_bg": "#192734", "card_border": "#262730"
    },
    "light": {
        "bg_color": "#FFFFFF", "text_color": "#0E1117", "primary_color": "#1d4ed8",
        "user_msg_bg": "#e0e7ff", "assistant_msg_bg": "#f0f2f6",
        "card_bg": "#f0f2f6", "card_border": "#e5e7eb"
    }
}

# --- Session State Initialization ---
if "theme" not in st.session_state:
    st.session_state.theme = "dark"  # Default to dark mode
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hey there! I'm Riley, your friendly Austin rideshare data analyst. I've got the inside scoop on over 2,000 recent group rides around our city – from late-night Sixth Street runs to airport shuttles and everything in between.\n\nWhat's got you curious about Austin's transportation patterns? Are you wondering about popular hangout spots, rush hour madness, or maybe how different age groups get around town? 🚗✨"}
    ]
if "query_times" not in st.session_state:
    st.session_state.query_times = []
if "conversation_context" not in st.session_state:
    st.session_state.conversation_context = {
        "topics_discussed": set(),
        "user_interests": [],
        "previous_filters": {},
        "follow_up_suggestions": [],
        "mentioned_stats": set(),
        "user_name": "",
        "name_usage_count": 0,
        "last_name_used": 0,
        "recent_response_patterns": [],
        "recent_questions": [],
        "last_question_asked": "",
        "last_question_topic": ""
    }


# --- Helper Functions ---
def _is_valid_plotly_figure(fig):
    """Check if the figure is a valid Plotly figure object."""
    if fig is None:
        return False
    # Check if it's a plotly figure
    return hasattr(fig, 'data') and hasattr(fig, 'layout') and (
        isinstance(fig, go.Figure) or 
        (hasattr(fig, '__dict__') and 'data' in fig.__dict__ and 'layout' in fig.__dict__)
    )

# --- Dynamic Theming ---
def apply_theme():
    theme = THEMES[st.session_state.theme]
    st.markdown(f"""
    <style>
        /* Core App Styling */
        .stApp {{
            background-color: {theme['bg_color']};
            color: {theme['text_color']};
        }}
        /* Main container styling */
        .main .block-container {{ padding-top: 2rem; padding-bottom: 2rem; }}
        h1, h2, h3, h4, h5, h6 {{ color: {theme['text_color']}; }}
        /* Chat bubbles */
        [data-testid="chat-message-container"] {{
            border-radius: 0.5rem; padding: 0.8rem; margin-bottom: 1rem;
        }}
        [data-testid="chat-message-container"] [data-testid="stMarkdown"] p {{
            margin: 0; font-size: 1rem; line-height: 1.6; color: {theme['text_color']};
        }}
        [data-testid="chat-message-container"]:has(div[data-testid="chat-avatar-user"]) {{
            background-color: {theme['user_msg_bg']};
        }}
        [data-testid="chat-message-container"]:has(div[data-testid="chat-avatar-assistant"]) {{
            background-color: {theme['assistant_msg_bg']};
        }}
        /* Expander styling */
        .stExpander {{
            border: 1px solid {theme['card_border']};
            border-radius: 0.5rem;
            background-color: {theme['card_bg']};
        }}
        .stExpander header {{ font-size: 1.1rem; font-weight: 600; color: {theme['text_color']}; }}
        /* Buttons */
        .stButton>button {{
            border-radius: 0.5rem; border: 1px solid {theme['primary_color']};
            background-color: transparent; color: {theme['primary_color']};
            transition: all 0.2s ease-in-out;
        }}
        .stButton>button:hover {{ background-color: {theme['primary_color']}; color: white; }}
        /* Title and Caption */
        h1 {{ text-align: center; font-weight: 700; }}
        .st-emotion-cache-10trblm {{ text-align: center; }}
    </style>
    """, unsafe_allow_html=True)

apply_theme()


# --- API Key and AI Configuration ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
configure_gemini(api_key=GEMINI_API_KEY)


# --- Data Loading ---
@st.cache_data(show_spinner="Loading rideshare data...")
def _cached_load() -> Dict[str, Any]:
    return load_datasets()

data = _cached_load()
trips, riders, demographics, meta = (
    data.get("trips"), data.get("riders"), data.get("demographics"), data.get("meta", {}),
)

# --- App Header ---
st.title("🚗 Riley - Your Austin Rideshare Buddy")
st.caption("ChatGPT meets Austin transportation data – let's explore the city together!")


# --- Controls and Filters Expander ---
with st.expander("⚙️ Controls & Filters"):
    # --- Theme Toggle ---
    current_theme_name = st.session_state.theme.capitalize()
    if st.button(f"Switch to { 'Light' if st.session_state.theme == 'dark' else 'Dark'} Mode"):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        st.rerun()

    st.markdown("---")
    st.markdown("Use the controls below to filter the dataset for your queries.")
    
    if st.button("🗑️ Start Fresh Chat", help="Clear conversation history and context"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hey there! I'm Riley, your friendly Austin rideshare data analyst. I've got the inside scoop on over 2,000 recent group rides around our city – from late-night Sixth Street runs to airport shuttles and everything in between.\n\nWhat's got you curious about Austin's transportation patterns? Are you wondering about popular hangout spots, rush hour madness, or maybe how different age groups get around town? 🚗✨"}
        ]
        st.session_state.conversation_context = {
            "topics_discussed": set(),
            "user_interests": [],
            "previous_filters": {},
            "follow_up_suggestions": [],
            "mentioned_stats": set(),
            "user_name": "",
            "name_usage_count": 0,
            "last_name_used": 0,
            "recent_response_patterns": [],
            "recent_questions": [],
            "last_question_asked": "",
            "last_question_topic": ""
        }
        st.rerun()

    if st.button("Give me something interesting to ask about"):
        sample_queries = [
            "What's the busiest time for rides around UT?",
            "Where do college students go on Friday nights?", 
            "Show me weekend hotspots for big groups",
            "What's the typical ride cost these days?",
            "Where are people going from the airport?",
            "Which neighborhoods are most popular for groups?",
            "When do people ride to Sixth Street?",
            "What time do folks head home from downtown?",
            "Tell me about late-night ride patterns",
            "Where do 20-somethings hang out?"
        ]
        st.session_state.pending_query = random.choice(sample_queries)
        st.rerun()

    # Filters and Data Status in columns
    col1, col2 = st.columns(2)
    with col1:
        if 'event_time' in trips.columns and not trips.empty:
            min_date, max_date = trips['event_time'].min().date(), trips['event_time'].max().date()
            st.session_state['date_filter'] = st.date_input(
                "Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date
            )
    with col2:
        if 'num_riders' in trips.columns and not trips['num_riders'].empty:
            min_r, max_r = int(trips['num_riders'].min()), int(trips['num_riders'].max())
            st.session_state['rider_filter'] = st.slider(
                "Group Size", min_value=min_r, max_value=max_r, value=(min_r, max_r)
            )

    st.subheader("Data Status")
    st.markdown(f"Trips loaded: **{len(trips):,}**")
    if meta.get("time_range"):
        st.caption(f"Time range: {meta['time_range']['min']:%Y-%m-%d} → {meta['time_range']['max']:%Y-%m-%d}")


# --- Chat Interface (chat-only) ---
# Replay history (text only)
for mi, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Process new user input
if prompt := st.chat_input("Ask a question...") or st.session_state.pop("pending_query", None):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing data..."):
            intent, entities = parse_query(prompt)
            context, figures = answer_query(trips, riders, demographics, intent, entities, st.session_state.conversation_context)
            
            # Update conversation context
            st.session_state.conversation_context["topics_discussed"].add(intent)
            if entities.get("place"):
                st.session_state.conversation_context["user_interests"].append(f"location:{entities['place']}")
            if entities.get("age_min") or entities.get("age_max"):
                st.session_state.conversation_context["user_interests"].append("demographics")
            
            # Track mentioned statistics to avoid repetition
            if "mentioned_stats" not in st.session_state.conversation_context:
                st.session_state.conversation_context["mentioned_stats"] = set()
                
            # Track name usage to avoid overuse
            if "name_usage_count" not in st.session_state.conversation_context:
                st.session_state.conversation_context["name_usage_count"] = 0
            if "last_name_used" not in st.session_state.conversation_context:
                st.session_state.conversation_context["last_name_used"] = 0
            
            # Extract user name from conversation if mentioned
            user_name = st.session_state.conversation_context.get("user_name", "")
            if not user_name:
                # Look for name in recent user messages
                for msg in st.session_state.messages[-3:]:
                    if msg["role"] == "user":
                        content = msg["content"].lower()
                        if "my name is" in content or "i'm" in content or "i am" in content:
                            words = content.split()
                            for i, word in enumerate(words):
                                if word in ["is", "i'm", "am"] and i + 1 < len(words):
                                    potential_name = words[i + 1].strip(".,!?").title()
                                    if potential_name.isalpha() and len(potential_name) > 1:
                                        st.session_state.conversation_context["user_name"] = potential_name
                                        break
                
            # Add key stats from this response to avoid future repetition
            for key, value in context.get("stats", {}).items():
                if any(word in key.lower() for word in ["distance", "cost", "trips"]):
                    st.session_state.conversation_context["mentioned_stats"].add(f"{key}:{value}")
            
            # Generate conversational response
            try:
                response_text = generate_response(
                    query=prompt, 
                    intent=intent, 
                    entities=entities, 
                    context=context,
                    conversation_history=st.session_state.messages,
                    conversation_context=st.session_state.conversation_context
                )
            except Exception as e:
                st.error(f"AI Error: {str(e)}")
                st.error("Please ensure GEMINI_API_KEY is properly configured in your environment variables.")
                st.stop()
            
            st.markdown(response_text)
            
            # Track what question was asked in the response to remember for next time
            response_lower = response_text.lower()
            if any(phrase in response_lower for phrase in ["group size", "how many people", "party size", "riders"]):
                st.session_state.conversation_context["last_question_topic"] = "group_sizes"
                st.session_state.conversation_context["last_question_asked"] = "about group sizes"
            elif any(phrase in response_lower for phrase in ["what time", "when", "peak hour", "timing"]):
                st.session_state.conversation_context["last_question_topic"] = "timing"
                st.session_state.conversation_context["last_question_asked"] = "about timing"
            elif any(phrase in response_lower for phrase in ["where", "location", "destination", "spots"]):
                st.session_state.conversation_context["last_question_topic"] = "locations"
                st.session_state.conversation_context["last_question_asked"] = "about locations"
            elif any(phrase in response_lower for phrase in ["age", "demographic", "old", "year"]):
                st.session_state.conversation_context["last_question_topic"] = "demographics"
                st.session_state.conversation_context["last_question_asked"] = "about demographics"
            
            # Track if name was used in this response
            user_name = st.session_state.conversation_context.get("user_name", "")
            if user_name and user_name.lower() in response_text.lower():
                st.session_state.conversation_context["name_usage_count"] += 1
                st.session_state.conversation_context["last_name_used"] = len(st.session_state.messages)
            
            # Save assistant message (text only)
            assistant_message = {"role": "assistant", "content": response_text}
            st.session_state.messages.append(assistant_message)
    st.rerun()
