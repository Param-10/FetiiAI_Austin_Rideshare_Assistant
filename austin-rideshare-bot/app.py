import os
import random
import time
from typing import Any, Dict

import streamlit as st
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
        {"role": "assistant", "content": "Hi! How can I help you analyze Austin's rideshare data today?"}
    ]
if "query_times" not in st.session_state:
    st.session_state.query_times = []


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
st.title("🤖 FetiiAI Rideshare Assistant")
st.caption("Ask me about group transportation trends in Austin, TX")


# --- Controls and Filters Expander ---
with st.expander("⚙️ Controls & Filters"):
    # --- Theme Toggle ---
    current_theme_name = st.session_state.theme.capitalize()
    if st.button(f"Switch to { 'Light' if st.session_state.theme == 'dark' else 'Dark'} Mode"):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        st.rerun()

    st.markdown("---")
    st.markdown("Use the controls below to filter the dataset for your queries.")

    if st.button("Try a sample question"):
        sample_queries = [
            "How many groups went to Moody Center last month?",
            "What are the top drop-off spots for 18–24 year-olds on Saturday nights?",
            "When do large groups (6+ riders) typically ride downtown?",
            "What are the peak pickup hours on weekends?",
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


# --- Chat Interface ---
# Replay history and display charts
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "figures" in msg:
            for fig in msg["figures"]:
                st.plotly_chart(fig, use_container_width=True)

# Process new user input
if prompt := st.chat_input("Ask a question...") or st.session_state.pop("pending_query", None):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing data..."):
            intent, entities = parse_query(prompt)
            context, figures = answer_query(trips, riders, demographics, intent, entities)
            response_text = generate_response(query=prompt, intent=intent, entities=entities, context=context)
            
            st.markdown(response_text)
            # Display new figures and save to session state
            assistant_message = {"role": "assistant", "content": response_text, "figures": []}
            for fig in figures:
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                    assistant_message["figures"].append(fig)
            st.session_state.messages.append(assistant_message)
    st.rerun()
