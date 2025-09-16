import os
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv

from data_processor import load_datasets
from ai_engine import (
    configure_gemini,
    parse_query,
    generate_response,
)
from analytics import answer_query

st.set_page_config(page_title="Austin Rideshare Intelligence Assistant", layout="wide")
load_dotenv()

# Get key from env first, then Streamlit secrets
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    try:
        GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]  # type: ignore[index]
    except Exception:
        GEMINI_API_KEY = None

configure_gemini(api_key=GEMINI_API_KEY)

@st.cache_data(show_spinner=False)
def _cached_load() -> Dict[str, Any]:
    return load_datasets()

# Sidebar
with st.sidebar:
    st.title("FetiiAI: Austin Intelligence")
    st.markdown("Ask about locations, time patterns, demographics, or operations.")
    sample_queries = [
        "How many groups went to Moody Center last month?",
        "What are the top drop-off spots for 18–24 year-olds on Saturday nights?",
        "When do large groups (6+ riders) typically ride downtown?",
        "What are the peak pickup hours on weekends?",
    ]
    if st.button("Use a random sample question"):
        import random
        st.session_state["pending_query"] = random.choice(sample_queries)

# Data
with st.spinner("Loading data..."):
    data = _cached_load()
    trips = data.get("trips")
    riders = data.get("riders")
    demographics = data.get("demographics")
    meta = data.get("meta", {})

with st.sidebar:
    st.subheader("Data Status")
    st.markdown(f"Trips loaded: **{len(trips):,}**")
    if meta.get("time_range"):
        st.caption(
            f"Time range: {meta['time_range']['min']} → {meta['time_range']['max']}"
        )
    if not GEMINI_API_KEY:
        st.warning("Gemini API key not set. Using local responses.")

st.title("Austin Rideshare Intelligence Assistant")
st.caption("Conversational insights from Fetii's Austin data")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask me about Austin rideshare patterns."}
    ]

# Replay history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
pending = st.session_state.pop("pending_query", None)
if pending:
    user_query = pending
else:
    user_query = st.chat_input("Ask a question")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            intent, entities = parse_query(user_query)
            context, figures = answer_query(trips, riders, demographics, intent, entities)
            response = generate_response(
                query=user_query,
                intent=intent,
                entities=entities,
                context=context,
            )
            st.markdown(response)
            for fig in figures:
                if fig is None:
                    continue
                # Plotly
                try:
                    import plotly.graph_objects as go
                    if isinstance(fig, go.Figure):
                        st.plotly_chart(fig, use_container_width=True)
                        continue
                except Exception:
                    pass
                # HTML string or folium-like object
                if isinstance(fig, str):
                    st.components.v1.html(fig, height=480, scrolling=False)
                elif hasattr(fig, "_repr_html_"):
                    try:
                        st.components.v1.html(fig._repr_html_(), height=480, scrolling=False)
                    except Exception:
                        st.write(fig)
                else:
                    st.write(fig)

        st.session_state.messages.append({"role": "assistant", "content": response})
        # Force rerun to show the chat input again
        st.rerun()
