# Austin Rideshare Intelligence Assistant

An interactive AI-powered chatbot that provides intelligent insights about Fetii's Austin rideshare data. Built for the **FetiiAI Hackathon.**

## Features

- **Natural Language Queries**: Ask questions in plain English about rideshare patterns
- **Intelligent Data Analysis**: AI-powered query understanding with Google Gemini Pro
- **Interactive Visualizations**: Dynamic charts and maps using Plotly and Folium
- **Conversational Interface**: Streamlit-based chat UI with persistent conversation history
- **Real-time Insights**: Analyze 2,000+ trips with location, time, and demographic data

### Sample Questions You Can Ask:

- *"When do large groups (6+ riders) typically ride downtown?"*
- *"What are the peak pickup hours on weekends?"*
- *"How many groups went to Moody Center last month?"*
- *"What are the top drop-off spots for 18–24 year-olds?"*
- *"Show me trip patterns over time"*

## Quick Start (Local Development)

### 1. Setup Environment

```bash
# Clone the repository
git clone <your-repo-url>
cd FetiiAI_Austin_Rideshare_Assistant

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -U pip
pip install -r requirements.txt
```

### 2. Configure API Key

Create a `.env` file in the project root:

```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

**Get your free Gemini API key:**

1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create a new API key
3. Copy and paste it into your `.env` file

### 3. Add Data

Place the provided data file in the project root:

- **Preferred**: `FetiiAI_Data_Austin.xlsx` (Excel format)
- **Alternative**: `FetiiAI_Data_Austin.xlsx - Trip Data.csv` (CSV format)

The app automatically detects and loads whichever format is available.

### 4. Run the Application

```bash
streamlit run austin-rideshare-bot/app.py
```

Open your browser to `http://localhost:8501` and start asking questions!

## Architecture

```
austin-rideshare-bot/
├── app.py                 # Main Streamlit application
├── data_processor.py      # Data loading, cleaning, and feature engineering
├── ai_engine.py          # Gemini AI integration and query parsing
├── analytics.py          # Data analysis and filtering logic
├── visualizations.py     # Plotly and Folium chart generation
├── requirements.txt      # Python dependencies
└── README.md            # This file

Project Root/
├── austin-rideshare-bot/ # Main application directory
├── FetiiAI_Data_Austin.xlsx         # Data file (Excel)
├── FetiiAI_Data_Austin.xlsx - Trip Data.csv  # Data file (CSV backup)
└── README.md            # Project overview
```

## Technical Stack

- **Frontend & Hosting**: Streamlit + Streamlit Community Cloud
- **Backend**: Python 3.9+, Pandas, NumPy
- **AI Integration**: Google Gemini Pro API (free tier)
- **Data Visualization**: Plotly (interactive charts), Folium (maps)
- **Data Processing**: Pandas, geospatial calculations
- **Development**: GitHub for version control and deployment

## Data Overview

The application analyzes **2,000 rideshare trips** with the following features:

- **Trip Details**: Pickup/dropoff locations, addresses, timestamps
- **Group Information**: Number of riders per trip (6-14 passengers)
- **Demographics**: Age group distributions
- **Temporal Data**: Trip dates from August-September 2025
- **Geospatial**: Austin-area coordinates with landmark detection

## License & Credits

Built for the **FetiiAI Hackathon** - September 2025

- **Data**: Fetii Austin rideshare trip dataset
- **AI**: Google Gemini Pro API
- **Hosting**: Streamlit Community Cloud
