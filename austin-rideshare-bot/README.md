## Austin Rideshare Intelligence Assistant

An interactive Streamlit chatbot that answers questions about Fetii's Austin rideshare data.

### Quickstart (Local)
1. Create a virtual environment and install deps:
```
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```
2. Add your Gemini key to a `.env` file:
```
GEMINI_API_KEY=your_key_here
```
(If deploying to Streamlit Cloud, put it in Secrets.)

3. Run the app:
```
streamlit run austin-rideshare-bot/app.py
```

### Data
Place the provided Excel file in the repo root as `FetiiAI_Data_Austin.xlsx`.
If you only have CSVs (e.g., `FetiiAI_Data_Austin.xlsx - Trip Data.csv`), the app will detect and load what it finds.

### Deployment (Streamlit Community Cloud)
- Connect this repo and select `austin-rideshare-bot/app.py` as the entry point.
- Ensure `requirements.txt` exists at the repo root (already included) and add `GEMINI_API_KEY` to app Secrets.

### Structure
```
austin-rideshare-bot/
  app.py
  data_processor.py
  ai_engine.py
  analytics.py
  visualizations.py
  requirements.txt
```
