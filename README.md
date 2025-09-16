# Austin Rideshare Intelligence Assistant

This repo contains a Streamlit chatbot that answers questions about Fetii's Austin rideshare data.

- App entrypoint: `austin-rideshare-bot/app.py`
- Requirements: `requirements.txt`
- Expected data: `FetiiAI_Data_Austin.xlsx` in repo root (CSV fallbacks supported)

## Local Setup
```
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# Add your Gemini key
echo "GEMINI_API_KEY=your_key_here" > .env

# Run
streamlit run austin-rideshare-bot/app.py
```

## Deployment (Streamlit Community Cloud)
- Connect this GitHub repo and set the entry point to `austin-rideshare-bot/app.py`.
- Add `GEMINI_API_KEY` in app Secrets.

## Project Structure
```
austin-rideshare-bot/
  app.py
  data_processor.py
  ai_engine.py
  analytics.py
  visualizations.py
  requirements.txt
```