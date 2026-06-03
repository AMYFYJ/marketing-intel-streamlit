# Marketing Intel Streamlit

A deployable Streamlit dashboard suite for digital advertising performance, competitor intelligence, demand pulse monitoring, and media mix planning.

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Data strategy

- Paid media performance uses a public campaign-performance dataset when available and falls back to generated sample data with the same business metrics.
- Competitor and demand modules use live public adapters with graceful fallbacks.
- Budget optimization starts with a 250k-row synthetic media-mix dataset and exposes connector interfaces for future real platform data.
