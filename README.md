# Marketing Intel Streamlit

A deployable Streamlit dashboard suite for digital advertising performance, competitor intelligence, demand pulse monitoring, and media mix planning.

Public repo: https://github.com/AMYFYJ/marketing-intel-streamlit

## What It Does

- **Performance**: paid-media KPI command center with campaign filters, period-over-period deltas, daily/weekly/monthly trends, platform-relative Scale/Optimize/Watch/Pause recommendations, a directional anomaly watchlist, and CSV export. Demo campaigns persist across days and are calibrated to believable benchmarks (~3x blended ROAS).
- **Competitor Intelligence**: competitor and creative monitoring across Meta Ad Library, TikTok Creative Center, YouTube, Reddit, and GDELT/news, with share-of-voice and creative theme/CTA analysis (word-boundary matching, placeholder links excluded) plus clickable live-search links.
- **Demand Pulse**: live demand monitoring with mentions-per-day trends, accelerating/steady/cooling momentum per keyword, VADER sentiment, and plain-language campaign angles.
- **Budget Optimizer**: allocates budget greedily along per-channel diminishing-return curves with risk-adjusted marginal ROAS; ROAS/CPA targets act as constraints and unservable budget is reported as unallocated. Uploaded CSVs get marginal returns fitted from daily spend variation when available.
- **Data Sources**: source attribution, live configuration status, and deployment notes.

## Data Sources

### Primary campaign dataset

The app is designed around the Kaggle **Digital Advertising Campaign Performance Dataset**:

https://www.kaggle.com/datasets/juniornsa/digital-advertising-campaign-performance-dataset

Place the downloaded CSV in `data/` using one of these names:

- `digital_advertising_campaign_performance.csv`
- `digital_ad_campaigns.csv`
- `paid_media_campaigns.csv`
- `campaign_performance.csv`

If no CSV is present, the app uses a deterministic fallback campaign sample with the same metric schema so the dashboard remains deployable. The sample models ~230 persistent campaigns with daily rows over two years, calibrated to believable paid-media economics (blended ROAS ~3x, platform ordering from efficient search to expensive B2B).

### Live and optional sources

- GDELT Doc API for news/category pulse: https://docs.gdeltproject.org/
- Reddit RSS search for public social mentions.
- YouTube Data API when `YOUTUBE_API_KEY` is configured.
- Meta Ad Library API when `META_ACCESS_TOKEN` is configured, plus public Ad Library search links when it is not.
- TikTok Creative Center live links: https://ads.tiktok.com/business/creativecenter/
- Optional `data/google_trends_export.csv` and `data/pinterest_trends_export.csv` files for trend-export workflows.

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Run tests:

```bash
TMPDIR=$PWD/.tmp python -m pytest tests
python scripts/smoke_test.py
```

## Optional Secrets

Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` locally or add the same keys in Streamlit Community Cloud secrets.

```toml
YOUTUBE_API_KEY = ""
META_ACCESS_TOKEN = ""
META_GRAPH_VERSION = "v21.0"
```

The app runs without these secrets. YouTube and Meta API calls will show `not configured` status and fall back to public links or other sources.

## Streamlit Community Cloud Deployment

1. Push this repo to GitHub.
2. In Streamlit Community Cloud, create a new app from `AMYFYJ/marketing-intel-streamlit`.
3. Set the app entrypoint to `streamlit_app.py`.
4. Add optional secrets if you have YouTube or Meta API credentials.
5. Deploy.

## Notes and Limitations

- Meta Ad Library API access is optional and separate from private Meta campaign API access.
- TikTok Creative Center does not expose a stable public API in this implementation, so the app provides live deep links rather than scraping.
- GDELT rate-limits frequent calls; the app surfaces `rate limited` status and uses Streamlit caching.
- The synthetic optimizer dataset is generated in memory and is not committed to the repo.
