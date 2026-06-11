# Marketing Intel Streamlit

A deployable Streamlit dashboard suite for digital advertising performance, competitor intelligence, demand pulse monitoring, and media mix planning.

Public repo: https://github.com/AMYFYJ/marketing-intel-streamlit

## What It Does

- **Performance**: paid-media KPI command center with campaign filters, period deltas, ROAS/CPA/CTR/CVR metrics, segment diagnostics, funnel health, campaign actions, and scenario projections.
- **Competitor Intelligence**: competitor and creative monitoring across Meta Ad Library, TikTok Creative Center, LinkedIn Ad Library, the EU-only X Ads Repository, YouTube, Reddit, and GDELT/news — with creative theme/CTA detection, share-of-voice, and next-test strategy recommendations.
- **Demand Pulse**: live demand monitoring for marketing topics across GDELT, Reddit, optional YouTube, and trend exports — with trend lifecycle/momentum staging (GDELT TimelineVolRaw daily volume), signal-confidence and corroboration scoring, demand-vs-baseline anomaly detection, and a capped, explained action queue.
- **Budget Optimizer**: goal-driven media planner — enter a budget and a goal (conversions, revenue, traffic, awareness, leads, app installs, or follower/engagement growth) to get the platform mix, recommended campaign type and audience per platform, budget allocation, and expected cost-per-goal and goal volume. Recommendations are reality-checked (platform-valid creative formats; upper-funnel goals never sold against retargeting pools), small budgets concentrate via a min-spend-per-platform floor, and a Returns & Sensitivity section shows per-platform response curves plus a budget sweep ("what would +$10k buy?"). Demand Pulse momentum and Competitor Intelligence pressure feed a strategic-context panel with an optional whitespace tilt. Synthetic benchmarks are calibrated to realistic industry ranges (platform ROAS ~1.5–4.5x). Also includes an Advanced ROAS/CPA optimizer and CSV/API connector paths. Follower growth isn't reported by ad APIs, so that goal uses ad engagement as a documented proxy.
- **Data Sources**: source attribution and deployment notes.

## Data Sources

### Primary campaign dataset

The app is designed around the Kaggle **Digital Advertising Campaign Performance Dataset**:

https://www.kaggle.com/datasets/juniornsa/digital-advertising-campaign-performance-dataset

Place the downloaded CSV in `data/` using one of these names:

- `digital_advertising_campaign_performance.csv`
- `digital_ad_campaigns.csv`
- `paid_media_campaigns.csv`
- `campaign_performance.csv`

If no CSV is present, the app uses a deterministic fallback campaign sample with the same metric schema so the dashboard remains deployable.

### Live and optional sources

- GDELT Doc API for news/category pulse: https://docs.gdeltproject.org/
- Reddit RSS search for public social mentions.
- YouTube Data API when `YOUTUBE_API_KEY` is configured.
- Meta Ad Library API when `META_ACCESS_TOKEN` is configured, plus public Ad Library search links when it is not.
- TikTok Creative Center live links: https://ads.tiktok.com/business/creativecenter/
- LinkedIn Ad Library live links for public advertiser, keyword, country, and date-range research: https://www.linkedin.com/ads/library/
- X Ads Repository live link for EU Digital Services Act ad transparency research: https://ads.twitter.com/ads-repository
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

The app runs without these secrets. YouTube and Meta API calls will show `not configured` status and fall back to public links, preview snapshots, exports, or other no-key sources.

## Streamlit Community Cloud Deployment

1. Push this repo to GitHub.
2. In Streamlit Community Cloud, create a new app from `AMYFYJ/marketing-intel-streamlit`.
3. Set the app entrypoint to `streamlit_app.py`.
4. Add optional secrets if you have YouTube or Meta API credentials.
5. Deploy.

## Notes and Limitations

- Meta Ad Library API access is optional and separate from private Meta campaign API access.
- TikTok Creative Center does not expose a stable public API in this implementation, so the app provides live deep links rather than scraping.
- LinkedIn Ad Library is supported as a live-link source; deeper API access requires LinkedIn approval.
- X Ads Repository coverage is EU/DSA-specific and should not be read as a global X competitor ad scan.
- GDELT rate-limits frequent calls; the app surfaces `rate limited` status and uses Streamlit caching.
- Public competitor and demand sources are directional signals, not verified media performance metrics.
- Follower growth is not reported by ad APIs, so the Budget Optimizer uses ad engagement as a documented proxy for that goal.
- The synthetic optimizer dataset is generated in memory and is not committed to the repo.
