# Marketing Intel Streamlit

A deployable Streamlit dashboard suite for digital advertising performance, competitor creative intelligence, demand pulse monitoring, and media mix planning.

Public repo: https://github.com/AMYFYJ/marketing-intel-streamlit

## Why This App Exists

Modern marketing teams have plenty of data, but the useful signals are scattered across ad platforms, public trend sources, competitor libraries, exports, and campaign spreadsheets. This app turns those disconnected inputs into a practical command center for answering:

- Which paid media segments are working, wasting budget, or showing fatigue?
- What are competitors saying in their ads, content, and public mentions?
- Which demand signals are timely enough to turn into campaigns?
- Where should the next budget dollar go?
- Which data sources are live, optional, missing, or limited?

The app is designed for marketers, founders, analysts, and growth teams who need fast directional intelligence without building a full warehouse or paid BI stack first.

## Pain Points It Solves

- **Fragmented reporting**: performance, competitor, demand, and budget planning usually live in separate tools.
- **Noisy trend monitoring**: public signals are hard to rank by urgency, source confidence, and campaign usefulness.
- **Manual competitor research**: ad libraries and social/news searches expose signals, but not clean creative briefs or next-test recommendations.
- **Unclear budget decisions**: ROAS, CPA, marginal return, risk, and saturation need to be evaluated together.
- **Optional API gaps**: dashboards often break when a paid API key is missing; this app degrades gracefully with preview data, public links, or no-key sources.
- **Low actionability**: raw tables are useful, but teams need recommendations, watchlists, and exportable briefs.

## Feature Overview

### Performance

The Performance tab is a paid media command center for the Kaggle **Digital Advertising Campaign Performance Dataset** or a deterministic fallback sample.

Use it to:

- Filter by date, platform, objective, industry, device, creative format, and budget tier.
- Compare spend, revenue, profit, ROAS, CPA, CTR, CVR, and conversions against the previous period.
- Spot highlights such as top segment, largest profit lift, fatigue watch, scale candidates, and budget at risk.
- Explore trend splits, rolling KPI pulse, day-of-week patterns, segment contribution, movement, and heatmaps.
- Triage campaigns with health scores, anomaly flags, action reasons, search, sorting, and CSV export.
- Model simple spend-change scenarios for directional planning.

### Competitor Intelligence

The Competitor Intelligence tab monitors observed competitor creative and messaging signals across Meta Ad Library, TikTok Creative Center, YouTube, Reddit, and GDELT/news.

Use it to:

- Enter competitors, themes, market, sources, and item limits from a compact command bar.
- Review source health so missing API keys or live-link-only sources are easy to understand.
- See signal summary cards for share-of-voice leader, top theme, top CTA, newest signal, and test-next ideas.
- Decode creative patterns by theme, CTA, sentiment, freshness, source confidence, and signal strength.
- Turn public signals into an action board of `Test next`, `Open source`, `Watch`, `Archive`, or `Fix source`.
- Export creative test briefs for planning and team handoff.

Competitive sources are treated as observed signals, not proof of competitor spend, CTR, CPA, or ROAS.

### Demand Pulse

The Demand Pulse tab turns public category, social, video, and trend-export signals into campaign hooks and content priorities.

Use it to:

- Monitor demand around keywords such as categories, customer problems, competitor terms, or channel themes.
- Combine no-key GDELT and Reddit signals with optional YouTube, Google Trends export, and Pinterest export inputs.
- Review a demand brief with active keywords, source coverage, rising topic, urgency, sentiment shift, audience language, and next move.
- Explore velocity, sentiment vs volume, keyword/source heatmaps, and freshness distribution.
- Classify signals by intent such as pain, question, comparison, purchase research, or general mention.
- Generate campaign hooks and export demand briefs while filtering out high-noise signals.

Demand Pulse surfaces directional public signals, not exact market size, spend, or conversion forecasts.

### Budget Optimizer

The Budget Optimizer tab uses a 250k-row synthetic media-mix dataset by default, with a CSV normalization path and connector stubs for future real data.

Use it to:

- Choose a business goal such as profitable growth, efficient acquisition, scaling winners, or waste reduction.
- Set scenario budget, planning horizon, target ROAS, target CPA, risk tolerance, objective focus, and excluded platforms.
- Review goal fit, primary constraint, expected revenue, profit, conversions, CPA, ROAS, budget to fund, and budget to limit.
- Compare recommended allocation by platform against the current spend mix.
- Inspect tradeoffs across expected CPA, expected ROAS, marginal ROAS, risk, saturation, and recommended shift.
- Export an allocation action plan with priority, decision, business reason, and next step by platform.

### Data Sources

The Data Sources tab documents what powers each workflow and how optional connectors behave.

Use it to:

- Confirm supported campaign CSV names.
- See which public/no-key sources are available.
- Understand which secrets unlock optional APIs.
- Review limitations such as Meta API access, TikTok Creative Center live links, GDELT rate limits, and synthetic optimizer data.

## Data Sources

### Primary Campaign Dataset

The app is designed around the Kaggle **Digital Advertising Campaign Performance Dataset**:

https://www.kaggle.com/datasets/juniornsa/digital-advertising-campaign-performance-dataset

Place the downloaded CSV in `data/` using one of these names:

- `digital_advertising_campaign_performance.csv`
- `digital_ad_campaigns.csv`
- `paid_media_campaigns.csv`
- `campaign_performance.csv`

If no CSV is present, the app uses a deterministic fallback campaign sample with the same metric schema so the dashboard remains deployable.

### Live and Optional Sources

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

The app runs without these secrets. YouTube and Meta API calls show `not configured` status and fall back to public links, preview snapshots, exports, or other no-key sources.

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
- Public competitor and demand sources are directional signals, not verified media performance metrics.
- The synthetic optimizer dataset is generated in memory and is not committed to the repo.
