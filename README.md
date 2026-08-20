# PumpPredictor

ML pipeline that predicts pump-like price/volume anomalies in penny stocks
(< $5, microcap) a few days before they occur, using daily market data,
promotion-history footprints, fundamentals, and (optionally) news text,
earnings-call transcripts, and social-media buzz.

The design is literature-driven — see [docs/RESEARCH.md](docs/RESEARCH.md) for
the full review and the paper-to-feature mapping. Headline findings baked in:

- **Market data dominates**: market cap, price level, and volume anomalies are
  the strongest predictors (arXiv:2412.18848); social/sentiment adds marginal lift.
- **Label definition** follows Nam & Frank (arXiv:2301.11403): a pump is a move
  where price **and** volume exceed baseline mean + 2σ within the next 5 days,
  plus absolute floors (+25% price, 3× volume) to reject flat-baseline artifacts.
- **Promotion history repeats**: previously spiked/promoted stocks are re-targeted
  (SEC microcap-fraud guidance) — the model gets prior-spike features.

## Quick start

```bash
pip install -r requirements.txt
python scripts/run_pipeline.py all        # universe -> download -> dataset -> train -> scan
```

Or step by step:

```bash
python scripts/run_pipeline.py universe   # screen listed symbols to penny universe
python scripts/run_pipeline.py download   # cache ~2y daily OHLCV per ticker
python scripts/run_pipeline.py dataset    # labeled (date, ticker) feature matrix
python scripts/run_pipeline.py train      # LightGBM + logistic baseline, time-split eval
python scripts/run_pipeline.py scan       # rank today's top pump candidates
```

Everything is configured in [config.yaml](config.yaml) (universe filters, label
thresholds, feature windows, model params).

## Layout

```
src/pump_predictor/
  universe.py        symbol directory -> price/liquidity/size screen
  market_data.py     OHLCV + fundamentals download & caching (yfinance)
  labeling.py        forward-looking pump labels (Nam & Frank rule + floors)
  features/
    technical.py     returns, volatility, volume z-scores, illiquidity,
                     spike history, quiet-period shape   (trained on)
    fundamental.py   market cap, float, short interest, sector, OTC flag
    text.py          LM + hype lexicons, optional FinBERT; scores news and
                     any transcripts you drop in data/text/<TICKER>/*.txt
    social.py        StockTwits buzz velocity, optional Reddit mentions
  dataset.py         assembles the labeled matrix
  model.py           time-split + purge-gap training, PR-AUC & precision@k
  scan.py            daily ranked candidate list (+ live text/social enrichment)
scripts/run_pipeline.py   CLI
tests/                    labeler correctness + no-lookahead guarantees
```

## Evaluation philosophy

Pumps are rare (~1-5% of ticker-days), so accuracy and ROC-AUC flatter the
model. The metrics that matter here:

- **PR-AUC** vs. the base rate
- **precision@k** — "of the top-k names flagged each day, how many pumped?"
  This mirrors how you'd actually use a screener.
- Time-based split with a purge gap (no forward-window leakage), never a
  random split.

## Data-source honesty

| Source | Status | Note |
|---|---|---|
| Daily OHLCV (yfinance) | full history | backbone of training |
| Fundamentals (yfinance) | current snapshot | not point-in-time; upgrade to SEC EDGAR facts |
| News headlines (yfinance) | recent only | live scans; archive accrues if you run daily |
| Earnings calls / PRs | bring your own | drop .txt into `data/text/<TICKER>/`; scored by lexicons/FinBERT. True OTC issuers rarely hold calls |
| StockTwits / Reddit | live only | free APIs have no history; the pipeline can accumulate its own archive over daily runs |
| OTC Markets promotion flag | paid feed | best label-quality upgrade if acquired |

## Disclaimers

This is a research/educational tool for studying market-manipulation anomalies.

- Not investment advice. Penny stocks are extremely high-risk: wide spreads
  (often 5-10%+ — enough to erase a predicted move), thin books, halts, and
  outright fraud. A model score is not an edge after costs until proven in a
  cost-aware backtest.
- Predicting pumps ≠ participating in them. Organizing, promoting, or knowingly
  trading as part of a pump-and-dump is securities fraud. The same signals this
  model learns are what regulators use to *detect* manipulation.
- yfinance data is unofficial and can be revised, delisted-survivorship-biased,
  and rate-limited. Validate before trusting any backtest built on it.
