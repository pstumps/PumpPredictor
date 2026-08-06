"""Daily scan: score today's universe and rank pump candidates.

Optionally enriches the top candidates with live text/social features
(news lexicon/FinBERT scores, StockTwits buzz). Those live features are shown
alongside the model score rather than fed into it, unless the model was trained
with matching columns — mixing untrained features into inference would be
garbage-in.
"""

from __future__ import annotations

import logging

import pandas as pd

from .dataset import feature_columns  # noqa: F401 - re-export for notebooks
from .features.social import reddit_features, stocktwits_features
from .features.technical import compute_technical_features
from .features.text import news_features, transcript_features
from .features.fundamental import compute_fundamental_features
from .market_data import download_ohlcv, fetch_fundamentals
from .model import load_model
from .universe import load_universe

log = logging.getLogger(__name__)


def run_scan(cfg) -> pd.DataFrame:
    model, feats = load_model(cfg)
    tickers = load_universe(cfg)
    ohlcv = download_ohlcv(cfg, tickers)
    fund = fetch_fundamentals(cfg, list(ohlcv))
    fund_X = compute_fundamental_features(fund)

    rows = []
    for t, df in ohlcv.items():
        if len(df) < 90:
            continue
        X = compute_technical_features(df, cfg).iloc[[-1]]
        X["ticker"] = t
        X["date"] = df.index[-1]
        X["last_close"] = float(df["Close"].iloc[-1])
        rows.append(X)
    live = pd.concat(rows).merge(
        fund_X.reset_index().rename(columns={"index": "ticker"}), on="ticker", how="left"
    )

    live["pump_score"] = model.predict_proba(live[feats])[:, 1]
    top = live.nlargest(int(cfg.scan.top_n), "pump_score")[
        ["ticker", "date", "last_close", "pump_score", "volume_z", "ret_5d",
         "prior_spikes_180d", "log_market_cap"]
    ].reset_index(drop=True)

    # Optional live enrichment on the short list only (one API call per ticker).
    enrich = []
    for t in top["ticker"]:
        e: dict[str, float] = {}
        if cfg.scan.enable_news:
            e.update(news_features(t))
            e.update(transcript_features(cfg, t))
        if cfg.scan.enable_social:
            e.update(stocktwits_features(t))
            e.update(reddit_features(t))
        enrich.append(e)
    if any(enrich):
        top = pd.concat([top, pd.DataFrame(enrich, index=top.index)], axis=1)

    return top
