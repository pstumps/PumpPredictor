"""Assemble the (date, ticker) feature matrix with pump labels."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .features.fundamental import compute_fundamental_features
from .features.technical import compute_technical_features
from .labeling import label_pumps
from .market_data import download_ohlcv, fetch_fundamentals
from .universe import load_universe

log = logging.getLogger(__name__)


def build_dataset(cfg, save: bool = True) -> pd.DataFrame:
    tickers = load_universe(cfg)
    ohlcv = download_ohlcv(cfg, tickers)
    fund = fetch_fundamentals(cfg, list(ohlcv))
    fund_X = compute_fundamental_features(fund)

    frames = []
    for t, df in ohlcv.items():
        if len(df) < 90:  # need warmup for rolling windows
            continue
        X = compute_technical_features(df, cfg)
        y = label_pumps(df, cfg)
        part = X.join(y)
        part["ticker"] = t
        frames.append(part)

    if not frames:
        raise RuntimeError("No tickers had enough history to build a dataset")

    data = pd.concat(frames)
    data.index.name = "date"
    data = data.reset_index().merge(
        fund_X.reset_index().rename(columns={"index": "ticker"}), on="ticker", how="left"
    )
    # Keep rows where the label is defined and core features have warmed up.
    data = data[data["label"].notna() & data["volatility_20d"].notna()]
    data["label"] = data["label"].astype(int)
    data = data.sort_values(["date", "ticker"]).reset_index(drop=True)

    pos = int(data["label"].sum())
    log.info("Dataset: %d rows, %d tickers, %d positives (%.2f%%)",
             len(data), data["ticker"].nunique(), pos, 100 * pos / max(len(data), 1))

    if save:
        out = Path(cfg.paths.dataset_file)
        out.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(out)
        log.info("Saved dataset -> %s", out)
    return data


FEATURE_EXCLUDE = {"date", "ticker", "label", "fwd_max_gain", "fwd_max_vol_mult"}


def feature_columns(data: pd.DataFrame) -> list[str]:
    return [c for c in data.columns if c not in FEATURE_EXCLUDE]
