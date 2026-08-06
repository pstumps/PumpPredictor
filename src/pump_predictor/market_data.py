"""Download and cache daily OHLCV plus basic fundamentals via yfinance."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]

# Fundamental snapshot fields kept from Ticker.info. NOTE: yfinance fundamentals
# are a *current* snapshot, not point-in-time history — shares outstanding and
# market cap are backfilled over the training window. Documented limitation;
# replace with a point-in-time source (e.g. SEC EDGAR facts API) for production.
INFO_FIELDS = [
    "marketCap", "sharesOutstanding", "floatShares", "shortPercentOfFloat",
    "sector", "exchange",
]


def download_ohlcv(cfg, tickers: list[str]) -> dict[str, pd.DataFrame]:
    """Download daily OHLCV for tickers, caching each to parquet."""
    cache = Path(cfg.paths.ohlcv_dir)
    cache.mkdir(parents=True, exist_ok=True)
    period = cfg.data.history_period
    bs = int(cfg.data.batch_size)

    result: dict[str, pd.DataFrame] = {}
    missing = []
    for t in tickers:
        f = cache / f"{t}.parquet"
        if f.exists():
            result[t] = pd.read_parquet(f)
        else:
            missing.append(t)

    for i in range(0, len(missing), bs):
        chunk = missing[i : i + bs]
        log.info("Downloading OHLCV %d-%d of %d", i + 1, i + len(chunk), len(missing))
        try:
            px = yf.download(chunk, period=period, progress=False, group_by="ticker",
                             auto_adjust=True, threads=True)
        except Exception as e:  # noqa: BLE001
            log.warning("Batch download failed: %s", e)
            continue
        for t in chunk:
            try:
                d = (px[t] if len(chunk) > 1 else px)[OHLCV_COLS].dropna(subset=["Close"])
            except KeyError:
                continue
            if d.empty:
                continue
            d = d[~d.index.duplicated(keep="last")].sort_index()
            d.to_parquet(cache / f"{t}.parquet")
            result[t] = d
        time.sleep(0.5)  # stay polite to the API

    log.info("OHLCV available for %d/%d tickers", len(result), len(tickers))
    return result


def fetch_fundamentals(cfg, tickers: list[str], refresh: bool = False) -> pd.DataFrame:
    """Fetch a fundamentals snapshot per ticker, cached to JSON."""
    cache = Path(cfg.paths.data_dir) / "fundamentals.json"
    data: dict[str, dict] = {}
    if cache.exists() and not refresh:
        data = json.loads(cache.read_text())

    todo = [t for t in tickers if t not in data]
    for n, t in enumerate(todo):
        try:
            info = yf.Ticker(t).info or {}
            data[t] = {k: info.get(k) for k in INFO_FIELDS}
        except Exception:  # noqa: BLE001 - missing fundamentals are fine
            data[t] = {}
        if n % 25 == 24:
            log.info("Fundamentals %d/%d", n + 1, len(todo))
            cache.write_text(json.dumps(data))
            time.sleep(0.5)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data))

    df = pd.DataFrame.from_dict(data, orient="index").reindex(tickers)
    df.index.name = "ticker"
    return df
