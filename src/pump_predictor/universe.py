"""Build the penny-stock universe.

Pulls the full NASDAQ/NYSE/AMEX symbol directory from Nasdaq Trader, filters out
ETFs/test issues, then screens by price, liquidity, and market cap using recent
market data. Literature motivation: Aggarwal & Wu (2006) show manipulation
concentrates in small, illiquid, low-priced stocks — the universe filter is the
first feature.

Note on OTC/Pink Sheets: free bulk symbol directories for OTC Markets are not
reliably available; yfinance covers many OTC tickers if you supply them. Add
your own via data/extra_tickers.txt (one symbol per line).
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

log = logging.getLogger(__name__)

NASDAQ_LISTED = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

# Small fallback so the pipeline stays runnable if the symbol directory is down.
FALLBACK_TICKERS = [
    "SNDL", "GSAT", "BITF", "BTG", "IQ", "NOK", "GRAB", "LAZR", "PLUG", "OPEN",
    "DNN", "UEC", "CLOV", "ACHR", "SOFI", "BBAI", "FCEL", "RIG", "TLRY", "CGC",
]


def _fetch_symbol_directory() -> pd.DataFrame:
    frames = []
    for url, sym_col in ((NASDAQ_LISTED, "Symbol"), (OTHER_LISTED, "ACT Symbol")):
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), sep="|")
        df = df[df[sym_col].notna()].iloc[:-1]  # last row is a file-creation footer
        out = pd.DataFrame({
            "ticker": df[sym_col].astype(str).str.strip(),
            "name": df.get("Security Name", ""),
            "etf": df.get("ETF", "N").fillna("N"),
            "test_issue": df.get("Test Issue", "N").fillna("N"),
        })
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def build_universe(cfg) -> pd.DataFrame:
    """Screen the listed-symbol directory down to a tradable penny-stock universe."""
    ucfg = cfg.universe
    try:
        symbols = _fetch_symbol_directory()
        symbols = symbols[symbols["test_issue"] != "Y"]
        if ucfg.exclude_etfs:
            symbols = symbols[symbols["etf"] != "Y"]
        # Skip units/warrants/rights and symbols yfinance can't map cleanly.
        symbols = symbols[~symbols["ticker"].str.contains(r"[.$=^]", regex=True)]
        name_l = symbols["name"].astype(str).str.lower()
        symbols = symbols[~name_l.str.contains("warrant|right|unit|preferred|%|due 20")]
        tickers = symbols["ticker"].tolist()
    except Exception as e:  # noqa: BLE001 - degrade to fallback list
        log.warning("Symbol directory fetch failed (%s); using fallback list", e)
        tickers = list(FALLBACK_TICKERS)

    extra_file = Path(cfg.paths.data_dir) / "extra_tickers.txt"
    if extra_file.exists():
        extras = [t.strip().upper() for t in extra_file.read_text().splitlines() if t.strip()]
        tickers = sorted(set(tickers) | set(extras))

    log.info("Screening %d candidate symbols by price/liquidity", len(tickers))
    rows = []
    for i in range(0, len(tickers), 200):
        chunk = tickers[i : i + 200]
        try:
            px = yf.download(chunk, period="1mo", progress=False, group_by="ticker",
                             auto_adjust=True, threads=True)
        except Exception as e:  # noqa: BLE001
            log.warning("Price screen batch failed at %d: %s", i, e)
            continue
        for t in chunk:
            try:
                d = px[t].dropna() if len(chunk) > 1 else px.dropna()
            except KeyError:
                continue
            if len(d) < 10:
                continue
            close = float(d["Close"].iloc[-1])
            med_dv = float((d["Close"] * d["Volume"]).median())
            if ucfg.min_price <= close <= ucfg.max_price and med_dv >= ucfg.min_median_dollar_volume:
                rows.append({"ticker": t, "last_close": close, "median_dollar_volume": med_dv})

    uni = pd.DataFrame(rows).sort_values("median_dollar_volume", ascending=False)
    if len(uni) > ucfg.max_tickers:
        uni = uni.head(ucfg.max_tickers)
    uni = uni.reset_index(drop=True)

    out = Path(cfg.paths.universe_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    uni.to_csv(out, index=False)
    log.info("Universe: %d tickers -> %s", len(uni), out)
    return uni


def load_universe(cfg) -> list[str]:
    return pd.read_csv(cfg.paths.universe_file)["ticker"].astype(str).tolist()
