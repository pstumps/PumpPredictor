"""Price/volume/microstructure features from daily OHLCV.

Every feature at date t uses data up to and including t only (no lookahead).

Literature mapping:
  * Volume z-score & multiples vs trailing baseline — the dissemination-phase
    volume surge is the canonical pump precursor (Kamps & Kleinberg 2018;
    Nam & Frank 2023; arXiv:2412.18848 found volume max a top XGBoost feature).
  * Volatility (incl. Parkinson high-low) — manipulation raises volatility
    (Aggarwal & Wu 2006).
  * Amihud illiquidity & high-low range — illiquid stocks are preferred targets
    (Aggarwal & Wu 2006); order-book spread features in arXiv:2412.18848.
  * Price level / distance from lows — low absolute price is the classic
    penny-pump target profile.
  * Prior spike history — repeat promotion is common; stocks pumped before get
    pumped again (SEC microcap fraud guidance).
  * Baseline flatness — the "quiet period" before the pump (Nam & Frank 2023).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..labeling import daily_avg_price


def compute_technical_features(df: pd.DataFrame, cfg) -> pd.DataFrame:
    f = cfg.features
    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    vol = df["Volume"].astype(float)
    dollar_vol = close * vol
    ret = close.pct_change()

    X = pd.DataFrame(index=df.index)

    # --- Returns over multiple horizons ---
    for w in f.ret_windows:
        X[f"ret_{w}d"] = close.pct_change(w)

    # --- Volatility ---
    for w in f.vol_windows:
        X[f"volatility_{w}d"] = ret.rolling(w).std()
    with np.errstate(divide="ignore", invalid="ignore"):
        park = (np.log(high / low) ** 2) / (4 * np.log(2))
    X["parkinson_vol_20d"] = np.sqrt(park.replace([np.inf, -np.inf], np.nan).rolling(20).mean())

    # --- Volume anomalies ---
    vw = int(f.volume_baseline_window)
    vmean, vstd = vol.rolling(vw).mean(), vol.rolling(vw).std()
    X["volume_z"] = (vol - vmean) / vstd.replace(0, np.nan)
    X["volume_ratio_5_60"] = vol.rolling(5).mean() / vol.rolling(60).mean().replace(0, np.nan)
    X["log_dollar_vol_20d"] = np.log1p(dollar_vol.rolling(20).mean())

    # --- Illiquidity / spread proxies ---
    X["amihud_20d"] = (ret.abs() / dollar_vol.replace(0, np.nan)).rolling(20).mean() * 1e6
    X["hl_range_20d"] = ((high - low) / close.replace(0, np.nan)).rolling(20).mean()
    X["zero_vol_days_20d"] = (vol == 0).rolling(20).sum()

    # --- Price level & position ---
    X["log_close"] = np.log(close.clip(lower=1e-4))
    X["below_1_dollar"] = (close < 1.0).astype(int)
    roll_max = close.rolling(252, min_periods=60).max()
    roll_min = close.rolling(252, min_periods=60).min()
    rng = (roll_max - roll_min).replace(0, np.nan)
    X["pos_52w"] = (close - roll_min) / rng
    X["dist_52w_high"] = close / roll_max - 1.0

    # --- Momentum shape ---
    X["updays_10d"] = (ret > 0).rolling(10).sum()
    dap = daily_avg_price(df)
    bap = dap.rolling(int(cfg.labeling.baseline_window)).mean()
    X["baseline_flatness"] = (
        dap.rolling(int(cfg.labeling.baseline_window)).std() / bap.replace(0, np.nan)
    )
    X["close_vs_baseline"] = close / bap.replace(0, np.nan) - 1.0

    # --- Promotion / spike history ---
    lb, sz = int(f.spike_lookback), float(f.spike_z)
    spike = ((vol - vmean) / vstd.replace(0, np.nan) > sz).astype(int)
    X["prior_spikes_180d"] = spike.rolling(lb, min_periods=vw).sum()
    # Days since last spike (capped at lookback).
    idx = np.arange(len(df), dtype=float)
    last_spike = pd.Series(np.where(spike.values == 1, idx, np.nan), index=df.index).ffill()
    X["days_since_spike"] = (idx - last_spike.values).clip(max=lb)
    X["days_since_spike"] = X["days_since_spike"].fillna(lb)

    return X
