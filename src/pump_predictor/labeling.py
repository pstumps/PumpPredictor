"""Pump-event labeling.

Implements the anomaly definition from Nam & Frank, "Detecting Pump&Dump Stock
Market Manipulation from Online Forums" (arXiv:2301.11403), adapted to a
forward-looking prediction target:

  * Daily average price DAP_t = (Open + High + Low + Close) / 4
  * Baseline = trailing `baseline_window` days ending at t (mean and std of
    DAP and Volume). Penny-stock baselines are typically flat ("quiet period").
  * Label(t) = 1 if, within the next `forward_window` days, BOTH
      max(DAP) >  BAP + z * std(DAP baseline)   and
      max(Vol) >  BAV + z * std(Vol baseline)
    AND the move clears absolute floors (min_forward_gain, min_volume_multiple).

The absolute floors are our addition: with a near-flat baseline, sigma -> 0 and
a 2-sigma exceedance can be a 2% blip. Requiring, e.g., +25% price and 3x volume
keeps only moves that look like actual pumps.

All baseline statistics use data up to and including day t; the event is sought
strictly after t — so the label is predictive, not descriptive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def daily_avg_price(df: pd.DataFrame) -> pd.Series:
    return (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4.0


def label_pumps(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Return DataFrame indexed like df with columns: label, fwd_max_gain, fwd_max_vol_mult.

    fwd_max_gain / fwd_max_vol_mult are kept for analysis and backtesting;
    only `label` is the training target.
    """
    lcfg = cfg.labeling
    bw, fw = int(lcfg.baseline_window), int(lcfg.forward_window)
    z = float(lcfg.z_threshold)

    dap = daily_avg_price(df)
    vol = df["Volume"].astype(float)

    bap = dap.rolling(bw).mean()
    bap_sd = dap.rolling(bw).std()
    bav = vol.rolling(bw).mean()
    bav_sd = vol.rolling(bw).std()

    # Forward max over the next fw days (exclusive of today):
    # reverse the series, take rolling max of the *past* fw values, shift, reverse back.
    fwd_max_dap = dap[::-1].rolling(fw, min_periods=1).max().shift(1)[::-1]
    fwd_max_vol = vol[::-1].rolling(fw, min_periods=1).max().shift(1)[::-1]

    price_z_hit = fwd_max_dap > (bap + z * bap_sd)
    vol_z_hit = fwd_max_vol > (bav + z * bav_sd)

    gain = fwd_max_dap / bap - 1.0
    vol_mult = fwd_max_vol / bav.clip(lower=1.0)
    gain_hit = gain >= float(lcfg.min_forward_gain)
    volmult_hit = vol_mult >= float(lcfg.min_volume_multiple)

    label = (price_z_hit & vol_z_hit & gain_hit & volmult_hit).astype(int)

    # Undefined where baseline or forward window incomplete.
    valid = bap.notna() & bap_sd.notna() & fwd_max_dap.notna()
    # The last fw rows have a truncated forward window — a pump might still be
    # coming that we can't see. Exclude them from training.
    tail = np.zeros(len(df), dtype=bool)
    tail[-fw:] = True
    valid &= ~pd.Series(tail, index=df.index)

    out = pd.DataFrame({
        "label": label.where(valid),
        "fwd_max_gain": gain.where(valid),
        "fwd_max_vol_mult": vol_mult.where(valid),
    }, index=df.index)
    return out
