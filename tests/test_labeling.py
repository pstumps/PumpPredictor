"""Labeler tests on synthetic data: a constructed pump must be labeled ahead of
the spike, and quiet series must produce no labels."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pump_predictor.config import load_config
from pump_predictor.labeling import label_pumps


def make_ohlcv(closes, volumes):
    closes = np.asarray(closes, dtype=float)
    idx = pd.bdate_range("2024-01-02", periods=len(closes))
    return pd.DataFrame({
        "Open": closes * 0.99, "High": closes * 1.02,
        "Low": closes * 0.98, "Close": closes,
        "Volume": np.asarray(volumes, dtype=float),
    }, index=idx)


def test_synthetic_pump_labeled_before_spike():
    cfg = load_config()
    rng = np.random.default_rng(0)
    n = 60
    closes = 1.0 + rng.normal(0, 0.005, n).cumsum() * 0.1
    volumes = np.full(n, 50_000.0) * rng.uniform(0.9, 1.1, n)
    # Pump on days 40-42: price triples, volume 20x.
    closes[40:43] = [2.0, 3.0, 2.5]
    volumes[40:43] = [1_000_000, 2_000_000, 800_000]

    labels = label_pumps(make_ohlcv(closes, volumes), cfg)
    # Days 35-39 look forward into the spike (forward_window=5) — must be labeled.
    assert labels["label"].iloc[36:40].eq(1).all(), labels["label"].iloc[30:45].tolist()
    # Long before the pump: no label.
    assert labels["label"].iloc[10:30].eq(0).all()


def test_quiet_series_has_no_labels():
    cfg = load_config()
    rng = np.random.default_rng(1)
    n = 80
    closes = 2.0 + rng.normal(0, 0.01, n)
    volumes = np.full(n, 100_000.0) * rng.uniform(0.8, 1.2, n)
    labels = label_pumps(make_ohlcv(closes, volumes), cfg)
    assert labels["label"].dropna().eq(0).all()


def test_tail_rows_unlabeled():
    cfg = load_config()
    n = 40
    df = make_ohlcv(np.full(n, 1.0), np.full(n, 10_000.0))
    labels = label_pumps(df, cfg)
    fw = int(cfg.labeling.forward_window)
    assert labels["label"].iloc[-fw:].isna().all()
