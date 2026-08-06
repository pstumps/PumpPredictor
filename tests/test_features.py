"""Feature tests: no lookahead — features at day t must not change when future
rows change — and text lexicon scoring sanity."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pump_predictor.config import load_config
from pump_predictor.features.technical import compute_technical_features
from pump_predictor.features.text import score_documents


def _random_ohlcv(n=300, seed=42):
    rng = np.random.default_rng(seed)
    closes = np.abs(2 + rng.normal(0, 0.05, n).cumsum())
    idx = pd.bdate_range("2023-01-02", periods=n)
    return pd.DataFrame({
        "Open": closes * rng.uniform(0.97, 1.0, n),
        "High": closes * rng.uniform(1.0, 1.05, n),
        "Low": closes * rng.uniform(0.95, 1.0, n),
        "Close": closes,
        "Volume": rng.integers(10_000, 200_000, n).astype(float),
    }, index=idx)


def test_no_lookahead():
    cfg = load_config()
    df = _random_ohlcv()
    X_full = compute_technical_features(df, cfg)

    mutated = df.copy()
    mutated.iloc[-40:, mutated.columns.get_loc("Close")] *= 10  # absurd future move
    mutated.iloc[-40:, mutated.columns.get_loc("Volume")] *= 50
    X_mut = compute_technical_features(mutated, cfg)

    cutoff = len(df) - 40
    pd.testing.assert_frame_equal(X_full.iloc[:cutoff], X_mut.iloc[:cutoff])


def test_hype_lexicon_scores_pump_language_higher():
    pump_text = ["URGENT alert!!! This hidden gem will explode and skyrocket 10x, buy now!"]
    normal_text = ["The company reported quarterly revenue in line with prior guidance."]
    s_pump = score_documents(pump_text)
    s_norm = score_documents(normal_text)
    assert s_pump["hype_score_mean"] > s_norm["hype_score_mean"]
    assert s_pump["exclaim_per_100w_mean"] > 0
