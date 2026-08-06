"""Fundamental / static features per ticker.

arXiv:2412.18848 found market-cap statistics the single most predictive family;
Aggarwal & Wu (2006) show small, low-float companies are preferred targets.

Caveat (documented in market_data.py): yfinance fundamentals are a current
snapshot, not point-in-time. Good enough to rank the cross-section; upgrade to
SEC EDGAR company facts for a production system.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_fundamental_features(fund: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=fund.index)
    mcap = pd.to_numeric(fund.get("marketCap"), errors="coerce")
    shares = pd.to_numeric(fund.get("sharesOutstanding"), errors="coerce")
    flt = pd.to_numeric(fund.get("floatShares"), errors="coerce")

    X["log_market_cap"] = np.log1p(mcap)
    X["log_shares_out"] = np.log1p(shares)
    X["float_ratio"] = (flt / shares).clip(0, 1)
    X["short_pct_float"] = pd.to_numeric(fund.get("shortPercentOfFloat"), errors="coerce")
    X["is_otc"] = (
        fund.get("exchange", pd.Series(index=fund.index, dtype=object))
        .astype(str).str.contains("PNK|OTC", case=False, na=False).astype(int)
    )
    # Sector as a stable categorical code (LightGBM handles int categories fine).
    sector = fund.get("sector", pd.Series(index=fund.index, dtype=object)).astype(str)
    X["sector_code"] = sector.astype("category").cat.codes
    return X
