"""Text features: news headlines, press releases, earnings-call transcripts.

Two scoring backends:
  1. Lexicons (always available):
     - Loughran-McDonald-style financial sentiment word subsets
       (positive / negative / uncertainty) — the standard finance NLP baseline.
     - A promotion/hype lexicon drawn from the stock-touting literature
       (Frieder & Zittrain 2007 spam studies; SEC fraudulent-promotion alerts):
       pump language is distinctive ("explode", "next 10x", "huge alert").
  2. FinBERT (optional, if `transformers` is installed): finance-tuned
     sentiment scores. Literature says tone adds a modest, short-lived signal
     (~1-day, decaying by day 5) — treat as a secondary feature family.

Sources of documents:
  * yfinance `Ticker.news` — recent headlines only, so usable for *live scans*,
    not for backfilled training history (documented limitation; a historical
    news archive with timestamps is the upgrade path).
  * data/text/<TICKER>/*.txt — drop earnings-call transcripts or press releases
    here; each file is scored and aggregated per ticker. True penny/OTC issuers
    rarely hold earnings calls, so press releases are the realistic input.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

# Compact subsets of the Loughran-McDonald dictionary (full lists: ~2k words;
# swap in the complete dictionary from the LM website for production).
LM_POSITIVE = {
    "achieve", "advance", "attain", "beneficial", "boost", "breakthrough",
    "efficient", "excellent", "exceptional", "gain", "growth", "improve",
    "innovative", "leading", "milestone", "opportunity", "outperform",
    "profitable", "record", "strong", "succeed", "success", "surpass", "win",
}
LM_NEGATIVE = {
    "adverse", "bankruptcy", "concern", "decline", "default", "deficit",
    "delist", "dilution", "impairment", "investigation", "lawsuit", "liability",
    "litigation", "loss", "restatement", "subpoena", "suspend", "weak",
    "writedown", "going concern", "deficiency",
}
LM_UNCERTAINTY = {
    "anticipate", "approximate", "believe", "could", "depend", "fluctuate",
    "indefinite", "may", "might", "possible", "risk", "uncertain", "variable",
}
HYPE = {
    "explode", "explosive", "skyrocket", "moon", "rocket", "breakout", "alert",
    "huge", "massive", "insane", "guaranteed", "urgent", "hot", "bonanza",
    "10x", "100x", "double", "triple", "undervalued", "hidden gem", "next big",
    "dont miss", "act now", "buy now", "takeover rumor", "short squeeze",
}

_word_re = re.compile(r"[a-z']+")


def _lexicon_scores(text: str) -> dict[str, float]:
    t = text.lower()
    words = _word_re.findall(t)
    n = max(len(words), 1)
    wset = set(words)

    def frac(lex: set[str]) -> float:
        hits = sum(1 for w in words if w in lex)
        hits += sum(1 for phrase in lex if " " in phrase and phrase in t)
        return hits / n

    return {
        "lm_positive": frac(LM_POSITIVE),
        "lm_negative": frac(LM_NEGATIVE),
        "lm_uncertainty": frac(LM_UNCERTAINTY),
        "hype_score": frac(HYPE),
        "exclaim_per_100w": t.count("!") / n * 100,
        "allcaps_frac": sum(1 for w in text.split() if len(w) > 2 and w.isupper()) / n,
    }


_finbert = None


def _finbert_scores(texts: list[str]) -> list[dict[str, float]] | None:
    """Optional FinBERT scoring; returns None when transformers isn't installed."""
    global _finbert
    try:
        if _finbert is None:
            from transformers import pipeline  # noqa: PLC0415 - lazy heavy import
            _finbert = pipeline("text-classification", model="ProsusAI/finbert",
                                top_k=None, truncation=True)
        out = []
        for res in _finbert(texts):
            d = {f"finbert_{r['label'].lower()}": r["score"] for r in res}
            out.append(d)
        return out
    except Exception as e:  # noqa: BLE001
        log.info("FinBERT unavailable (%s); using lexicons only", e)
        return None


def score_documents(texts: list[str]) -> dict[str, float]:
    """Aggregate feature dict over a list of documents (mean of per-doc scores)."""
    if not texts:
        return {}
    rows = [_lexicon_scores(t) for t in texts]
    fb = _finbert_scores([t[:2000] for t in texts])
    if fb:
        for r, f in zip(rows, fb):
            r.update(f)
    df = pd.DataFrame(rows)
    agg = df.mean().add_suffix("_mean").to_dict()
    agg["doc_count"] = float(len(texts))
    agg["hype_score_max"] = float(df["hype_score"].max())
    return agg


def news_features(ticker: str) -> dict[str, float]:
    """Live news headline features via yfinance (recent items only)."""
    try:
        import yfinance as yf  # noqa: PLC0415
        items = yf.Ticker(ticker).news or []
    except Exception:  # noqa: BLE001
        return {}
    texts = []
    for it in items:
        content = it.get("content", it)
        title = content.get("title") or ""
        summary = content.get("summary") or ""
        if title or summary:
            texts.append(f"{title}. {summary}")
    feats = score_documents(texts)
    return {f"news_{k}": v for k, v in feats.items()}


def transcript_features(cfg, ticker: str) -> dict[str, float]:
    """Score user-supplied transcripts/press releases in data/text/<TICKER>/."""
    d = Path(cfg.paths.text_dir) / ticker.upper()
    if not d.is_dir():
        return {}
    texts = [p.read_text(encoding="utf-8", errors="ignore") for p in sorted(d.glob("*.txt"))]
    feats = score_documents(texts)
    return {f"doc_{k}": v for k, v in feats.items()}
