"""Social-media buzz features (live scan only).

Literature: message-volume surges precede pumps more reliably than message
sentiment (arXiv:2412.18848 found social features marginal vs market data;
Nam & Frank got F1 62% from Reddit text alone). We therefore emphasize *counts*
and buzz intensity over polarity.

StockTwits: public unauthenticated JSON API, per-symbol stream.
Reddit: optional, requires PRAW credentials in env
  (REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET / REDDIT_USER_AGENT).

Historical backfill of social data is the hard part of this project — free APIs
only expose recent messages, so these features apply to live scans. To train on
them you need an archive (e.g. academic Pushshift access or a paid provider).
"""

from __future__ import annotations

import datetime as dt
import logging
import os

import requests

log = logging.getLogger(__name__)

ST_URL = "https://api.stocktwits.com/api/2/streams/symbol/{sym}.json"


def stocktwits_features(ticker: str) -> dict[str, float]:
    try:
        r = requests.get(ST_URL.format(sym=ticker), timeout=10,
                         headers={"User-Agent": "pump-predictor-research"})
        if r.status_code != 200:
            return {}
        msgs = r.json().get("messages", [])
    except Exception:  # noqa: BLE001
        return {}
    if not msgs:
        return {}

    now = dt.datetime.now(dt.timezone.utc)
    ages_h, bull, bear = [], 0, 0
    for m in msgs:
        try:
            created = dt.datetime.strptime(m["created_at"], "%Y-%m-%dT%H:%M:%SZ")
            ages_h.append((now - created.replace(tzinfo=dt.timezone.utc)).total_seconds() / 3600)
        except Exception:  # noqa: BLE001
            pass
        sent = ((m.get("entities") or {}).get("sentiment") or {}).get("basic")
        bull += sent == "Bullish"
        bear += sent == "Bearish"

    n = len(msgs)
    feats = {
        "st_msgs_24h": float(sum(1 for a in ages_h if a <= 24)),
        "st_bull_ratio": bull / max(bull + bear, 1),
    }
    if len(ages_h) >= 2:
        span_h = max(max(ages_h) - min(ages_h), 0.5)
        feats["st_msgs_per_hour"] = n / span_h  # stream velocity = buzz intensity
    return feats


def reddit_features(ticker: str, subreddits=("pennystocks", "wallstreetbets")) -> dict[str, float]:
    cid = os.environ.get("REDDIT_CLIENT_ID")
    if not cid:
        return {}
    try:
        import praw  # noqa: PLC0415 - optional dependency

        reddit = praw.Reddit(
            client_id=cid,
            client_secret=os.environ["REDDIT_CLIENT_SECRET"],
            user_agent=os.environ.get("REDDIT_USER_AGENT", "pump-predictor-research"),
        )
        mentions = score_sum = 0
        for sub in subreddits:
            for post in reddit.subreddit(sub).search(ticker, time_filter="week", limit=50):
                mentions += 1
                score_sum += post.score
        return {"reddit_mentions_7d": float(mentions), "reddit_score_7d": float(score_sum)}
    except Exception as e:  # noqa: BLE001
        log.info("Reddit features unavailable: %s", e)
        return {}
