# Literature Review: Predicting Penny-Stock Pumps

What the research says about which parameters signal an imminent pump, and how
each finding maps to features in this codebase.

## 1. The reference paper

**[Pump-and-Dump Detection (arXiv:2412.18848)](https://arxiv.org/html/2412.18848v1)** —
real-time prediction of crypto pump targets from Telegram channels, exchange
order books, and market data. Key transferable findings:

- **Traditional market data beat social data.** Across all four XGBoost
  variants, the dominant features were **market-cap statistics** (median, max,
  min, std), **closing-price extremes**, and **trading-volume maxima**. Social
  and sentiment features (LunarCrush interactions, social dominance) added only
  marginal lift. → our `log_market_cap`, price-level, and volume features.
- **Targets are tiny.** Median market cap of pumped coins: **$2.7M**; 95.7%
  under $60M. Pre-filtering the universe by size is essential. → the universe
  screen (`max_market_cap`, `max_price`) is itself the first model input.
- **Microstructure anomalies arrive late.** Bid-ask spread, order imbalance,
  and order-flow anomalies appeared only **seconds** before crypto pumps.
  Equity pumps unfold over days-to-months, so daily illiquidity/spread proxies
  (Amihud, high-low range) remain useful, but tick-level order-book feeds are
  not the first thing to build.
- **Z-score anomaly framing works**: score short-term metrics against their
  historical mean/σ. → our `volume_z`, baseline-deviation features, and the
  labeling rule itself.

## 2. Equity pump-and-dump (the direct analogues)

**[Nam & Frank 2023, "Detecting Pump&Dump Stock Market Manipulation from Online
Forums" (arXiv:2301.11403)](https://arxiv.org/abs/2301.11403)** — the closest
published work to this project: penny stocks + Reddit.

- **Pump-event definition** (adopted here as the labeling rule): daily average
  price `DAP = (O+H+L+C)/4`; a pump is when price **and** volume exceed the
  **5-day baseline mean + 2σ**. We add absolute floors (+25% price, 3× volume)
  because near-flat baselines make σ→0 and 2σ trivial.
- Penny-stock baselines are **flat before the pump** ("quiet period") → our
  `baseline_flatness` feature.
- **Dumps complete within ~4 days** of the peak (citing Sabherwal et al.) →
  our 5-day forward labeling window.
- Text-only NLP models on forum posts reached **85% accuracy / F1 62%** —
  strong evidence that promotion language is predictive. Their most impactful
  words were hype/agreement terms ("buy", "go", sector names).

**[Aggarwal & Wu 2006, "Stock Market Manipulations" (J. Business)](https://www.researchgate.net/publication/24103583_Stock_Market_Manipulations)** —
foundational empirical study of SEC manipulation cases:

- Manipulated stocks are **small, illiquid, low-priced**, with low analyst
  coverage/institutional ownership.
- Manipulation **raises volatility, liquidity, and returns** during the pump;
  prices fall in the post-period. → volatility, turnover, illiquidity features.
- Equity pump episodes often last **weeks to months** (vs. minutes in crypto),
  which is why daily-frequency features can work at all.

**[Detecting Pump-and-Dumps with Crypto-Assets (MDPI Econometrics 2023)](https://www.mdpi.com/2225-1146/11/3/22)**
and **[survey of P&D detection ML (MDPI 2023)](https://www.mdpi.com/1999-5903/15/8/267)**:

- Confirm the **class-imbalance problem** dominates model design (pumps are
  1-5% of samples) → class weighting + PR-AUC/precision@k evaluation.
- Note **insiders' anticipated purchases**: volume creeps up *before* the
  public announcement as insiders accumulate → trailing volume-trend features
  (`volume_ratio_5_60`) target exactly this accumulation footprint.
- Pumps concentrate in assets with market cap **below $50M**.

## 3. Promotion campaigns — the strongest single signal

**Stock-spam studies** (Frieder & Zittrain 2007 "Spam Works"; Hanke & Hauser
2008) and **[SEC fraudulent-promotion alerts](https://www.investor.gov/introduction-investing/general-resources/news-alerts/alerts-bulletins/investor-alerts/investor-25)** /
**[microcap fraud guidance](https://www.investor.gov/additional-resources/spotlight/microcap-fraud)**:

- An **active promotion campaign** (spam email, paid touts, newsletter blasts)
  is the most direct precursor of an equity pump; touted stocks spike in
  volume/price during campaigns and reverse after.
- **Repeat targeting is common** — stocks promoted once get promoted again, and
  the SEC lists "previously suspended/promoted" as a top red flag → our
  `prior_spikes_180d` / `days_since_spike` features are the price-history proxy.
- **[OTC Markets' Promotion Flag and Caveat Emptor designations](https://blog.otcmarkets.com/2017/12/06/otc-markets-group-establishes-a-stock-promotion-policy/)**
  (live since 2018) are exchange-curated labels of active campaigns — the best
  available *ground-truth-adjacent* label source for equity pumps. Their
  Compliance Data Feed is paid; if you get access, join it as both a feature
  and a label refinement.

## 4. Text: earnings calls, news, filings

- **FinBERT** ([Huang et al. 2020, arXiv:2006.08097](https://arxiv.org/pdf/2006.08097))
  is the standard finance-tuned sentiment model (88.7% tone accuracy on
  AnalystTone); **Loughran-McDonald** dictionaries are the lexicon baseline.
- Replications ([e.g. earnings-sentiment studies](https://github.com/rj694/earnings-sentiment))
  find call-tone → returns correlation of only **ρ≈0.3 at 1 day, gone by day
  5**. Tone is a *secondary* feature family — consistent with the reference
  paper's finding that market data dominates.
- **Practical constraint**: true penny/OTC issuers rarely hold earnings calls.
  The realistic text stream is **press releases and SEC filings** — and for
  pumps specifically, promotional *third-party* content matters more than
  issuer content. A news-volume burst (many promo articles in days) is itself
  a dissemination-phase signal → `news_doc_count`, `hype_score`, lexicon scores.
- Red-flag language worth flagging from filings: going-concern, dilution,
  shell-company status, reverse splits, S-1/424B registrations before spikes.

## 5. Social media

- **[Nam & Frank](https://arxiv.org/abs/2301.11403)** (Reddit, stocks) and the
  Telegram literature agree: **message *volume* surges are more reliable than
  message *sentiment*** — buzz intensity is the signal → StockTwits message
  velocity and Reddit mention counts, z-scored, over polarity scores.
- Twitter/X studies ([sentiment→returns ML](https://www.jsr.org/hs/index.php/path/article/view/7416),
  [Reddit/Twitter volatility](https://www.researchgate.net/publication/396206198_Analyzing_the_Impact_of_Reddit_and_Twitter_Sentiment_on_Short-Term_Stock_Volatility))
  report F1 up to ~0.8-0.9 for *direction* on liquid names, but penny-stock
  results are far noisier ([data sparsity + manipulation](https://medium.com/@zhonghong9998/can-machine-learning-predict-penny-stocks-a-risk-reward-analysis-8af1b098bfe9)).
- **The hard engineering problem is historical backfill**: free APIs expose
  only recent messages. Training on social features requires an archive
  (academic Pushshift access, paid Twitter/X firehose, or scraping+storing
  going forward). The pipeline therefore uses social features for **live
  scoring**, and accumulates its own archive over time.

## 6. Consolidated feature ranking (what to build first)

| Rank | Feature family | Evidence strength | In codebase |
|---|---|---|---|
| 1 | Size/price filters (market cap, price, float) | Very strong (every study) | universe screen + fundamentals |
| 2 | Volume anomalies (z-score, multi-window ratios) | Very strong | `volume_z`, `volume_ratio_5_60` |
| 3 | Promotion history (prior spikes, repeat targeting) | Strong (SEC, spam lit.) | `prior_spikes_180d`, `days_since_spike` |
| 4 | Illiquidity (Amihud, spreads, zero-volume days) | Strong (Aggarwal & Wu) | `amihud_20d`, `hl_range_20d` |
| 5 | Volatility & quiet-period shape | Strong | `volatility_*`, `baseline_flatness` |
| 6 | Promotion/news text burst + hype language | Strong but data-limited | `text.py` lexicons + FinBERT hook |
| 7 | Social buzz volume | Medium (live only w/o archive) | `social.py` |
| 8 | Sentiment polarity (calls, news, posts) | Weak-medium, short-lived | FinBERT/LM scores |
| 9 | Order-book microstructure | Strong in crypto, impractical daily | high-low & Amihud proxies only |

## 7. Sources

- https://arxiv.org/html/2412.18848v1 — reference paper (crypto P&D prediction)
- https://arxiv.org/abs/2301.11403 — Nam & Frank, stock P&D from forums
- https://www.researchgate.net/publication/24103583_Stock_Market_Manipulations — Aggarwal & Wu
- https://www.mdpi.com/2225-1146/11/3/22 — imbalanced P&D detection, insider anticipation
- https://www.mdpi.com/1999-5903/15/8/267 — ML P&D detection survey
- https://link.springer.com/article/10.1186/s40163-018-0093-5 — Kamps & Kleinberg, defining pumps
- https://arxiv.org/pdf/2006.08097 — FinBERT
- https://arxiv.org/pdf/2503.01886 — deep learning on earnings-call transcripts (survey)
- https://www.investor.gov/additional-resources/spotlight/microcap-fraud — SEC red flags
- https://blog.otcmarkets.com/2017/12/06/otc-markets-group-establishes-a-stock-promotion-policy/ — promotion flag
- https://www.sciencedirect.com/science/article/abs/pii/S0957417422014555 — manipulation-detection SLR
