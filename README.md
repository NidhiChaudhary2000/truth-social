# Trump on Truth Social as a complex system

Course project code (Complex Systems, option 15: self-proposed). It analyses Trump's Truth Social posts as a complex system and tests what they couple to in the real world: markets, geopolitics, and public attention to politics **and to culture** (brands, sport, entertainment, health, places). The notebook keeps only the analyses that give a clear result against a null model. Each Part ends with a *Findings* section in markdown that interprets the numbers in complex-systems terms, and a closing *Synthesis* pulls them together.

## Quick start

```bash
pip install -r requirements.txt
```

* **The `data/` folder ships pre-filled.** It holds the Truth Social archive, the prices and the Wikipedia pageviews as of 29 Oct 2026, so Run All works straight away and offline. Set `REFRESH = True` in section 0 to pull fresh data.
* The notebook ships **executed on the real data**, so every output is visible before you run anything.
* A full run takes about 15–25 minutes, mostly the null models and the two-timescale Hawkes fit. `FAST = True` takes about 5 minutes.
* **Analysis window:** posts from **28 Apr 2022** (`START`). The account posted once on 14 Feb 2022 and then stayed silent for 73 days. That single gap would inflate burstiness from 0.45 to 0.79, so it is excluded.

### Data options (section 0)
| Setting | Effect |
|---|---|
| `USE_WIKI = True` | Use the Wikipedia pageview data (Part F). Uncached articles are downloaded anonymously at about 10 requests/min. |

## Structure of the notebook

| Part | Question | Main methods |
|---|---|---|
| A | How does he post? | Burstiness and memory (with circadian rescaling), bursty trains, CSN power-law vs lognormal, DFA, 1- and 2-kernel Hawkes with a time-rescaling test, rolling branching ratio |
| B | What gets attention? | Heavy-tailed engagement, engagement regression, Zipf and Heaps laws, repost concentration and the preferential-attachment kernel |
| C | Whom does he talk about? | Co-mention network vs degree-preserving nulls (small world, assortativity, rich club), disparity backbone, Louvain with a topology null and stability checks, temporal re-wiring, valence |
| D | Do markets react? | Intraday event study with time-of-day-matched placebos and FDR, case studies |
| E | How does he treat countries? | Stance time series, change points, office effect at the inauguration, multiplex war/trade/diplomacy network |
| F | What else do his posts move? | Wikipedia attention for 24 foreign editions, 16 issues and 24 culture targets (placebo event responses per article); brand stocks; cascade-size tails; Crane–Sornette endogenous vs exogenous bursts |
| G | A mechanism | Deffuant bounded-confidence model driven by his real posting record, with a phase diagram |

## Analyses that were dropped

These were removed from the notebook because they gave null or uninformative results. Their code stays in `tscs/` so you can mention them in the report as robustness checks or as negative results:

* signed network and structural balance;
* transfer entropy and Granger tests against policy uncertainty, approval and markets, and the "Trump put";
* RMT and MST of the market;
* the daily event study (the effect is intraday only, and the notebook prints the daily contrast) and the country-ETF study;
* GDELT event and media-tone coupling (`tscs/gdelt_events.py`, `fetch_gdelt*.py`, `gdelt_events_bigquery.sql`);
* Polymarket prediction markets (`EXT.polymarket_event`);
* Wikipedia edit counts. These were skipped to keep the number of requests small. `EXT.wiki_edits` exists if you want to add them.

## Files

| File | What it does |
|---|---|
| `truth_social_complexity.ipynb` | The analysis notebook (Parts A–G plus synthesis), executed on the real data |
| `tscs/data.py` | Downloads, caching and cleaning (ReTruth parsing, mojibake repair); maps posts to trading days |
| `tscs/text.py` | Entity dictionary, topic lexicons, conflict/trade/diplomacy frames, sentence-level sentiment |
| `tscs/pointproc.py` | Burstiness, memory, bursty trains, CSN power-law fitting, DFA, 1- and K-kernel Hawkes MLE, Zipf/Heaps |
| `tscs/networks.py` | Co-mention network, null models, small world, rich club, disparity backbone, communities, temporal networks |
| `tscs/markets.py` | Event studies with placebo nulls, intraday study, FDR |
| `tscs/geo.py` | Country stance and regime shifts, multiplex network, spike detection, burst shapes, opinion-dynamics model |
| `tscs/culture.py` | Culture targets (article, domain, regex), brand tickers, per-article placebo event responses, brand-stock CAR z-scores |
| `tscs/external.py` | Wikipedia pageviews (throttled, cached), plus optional GDELT DOC and Polymarket helpers |
| `tscs/gdelt_events.py` | Optional: GDELT Event Database via daily files or BigQuery (not used in the notebook) |
| `tscs/synthetic.py` | Synthetic data generator, used for validation |

## Data sources

* **Posts:** CNN's archive at `https://ix.cnn.io/data/truth-social/truth_archive.json`, updated every 5 minutes and based on github.com/stiles/trump-truth-social-archive. If the download fails, save that file by hand as `data/truth_archive.json`.
* **Markets:** `yfinance`. Daily data runs from 2022. Hourly data covers **only the last 730 days**, so re-running after about April 2027 loses the April 2025 tariff case study in hourly resolution. Keep the shipped `data/prices_60m.csv`.
* **Brand stocks:** CBRL, AEO, KO, AAPL, INTC and TSLA from `yfinance`, with SPY as the market (`data/prices_brands_1d.csv`).
* **Wikipedia pageviews** (Wikimedia REST API, no key): the *Donald Trump* article in 24 language editions, 16 global-issue articles, and 24 culture articles (see `tscs/culture.py`).

## Data cleaning decisions

1. **ReTruths** are about 20% of posts. In the archive they look like `RT @handleTEXT` (the handle is glued to the text) or `RT: https://truthsocial.com/users/X/statuses/ID`. The reposted handle is recovered by longest match against handles seen in unambiguous places. About 39% of ReTruths are self-reposts. Reposts are excluded from text analyses and kept for timing and amplification (B3).
2. **Media or link-only posts** are about 41% and have no text of his own.
3. **Quotation posts** (`“…”` headlines and TV quotes) are about 9% of original posts. They are kept in `orig` (attention, topics, culture mentions) and excluded from `own`, which is used for his sentiment and stance (C6, E).
4. **Tagging precision.** Hand-check about 100 random posts for entities, topics and sentiment, and report the precision. Aliases live in `tscs/text.py` and culture regexes in `tscs/culture.py`. Bump `TAG_VERSION` after editing the aliases so the cache is rebuilt.
5. **Text-encoding errors.** About 2.5% of posts contain mojibake such as `Worldâ€™s` (UTF-8 decoded as Latin-1 upstream). It is repaired in `data.fix_mojibake`.
6. **Determinism.** All seeds are fixed, and network construction sorts its edges, so Louvain gives the same communities on every run.