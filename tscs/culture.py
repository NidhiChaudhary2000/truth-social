"""Beyond finance and politics: brands, sport, entertainment, health & science, places & heritage.

Response variables (all free, daily):
* Wikipedia pageviews of the target's English article  -> public attention
* Wikipedia edits of the same article                   -> collective editing activity (the 'crowd' rewriting
                                                           the shared record)
* stock price of the brand's company (when listed)      -> consumer-market reaction

Event = a calendar day (US/Eastern) on which an original post mentions the target (de-clustered: events at
least `gap` days apart). For each event, the response is the change in log views from the 7 days before to
days 0-2, compared with the same statistic on placebo days of the same article.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Wikipedia article -> (domain, regex on post text, case_sensitive)
CULTURE = {
    "Cracker Barrel": ("brands", r"Cracker Barrel", True),
    "American Eagle Outfitters": ("brands", r"American Eagle", True),
    "Coca-Cola": ("brands", r"Coca-Cola|\bCoke\b", True),
    "Jaguar Cars": ("brands", r"\bJaguar\b", True),
    "Apple Inc.": ("brands", r"\bApple\b|Tim Cook", True),
    "TikTok": ("brands", r"TikTok", True),
    "Intel": ("brands", r"\bIntel\b", True),
    "Tesla, Inc.": ("brands", r"\bTesla\b", True),
    "Washington Commanders": ("sport", r"Commanders|Redskins", True),
    "LIV Golf": ("sport", r"\bLIV\b", True),
    "Ultimate Fighting Championship": ("sport", r"\bUFC\b", True),
    "Transgender people in sports": ("sport", r"transgender|trans athlete|men in women|men playing in women", False),
    "Sydney Sweeney": ("entertainment", r"Sydney Sweeney", True),
    "Jimmy Kimmel": ("entertainment", r"Kimmel", True),
    "Bruce Springsteen": ("entertainment", r"Springsteen", True),
    "Taylor Swift": ("entertainment", r"Taylor Swift", True),
    "Kennedy Center": ("entertainment", r"Kennedy Center", True),
    "Paracetamol": ("health & science", r"Tylenol|acetaminophen", False),
    "Autism": ("health & science", r"autism|autistic", False),
    "Vaccine": ("health & science", r"vaccin", False),
    "Alcatraz Federal Penitentiary": ("places & heritage", r"Alcatraz", True),
    "Gulf of Mexico": ("places & heritage", r"Gulf of America|Gulf of Mexico", True),
    "Columbus Day": ("places & heritage", r"Columbus Day|Christopher Columbus", True),
    "Penny (United States coin)": ("places & heritage", r"\bpenn(?:y|ies)\b", False),
}
BRAND_TICKERS = {"Cracker Barrel": "CBRL", "American Eagle Outfitters": "AEO", "Coca-Cola": "KO", "Apple Inc.": "AAPL",
                 "Intel": "INTC", "Tesla, Inc.": "TSLA",
                 "Sydney Sweeney": "AEO"}  # the Sydney Sweeney post was about her American Eagle advert


def decluster(days, gap=5):
    days = pd.DatetimeIndex(sorted(set(pd.DatetimeIndex(days))))
    keep, last = [], None
    for d in days:
        if last is None or (d - last).days >= gap:
            keep.append(d)
            last = d
    return pd.DatetimeIndex(keep)


def mention_days(orig, regex, case=True, gap=5):
    m = orig["text_nourl"].str.contains(regex, case=case, regex=True)
    return decluster(orig.loc[m, "t_et"].dt.tz_localize(None).dt.normalize(), gap)


def _window_stats(y, i, resp, pre, pretrend):
    base = np.nanmean(y[i + pre[0]: i + pre[1] + 1])
    return (np.nanmean(y[i + resp[0]: i + resp[1] + 1]) - base,
            np.nanmean(y[i + pretrend[0]: i + pretrend[1] + 1]) - np.nanmean(y[i + pre[0]: i + pretrend[0]]))


def event_responses(series: pd.Series, days, resp=(0, 2), pre=(-10, -1), pretrend=(-3, -1), horizon=7,
                    log=True, n_placebo=2000, seed=0):
    """Event-level responses of one daily series + an article-level placebo test.

    delta     : mean log level on days resp minus mean over days pre   (x100 ~ % change)
    pre_delta : mean of days -3..-1 minus days -10..-4  (pre-trend: is attention already rising?)
    z         : delta standardised by the placebo distribution of the same article
    excess    : extra raw counts over days 0..horizon-1 relative to the pre-window median (cascade size)"""
    s = series.asfreq("D").astype(float)
    raw = s.to_numpy()
    y = np.log1p(raw) if log else raw
    n = len(y)
    pos = [s.index.get_loc(d) for d in pd.DatetimeIndex(days) if d in s.index]
    pos = [i for i in pos if i + pre[0] >= 0 and i + max(resp[1], horizon) < n]
    excl = np.zeros(n, bool)
    for i in pos:
        excl[max(0, i - 10): i + 11] = True
    cand = [i for i in range(-pre[0], n - max(resp[1], horizon) - 1) if not excl[i]]
    pl = np.array([_window_stats(y, i, resp, pre, pretrend)[0] for i in cand])
    pl = pl[np.isfinite(pl)]
    rows = []
    for i in pos:
        d, pdl = _window_stats(y, i, resp, pre, pretrend)
        base_raw = np.nanmedian(raw[i + pre[0]: i + pre[1] + 1])
        rows.append(dict(day=s.index[i], delta=d, pre_delta=pdl, z=(d - pl.mean()) / (pl.std() + 1e-12),
                         excess=np.nansum(raw[i: i + horizon] - base_raw), baseline=base_raw))
    ev = pd.DataFrame(rows)
    if not len(ev) or not len(pl):
        return ev, {}
    rng = np.random.default_rng(seed)
    means = rng.choice(pl, (n_placebo, len(ev))).mean(1)
    obs = ev.delta.mean()
    p = (np.sum(np.abs(means - means.mean()) >= abs(obs - means.mean())) + 1) / (n_placebo + 1)
    summ = dict(n_events=len(ev), mean_delta=obs, pct_change=100 * (np.exp(obs) - 1) if log else np.nan,
                median_pct=100 * (np.exp(ev.delta.median()) - 1) if log else np.nan, mean_z=ev.z.mean(),
                p=p, pre_trend=ev.pre_delta.mean(), max_pct=100 * (np.exp(ev.delta.max()) - 1) if log else np.nan,
                total_excess=ev.excess.clip(lower=0).sum())
    return ev, summ


def stock_event_z(rets: pd.DataFrame, ticker: str, event_days, market="SPY", car=(0, 1)):
    """Market-model abnormal return over trading days car[0]..car[1] after each event, as a z-score
    against the stock's own distribution of such CARs on non-event days."""
    r = rets[[ticker, market]].dropna()
    days = r.index
    pos = sorted({int(np.searchsorted(days.values, np.datetime64(d), side="left")) for d in pd.DatetimeIndex(event_days)})
    pos = [p for p in pos if 0 <= p and p + car[1] < len(days)]
    mask = np.zeros(len(days), bool)
    for p in pos:
        mask[max(0, p - 2): p + car[1] + 3] = True
    b, a = np.polyfit(r[market].values[~mask], r[ticker].values[~mask], 1)
    ar = r[ticker].values - (a + b * r[market].values)
    cs = np.concatenate([[0], np.cumsum(ar)])
    carv = np.array([cs[min(i + car[1] + 1, len(days))] - cs[i + car[0]] for i in range(len(days))])
    null = carv[~mask][: len(days) - car[1] - 1]
    return pd.DataFrame([dict(day=days[p], ticker=ticker, CAR=carv[p], z=(carv[p] - null.mean()) / null.std(),
                              pct_rank=(null < carv[p]).mean()) for p in pos])
