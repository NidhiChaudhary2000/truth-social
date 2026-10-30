"""Synthetic stand-in data with the *same schemas* as the real downloads.

Purpose: let the notebook run end-to-end without internet (DEMO mode) and act as
a validation test-bed -- the generator plants known effects (Hawkes branching
ratio, circadian rhythm, tariff-post shocks to markets, signed allies/enemies
structure), so we can check that every estimator recovers them.

NEVER report results from this data as findings. Set DEMO = False for the real analysis.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from .data import COUNTRY_ETF, COMPANY_TICKER, all_daily_tickers, ET

TRUE_PARAMS = dict(branching_ratio=0.55, decay_hours=0.5, mean_daily_posts=16)

ALLIES = ["Republicans", "MAGA", "JD Vance", "Benjamin Netanyahu", "Fox News", "Mike Johnson", "Elon Musk"]
ENEMIES = ["Democrats", "Joe Biden", "Kamala Harris", "Fake News Media", "Gavin Newsom", "Jerome Powell",
           "Jack Smith", "CNN", "Barack Obama", "Nancy Pelosi", "Chuck Schumer", "New York Times", "Liz Cheney"]
NAMES = {"Republicans": "Republicans", "MAGA": "MAGA", "JD Vance": "JD Vance", "Benjamin Netanyahu": "Bibi Netanyahu",
         "Fox News": "Fox News", "Mike Johnson": "Speaker Mike Johnson", "Elon Musk": "Elon Musk",
         "Democrats": "Radical Left Democrats", "Joe Biden": "Sleepy Joe Biden", "Kamala Harris": "Kamala",
         "Fake News Media": "Fake News", "Gavin Newsom": "Newscum", "Jerome Powell": "Jerome Powell",
         "Jack Smith": "Deranged Jack Smith", "CNN": "CNN", "Barack Obama": "Obama", "Nancy Pelosi": "Pelosi",
         "Chuck Schumer": "Schumer", "New York Times": "New York Times", "Liz Cheney": "Liz Cheney"}
COUNTRIES = ["China", "Canada", "Mexico", "European Union", "Japan", "India", "Brazil", "South Korea", "Vietnam",
             "Switzerland", "Taiwan"]
CNAME = {"European Union": "the European Union"}
POS_ADJ = ["great", "tremendous", "incredible", "strong", "fantastic", "brave", "highly respected"]
NEG_ADJ = ["crooked", "corrupt", "radical", "failed", "weak", "pathetic", "deranged", "incompetent"]
HANDLES = ["WhiteHouse", "FoxNews", "JDVance", "elonmusk", "seanhannity", "DonaldJTrumpJr", "TeamTrump"]


GEO_C = ["China", "Russia", "Ukraine", "Israel", "Iran", "Canada", "Mexico", "India", "Greenland", "Venezuela",
         "Japan", "Germany"]
GEO_P = np.array([5, 5, 5, 4, 4, 4, 3, 2, 2, 2, 1, 1], float); GEO_P /= GEO_P.sum()
GEO_LEADER = {"China": "President Xi", "Russia": "Putin", "Ukraine": "Zelensky", "Israel": "Bibi Netanyahu",
              "Canada": "Mark Carney", "Mexico": "Sheinbaum", "India": "Modi", "Venezuela": "Maduro"}


def _p_hostile(c, yr):
    """Planted stance regimes: Canada and Russia turn hostile in 2025; Israel stays friendly; Iran hostile."""
    if c == "Canada":
        return 0.2 if yr < 2025.0 else 0.8
    if c == "Russia":
        return 0.3 if yr < 2025.5 else 0.85
    if c == "Israel":
        return 0.1
    if c == "Iran":
        return 0.9
    return 0.5


def _circadian():
    h = np.arange(24)
    prof = 0.15 + np.exp(-0.5 * ((h - 10) / 3.5) ** 2) + 0.9 * np.exp(-0.5 * ((h - 22) / 2.0) ** 2)
    return prof / prof.mean()


def simulate_hawkes(T_hours, mu_fn, alpha, beta, rng):
    """Ogata thinning for lambda(t) = mu(t) + alpha sum exp(-beta (t-ti)). mu_fn vectorised."""
    t, ev, A = 0.0, [], 0.0
    mu_max = mu_fn(np.linspace(0, T_hours, 20000)).max() * 1.05
    while t < T_hours:
        lam_bar = mu_max + alpha * A + alpha  # A decays, so this bounds the intensity until next event
        w = rng.exponential(1 / lam_bar)
        A *= np.exp(-beta * w)
        t += w
        if t >= T_hours:
            break
        lam = mu_fn(np.array([t]))[0] + alpha * A
        if rng.random() * lam_bar <= lam:
            ev.append(t)
            A += 1.0
    return np.array(ev)


def _sentence(kind, rng, ent=None):
    if kind == "ally":
        return rng.choice([f"Congratulations to {ent} on a {rng.choice(POS_ADJ)} job!",
                           f"{ent} is {rng.choice(POS_ADJ)} and a true Patriot. Thank you!",
                           f"Great meeting with {ent}, a {rng.choice(POS_ADJ)} fighter for our Country!"])
    if kind == "enemy":
        return rng.choice([f"The {rng.choice(NEG_ADJ)} {ent} are destroying our Country!",
                           f"{ent} is a total disaster, {rng.choice(NEG_ADJ)} and dangerous!",
                           f"{ent} pushed the Witch Hunt Hoax. So corrupt and pathetic!"])
    return ""


def make_post(rng, date, cfg):
    sents, topics = [], []
    r = rng.random()
    tariff_p = cfg["tariff_p"]
    if r < tariff_p:
        cs = rng.choice(COUNTRIES, size=rng.choice([1, 1, 1, 2]), replace=False, p=cfg["country_p"])
        names = " and ".join(CNAME.get(c, c) for c in cs)
        sents.append(rng.choice([f"{names} has been ripping us off for years. Tariffs of {rng.choice([10, 25, 50, 100])}% start NOW!",
                                 f"Reciprocal Tariffs on {names} will be very big. The trade deficit is a disaster!",
                                 f"We are close to a great Trade Deal with {names}. The Stock Market will boom!"]))
        topics.append("tariffs")
        if rng.random() < 0.3:
            sents.append(_sentence("enemy", rng, NAMES[rng.choice(["Jerome Powell", "Democrats", "Joe Biden"])]))
    elif r < tariff_p + 0.35:
        e = rng.choice(ENEMIES)
        sents.append(_sentence("enemy", rng, NAMES[e]))
        if rng.random() < 0.45:  # contrast with an ally -> opposite signs in one post
            sents.append(_sentence("ally", rng, NAMES[rng.choice(ALLIES)]))
        if rng.random() < 0.35:  # another enemy -> same-sign pair
            e2 = rng.choice([x for x in ENEMIES if x != e])
            sents.append(_sentence("enemy", rng, NAMES[e2]))
    elif r < tariff_p + 0.55:
        a = rng.choice(ALLIES)
        sents.append(_sentence("ally", rng, NAMES[a]))
        if rng.random() < 0.3:
            sents.append(rng.choice(["has my Complete and Total Endorsement!", "MAKE AMERICA GREAT AGAIN!"]))
        if rng.random() < 0.3:
            sents.append(_sentence("ally", rng, NAMES[rng.choice([x for x in ALLIES if x != a])]))
    elif r < tariff_p + 0.65 and rng.random() < 0.6:  # geopolitical posts with time-varying stance
        c = rng.choice(GEO_C, p=GEO_P)
        yr = date.year + date.dayofyear / 366
        hostile = rng.random() < _p_hostile(c, yr)
        leader = GEO_LEADER.get(c, c)
        if hostile:
            sents.append(rng.choice([f"{c} must stop the war and the killing NOW. Very bad and dangerous!",
                                     f"{c} is treating us very badly. A total disaster, we will respond with strength!",
                                     f"Missiles and bombs from {c}, pathetic and dangerous!"]))
        else:
            sents.append(rng.choice([f"Great call with {leader} of {c}. A wonderful relationship and talks on peace!",
                                     f"Tremendous meeting with {leader}. {c} is a great friend, we will make a great deal!",
                                     f"Congratulations to {c} on the ceasefire agreement. Peace through strength!"]))
    elif r < tariff_p + 0.65:
        sents.append(rng.choice(["The Border is now closed to Illegal Aliens. Deportations are working!",
                                 "Putin and Zelensky must make a deal. The war in Ukraine must stop!",
                                 "Iran will never have a Nuclear Weapon!",
                                 "Interest Rates should be cut NOW. Too Late Powell is a disaster!",
                                 "The Economy is BOOMING, Jobs are up, Inflation is down!"]))
    else:
        sents.append(rng.choice(["MAKE AMERICA GREAT AGAIN!", "Thank you!", "Watch tonight at 9PM on Fox News!",
                                 "Happy Birthday to a great Patriot!", "WE WILL WIN!"]))
    text = " ".join(sents)
    if rng.random() < 0.25:
        text = text.upper()
    html_parts = []
    if rng.random() < 0.08:
        h = rng.choice(HANDLES)
        html_parts.append(f'<span class="h-card"><a href="https://truthsocial.com/@{h}" class="u-url mention">@<span>{h}</span></a></span> ')
    content = "<p>" + "".join(html_parts) + text + "</p>"
    rr = rng.random()
    if rr < 0.15:
        content = f"<p>RT @{rng.choice(HANDLES)} {text}</p>"
    elif rr < 0.20:
        content = ""
    return content, ("tariffs" in topics)


def generate(data_dir="data_demo", start="2022-02-14", end="2026-10-28", seed=42) -> dict:
    rng = np.random.default_rng(seed)
    os.makedirs(data_dir, exist_ok=True)
    t0 = pd.Timestamp(start, tz=ET)
    T = (pd.Timestamp(end, tz=ET) - t0).total_seconds() / 3600
    prof = _circadian()
    n_br, beta = TRUE_PARAMS["branching_ratio"], 1 / TRUE_PARAMS["decay_hours"]
    alpha = n_br * beta
    base_rate = TRUE_PARAMS["mean_daily_posts"] / 24 * (1 - n_br)
    start_h = t0.hour

    def mu_fn(th):
        days = th / 24
        trend = 0.7 + 0.6 / (1 + np.exp(-(days - 900) / 120))  # more active after ~2024
        return base_rate * trend * prof[((start_h + np.floor(th)).astype(int)) % 24]

    th = simulate_hawkes(T, mu_fn, alpha, beta, rng)
    times = t0 + pd.to_timedelta(th, unit="h")
    posts, tariff_flags = [], []
    for i, t in enumerate(times):
        yr = t.year + t.dayofyear / 366
        tariff_p = 0.03 if yr < 2025 else 0.16 if yr < 2025.6 else 0.08
        cp = np.array([6, 4, 4, 3, 2, 2, 1, 1, 1, 0.5, 0.5])
        cfg = dict(tariff_p=tariff_p, country_p=cp / cp.sum())
        content, is_tariff = make_post(rng, t, cfg)
        fav = int(rng.lognormal(9.3, 0.9) + (rng.pareto(1.8) * 4000 if rng.random() < 0.3 else 0))
        posts.append(dict(id=str(10**17 + i), created_at=t.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                          content=content, url=f"https://truthsocial.com/@realDonaldTrump/{10**17 + i}",
                          media=[], replies_count=int(fav * rng.uniform(0.1, 0.4)),
                          reblogs_count=int(fav * rng.uniform(0.15, 0.35)), favourites_count=fav))
        tariff_flags.append(is_tariff)
    with open(os.path.join(data_dir, "truth_archive.json"), "w") as f:
        json.dump(posts, f)

    # ---------------- markets (daily) ----------------
    days = pd.bdate_range("2022-01-03", end)
    tick = all_daily_tickers()
    n = len(days)
    tariff_times = times[np.array(tariff_flags)]
    from .data import trading_day_of
    tdays = trading_day_of(pd.Series(tariff_times), days).dropna()
    tariff_count = pd.Series(1, index=tdays.values).groupby(level=0).sum().reindex(days).fillna(0).to_numpy()
    # country-specific tariff days (re-parse the generated text)
    country_days = {}
    for c in COUNTRIES:
        tt = [t for t, p, fl in zip(times, posts, tariff_flags) if fl and CNAME.get(c, c) in p["content"]]
        country_days[c] = set(trading_day_of(pd.Series(tt), days).dropna()) if tt else set()

    vol = np.empty(n)
    v = 0.009
    mkt = np.empty(n)
    for i in range(n):  # GARCH(1,1)-like market factor
        shock = rng.standard_t(4) / np.sqrt(2)
        sig = v * (1.8 if tariff_count[i] > 0 else 1.0)
        mkt[i] = 0.0003 + sig * shock - (0.004 if tariff_count[i] > 0 else 0)
        v = min(np.sqrt(0.000004 + 0.07 * (v * shock) ** 2 + 0.88 * v ** 2), 0.035)
        vol[i] = v
    rets = {}
    sector_f = {k: rng.normal(0, 0.006, n) for k in "abc"}
    for j, tk in enumerate(tick):
        beta_i = rng.uniform(0.5, 1.4)
        sf = sector_f["abc"[j % 3]] * rng.uniform(0.3, 1.0)
        idio = rng.standard_t(4, n) * rng.uniform(0.004, 0.015) / np.sqrt(2)
        r = beta_i * mkt + sf + idio
        if tk in ("TLT", "GLD", "UUP"):
            r = -0.2 * mkt + 0.6 * idio
        if tk.endswith("=X"):
            r = 0.1 * mkt + 0.4 * idio
        if tk == "BTC-USD":
            r = 1.5 * mkt + 2.5 * idio
        rets[tk] = r
    for c, (etf, fx) in COUNTRY_ETF.items():
        if c in country_days:
            m = days.isin(list(country_days[c]))
            if etf in rets:
                rets[etf] = rets[etf] - 0.006 * m
            if fx in rets and fx not in ("EURUSD=X", "GBPUSD=X"):
                rets[fx] = rets[fx] + 0.003 * m  # local currency weakens vs USD
    px = pd.DataFrame({k: 100 * np.exp(np.cumsum(v_)) for k, v_ in rets.items()}, index=days)
    vix = np.clip(6 + 1300 * vol * np.exp(rng.normal(0, 0.05, n)) + 3 * tariff_count.clip(0, 3), 9, 80)
    px["^VIX"] = vix
    px.index.name = "time"
    px.to_csv(os.path.join(data_dir, "prices_1d.csv"))

    # ---------------- markets (hourly, last ~729 days) ----------------
    hdays = days[days >= days[-1] - pd.Timedelta(days=729)]
    bars = [pd.Timestamp(d.date()).tz_localize(ET) + pd.Timedelta(hours=9.5) + pd.Timedelta(hours=k)
            for d in hdays for k in range(7)]
    bars = pd.DatetimeIndex(bars)
    nb = len(bars)
    hr = {k: rng.standard_t(4, nb) * s / np.sqrt(2) for k, s in
          [("SPY", 0.003), ("QQQ", 0.004), ("BTC-USD", 0.008), ("DJT", 0.012)]}
    hr["QQQ"] = 1.2 * hr["SPY"] + hr["QQQ"] * 0.5
    tar_in_hours = tariff_times[(tariff_times >= bars[0]) & (tariff_times < bars[-1] + pd.Timedelta("1h"))]
    pos = np.searchsorted(bars.values, tar_in_hours.values, side="right") - 1
    for p_ in pos:
        if 0 <= p_ < nb - 2:
            for k, amp in [("SPY", 0.004), ("QQQ", 0.005), ("BTC-USD", 0.008)]:
                sgn = -1 if rng.random() < 0.7 else 1
                hr[k][p_ + 1] += sgn * amp * abs(rng.normal(1, 0.3))
    hpx = pd.DataFrame({k: 100 * np.exp(np.cumsum(v_)) for k, v_ in hr.items()}, index=bars)
    hpx["^VIX"] = np.clip(20 * np.exp(-3 * np.cumsum(hr["SPY"])), 9, 80)
    hpx.index.name = "time"
    hpx.to_csv(os.path.join(data_dir, "prices_60m.csv"))

    # ---------------- FRED + approval ----------------
    cal = pd.date_range(days[0], days[-1], freq="D")
    tc_cal = pd.Series(tariff_count, index=days).reindex(cal).fillna(0).to_numpy()
    epu = np.exp(np.log(120) + np.convolve(rng.normal(0, 0.35, len(cal)), np.ones(3) / 3, "same"))
    epu[1:] *= np.exp(0.25 * np.minimum(tc_cal[:-1], 4))  # tariff posts raise next-day uncertainty
    fred = pd.DataFrame({"VIXCLS": pd.Series(vix, index=days).reindex(cal),
                         "USEPUINDXD": epu,
                         "DGS10": 3.5 + np.cumsum(rng.normal(0, 0.03, len(cal))),
                         "DTWEXBGS": 120 + np.cumsum(rng.normal(0, 0.2, len(cal)))}, index=cal)
    fred.index.name = "date"
    fred.to_csv(os.path.join(data_dir, "fred.csv"))
    net = -5 + np.cumsum(rng.normal(0, 0.25, len(cal)))
    pd.DataFrame({"date": cal, "approve": 45 + net / 2, "disapprove": 50 - net / 2}).to_csv(
        os.path.join(data_dir, "approval.csv"), index=False)
    _external(data_dir, posts, times, tariff_flags, cal, rng)
    return dict(n_posts=len(posts), true=TRUE_PARAMS, data_dir=data_dir)


def _external(data_dir, posts, times, tariff_flags, cal, rng):
    """Synthetic Wikipedia / GDELT / political files in the loaders' cache format, with planted effects:
    issue attention jumps x3 after posts on the issue and relaxes as t^-1; the language edition of a
    mentioned country gets +0.3 log-views for 3 days; a hostile post lowers that country's media tone by 0.6."""
    from . import external as EXT, geo as GEO, text as TX
    from .data import html_to_text
    day = pd.DatetimeIndex(times.tz_convert(ET).tz_localize(None).normalize())
    texts = [html_to_text(p["content"]) for p in posts]
    ents = [TX.find_entities(t) for t in texts]
    sent = np.array([TX.lexicon_score(t) for t in texts])
    idx = {d: i for i, d in enumerate(cal)}
    n = len(cal)

    def daily(mask):
        v = np.zeros(n)
        for d, m in zip(day, mask):
            if m and d in idx:
                v[idx[d]] += 1
        return v

    kern = 1.0 / (1 + np.arange(60)) ** 1.0
    os.makedirs(os.path.join(data_dir, "wiki"), exist_ok=True)
    for art, (kind, trig) in EXT.ISSUE_ARTICLES.items():
        mask = np.array(tariff_flags) if kind == "topic" and trig == "tariffs_trade" else \
            np.array(["Border" in t for t in texts]) if kind == "topic" else np.array([any(e in es for e in trig) for es in ents])
        imp = np.convolve((daily(mask) > 0).astype(float), kern)[:n] * 2.0
        lv = np.log(rng.uniform(2e3, 3e4)) + np.cumsum(rng.normal(0, 0.03, n)) * 0.3 + rng.normal(0, 0.12, n)
        v = np.exp(lv) * (1 + imp)
        pd.Series(np.round(v), index=pd.DatetimeIndex(cal, name="date"), name=art).to_csv(
            os.path.join(data_dir, "wiki", f"en.wikipedia__{EXT._safe(art)}.csv"))
    links = {lg: "Donald Trump" for lg in EXT.LANG_COUNTRIES}
    links["en"] = "Donald Trump"
    json.dump(links, open(os.path.join(data_dir, "wiki", "langlinks__Donald_Trump.json"), "w"))
    common = np.cumsum(rng.normal(0, 0.05, n)) * 0.3 + 0.2 * np.log1p(daily(np.ones(len(day), bool)))
    for lg, cs in EXT.LANG_COUNTRIES.items():
        members = {e for c in cs for e in GEO.COUNTRY_GROUPS.get(c, [c])}
        treat = daily(np.array([bool(members & set(es)) for es in ents]))
        bump = np.convolve((treat > 0).astype(float), np.ones(3))[:n] * 0.3
        lv = np.log(rng.uniform(5e2, 2e4)) + common + rng.normal(0, 0.1, n) + bump
        pd.Series(np.round(np.exp(lv)), index=pd.DatetimeIndex(cal, name="date"), name=lg).to_csv(
            os.path.join(data_dir, "wiki", f"{lg}.wikipedia__Donald_Trump.csv"))
    from . import culture as CU  # culture articles: baseline only (synthetic posts do not mention them)
    for art in CU.CULTURE:
        lv = np.log(rng.uniform(1e3, 3e4)) + np.cumsum(rng.normal(0, 0.03, n)) * 0.3 + rng.normal(0, 0.15, n)
        pd.Series(np.round(np.exp(lv)), index=pd.DatetimeIndex(cal, name="date"), name=art).to_csv(
            os.path.join(data_dir, "wiki", f"en.wikipedia__{EXT._safe(art)}.csv"))
    bdays = pd.bdate_range(cal[0], cal[-1])
    bp = {tk: 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, len(bdays)))) for tk in set(CU.BRAND_TICKERS.values()) | {"SPY"}}
    pd.DataFrame(bp, index=pd.DatetimeIndex(bdays, name="time")).to_csv(os.path.join(data_dir, "prices_brands_1d.csv"))
    os.makedirs(os.path.join(data_dir, "gdelt"), exist_ok=True)
    gcommon = rng.normal(0, 0.3, n)
    for c, code in EXT.GDELT_COUNTRIES.items():
        members = set(GEO.COUNTRY_GROUPS.get(c, [c]))
        hostile = daily(np.array([bool(members & set(es)) and s_ < -0.2 for es, s_ in zip(ents, sent)]))
        eff = np.convolve((hostile > 0).astype(float), np.ones(3))[:n].clip(0, 1) * -0.6
        tone = -2 + gcommon + rng.normal(0, 0.35, n) + eff
        pd.Series(tone, index=pd.DatetimeIndex(cal, name="date"), name=f"tone_{code}").to_csv(
            os.path.join(data_dir, "gdelt", f"timelinetone__tone_{code}.csv"))
    from . import gdelt_events as GE  # bilateral event data: C->US cooperation drops after hostile posts about C
    rows = []
    for c, code in GE.CAMEO.items():
        members = set(GEO.COUNTRY_GROUPS.get(c, [c]))
        hostile = daily(np.array([bool(members & set(es)) and s_ < -0.2 for es, s_ in zip(ents, sent)]))
        eff = np.convolve((hostile > 0).astype(float), np.ones(3))[:n].clip(0, 1)
        base = rng.uniform(-0.5, 2.5)
        for dirn, k in [("C2US", -1.2), ("US2C", -0.8)]:
            g = base + rng.normal(0, 0.8, n) + k * eff
            cs = np.clip(0.3 + rng.normal(0, 0.08, n) + 0.12 * eff * (k < 0), 0, 1)
            rows.append(pd.DataFrame({"day": cal, "country": c, "direction": dirn, "n_events": rng.poisson(60, n),
                                      "mentions": rng.poisson(400, n), "goldstein": g, "tone": -2 + 0.3 * g,
                                      "conflict_share": cs, "material_conflict_share": cs / 3,
                                      "threaten_share": np.clip(0.05 + 0.1 * eff * (k < 0) + rng.normal(0, 0.02, n), 0, 1),
                                      "coerce_share": np.clip(0.08 + rng.normal(0, 0.02, n), 0, 1),
                                      "consult_share": np.clip(0.3 - 0.1 * eff + rng.normal(0, 0.05, n), 0, 1)}))
    ev = pd.concat(rows)  # mimic a BigQuery export: CAMEO country codes
    ev["country"] = ev["country"].map(GE.CAMEO)
    ev.to_csv(os.path.join(data_dir, "gdelt_events_daily.csv"), index=False)
    os.makedirs(os.path.join(data_dir, "political"), exist_ok=True)
    lg = np.cumsum(rng.normal(0, 0.04, n))
    pd.DataFrame({"date": cal, "value": 1 / (1 + np.exp(-lg))}).to_csv(
        os.path.join(data_dir, "political", "house_2026_dem_prob.csv"), index=False)
