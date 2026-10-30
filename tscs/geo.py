"""Part F: geopolitics, global issues and political leaning.

* Country stance dynamics: Trump's sentence-level valence towards each country (and its leader)
  over time, friend/foe switches, and regime shifts found by binary segmentation with a
  permutation test.
* Multiplex geopolitical network with conflict / trade / diplomacy layers (Battiston, Nicosia &
  Latora 2014; De Domenico et al. 2013): edge overlap, inter-layer correlation, multiplex
  participation coefficient, and a per-country "issue linkage" test (are war and trade framed together?).
* Panel event study with cross-sectional demeaning (removes global news shocks) and placebo
  inference. Used for foreign media tone (GDELT) and foreign public attention (Wikipedia language
  editions).
* Attention bursts: endogenous vs exogenous relaxation (Crane & Sornette 2008, PNAS 105:15649).
* Opinion dynamics: bounded-confidence (Deffuant et al. 2000) model on a scale-free network with
  a broadcasting "media" agent whose stance and activity are the *real* post series
  (cf. mass-media models, Gonzalez-Avella et al. 2005).
"""
from __future__ import annotations

import itertools
from collections import Counter, defaultdict

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

COUNTRY_GROUPS = {
    "China": ["China", "Xi Jinping"], "Russia": ["Russia", "Vladimir Putin"],
    "Ukraine": ["Ukraine", "Volodymyr Zelensky"], "Canada": ["Canada", "Justin Trudeau", "Mark Carney"],
    "Mexico": ["Mexico", "Claudia Sheinbaum"], "Israel": ["Israel", "Benjamin Netanyahu"],
    "Iran": ["Iran", "Ali Khamenei"], "European Union": ["European Union"], "India": ["India", "Narendra Modi"],
    "Venezuela": ["Venezuela", "Nicolas Maduro"], "Brazil": ["Brazil", "Lula"],
    "Japan": ["Japan", "Shigeru Ishiba", "Sanae Takaichi"], "South Korea": ["South Korea"],
    "North Korea": ["North Korea", "Kim Jong Un"], "Germany": ["Germany"], "France": ["France", "Emmanuel Macron"],
    "United Kingdom": ["United Kingdom", "Keir Starmer"], "Gaza": ["Gaza"], "Greenland": ["Greenland"],
    "Denmark": ["Denmark"], "Panama": ["Panama"], "Saudi Arabia": ["Saudi Arabia", "Mohammed bin Salman"],
    "Qatar": ["Qatar"], "Syria": ["Syria"], "Turkey": ["Turkey", "Recep Erdogan"], "Argentina": ["Argentina", "Javier Milei"],
    "Colombia": ["Colombia", "Gustavo Petro"], "South Africa": ["South Africa", "Cyril Ramaphosa"],
    "Hungary": ["Hungary", "Viktor Orban"], "Taiwan": ["Taiwan"], "Vietnam": ["Vietnam"], "Switzerland": ["Switzerland"],
    "Australia": ["Australia"], "Poland": ["Poland"], "Cuba": ["Cuba"], "Afghanistan": ["Afghanistan"],
}
_ENT2COUNTRY = {e: c for c, es in COUNTRY_GROUPS.items() for e in es}


# --------------------------------------------------------------------------
# Stance
# --------------------------------------------------------------------------
def country_mentions(df) -> pd.DataFrame:
    """Long table: one row per (post, country) with the post's stance towards that country
    (mean sentence valence over the country entity and its leader) and the post's frames."""
    rows = []
    frame_cols = [c for c in df.columns if c.startswith("frame_")]
    for i, (t, es, fr) in enumerate(zip(df["t_et"], df["ent_sent"], df[frame_cols].to_numpy() if frame_cols else
                                        [[]] * len(df))):
        acc = defaultdict(list)
        for e, s in es.items():
            if e in _ENT2COUNTRY:
                acc[_ENT2COUNTRY[e]].append(s)
        for c, v in acc.items():
            r = dict(post=df.index[i], t_et=t, country=c, stance=float(np.mean(v)), n_countries=len(acc))
            r.update({fc: bool(x) for fc, x in zip(frame_cols, fr)})
            rows.append(r)
    out = pd.DataFrame(rows)
    out["date"] = out["t_et"].dt.tz_localize(None).dt.normalize()
    return out


def stance_panel(cm, freq="MS", min_mentions=3, top=18):
    top_c = cm["country"].value_counts().index[:top]
    g = cm[cm.country.isin(top_c)].groupby([pd.Grouper(key="date", freq=freq), "country"])["stance"]
    mean, n = g.mean().unstack(), g.size().unstack().fillna(0)
    return mean.where(n >= min_mentions)[top_c], n[top_c]


def _best_split(x, min_size):
    n = len(x)
    best = (0.0, None)
    for k in range(min_size, n - min_size + 1):
        a, b = x[:k], x[k:]
        s = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) if min(len(a), len(b)) > 1 else np.inf
        t = abs(a.mean() - b.mean()) / (s + 1e-12)
        if t > best[0]:
            best = (t, k)
    return best


def changepoints(series: pd.Series, min_size=3, n_perm=500, alpha=0.05, seed=0, max_cp=4) -> list:
    """Binary segmentation for shifts in the mean, each split accepted only if its max |t| beats
    a permutation null (shuffle within segment). Returns the index labels of the change points."""
    s = series.dropna()
    x = s.to_numpy(float)
    rng = np.random.default_rng(seed)
    segs, cps = [(0, len(x))], []
    while segs and len(cps) < max_cp:
        lo, hi = segs.pop(0)
        seg = x[lo:hi]
        if len(seg) < 2 * min_size:
            continue
        t, k = _best_split(seg, min_size)
        if k is None:
            continue
        null = [_best_split(rng.permutation(seg), min_size)[0] for _ in range(n_perm)]
        p = (np.sum(np.array(null) >= t) + 1) / (n_perm + 1)
        if p < alpha:
            cps.append((lo + k, p))
            segs += [(lo, lo + k), (lo + k, hi)]
    return [(s.index[i], p) for i, p in sorted(cps)]


def stance_regimes(panel: pd.DataFrame, min_months=8, **kw) -> pd.DataFrame:
    rows = []
    for c in panel.columns:
        s = panel[c].dropna()
        if len(s) < min_months:
            continue
        cps = changepoints(s, **kw)
        bounds = [s.index[0]] + [d for d, _ in cps] + [s.index[-1] + pd.Timedelta("1D")]
        means = [s[(s.index >= a) & (s.index < b)].mean() for a, b in zip(bounds[:-1], bounds[1:])]
        rows.append(dict(country=c, months=len(s), mean_stance=s.mean(), n_changepoints=len(cps),
                         changepoints=", ".join(f"{d:%Y-%m}" for d, _ in cps),
                         regime_means=" → ".join(f"{m:+.2f}" for m in means),
                         sign_flips=int(np.sum(np.diff(np.sign(np.array(means)[np.abs(means) > 0.05])) != 0))))
    return pd.DataFrame(rows).set_index("country")


# --------------------------------------------------------------------------
# Multiplex network (conflict / trade / diplomacy)
# --------------------------------------------------------------------------
LAYERS = ("conflict", "trade", "diplomacy")


def multiplex(cm, min_mentions=10, layers=LAYERS):
    keep = cm["country"].value_counts()
    keep = set(keep[keep >= min_mentions].index)
    cm = cm[cm.country.isin(keep)]
    strength = pd.DataFrame({L: cm[cm[f"frame_{L}"]].groupby("country").size() for L in layers}).reindex(sorted(keep)).fillna(0)
    G = {}
    for L in layers:
        g = nx.Graph()
        g.add_nodes_from(keep)
        sub = cm[cm[f"frame_{L}"]]
        for _, grp in sub.groupby("post"):
            for a, b in itertools.combinations(sorted(set(grp.country)), 2):
                w = g[a][b]["weight"] + 1 if g.has_edge(a, b) else 1
                g.add_edge(a, b, weight=w)
        G[L] = g
    return G, strength


def participation_coefficient(strength: pd.DataFrame) -> pd.Series:
    """P_i = M/(M-1) * (1 - sum_a (k_i^a / o_i)^2): 0 = active in one layer only, 1 = evenly spread."""
    M = strength.shape[1]
    o = strength.sum(1)
    return (M / (M - 1)) * (1 - ((strength.div(o, axis=0)) ** 2).sum(1))


def multiplex_stats(G: dict) -> dict:
    Ls = list(G)
    E = {L: {frozenset(e) for e in G[L].edges()} for L in Ls}
    allE = set().union(*E.values())
    overlap = np.mean([sum(e in E[L] for L in Ls) for e in allE]) / len(Ls) if allE else np.nan
    jac = {f"{a}~{b}": len(E[a] & E[b]) / max(1, len(E[a] | E[b])) for a, b in itertools.combinations(Ls, 2)}
    return dict(edges_per_layer={L: len(E[L]) for L in Ls}, global_edge_overlap=overlap, layer_jaccard=jac)


def issue_linkage(cm, a="conflict", b="trade", min_posts=15) -> pd.DataFrame:
    """Per country: odds ratio that a post uses frame b given it uses frame a (Fisher exact test)."""
    rows = []
    for c, g in cm.groupby("country"):
        if len(g) < min_posts:
            continue
        A, B = g[f"frame_{a}"].to_numpy(), g[f"frame_{b}"].to_numpy()
        tab = [[np.sum(A & B), np.sum(A & ~B)], [np.sum(~A & B), np.sum(~A & ~B)]]
        orr, p = stats.fisher_exact(tab)
        rows.append(dict(country=c, n_posts=len(g), both=tab[0][0], only_a=tab[0][1], only_b=tab[1][0], odds_ratio=orr, p=p))
    return pd.DataFrame(rows).set_index("country").sort_values("odds_ratio", ascending=False)


# --------------------------------------------------------------------------
# Panel event study
# --------------------------------------------------------------------------
def decluster_days(days, gap=3):
    days = pd.DatetimeIndex(sorted(set(pd.DatetimeIndex(days))))
    keep, last = [], None
    for d in days:
        if last is None or (d - last).days >= gap:
            keep.append(d)
            last = d
    return pd.DatetimeIndex(keep)


def panel_event_study(Y: pd.DataFrame, events: dict, window=(-5, 10), resp=(0, 2), pre=(-7, -1),
                      log=False, cross_demean=True, n_placebo=2000, seed=0, min_events=3) -> dict:
    """Y: daily DataFrame (date x unit). events: unit -> event dates.

    y -> (log) -> minus the cross-sectional mean of all units that day (removes common shocks such as
    world-wide Trump news) -> each event's response = mean over `resp` days minus mean over `pre` days,
    divided by the unit's sd of daily changes (so units are comparable). Placebo: same number of
    random non-event dates per unit."""
    Y = Y.sort_index().asfreq("D")
    if log:
        Y = np.log1p(Y)
    if cross_demean:
        Y = Y.sub(Y.mean(axis=1), axis=0)
    rng = np.random.default_rng(seed)
    offs = np.arange(window[0], window[1] + 1)
    paths, ev_vals, pl_vals, rows = [], [], [], []
    for u, days in events.items():
        if u not in Y:
            continue
        y = Y[u].to_numpy(float)
        sd = np.nanstd(np.diff(y))
        if not np.isfinite(sd) or sd == 0:
            continue
        n = len(y)
        idx_all = np.arange(n)

        def stat(i):
            if i + min(pre[0], window[0]) < 0 or i + max(resp[1], window[1]) >= n:
                return np.nan
            base = np.nanmean(y[i + pre[0]: i + pre[1] + 1])
            return (np.nanmean(y[i + resp[0]: i + resp[1] + 1]) - base) / sd

        pos = [Y.index.get_loc(d) for d in pd.DatetimeIndex(days) if d in Y.index]
        vals = np.array([stat(i) for i in pos])
        good = np.isfinite(vals)
        if good.sum() < min_events:
            continue
        pos = np.array(pos)[good]
        vals = vals[good]
        for i in pos:
            base = np.nanmean(y[i + pre[0]: i + pre[1] + 1])
            paths.append((y[i + offs] - base) / sd)
        excl = np.zeros(n, bool)
        for i in pos:
            excl[max(0, i - 3): i + resp[1] + 4] = True
        cand = idx_all[~excl & np.isfinite(y)]
        cvals = np.array([stat(i) for i in cand])
        cvals = cvals[np.isfinite(cvals)]
        pl = rng.choice(cvals, (n_placebo, len(vals))).mean(1)
        p = (np.sum(np.abs(pl - pl.mean()) >= abs(vals.mean() - pl.mean())) + 1) / (n_placebo + 1)
        rows.append(dict(unit=u, n_events=len(vals), mean_response_sd=vals.mean(), placebo_mean=pl.mean(), p=p))
        ev_vals.append(vals)
        pl_vals.append(cvals)
    table = pd.DataFrame(rows).set_index("unit") if rows else pd.DataFrame()
    pooled = {}
    if ev_vals:
        allv = np.concatenate(ev_vals)
        pl = np.array([np.mean(np.concatenate([rng.choice(c, len(v)) for c, v in zip(pl_vals, ev_vals)]))
                       for _ in range(max(200, n_placebo // 5))])
        pooled = dict(n_events=len(allv), mean_response_sd=allv.mean(), placebo_mean=pl.mean(), placebo_sd=pl.std(),
                      z=(allv.mean() - pl.mean()) / (pl.std() + 1e-12),
                      p=(np.sum(np.abs(pl - pl.mean()) >= abs(allv.mean() - pl.mean())) + 1) / (len(pl) + 1))
    path = pd.DataFrame({"offset": offs, "mean": np.nanmean(paths, 0) if paths else np.nan,
                         "se": (np.nanstd(paths, 0) / np.sqrt(len(paths))) if paths else np.nan})
    return dict(table=table, pooled=pooled, path=path)


# --------------------------------------------------------------------------
# Attention bursts: endogenous vs exogenous (Crane & Sornette 2008)
# --------------------------------------------------------------------------
def find_spikes(s: pd.Series, k=4.0, window=30, sep=14):
    """Days whose log-views exceed the trailing median by k robust SDs and are local maxima (+-sep days)."""
    x = np.log1p(s.asfreq("D").interpolate(limit_area="inside"))
    med = x.rolling(window, min_periods=10).median().shift(1)
    mad = (x - med).abs().rolling(window, min_periods=10).median().shift(1) * 1.4826 + 1e-6
    z = (x - med) / mad
    cand = z[z > k].index
    peaks = []
    for d in cand:
        w = x[(x.index >= d - pd.Timedelta(days=sep)) & (x.index <= d + pd.Timedelta(days=sep))]
        if x[d] >= w.max():
            peaks.append(d)
    return pd.DatetimeIndex(peaks)


def burst_shape(s: pd.Series, peak, horizon=30, base_window=30):
    """Relaxation exponent gamma from excess(t) ~ t^-gamma after the peak, and the share of the
    burst's excess attention that arrives *before* the peak (precursor share: ~0 for an exogenous
    shock, larger for endogenous, word-of-mouth growth)."""
    s = s.asfreq("D")
    i = s.index.get_loc(peak)
    if i < base_window + 7 or i + horizon >= len(s):
        return np.nan, np.nan, np.nan
    base = np.nanmedian(s.iloc[i - base_window - 7: i - 7])
    after = s.iloc[i + 1: i + horizon + 1].to_numpy() - base
    before = s.iloc[i - 7: i].to_numpy() - base
    peak_ex = s.iloc[i] - base
    t = np.arange(1, horizon + 1)
    ok = after > 0.02 * peak_ex
    gamma = -np.polyfit(np.log(t[ok]), np.log(after[ok]), 1)[0] if ok.sum() >= 5 else np.nan
    total = peak_ex + np.clip(after, 0, None).sum() + np.clip(before, 0, None).sum()
    return gamma, np.clip(before, 0, None).sum() / total, peak_ex / total


# --------------------------------------------------------------------------
# Opinion dynamics with a broadcasting agent
# --------------------------------------------------------------------------
def bimodality_coefficient(x):
    n = len(x)
    g, k = stats.skew(x), stats.kurtosis(x)
    return (g ** 2 + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))


def deffuant_media(stance, reach, N=1000, m=3, eps=0.3, eps_media=None, mu=0.3, mu_media=0.05,
                   follower_frac=0.3, hub_followers=True, seed=0, x0=None, record_every=5):
    """Bounded-confidence opinions x in [-1, 1] on a Barabasi-Albert network.

    Each day: N/2 random peer interactions (agents within eps move towards each other by mu), then the
    media agent broadcasts the real daily stance s(t); each follower is exposed with probability
    reach(t) and, if within eps_media, moves towards s(t) by mu_media. Days with no stance (NaN) have
    no broadcast. Returns per-day summary and the final opinions."""
    rng = np.random.default_rng(seed)
    G = nx.barabasi_albert_graph(N, m, seed=seed)
    E = np.array(G.edges())
    deg = np.array([d for _, d in G.degree()], float)
    x = rng.uniform(-1, 1, N) if x0 is None else np.array(x0, float).copy()
    eps_media = eps if eps_media is None else eps_media
    nf = int(follower_frac * N)
    p = deg / deg.sum() if hub_followers else None
    followers = np.zeros(N, bool)
    if nf:
        followers[rng.choice(N, nf, replace=False, p=p)] = True
    stance, reach = np.asarray(stance, float), np.asarray(reach, float)
    rec = []
    for t in range(len(stance)):
        idx = rng.integers(0, len(E), N // 2)
        i, j = E[idx, 0], E[idx, 1]
        d = x[j] - x[i]
        ok = np.abs(d) < eps
        dx = np.zeros(N)
        np.add.at(dx, i[ok], mu * d[ok])
        np.add.at(dx, j[ok], -mu * d[ok])
        x = np.clip(x + dx, -1, 1)
        if np.isfinite(stance[t]) and reach[t] > 0 and nf:
            exp_ = followers & (rng.random(N) < reach[t])
            dd = stance[t] - x
            mv = exp_ & (np.abs(dd) < eps_media)
            x[mv] += mu_media * dd[mv]
        if t % record_every == 0:
            rec.append((t, x.mean(), x.std(), x[followers].mean() if nf else np.nan, x[~followers].mean(),
                        bimodality_coefficient(x)))
    hist = pd.DataFrame(rec, columns=["t", "mean", "sd", "mean_followers", "mean_others", "bimodality"])
    return hist, x, followers
