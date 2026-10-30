"""Coupling between posts and markets / macro-political series.

* daily post features aligned to trading days
* event studies (daily market-model CARs, intraday hourly responses) with
  placebo/permutation nulls instead of parametric t-tests
* transfer entropy (Schreiber 2000, PRL 85:461) with time-shift surrogates
* Granger causality F-tests, lagged cross-correlations
* random-matrix theory of the cross-asset correlation matrix (Laloux et al. 1999;
  Plerou et al. 1999) and the "market mode"
* minimum spanning trees of assets (Mantegna 1999; Onnela et al. 2003)
* Benjamini-Hochberg false-discovery-rate control
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

from .data import trading_day_of


def log_returns(px: pd.DataFrame) -> pd.DataFrame:
    return np.log(px).diff()


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
def daily_features(posts: pd.DataFrame, trading_days: pd.DatetimeIndex, topic_cols=None) -> pd.DataFrame:
    """Aggregate posts onto the trading day whose close first reflects them."""
    p = posts.copy()
    p["tday"] = trading_day_of(p["t_et"], trading_days)
    p = p.dropna(subset=["tday"])
    topic_cols = topic_cols or [c for c in p.columns if c.startswith("topic_")]
    g = p.groupby("tday")
    f = pd.DataFrame({"n_posts": g.size(), "n_original": g["is_repost"].apply(lambda s: (~s).sum()),
                      "mean_sent": g["sent"].mean(), "n_neg": g["sent"].apply(lambda s: (s < -0.2).sum()),
                      "caps": g["caps_ratio"].mean()})
    for c in topic_cols:
        f[c.replace("topic_", "n_")] = g[c].sum()
    f = f.reindex(pd.DatetimeIndex(trading_days)).fillna({c: 0 for c in f.columns if c.startswith("n_")})
    return f


def decluster(times: pd.Series, gap="3h") -> pd.Series:
    """Keep the first event of each cluster (events closer than `gap` belong to the same cluster)."""
    t = pd.to_datetime(times).sort_values()
    keep = t.diff().isna() | (t.diff() > pd.Timedelta(gap))
    return t[keep]


# --------------------------------------------------------------------------
# Daily event study
# --------------------------------------------------------------------------
def event_study_daily(rets: pd.DataFrame, event_days, asset: str, market: str | None = "SPY",
                      window=(-2, 5), car_window=(0, 1), n_placebo=2000, seed=0, min_sep=None) -> dict:
    """Market-model abnormal returns around event days, with a placebo null.

    AR_t = r_t - (a + b r_m,t) with (a, b) estimated on days outside all event windows.
    If asset == market (or market is None), a constant-mean model is used.
    Placebo: same number of random non-event days -> distribution of mean CAR.
    """
    r = rets[[asset] + ([market] if market and market != asset else [])].dropna()
    days = r.index
    ev = pd.DatetimeIndex(sorted(set(pd.DatetimeIndex(event_days)) & set(days)))
    min_sep = min_sep if min_sep is not None else (car_window[1] - car_window[0] + 1)
    pos_all = np.array([days.get_loc(d) for d in ev])
    pos, last = [], -10**9  # non-overlapping events
    for p_ in pos_all:
        if p_ - last >= min_sep:
            pos.append(p_)
            last = p_
    pos = np.array(pos, int)
    in_win = np.zeros(len(days), bool)
    for p_ in pos_all:
        in_win[max(0, p_ + window[0]): p_ + window[1] + 1] = True
    y = r[asset].to_numpy()
    if market and market != asset:
        x = r[market].to_numpy()
        b, a = np.polyfit(x[~in_win], y[~in_win], 1)
        ar = y - (a + b * x)
    else:
        a, b = y[~in_win].mean(), 0.0
        ar = y - a
    ar = pd.Series(ar, index=days)
    sd = ar[~in_win].std()
    offs = np.arange(window[0], window[1] + 1)
    valid = [p_ for p_ in pos if p_ + window[0] >= 0 and p_ + window[1] < len(days)]
    M = np.array([[ar.iloc[p_ + o] for o in offs] for p_ in valid]) if valid else np.empty((0, len(offs)))
    path = pd.DataFrame({"offset": offs, "mean_AR": M.mean(0) if len(M) else np.nan,
                         "CAAR": np.cumsum(M.mean(0)) if len(M) else np.nan})
    c0, c1 = car_window
    car = M[:, (offs >= c0) & (offs <= c1)].sum(1) if len(M) else np.array([])
    rng = np.random.default_rng(seed)
    cand = np.where(~in_win)[0]
    cand = cand[(cand + c0 >= 0) & (cand + c1 < len(days))]
    arv = ar.to_numpy()
    csum = np.concatenate([[0], np.cumsum(arv)])
    car_all = np.full(len(arv), np.nan)
    car_all[cand] = csum[cand + c1 + 1] - csum[cand + c0]
    plac = (car_all[rng.choice(cand, (n_placebo, len(car)))].mean(1) if len(car) else np.array([np.nan]))
    m = car.mean() if len(car) else np.nan
    p = (np.sum(np.abs(plac - plac.mean()) >= abs(m - plac.mean())) + 1) / (len(plac) + 1)
    # also absolute AR (volatility) response on event day vs placebo
    absm = np.mean(np.abs(M[:, offs == 0])) if len(M) else np.nan
    absplac = (np.abs(arv[rng.choice(cand, (n_placebo, len(car)))]).mean(1) if len(car) else np.array([np.nan]))
    p_abs = (np.sum(absplac >= absm) + 1) / (len(absplac) + 1)
    return dict(asset=asset, n_events=len(car), beta=b, mean_CAR=m, CAR_sd_units=m / (sd * np.sqrt(c1 - c0 + 1)),
                p_placebo=p, mean_absAR0=absm, absAR_ratio=absm / np.mean(absplac), p_abs=p_abs,
                path=path, cars=car, placebo=plac)


# --------------------------------------------------------------------------
# Intraday event study (hourly bars)
# --------------------------------------------------------------------------
def intraday_event_study(px_h: pd.Series, event_times, horizons=(1, 2, 3, 6), n_placebo=2000, seed=0) -> pd.DataFrame:
    """Response of an asset in the h hourly bars after a post published during regular trading hours.

    r_h = log P(close of bar containing t + h-1 bars) - log P(close of previous bar).
    Placebo bars are drawn with the same bar-of-day mix as the events, which controls for the
    intraday U-shaped volatility profile.
    """
    s = px_h.dropna()
    s = s[(s.index.time >= pd.Timestamp("09:30").time()) & (s.index.time < pd.Timestamp("16:00").time())]
    lp = np.log(s.to_numpy())
    idx = s.index
    day = idx.normalize()
    bar_of_day = pd.Series(idx.hour * 60 + idx.minute).groupby(day).rank(method="first").to_numpy().astype(int)
    ev = pd.DatetimeIndex(pd.to_datetime(event_times))
    if ev.tz is None:
        ev = ev.tz_localize("America/New_York")
    ev = ev.tz_convert(idx.tz)
    pos = np.searchsorted(idx.values, ev.values, side="right") - 1  # bar containing t
    ok = (pos >= 1) & (pos < len(idx))
    ok &= np.array([(idx[p].normalize() == e.normalize()) and (e - idx[p] < pd.Timedelta("1h"))
                    if 0 <= p < len(idx) else False for p, e in zip(pos, ev)])
    pos = pos[ok]
    rng = np.random.default_rng(seed)
    rows = []
    for h in horizons:
        def resp(p_):
            q = p_ + h - 1
            if q >= len(idx) or day[q] != day[p_] or p_ - 1 < 0:
                return np.nan
            return lp[q] - lp[p_ - 1]
        r_ev = np.array([resp(p_) for p_ in pos])
        good = np.isfinite(r_ev)
        r_ev, bod = r_ev[good], bar_of_day[pos[good]]
        excl = np.zeros(len(idx), bool)  # exclude bars overlapping any event window
        for p_ in pos:
            excl[max(0, p_ - h): p_ + h + 1] = True
        cand_by_bar = {b: np.where((bar_of_day == b) & ~excl)[0] for b in np.unique(bod)}
        pl_mean, pl_abs = [], []
        all_cand = np.concatenate(list(cand_by_bar.values())) if cand_by_bar else np.array([], int)
        resp_cache = {}
        for _ in range(n_placebo):
            draws = np.concatenate([rng.choice(cand_by_bar[b], np.sum(bod == b)) for b in np.unique(bod)
                                    if len(cand_by_bar[b])]) if len(all_cand) else np.array([], int)
            vals = []
            for q in draws:
                if q not in resp_cache:
                    resp_cache[q] = resp(q)
                vals.append(resp_cache[q])
            vals = np.array(vals)
            vals = vals[np.isfinite(vals)]
            pl_mean.append(vals.mean() if len(vals) else np.nan)
            pl_abs.append(np.abs(vals).mean() if len(vals) else np.nan)
        pl_mean, pl_abs = np.array(pl_mean), np.array(pl_abs)
        m, am = r_ev.mean(), np.abs(r_ev).mean()
        rows.append(dict(h=h, n=len(r_ev), mean_ret_bp=1e4 * m, mean_abs_bp=1e4 * am,
                         placebo_abs_bp=1e4 * np.nanmean(pl_abs), abs_ratio=am / np.nanmean(pl_abs),
                         p_abs=(np.sum(pl_abs >= am) + 1) / (n_placebo + 1),
                         p_mean=(np.sum(np.abs(pl_mean - np.nanmean(pl_mean)) >= abs(m - np.nanmean(pl_mean))) + 1)
                         / (n_placebo + 1)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Information flow
# --------------------------------------------------------------------------
def _discretise(x, bins):
    x = np.asarray(x, float)
    qs = np.unique(np.nanquantile(x, np.linspace(0, 1, bins + 1)[1:-1]))
    return np.digitize(x, qs, right=True)


def _te(xs, ys, k=1, l=1):
    """TE_{X->Y} in bits with history lengths k (target) and l (source)."""
    n = len(ys)
    m = max(k, l)
    yf = ys[m:]
    yp = np.stack([ys[m - i - 1: n - i - 1] for i in range(k)], 1)
    xp = np.stack([xs[m - i - 1: n - i - 1] for i in range(l)], 1)
    base = max(ys.max(), xs.max()) + 1

    def code(cols):
        c = np.zeros(len(cols), np.int64)
        for j in range(cols.shape[1]):
            c = c * base + cols[:, j]
        return c

    c_yp, c_xp = code(yp), code(xp)
    c_yp_xp = c_yp * (base ** l) + c_xp
    c_yf_yp = yf * (base ** k) + c_yp
    c_all = yf * (base ** (k + l)) + c_yp_xp

    def H(c):
        _, cnt = np.unique(c, return_counts=True)
        p = cnt / cnt.sum()
        return -np.sum(p * np.log2(p))

    # TE = H(Yf,Yp) - H(Yp) - H(Yf,Yp,Xp) + H(Yp,Xp)
    return H(c_yf_yp) - H(c_yp) - H(c_all) + H(c_yp_xp)


def transfer_entropy(x, y, bins=3, k=1, l=1, n_surr=500, seed=0) -> dict:
    """TE from x to y with circular time-shift surrogates of x (preserve x's autocorrelation)."""
    d = pd.concat([pd.Series(np.asarray(x, float)), pd.Series(np.asarray(y, float))], axis=1).dropna()
    xs, ys = _discretise(d.iloc[:, 0], bins), _discretise(d.iloc[:, 1], bins)
    te = _te(xs, ys, k, l)
    rng = np.random.default_rng(seed)
    n = len(xs)
    surr = np.array([_te(np.roll(xs, rng.integers(20, n - 20)), ys, k, l) for _ in range(n_surr)])
    return dict(TE_bits=te, TE_eff=te - surr.mean(), z=(te - surr.mean()) / (surr.std() + 1e-15),
                p=(np.sum(surr >= te) + 1) / (n_surr + 1), n=n)


def te_both_ways(x, y, names=("posts", "market"), **kw) -> pd.DataFrame:
    a = transfer_entropy(x, y, **kw)
    b = transfer_entropy(y, x, **kw)
    return pd.DataFrame([dict(direction=f"{names[0]} -> {names[1]}", **a),
                         dict(direction=f"{names[1]} -> {names[0]}", **b)])


def granger(x, y, maxlag=5) -> pd.DataFrame:
    """Does x Granger-cause y? F-test of lags of x in an OLS of y on its own lags (+ x lags)."""
    d = pd.concat([pd.Series(np.asarray(y, float), name="y"), pd.Series(np.asarray(x, float), name="x")],
                  axis=1).dropna().to_numpy()
    rows = []
    for p in range(1, maxlag + 1):
        Y = d[p:, 0]
        lagsY = np.column_stack([d[p - i: len(d) - i, 0] for i in range(1, p + 1)])
        lagsX = np.column_stack([d[p - i: len(d) - i, 1] for i in range(1, p + 1)])
        Xr = np.column_stack([np.ones(len(Y)), lagsY])
        Xu = np.column_stack([Xr, lagsX])
        rss_r = np.sum((Y - Xr @ np.linalg.lstsq(Xr, Y, rcond=None)[0]) ** 2)
        rss_u = np.sum((Y - Xu @ np.linalg.lstsq(Xu, Y, rcond=None)[0]) ** 2)
        df1, df2 = p, len(Y) - Xu.shape[1]
        F = ((rss_r - rss_u) / df1) / (rss_u / df2)
        rows.append(dict(lag=p, F=F, p=stats.f.sf(F, df1, df2)))
    return pd.DataFrame(rows)


def lagged_xcorr(x, y, maxlag=10) -> pd.DataFrame:
    """corr(x_t, y_{t+lag}); positive lag = x leads y."""
    x = pd.Series(np.asarray(x, float))
    y = pd.Series(np.asarray(y, float))
    rows = [dict(lag=L, r=x.corr(y.shift(-L))) for L in range(-maxlag, maxlag + 1)]
    df = pd.DataFrame(rows)
    df["ci95"] = 1.96 / np.sqrt(x.notna().sum())
    return df


# --------------------------------------------------------------------------
# Random matrix theory + MST
# --------------------------------------------------------------------------
def rmt_spectrum(rets: pd.DataFrame) -> dict:
    R = rets.dropna()
    Z = (R - R.mean()) / R.std()
    T, N = Z.shape
    C = np.corrcoef(Z.to_numpy().T)
    w, V = np.linalg.eigh(C)
    q = N / T
    lam_minus, lam_plus = (1 - np.sqrt(q)) ** 2, (1 + np.sqrt(q)) ** 2
    return dict(eigvals=w, eigvecs=V, lambda_plus=lam_plus, lambda_minus=lam_minus, q=q, N=N, T=T,
                n_signal=int(np.sum(w > lam_plus)), market_mode_share=w[-1] / N,
                market_vector=pd.Series(V[:, -1] * np.sign(V[:, -1].sum()), index=R.columns), C=C)


def marchenko_pastur_pdf(x, q):
    lm, lp = (1 - np.sqrt(q)) ** 2, (1 + np.sqrt(q)) ** 2
    out = np.zeros_like(x, float)
    m = (x > lm) & (x < lp)
    out[m] = np.sqrt((lp - x[m]) * (x[m] - lm)) / (2 * np.pi * q * x[m])
    return out


def rolling_market_mode(rets: pd.DataFrame, window=60) -> pd.Series:
    R = rets.dropna()
    out = {}
    for i in range(window, len(R) + 1):
        C = np.corrcoef(R.iloc[i - window: i].to_numpy().T)
        out[R.index[i - 1]] = np.linalg.eigvalsh(C)[-1] / R.shape[1]
    return pd.Series(out, name="lambda_max_share")


def mst_from_returns(rets: pd.DataFrame):
    C = rets.dropna().corr()
    D = np.sqrt(2 * (1 - C.clip(-1, 1)))
    G = nx.Graph()
    cols = list(C.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            G.add_edge(cols[i], cols[j], weight=float(D.iloc[i, j]), corr=float(C.iloc[i, j]))
    T = nx.minimum_spanning_tree(G)
    L = np.mean([d["weight"] for *_, d in T.edges(data=True)])
    return T, float(L), C


def regime_comparison(rets: pd.DataFrame, mask: pd.Series, n_boot=500, seed=0) -> dict:
    """Mean pairwise correlation and normalised MST length on 'high' days vs random day subsets."""
    R = rets.dropna()
    mask = mask.reindex(R.index).fillna(False).astype(bool)

    def stats_of(sub):
        C = sub.corr().to_numpy()
        iu = np.triu_indices_from(C, 1)
        _, L, _ = mst_from_returns(sub)
        return C[iu].mean(), L

    hi = stats_of(R[mask])
    lo = stats_of(R[~mask])
    rng = np.random.default_rng(seed)
    k = int(mask.sum())
    boot = np.array([stats_of(R.iloc[np.sort(rng.choice(len(R), k, replace=False))]) for _ in range(n_boot)])
    return dict(n_high=k, mean_corr_high=hi[0], mean_corr_low=lo[0], mst_len_high=hi[1], mst_len_low=lo[1],
                p_corr=(np.sum(boot[:, 0] >= hi[0]) + 1) / (n_boot + 1),
                p_mst=(np.sum(boot[:, 1] <= hi[1]) + 1) / (n_boot + 1))


def bh_fdr(p, q=0.05):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    adj[order] = np.minimum.accumulate((p[order] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(adj, 1), adj <= q
