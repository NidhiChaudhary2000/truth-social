"""Temporal complexity of the posting process.

* burstiness B and memory M (Goh & Barabasi 2008, EPL 81:48002)
* bursty trains / burst-size distribution (Karsai et al. 2012, Sci. Rep. 2:397)
* heavy-tail fitting with the Clauset-Shalizi-Newman (2009) recipe:
  xmin by KS minimisation, MLE exponent, semi-parametric bootstrap p-value,
  Vuong likelihood-ratio tests against lognormal / exponential.
* detrended fluctuation analysis (Peng et al. 1994) -> Hurst-like exponent
* self-exciting (Hawkes) point process MLE with exponential kernel and an
  optional circadian baseline; branching ratio n = alpha/beta measures the
  fraction of endogenously triggered posts (Filimonov & Sornette 2012).
* Zipf and Heaps laws for the vocabulary.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize, special, stats


# --------------------------------------------------------------------------
# Inter-event statistics
# --------------------------------------------------------------------------
def to_seconds(t: pd.Series) -> np.ndarray:
    """Epoch seconds, independent of the datetime resolution (ns/us) pandas uses."""
    t = pd.to_datetime(pd.Series(t))
    if t.dt.tz is None:
        t = t.dt.tz_localize("UTC")
    return (t - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy()


def interevent_times(t: pd.Series, unit: str = "h") -> np.ndarray:
    ts = np.sort(to_seconds(t))
    dt = np.diff(ts)
    scale = {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
    return dt / scale


def burstiness(tau: np.ndarray) -> float:
    """B = (sigma - mu)/(sigma + mu): -1 periodic, 0 Poisson, -> 1 extremely bursty."""
    m, s = np.mean(tau), np.std(tau)
    return float((s - m) / (s + m))


def burstiness_finite(tau: np.ndarray) -> float:
    """Finite-size corrected burstiness (Kim & Jo 2016, PRE 94:032311)."""
    n = len(tau)
    r = np.std(tau) / np.mean(tau)
    return float((np.sqrt(n + 1) * r - np.sqrt(n - 1)) / ((np.sqrt(n + 1) - 2) * r + np.sqrt(n - 1)))


def memory_coefficient(tau: np.ndarray) -> float:
    """Pearson correlation between consecutive inter-event times."""
    return float(np.corrcoef(tau[:-1], tau[1:])[0, 1])


def burst_sizes(t_seconds: np.ndarray, window: float) -> np.ndarray:
    """Number of events in each bursty train: consecutive events separated by <= window."""
    dt = np.diff(np.sort(t_seconds))
    breaks = np.where(dt > window)[0]
    edges = np.concatenate([[-1], breaks, [len(t_seconds) - 1]])
    return np.diff(edges)


def circadian_rescale(t_et: pd.Series) -> np.ndarray:
    """Time-rescale event times by the empirical weekly activity profile.

    Replaces clock time by 'activity time' A(t) = integral of r(s) ds where r is the
    mean rate per hour-of-week (normalised to mean 1). Burstiness that survives this
    is not just the day/night cycle (Jo et al. 2012, NJP 14:013055).
    Returns rescaled event times in hours.
    """
    t = pd.to_datetime(t_et).sort_values()
    how = (t.dt.dayofweek * 24 + t.dt.hour).to_numpy()
    counts = np.bincount(how, minlength=168).astype(float) + 0.5
    rate = counts / counts.mean()
    start = t.iloc[0].floor("h")
    hrs = (t - start).dt.total_seconds().to_numpy() / 3600.0
    n_hours = int(np.ceil(hrs.max())) + 1
    grid_how = ((start.dayofweek * 24 + start.hour) + np.arange(n_hours)) % 168
    cum = np.concatenate([[0], np.cumsum(rate[grid_how])])
    k = np.floor(hrs).astype(int)
    return cum[k] + rate[grid_how[k]] * (hrs - k)


# --------------------------------------------------------------------------
# Power laws (Clauset, Shalizi & Newman 2009, SIAM Review 51:661)
# --------------------------------------------------------------------------
@dataclass
class PowerLawFit:
    alpha: float
    xmin: float
    n_tail: int
    ks: float
    sigma: float
    discrete: bool
    p_value: float | None = None
    lr_lognormal: tuple | None = None   # (normalised LR, p); LR>0 favours power law
    lr_exponential: tuple | None = None

    def summary(self) -> dict:
        d = dict(alpha=self.alpha, alpha_se=self.sigma, xmin=self.xmin, n_tail=self.n_tail, KS=self.ks,
                 p_gof=self.p_value)
        if self.lr_lognormal:
            d.update(LR_vs_lognormal=self.lr_lognormal[0], p_LR_lognormal=self.lr_lognormal[1])
        if self.lr_exponential:
            d.update(LR_vs_exponential=self.lr_exponential[0], p_LR_exponential=self.lr_exponential[1])
        return d


def _alpha_mle(x, xmin, discrete):
    if discrete:  # standard approximation for discrete data (CSN eq. 3.7)
        return 1 + len(x) / np.sum(np.log(x / (xmin - 0.5)))
    return 1 + len(x) / np.sum(np.log(x / xmin))


def _ks(x, xmin, alpha, discrete):
    x = np.sort(x)
    n = len(x)
    if discrete:
        xs = np.unique(x)
        emp = np.searchsorted(x, xs, side="right") / n
        theo = 1 - special.zeta(alpha, xs + 1) / special.zeta(alpha, xmin)
        return float(np.max(np.abs(emp - theo)))
    emp = np.arange(1, n + 1) / n
    theo = 1 - (x / xmin) ** (1 - alpha)
    return float(np.max(np.abs(emp - theo)))


def fit_powerlaw(x, discrete=False, xmin=None, n_xmin=60, min_tail=50) -> PowerLawFit:
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x > 0)]
    if xmin is None:
        cand = np.unique(np.quantile(x, np.linspace(0, 0.98, n_xmin)))
        if discrete:
            cand = np.unique(np.maximum(1, np.round(cand)))
        best = None
        for xm in cand:
            tail = x[x >= xm]
            if len(tail) < min_tail:
                continue
            a = _alpha_mle(tail, xm, discrete)
            if not np.isfinite(a) or a <= 1:
                continue
            d = _ks(tail, xm, a, discrete)
            if best is None or d < best[0]:
                best = (d, xm, a)
        if best is None:
            if min_tail > 10:
                return fit_powerlaw(x, discrete, None, n_xmin, min_tail=10)
            return PowerLawFit(np.nan, np.nan, 0, np.nan, np.nan, discrete)
        d, xmin, a = best
    else:
        tail = x[x >= xmin]
        a = _alpha_mle(tail, xmin, discrete)
        d = _ks(tail, xmin, a, discrete)
    n = int(np.sum(x >= xmin))
    return PowerLawFit(alpha=float(a), xmin=float(xmin), n_tail=n, ks=float(d), sigma=float((a - 1) / np.sqrt(n)),
                       discrete=discrete)


def _sample_powerlaw(n, alpha, xmin, discrete, rng):
    u = rng.random(n)
    if discrete:  # CSN approximation (eq. D.6)
        return np.floor((xmin - 0.5) * (1 - u) ** (-1 / (alpha - 1)) + 0.5)
    return xmin * (1 - u) ** (-1 / (alpha - 1))


def powerlaw_pvalue(x, fit: PowerLawFit, n_boot=100, seed=0) -> float:
    """Semi-parametric bootstrap goodness-of-fit p-value (p > 0.1 => power law plausible)."""
    if not np.isfinite(fit.alpha):
        return np.nan
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x > 0)]
    body = x[x < fit.xmin]
    n, ptail = len(x), np.mean(x >= fit.xmin)
    worse = 0
    for _ in range(n_boot):
        nt = rng.binomial(n, ptail)
        synth = np.concatenate([_sample_powerlaw(nt, fit.alpha, fit.xmin, fit.discrete, rng),
                                rng.choice(body, n - nt) if len(body) else np.array([])])
        f = fit_powerlaw(synth, discrete=fit.discrete, n_xmin=30)
        worse += f.ks >= fit.ks
    fit.p_value = worse / n_boot
    return fit.p_value


def _loglik_pl(x, xmin, alpha, discrete):
    if discrete:
        return -alpha * np.log(x) - np.log(special.zeta(alpha, xmin))
    return np.log(alpha - 1) - np.log(xmin) - alpha * np.log(x / xmin)


def _fit_lognormal_tail(x, xmin):
    lx = np.log(x)

    def nll(p):
        mu, ls = p
        s = np.exp(ls)
        logsf = stats.norm.logsf((np.log(xmin) - mu) / s)
        return -np.sum(stats.norm.logpdf(lx, mu, s) - lx - logsf)

    r = optimize.minimize(nll, [np.mean(lx), np.log(np.std(lx) + 1e-6)], method="Nelder-Mead")
    mu, s = r.x[0], np.exp(r.x[1])
    ll = stats.norm.logpdf(lx, mu, s) - lx - stats.norm.logsf((np.log(xmin) - mu) / s)
    return ll, (mu, s)


def compare_tails(x, fit: PowerLawFit):
    """Vuong normalised log-likelihood ratios (CSN section 5). R>0 favours the power law."""
    if not np.isfinite(fit.alpha):
        fit.lr_lognormal = fit.lr_exponential = (np.nan, np.nan)
        return {"lognormal": (np.nan, np.nan), "exponential": (np.nan, np.nan)}
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x >= fit.xmin)]
    lp = _loglik_pl(x, fit.xmin, fit.alpha, fit.discrete)
    out = {}
    lln, _ = _fit_lognormal_tail(x, fit.xmin)
    lam = 1 / (np.mean(x) - fit.xmin + 1e-12)
    lle = np.log(lam) - lam * (x - fit.xmin)
    for name, lo in [("lognormal", lln), ("exponential", lle)]:
        d = lp - lo
        R = d.sum()
        sd = d.std(ddof=1) * np.sqrt(len(d))
        z = R / sd if sd > 0 else 0.0
        p = 2 * stats.norm.sf(abs(z))
        out[name] = (float(z), float(p))
    fit.lr_lognormal, fit.lr_exponential = out["lognormal"], out["exponential"]
    return out


def ccdf(x):
    x = np.sort(np.asarray(x, float))
    x = x[np.isfinite(x) & (x > 0)]
    y = 1 - np.arange(len(x)) / len(x)
    return x, y


# --------------------------------------------------------------------------
# Detrended fluctuation analysis
# --------------------------------------------------------------------------
def dfa(series, scales=None, order=1):
    """Returns (scales, F(s), alpha). alpha=0.5 uncorrelated, >0.5 persistent long-range correlations."""
    x = np.asarray(series, float)
    y = np.cumsum(x - x.mean())
    n = len(y)
    if scales is None:
        scales = np.unique(np.logspace(np.log10(8), np.log10(n // 4), 20).astype(int))
    F = []
    for s in scales:
        m = n // s
        segs = y[: m * s].reshape(m, s)
        tt = np.arange(s)
        V = np.vander(tt, order + 1)
        coef, *_ = np.linalg.lstsq(V, segs.T, rcond=None)
        resid = segs.T - V @ coef
        F.append(np.sqrt(np.mean(resid ** 2)))
    F = np.array(F)
    alpha = np.polyfit(np.log(scales), np.log(F), 1)[0]
    return np.asarray(scales), F, float(alpha)


# --------------------------------------------------------------------------
# Hawkes process
# --------------------------------------------------------------------------
@dataclass
class HawkesFit:
    mu: float
    alpha: float
    beta: float
    loglik: float
    n_events: int
    T: float
    baseline: str

    @property
    def branching_ratio(self):
        return self.alpha / self.beta

    @property
    def decay_time(self):
        return 1 / self.beta

    def aic(self):
        return 2 * 3 - 2 * self.loglik


def _excitation_loop(t, b):
    import math
    tl = t.tolist()
    out = [0.0] * len(tl)
    a = 0.0
    for i in range(1, len(tl)):
        a = math.exp(-b * (tl[i] - tl[i - 1])) * (1.0 + a)
        out[i] = a
    return np.array(out)


def excitation_sum(t, b):
    """A_i = sum_{j<i} exp(-b (t_i - t_j)).

    Slow kernels (few blocks): vectorised blocks, numerically stable. Fast kernels (many blocks):
    the O(n) scalar recursion A_i = exp(-b dt_i) (1 + A_{i-1}) is cheaper."""
    t = np.asarray(t, float)
    n = len(t)
    if n > 1 and b * (t[-1] - t[0]) / 500.0 > n / 20:
        return _excitation_loop(t, b)
    A = np.zeros(n)
    L = 500.0 / b  # block length so that exp(b*L) cannot overflow
    edges = np.searchsorted(t, np.arange(t[0], t[-1] + L, L), side="left")
    edges = np.unique(np.concatenate([edges, [n]]))
    carry, t_prev_ref = 0.0, t[0]
    for k in range(len(edges) - 1):
        i0, i1 = edges[k], edges[k + 1]
        if i0 >= i1:
            continue
        tau = t[i0]
        carry *= np.exp(-b * (tau - t_prev_ref))  # decay carry from previous reference time
        w = np.exp(b * (t[i0:i1] - tau))
        cs = np.concatenate([[0.0], np.cumsum(w)[:-1]])  # sum over earlier events in block
        A[i0:i1] = np.exp(-b * (t[i0:i1] - tau)) * (carry + cs)
        carry = carry + w.sum()  # now referenced to tau
        t_prev_ref = tau
    return A


def _hawkes_nll(params, t, T, s_t, S_T):
    """Exponential-kernel Hawkes: lambda(t) = mu*s(t) + alpha*sum exp(-beta(t-ti)).

    s_t: baseline modulation at event times; S_T: integral of s over [0,T].
    """
    mu, a, b = np.exp(params)
    if a >= b * 0.999:  # enforce stationarity n<1
        return 1e12
    A = excitation_sum(t, b)
    lam = mu * s_t + a * A
    if np.any(lam <= 0):
        return 1e12
    comp = mu * S_T + (a / b) * np.sum(1 - np.exp(-b * (T - t)))
    return -(np.sum(np.log(lam)) - comp)


def _baseline(t_hours, start_how, T, profile):
    if profile is None:
        return np.ones_like(t_hours), T
    hrs = np.arange(int(np.ceil(T)) + 1)
    s_grid = profile[(start_how + hrs) % len(profile)]
    s_t = profile[(start_how + np.floor(t_hours).astype(int)) % len(profile)]
    S_T = s_grid[:-1].sum() + s_grid[-1] * (T - np.floor(T))
    return s_t, S_T


def fit_hawkes(t_et: pd.Series, circadian: bool = True, jitter_seed: int = 0) -> HawkesFit:
    """MLE of an exponential Hawkes process on event times (units: hours).

    With circadian=True the baseline is mu * s(hour-of-week) where s is the
    empirical weekly profile (mean 1). This guards against the well-known bias
    in which an unmodelled seasonal baseline masquerades as self-excitation.
    """
    t = pd.to_datetime(t_et).sort_values()
    start = t.iloc[0].floor("h")
    th = (t - start).dt.total_seconds().to_numpy() / 3600.0
    rng = np.random.default_rng(jitter_seed)
    th = np.sort(th + rng.uniform(0, 1 / 3600, len(th)))  # break exact ties (same second)
    T = th[-1] + 1e-6
    profile = None
    start_how = int(start.dayofweek * 24 + start.hour)
    if circadian:
        how = (t.dt.dayofweek * 24 + t.dt.hour).to_numpy()
        c = np.bincount(how, minlength=168).astype(float) + 0.5
        profile = c / c.mean()
    s_t, S_T = _baseline(th, start_how, T, profile)
    n = len(th)
    best = None
    for b0 in [0.5, 2.0, 8.0]:
        for n0 in [0.3, 0.6]:
            x0 = np.log([n * (1 - n0) / T, n0 * b0, b0])
            r = optimize.minimize(_hawkes_nll, x0, args=(th, T, s_t, S_T), method="Nelder-Mead",
                                  options=dict(maxiter=2000, xatol=1e-5, fatol=1e-4))
            if best is None or r.fun < best.fun:
                best = r
    mu, a, b = np.exp(best.x)
    fit = HawkesFit(mu, a, b, -best.fun, n, T, "circadian" if circadian else "constant")
    fit._th, fit._s_t, fit._profile, fit._start_how = th, s_t, profile, start_how  # for diagnostics
    return fit


def hawkes_residuals(fit: HawkesFit) -> np.ndarray:
    """Time-rescaling theorem: compensator increments should be i.i.d. Exp(1) if the model is right."""
    th = fit._th
    prof = fit._profile
    if prof is None:
        base_cum = fit.mu * th
    else:
        hrs = np.arange(int(np.ceil(th[-1])) + 2)
        s_grid = prof[(fit._start_how + hrs) % len(prof)]
        cum = np.concatenate([[0], np.cumsum(s_grid)])
        k = np.floor(th).astype(int)
        base_cum = fit.mu * (cum[k] + s_grid[k] * (th - k))
    a, b = fit.alpha, fit.beta
    # Lambda(t_i) = base + (a/b) * sum_{j<i} (1 - exp(-b(t_i - t_j)))
    A = excitation_sum(th, b)
    exc_cum = (a / b) * (np.arange(len(th)) - A)
    Lam = base_cum + exc_cum
    return np.diff(Lam)


def rolling_hawkes(t_et: pd.Series, window="90D", step="30D", circadian=True, min_events=150):
    t = pd.to_datetime(t_et).sort_values()
    starts = pd.date_range(t.iloc[0].floor("D"), t.iloc[-1] - pd.Timedelta(window), freq=step)
    rows = []
    for s in starts:
        e = s + pd.Timedelta(window)
        sub = t[(t >= s) & (t < e)]
        if len(sub) < min_events:
            continue
        f = fit_hawkes(sub, circadian=circadian)
        rows.append(dict(start=s, mid=s + pd.Timedelta(window) / 2, n_events=len(sub), mu=f.mu,
                         branching_ratio=f.branching_ratio, decay_min=60 / f.beta))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Linguistic laws
# --------------------------------------------------------------------------
def zipf(token_lists):
    from collections import Counter
    c = Counter(w for toks in token_lists for w in toks)
    freq = np.array(sorted(c.values(), reverse=True), float)
    ranks = np.arange(1, len(freq) + 1)
    sel = (ranks >= 10) & (ranks <= min(5000, len(freq)))
    slope = np.polyfit(np.log(ranks[sel]), np.log(freq[sel]), 1)[0]
    return ranks, freq, float(-slope), c


def heaps(token_lists):
    seen, N, V = set(), [], []
    n = 0
    for toks in token_lists:
        for w in toks:
            n += 1
            seen.add(w)
        N.append(n)
        V.append(len(seen))
    N, V = np.array(N), np.array(V)
    sel = N > 1000
    beta = np.polyfit(np.log(N[sel]), np.log(V[sel]), 1)[0]
    return N, V, float(beta)


# --------------------------------------------------------------------------
# Multi-timescale Hawkes (sum of exponential kernels)
# --------------------------------------------------------------------------
@dataclass
class HawkesMultiFit:
    mu: float
    alphas: np.ndarray
    betas: np.ndarray
    loglik: float
    n_events: int
    T: float
    baseline: str

    @property
    def branching_ratio(self):
        return float(np.sum(self.alphas / self.betas))

    @property
    def components(self):
        """Per-kernel branching ratio and decay time (minutes), fastest first."""
        return pd.DataFrame({"branching_ratio": self.alphas / self.betas, "decay_minutes": 60 / self.betas})

    def aic(self):
        return 2 * (1 + 2 * len(self.betas)) - 2 * self.loglik


def _nll_multi(params, t, T, s_t, S_T, K):
    p = np.exp(params)
    mu, a, b = p[0], p[1:1 + K], p[1 + K:]
    if np.any(np.diff(b) >= 0) or np.sum(a / b) >= 0.999:  # betas strictly decreasing (fast -> slow), n < 1
        return 1e12
    if b[0] > 600 or b[-1] < 1 / (24 * 14):  # decay times between 6 seconds and 2 weeks
        return 1e12
    lam = mu * s_t
    comp = mu * S_T
    for k in range(K):
        lam = lam + a[k] * excitation_sum(t, b[k])
        comp += (a[k] / b[k]) * np.sum(1 - np.exp(-b[k] * (T - t)))
    if np.any(lam <= 0):
        return 1e12
    return -(np.sum(np.log(lam)) - comp)


def fit_hawkes_multi(t_et: pd.Series, K: int = 2, circadian: bool = True, init_decays_min=(2.0, 60.0),
                     jitter_seed: int = 0) -> HawkesMultiFit:
    """Hawkes process with K exponential kernels (e.g. a fast 'reposting spree' and a slow 'topic' timescale).
    A single exponential cannot fit both; this is checked with AIC and the time-rescaling KS test."""
    t = pd.to_datetime(t_et).sort_values()
    start = t.iloc[0].floor("h")
    th = (t - start).dt.total_seconds().to_numpy() / 3600.0
    th = np.sort(th + np.random.default_rng(jitter_seed).uniform(0, 1 / 3600, len(th)))
    T = th[-1] + 1e-6
    start_how = int(start.dayofweek * 24 + start.hour)
    profile = None
    if circadian:
        how = (t.dt.dayofweek * 24 + t.dt.hour).to_numpy()
        c = np.bincount(how, minlength=168).astype(float) + 0.5
        profile = c / c.mean()
    s_t, S_T = _baseline(th, start_how, T, profile)
    n = len(th)
    decays = np.geomspace(init_decays_min[0], init_decays_min[-1], K) / 60.0  # hours
    best = None
    for n0 in [0.4, 0.7]:
        for shift in [0.5, 1.0, 2.0]:
            b0 = 1 / (decays * shift)
            a0 = (n0 / K) * b0
            x0 = np.log(np.concatenate([[n * (1 - n0) / T], a0, b0]))
            r = optimize.minimize(_nll_multi, x0, args=(th, T, s_t, S_T, K), method="Nelder-Mead",
                                  options=dict(maxiter=4000, maxfev=4000, xatol=1e-4, fatol=1e-3))
            if best is None or r.fun < best.fun:
                best = r
    p = np.exp(best.x)
    fit = HawkesMultiFit(p[0], p[1:1 + K], p[1 + K:], -best.fun, n, T, "circadian" if circadian else "constant")
    fit._th, fit._profile, fit._start_how = th, profile, start_how
    return fit


def hawkes_multi_residuals(fit: HawkesMultiFit) -> np.ndarray:
    th, prof = fit._th, fit._profile
    if prof is None:
        base_cum = fit.mu * th
    else:
        hrs = np.arange(int(np.ceil(th[-1])) + 2)
        s_grid = prof[(fit._start_how + hrs) % len(prof)]
        cum = np.concatenate([[0], np.cumsum(s_grid)])
        k = np.floor(th).astype(int)
        base_cum = fit.mu * (cum[k] + s_grid[k] * (th - k))
    Lam = base_cum.copy()
    for a, b in zip(fit.alphas, fit.betas):
        Lam += (a / b) * (np.arange(len(th)) - excitation_sum(th, b))
    return np.diff(Lam)
