"""Data acquisition and cleaning.

Sources
-------
* Truth Social posts: the auto-updating archive maintained by CNN
  (https://ix.cnn.io/data/truth-social/truth_archive.json), originally built by
  Matt Stiles (github.com/stiles/trump-truth-social-archive).
  Fields: id, created_at (ISO-8601, UTC), content (HTML), url, media,
  replies_count, reblogs_count, favourites_count.
* Market prices: Yahoo Finance through the `yfinance` package (daily + hourly).
* Macro / uncertainty series: FRED CSV endpoint (no API key needed), e.g.
  VIXCLS (VIX close) and USEPUINDXD (daily Economic Policy Uncertainty index).
* Optional: a user-supplied approval-rating CSV (date, approve, disapprove).

Everything downloaded is cached in `data_dir` so the notebook is reproducible
offline once the first run has succeeded.
"""
from __future__ import annotations

import html
import io
import json
import os
import re
import time
import urllib.request

import numpy as np
import pandas as pd

ET = "America/New_York"

POST_URLS = [
    "https://ix.cnn.io/data/truth-social/truth_archive.json",
    "https://stilesdata.com/trump-truth-social-archive/truth_archive.json",  # stale after Oct 2025
]

# --------------------------------------------------------------------------
# Asset universe. Grouped so later analyses can pick what they need.
# --------------------------------------------------------------------------
BROAD = ["SPY", "QQQ", "IWM", "DIA", "^VIX", "TLT", "GLD", "USO", "UUP", "BTC-USD", "DJT"]
SECTORS = ["XLK", "XLF", "XLE", "XLI", "XLV", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC"]
COUNTRY_ETF = {  # country entity -> (equity ETF, FX ticker quoted as local currency per USD where possible)
    "China": ("FXI", "CNY=X"),
    "Mexico": ("EWW", "MXN=X"),
    "Canada": ("EWC", "CAD=X"),
    "European Union": ("EZU", "EURUSD=X"),
    "Japan": ("EWJ", "JPY=X"),
    "India": ("INDA", "INR=X"),
    "Brazil": ("EWZ", "BRL=X"),
    "South Korea": ("EWY", "KRW=X"),
    "Taiwan": ("EWT", "TWD=X"),
    "Vietnam": ("VNM", None),
    "Germany": ("EWG", None),
    "United Kingdom": ("EWU", "GBPUSD=X"),
    "Switzerland": ("EWL", "CHF=X"),
}
COMPANY_TICKER = {
    "Apple": "AAPL", "Tesla": "TSLA", "Amazon": "AMZN", "Intel": "INTC", "Nvidia": "NVDA",
    "Boeing": "BA", "Walmart": "WMT", "Ford": "F", "General Motors": "GM", "Meta": "META",
    "Google": "GOOGL", "Microsoft": "MSFT", "Disney": "DIS", "Comcast": "CMCSA",
    "Lockheed Martin": "LMT", "Mattel": "MAT", "Harley-Davidson": "HOG", "US Steel": "X",
    "Coca-Cola": "KO", "Pfizer": "PFE", "Exxon": "XOM", "Paramount": "PARA",
}
FRED_SERIES = {"VIXCLS": "VIX close (CBOE)", "USEPUINDXD": "Economic Policy Uncertainty (daily)",
               "DGS10": "10y Treasury yield", "DTWEXBGS": "Broad USD index"}


def all_daily_tickers() -> list[str]:
    t = BROAD + SECTORS + [v[0] for v in COUNTRY_ETF.values()]
    t += [v[1] for v in COUNTRY_ETF.values() if v[1]]
    t += list(COMPANY_TICKER.values())
    return list(dict.fromkeys(t))


# --------------------------------------------------------------------------
# Download helpers
# --------------------------------------------------------------------------
def _http_get(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research; university project)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def download_posts(data_dir: str = "data", refresh: bool = False) -> str:
    """Download the raw archive (JSON) once and cache it. Returns local path."""
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "truth_archive.json")
    if os.path.exists(path) and not refresh:
        return path
    last_err = None
    for url in POST_URLS:
        try:
            raw = _http_get(url)
            json.loads(raw)  # validate
            with open(path, "wb") as f:
                f.write(raw)
            print(f"downloaded {len(raw)/1e6:.1f} MB from {url}")
            return path
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"  failed {url}: {e}")
    raise RuntimeError(f"Could not download the archive ({last_err}). Download it manually "
                       f"from {POST_URLS[0]} and save it as {path}.")


def download_prices(tickers, data_dir="data", start="2022-01-01", end=None, interval="1d",
                    refresh=False, fname=None) -> pd.DataFrame:
    """Adjusted close prices from Yahoo Finance (cached as CSV).

    interval='1d' uses [start, end]; interval='60m' is limited by Yahoo to the last 730 days.
    """
    os.makedirs(data_dir, exist_ok=True)
    fname = fname or f"prices_{interval}.csv"
    path = os.path.join(data_dir, fname)
    if os.path.exists(path) and not refresh:
        return read_prices(path)
    import yfinance as yf  # imported lazily so demo mode does not need it

    frames = []
    chunks = [tickers[i:i + 15] for i in range(0, len(tickers), 15)]
    for ch in chunks:
        for attempt in range(3):
            try:
                kw = dict(interval=interval, auto_adjust=True, progress=False, threads=True)
                if interval == "1d":
                    df = yf.download(ch, start=start, end=end, **kw)
                else:
                    df = yf.download(ch, period="729d", **kw)
                break
            except Exception as e:  # noqa: BLE001
                print("  yfinance error, retrying:", e)
                time.sleep(5 * (attempt + 1))
        else:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            close = df["Close"]
        else:
            close = df[["Close"]].rename(columns={"Close": ch[0]})
        frames.append(close)
        time.sleep(1)
    px = pd.concat(frames, axis=1)
    px = px.loc[:, ~px.columns.duplicated()].dropna(how="all", axis=1)
    px.index.name = "time"
    px.to_csv(path)
    missing = sorted(set(tickers) - set(px.columns))
    if missing:
        print("  no data for:", missing)
    return read_prices(path)


def read_prices(path: str) -> pd.DataFrame:
    px = pd.read_csv(path, index_col=0)
    idx = pd.to_datetime(px.index, utc=True)
    # daily bars: keep naive calendar dates; intraday: convert to US/Eastern
    if "1d" in os.path.basename(path):
        px.index = pd.DatetimeIndex(idx.tz_convert(None).normalize())
    else:
        px.index = idx.tz_convert(ET)
    return px.sort_index().apply(pd.to_numeric, errors="coerce")


def download_fred(series=tuple(FRED_SERIES), data_dir="data", refresh=False) -> pd.DataFrame:
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "fred.csv")
    if os.path.exists(path) and not refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    out = {}
    for s in series:
        try:
            raw = _http_get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}")
            df = pd.read_csv(io.BytesIO(raw))
            ser = pd.to_numeric(df.iloc[:, 1], errors="coerce")
            ser.index = pd.DatetimeIndex(pd.to_datetime(df.iloc[:, 0]))
            out[s] = ser
        except Exception as e:  # noqa: BLE001
            print(f"  FRED {s} failed: {e}")
    fred = pd.DataFrame(out)
    fred.index.name = "date"
    fred.to_csv(path)
    return fred


def load_approval(data_dir="data") -> pd.DataFrame | None:
    """Optional: data/approval.csv with columns date, approve, disapprove (any poll average).

    Good sources: Silver Bulletin, RealClearPolling, NYT, or the Economist poll averages
    (export/copy the daily series). Returns None if the file is absent.
    """
    path = os.path.join(data_dir, "approval.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    df["net"] = df["approve"] - df["disapprove"]
    return df


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------
_TAG = re.compile(r"<[^>]+>")
_MENTION_HTML = re.compile(r'class="u-url mention">@<span>([A-Za-z0-9_]+)</span>')
_MENTION_TXT = re.compile(r"(?<![\w/])@([A-Za-z0-9_]{2,30})")
_TS_STATUS = re.compile(r"https?://truthsocial\.com/(?:users/\w+/statuses|@\w+)/\d+")  # glued self-repost links
_URL = re.compile(r"https?://[^\s\"“”]+")
_RT = re.compile(r"^RT[:\s]*@?")
_QUOTE_START = re.compile(r"^\s*[“\"]")


def _clean_handle_contexts(texts):
    """Handles seen in unambiguous contexts: mentions inside a post (not the glued 'RT @handleText' prefix),
    followed by a space, punctuation or end of text; plus reposts that consist of the handle only."""
    rx = re.compile(r"(?<![\w/])@([A-Za-z0-9_]{2,30})(?=$|[\s.,:;!?)\]\"'’”…-])")
    from collections import Counter
    c = Counter()
    for t in texts:
        if t.startswith("RT"):
            body = _RT.sub("", t, count=1)
            if re.fullmatch(r"[A-Za-z0-9_]{2,30}", body.strip()):
                c[body.strip().lower()] += 1
            t = re.sub(r"^RT[:\s]*@?[A-Za-z0-9_]+", "", t)  # drop the ambiguous glued prefix
        c.update(h.lower() for h in rx.findall(t))
    return c


def parse_repost(text, known):
    """In the archive a ReTruth looks like 'RT @handleTEXT...' (handle glued to the text) or
    'RT: https://truthsocial.com/users/X/statuses/ID TEXT'. Returns (reposted_handle, reposted_text)."""
    if not text.startswith("RT"):
        return None, None
    m = _TS_STATUS.match(_RT.sub("", text, count=1))
    if m:  # 'RT: https://truthsocial.com/users/realDonaldTrump/statuses/...'
        user = re.search(r"/(?:users/|@)(\w+)/", m.group(0))
        return (user.group(1).lower() if user else None), text[len(_RT.match(text).group(0)) + m.end():].strip()
    body = _RT.sub("", text, count=1)
    tok = re.match(r"[A-Za-z0-9_]{1,40}", body)
    if not tok:
        return None, body.strip()
    tok = tok.group(0)
    for k in range(min(len(tok), 30), 1, -1):  # longest known handle that is a prefix of the glued token
        if tok[:k].lower() in known:
            return tok[:k].lower(), body[k:].strip()
    m = re.match(r"(.*?[a-z0-9_])([A-Z]{2,}.*)$", tok)  # 'pjcolbeckFACT' -> 'pjcolbeck' + 'FACT...'
    if m and len(m.group(1)) >= 3:
        return m.group(1).lower(), body[len(m.group(1)):].strip()
    return tok.lower(), body[len(tok):].strip()  # unknown: assume the whole token is the handle


_MOJIBAKE = re.compile("[\u00c2-\u00f4][\u0080-\u00bf]{1,3}")


def fix_mojibake(s: str) -> str:
    """Repair UTF-8 text that was decoded as Latin-1 somewhere upstream ('Worldâ\x80\x99s' -> 'World’s').
    Only byte sequences that form valid UTF-8 are replaced, so correct text is left untouched."""
    def rep(m):
        try:
            return m.group(0).encode("latin-1").decode("utf-8")
        except UnicodeError:
            return m.group(0)
    return _MOJIBAKE.sub(rep, s) if ("\u00e2" in s or "\u00c3" in s or "\u00c2" in s or "\u00f0" in s) else s


def html_to_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = re.sub(r"<br\s*/?>|</p>", "\n", s)
    s = _TAG.sub("", s)
    return fix_mojibake(html.unescape(s)).replace("\u00a0", " ").strip()


def load_posts(data_dir="data", refresh=False, start=None, end=None) -> pd.DataFrame:
    """Load + clean the archive. One row per post, sorted by time.

    Added columns: t_utc, t_et, text, text_nourl, mentions, is_repost, repost_of, is_self_repost,
    is_quote, media_only, n_words, caps_ratio, n_excl, hour_et, weekday.
    """
    path = download_posts(data_dir, refresh)
    with open(path, "rb") as f:
        raw = json.load(f)
    df = pd.DataFrame(raw)
    df = df.drop_duplicates("id")
    df["t_utc"] = pd.to_datetime(df["created_at"], utc=True, format="ISO8601")
    df = df.sort_values("t_utc").reset_index(drop=True)
    df["t_et"] = df["t_utc"].dt.tz_convert(ET)
    df["content"] = df["content"].fillna("")
    df["text"] = df["content"].map(html_to_text)
    # ReTruths start with "RT" ("RT @user..." with the handle glued to the text, or "RT: <truthsocial link>")
    df["is_repost"] = df["text"].str.match(r"^RT[\s:@]")
    known = _clean_handle_contexts(df["text"])
    rp = [parse_repost(t, known) if r else (None, None) for t, r in zip(df["text"], df["is_repost"])]
    df["repost_of"] = [h for h, _ in rp]
    df["repost_text"] = [x for _, x in rp]
    df["is_self_repost"] = df["repost_of"].eq("realdonaldtrump")
    # for reposts, analyse the reposted text without the 'RT @handle' prefix
    base = [x if r and x is not None else t for t, r, x in zip(df["text"], df["is_repost"], df["repost_text"])]
    df["text_nourl"] = [_URL.sub(" ", _TS_STATUS.sub(" ", b)).strip() for b in base]
    df["text_nourl"] = df["text_nourl"].str.replace(r"\s{2,}", " ", regex=True)
    df["mentions"] = [sorted(set(_MENTION_HTML.findall(c)) | {h for h in _MENTION_TXT.findall(t) if h.lower() in known})
                      for c, t in zip(df["content"], df["text_nourl"])]
    df["media_only"] = df["text_nourl"].str.len() == 0
    # posts that are mostly a quotation of someone else (headlines, TV quotes): not his own voice
    qlen = df["text_nourl"].str.extract(r"^\s*[“\"]([^”\"]*)[”\"]?", expand=False).fillna("").str.len()
    df["is_quote"] = df["text_nourl"].str.match(_QUOTE_START.pattern) & (qlen >= 0.6 * df["text_nourl"].str.len())
    for c in ["replies_count", "reblogs_count", "favourites_count"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    words = df["text_nourl"].str.findall(r"[A-Za-z']+")
    df["n_words"] = words.map(len)
    letters = df["text_nourl"].str.count(r"[A-Za-z]")
    df["caps_ratio"] = (df["text_nourl"].str.count(r"[A-Z]") / letters.replace(0, np.nan)).fillna(0)
    df["n_excl"] = df["text_nourl"].str.count("!")
    df["hour_et"] = df["t_et"].dt.hour
    df["weekday"] = df["t_et"].dt.dayofweek
    if start is not None:
        df = df[df["t_utc"] >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        df = df[df["t_utc"] <= pd.Timestamp(end, tz="UTC")]
    return df.reset_index(drop=True)


def trading_day_of(t_et: pd.Series, trading_days: pd.DatetimeIndex, close_hour: int = 16) -> pd.Series:
    """Map each post to the first trading-day *close* after it was published.

    A daily close-to-close return on day d spans close(d-1) -> close(d), so a post
    at 18:00 on Monday is first priced in Tuesday's return; a Saturday post in
    Monday's return. Returns naive dates (NaT if beyond the last trading day).
    """
    td = pd.DatetimeIndex(sorted(trading_days))
    naive = t_et.dt.tz_localize(None)
    close_times = td + pd.Timedelta(hours=close_hour)
    pos = np.searchsorted(close_times.values, naive.values, side="left")
    out = pd.Series(pd.NaT, index=t_et.index, dtype="datetime64[ns]")
    ok = pos < len(td)
    out[ok] = td.values[pos[ok]]
    return out
