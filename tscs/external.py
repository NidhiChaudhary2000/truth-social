"""External real-world response data for Part F (all free, no API keys). Every download is cached.

* Wikipedia pageviews (Wikimedia REST API): public attention to issues, and the attention that
  different *language communities* pay to the "Donald Trump" article (a proxy for foreign publics).
* GDELT DOC 2.0 API: daily average tone of each country's national news coverage that mentions
  Trump (a proxy for how relations are framed abroad). GDELT rate-limits hard (about 1 request per
  5 s), so this step is slow the first time and can be skipped (USE_GDELT = False).
* Political series: any CSVs you drop in data/political/ (columns: date,value), e.g. approval,
  generic ballot, or prediction-market probabilities. polymarket_history() tries to fetch these
  from Polymarket's public API (experimental: endpoints are not officially documented as stable).
"""
from __future__ import annotations

import glob
import json
import os
import re
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

# Wikimedia rate-limits by client identity (2026 policy): a User-Agent WITH contact info (email or URL) gets
# ~200 requests/min, an anonymous one only ~10/min. Set CONTACT (e.g. in the notebook config) to your email.
CONTACT = None


def user_agent():
    c = f"; {CONTACT}" if CONTACT else ""
    return f"TruthSocialComplexity/1.1 (university complex-systems project{c}) python-urllib"


_WIKI = {"last": 0.0}


def _wiki_interval():
    return 0.4 if CONTACT else 6.5   # stay under 200/min (identified) or 10/min (anonymous)

# language edition -> country entities whose mention counts as a "treatment" for that language community
LANG_COUNTRIES = {
    "de": ["Germany"], "fr": ["France"], "es": ["Mexico", "Venezuela", "Colombia", "Argentina", "Cuba"],
    "zh": ["China", "Taiwan"], "ja": ["Japan"], "ru": ["Russia"], "uk": ["Ukraine"], "he": ["Israel"],
    "fa": ["Iran"], "hi": ["India"], "pt": ["Brazil"], "ko": ["South Korea", "North Korea"],
    "da": ["Denmark", "Greenland"], "pl": ["Poland"], "tr": ["Turkey"], "hu": ["Hungary"], "vi": ["Vietnam"],
    "ar": ["Saudi Arabia", "Qatar", "Syria", "Gaza"],
    "it": [], "nl": [], "sv": [], "id": [], "fi": [], "cs": [],  # control languages (no target country)
}

# English Wikipedia issue articles -> entities/topics whose posts are the "treatment"
ISSUE_ARTICLES = {
    "Greenland": ("entity", ["Greenland", "Denmark"]),
    "Panama_Canal": ("entity", ["Panama"]),
    "Tariff": ("topic", "tariffs_trade"),
    "Gaza_Strip": ("entity", ["Gaza"]),
    "Russian_invasion_of_Ukraine": ("entity", ["Ukraine", "Volodymyr Zelensky"]),
    "Iran": ("entity", ["Iran", "Ali Khamenei"]),
    "Venezuela": ("entity", ["Venezuela", "Nicolas Maduro"]),
    "Federal_Reserve": ("entity", ["Federal Reserve", "Jerome Powell"]),
    "Jeffrey_Epstein": ("entity", ["Jeffrey Epstein"]),
    "Elon_Musk": ("entity", ["Elon Musk", "@elonmusk"]),
    "NATO": ("entity", ["NATO"]),
    "U.S._Immigration_and_Customs_Enforcement": ("topic", "immigration"),
    "Vladimir_Putin": ("entity", ["Vladimir Putin"]),
    "Volodymyr_Zelenskyy": ("entity", ["Volodymyr Zelensky"]),
    "Benjamin_Netanyahu": ("entity", ["Benjamin Netanyahu"]),
    "Xi_Jinping": ("entity", ["Xi Jinping"]),
}

# country entity -> GDELT sourcecountry name (no spaces)
GDELT_COUNTRIES = {
    "China": "china", "Mexico": "mexico", "Canada": "canada", "India": "india", "Japan": "japan",
    "Russia": "russia", "Ukraine": "ukraine", "Israel": "israel", "Iran": "iran", "Germany": "germany",
    "France": "france", "United Kingdom": "unitedkingdom", "Brazil": "brazil", "South Korea": "southkorea",
    "Denmark": "denmark", "Venezuela": "venezuela", "Turkey": "turkey", "Saudi Arabia": "saudiarabia",
    "Australia": "australia", "South Africa": "southafrica",
}


def _get(url, timeout=60, retries=4, pause=2.0, json_out=True):
    """GET with retries. On HTTP 429 it waits for the server's Retry-After (default 60 s)."""
    last = None
    for a in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": user_agent()})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read()
            return json.loads(raw) if json_out else raw
        except urllib.error.HTTPError as e:
            last = e
            if e.code == 404:
                return None
            if e.code == 429:
                wait = float(e.headers.get("Retry-After") or 60) + 1
                print(f"    rate-limited by {urllib.parse.urlparse(url).netloc}; waiting {wait:.0f}s")
                time.sleep(wait)
            else:
                time.sleep(pause * (2 ** a))
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(pause * (2 ** a))
    raise RuntimeError(f"GET failed after {retries} tries: {url} ({last})")


def _safe(name):
    """File-system-safe, collision-free name (non-Latin titles such as 'Дональд Трамп' stay distinct)."""
    import hashlib
    ascii_part = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)[:80]
    if re.fullmatch(r"[A-Za-z0-9_.-]+", name) and len(name) <= 80:
        return ascii_part
    return f"{ascii_part}_{hashlib.md5(name.encode('utf8')).hexdigest()[:8]}"


# --------------------------------------------------------------------------
# Wikipedia
# --------------------------------------------------------------------------
def wiki_pageviews(article, project="en.wikipedia", start="2022-01-01", end=None, data_dir="data",
                   refresh=False) -> pd.Series:
    d = os.path.join(data_dir, "wiki")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{_safe(project)}__{_safe(article)}.csv")
    if os.path.exists(path) and not refresh:
        s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
        s.index = pd.DatetimeIndex(s.index)
        return s
    end = end or (pd.Timestamp.today() - pd.Timedelta("2D")).strftime("%Y-%m-%d")
    wait = _WIKI["last"] + _wiki_interval() - time.time()
    if wait > 0:
        time.sleep(wait)
    _WIKI["last"] = time.time()
    a = urllib.parse.quote(article.replace(" ", "_"), safe="")
    url = (f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{project}/all-access/user/"
           f"{a}/daily/{pd.Timestamp(start):%Y%m%d}00/{pd.Timestamp(end):%Y%m%d}00")
    js = _get(url)
    items = (js or {}).get("items", [])
    s = pd.Series({pd.to_datetime(it["timestamp"][:8]): it["views"] for it in items}, name=article, dtype=float)
    s.index.name = "date"
    s.to_csv(path)
    return s


def wiki_edits(article, project="en.wikipedia", start="2022-01-01", end=None, data_dir="data", refresh=False) -> pd.Series:
    """Daily number of edits to a Wikipedia article (all editor types) - collective editing activity."""
    d = os.path.join(data_dir, "wiki")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"edits__{_safe(project)}__{_safe(article)}.csv")
    if os.path.exists(path) and not refresh:
        s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
        s.index = pd.DatetimeIndex(s.index)
        return s
    end = end or (pd.Timestamp.today() - pd.Timedelta("2D")).strftime("%Y-%m-%d")
    wait = _WIKI["last"] + _wiki_interval() - time.time()
    if wait > 0:
        time.sleep(wait)
    _WIKI["last"] = time.time()
    a = urllib.parse.quote(article.replace(" ", "_"), safe="")
    url = (f"https://wikimedia.org/api/rest_v1/metrics/edits/per-page/{project}/{a}/all-editor-types/daily/"
           f"{pd.Timestamp(start):%Y%m%d}/{pd.Timestamp(end):%Y%m%d}")
    js = _get(url)
    res = [r for it in (js or {}).get("items", []) for r in it.get("results", [])]
    s = pd.Series({pd.to_datetime(r["timestamp"][:10]): r["edits"] for r in res}, name=article, dtype=float)
    idx = pd.date_range(start, end, freq="D")
    s = s.reindex(idx).fillna(0)  # days without edits are absent from the API response
    s.index.name = "date"
    s.to_csv(path)
    return s


def wiki_langlinks(title="Donald_Trump", data_dir="data", refresh=False) -> dict:
    d = os.path.join(data_dir, "wiki")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"langlinks__{_safe(title)}.json")
    if os.path.exists(path) and not refresh:
        return json.load(open(path))
    url = ("https://en.wikipedia.org/w/api.php?action=query&prop=langlinks&lllimit=500&format=json"
           f"&formatversion=2&titles={urllib.parse.quote(title)}")
    js = _get(url)
    links = {ll["lang"]: ll["title"] for ll in js["query"]["pages"][0].get("langlinks", [])}
    links["en"] = title.replace("_", " ")
    json.dump(links, open(path, "w"), ensure_ascii=False)
    return links


def foreign_trump_attention(langs=tuple(LANG_COUNTRIES), data_dir="data", start="2022-01-01") -> pd.DataFrame:
    """Daily views of the 'Donald Trump' article in each language edition (date x language)."""
    links = wiki_langlinks("Donald_Trump", data_dir)
    out = {}
    for lg in langs:
        if lg not in links:
            continue
        try:
            out[lg] = wiki_pageviews(links[lg], f"{lg}.wikipedia", start=start, data_dir=data_dir)
        except Exception as e:  # noqa: BLE001
            print(f"  {lg}: {e}")
    return pd.DataFrame(out).sort_index()


def issue_attention(articles=tuple(ISSUE_ARTICLES), data_dir="data", start="2022-01-01") -> pd.DataFrame:
    out = {}
    for a in articles:
        try:
            out[a] = wiki_pageviews(a, "en.wikipedia", start=start, data_dir=data_dir)
        except Exception as e:  # noqa: BLE001
            print(f"  {a}: {e}")
    return pd.DataFrame(out).sort_index()


# --------------------------------------------------------------------------
# GDELT
# --------------------------------------------------------------------------
_GDELT = {"last": 0.0}
GDELT_MIN_INTERVAL = 12.0   # seconds between requests (GDELT asks for >= 5 s; heavy queries need more)


class GdeltThrottled(Exception):
    pass


def _gdelt_request(url):
    """One polite request. Raises GdeltThrottled on HTTP 429 or GDELT's plain-text 'please limit' reply.
    No automatic retries: hammering a throttled endpoint only extends the block."""
    wait = _GDELT["last"] + GDELT_MIN_INTERVAL - time.time()
    if wait > 0:
        time.sleep(wait)
    _GDELT["last"] = time.time()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": user_agent()})
        with urllib.request.urlopen(req, timeout=120) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        if e.code == 429:
            raise GdeltThrottled(str(e)) from e
        raise
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        txt = raw[:300].decode("utf8", "ignore")
        if "limit" in txt.lower() or not txt.strip():
            raise GdeltThrottled(txt.strip() or "empty reply")
        raise RuntimeError(f"non-JSON reply: {txt}")


def _gdelt_parse(js):
    tl = (js or {}).get("timeline", [])
    if not tl:
        return pd.Series(dtype=float)
    data = tl[0].get("data", [])
    return pd.Series({pd.to_datetime(x["date"], format="%Y%m%dT%H%M%SZ"): x["value"] for x in data}, dtype=float)


def _read_cached(path):
    s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    s.index = pd.DatetimeIndex(s.index)
    return s


def _coverage(s, a, b):
    if not len(s):
        return 0.0
    return s[(s.index >= a) & (s.index <= b)].index.normalize().nunique() / max(1, (b - a).days + 1)


def gdelt_timeline(query, mode="timelinetone", start="2022-02-01", end=None, name=None, data_dir="data",
                   refresh=False, min_coverage=0.8) -> pd.Series:
    """Daily GDELT DOC 2.0 timeline for `query`, cached in data/gdelt/.

    1. Use the cached series if a previous download completed (partial caches are refetched).
    2. Otherwise ask for the full range in ONE request.
    3. Only if that reply is valid but truncated, fall back to yearly chunks, each cached separately so
       progress survives interruptions.
    Raises GdeltThrottled when GDELT rate-limits, so the caller can pause instead of hammering it."""
    d = os.path.join(data_dir, "gdelt")
    os.makedirs(os.path.join(d, "chunks"), exist_ok=True)
    key = _safe(name or query)
    path = os.path.join(d, f"{mode}__{key}.csv")
    a0 = pd.Timestamp(start)
    end = pd.Timestamp(end) if end else pd.Timestamp.today().normalize() - pd.Timedelta("1D")
    done = path + ".complete"  # written only after a successful full download (older partial caches lack it)
    if os.path.exists(path) and not refresh:
        cached = _read_cached(path)
        if os.path.exists(done):
            return cached
        if _coverage(cached, a0, min(end, cached.index.max())) >= 0.9:  # cache from an older version: accept if complete
            open(done, "w").write("accepted existing cache\n")
            return cached

    def url(a, b):
        return ("https://api.gdeltproject.org/api/v2/doc/doc?format=json&mode=" + mode +
                f"&startdatetime={a:%Y%m%d}000000&enddatetime={b:%Y%m%d}235959&query=" + urllib.parse.quote(query))

    chunked = path + ".chunked"  # remembers that the full-range request comes back truncated for this query
    s = pd.Series(dtype=float) if os.path.exists(chunked) else _gdelt_parse(_gdelt_request(url(a0, end)))
    if _coverage(s, a0, end) < min_coverage:  # valid but truncated reply -> yearly chunks
        open(chunked, "w").write("full-range reply was truncated\n")
        parts, a = [s], a0
        while a <= end:
            b = min(a + pd.DateOffset(years=1) - pd.Timedelta("1D"), end)
            cpath = os.path.join(d, "chunks", f"{mode}__{key}__{a:%Y%m%d}_{b:%Y%m%d}.csv")
            if os.path.exists(cpath):
                parts.append(_read_cached(cpath))
            else:
                c = _gdelt_parse(_gdelt_request(url(a, b)))
                if len(c):
                    c.rename("value").rename_axis("date").to_csv(cpath)
                parts.append(c)
            a = b + pd.Timedelta("1D")
        s = pd.concat([p for p in parts if len(p)]).sort_index() if any(len(p) for p in parts) else s
        s = s[~s.index.duplicated()]
    s = s.resample("D").mean() if len(s) else s
    s.name, s.index.name = (name or query), "date"
    if len(s):
        s.to_csv(path)
        with open(done, "w") as f:
            f.write(f"{a0:%Y-%m-%d} {end:%Y-%m-%d} coverage={_coverage(s, a0, end):.2f}\n")
    return s


def gdelt_country_tone(countries=None, data_dir="data", start="2022-02-01", download=True, cooldown=180,
                       max_throttles=2) -> pd.DataFrame:
    """Daily average tone of each country's national media coverage mentioning Trump (date x country).

    Already-downloaded countries come from the cache. When GDELT throttles, waits `cooldown` seconds and
    retries; after `max_throttles` consecutive throttles it stops and returns what it has. Run
    `python fetch_gdelt.py` in a terminal to download the rest slowly in the background."""
    countries = countries or list(GDELT_COUNTRIES)
    out, missing, strikes = {}, [], 0
    for i, c in enumerate(countries):
        code = GDELT_COUNTRIES[c]
        path = os.path.join(data_dir, "gdelt", f"timelinetone__tone_{code}.csv")
        if not download:
            if os.path.exists(path):
                out[c] = _read_cached(path)
            else:
                missing.append(c)
            continue
        while True:
            try:
                s = gdelt_timeline(f"trump sourcecountry:{code}", "timelinetone", start=start, name=f"tone_{code}",
                                   data_dir=data_dir)
                strikes = 0
                if len(s):
                    out[c] = s
                    print(f"  GDELT tone: {c} ok ({len(s)} days)")
                else:
                    print(f"  GDELT tone: {c} returned no data")
                break
            except GdeltThrottled as e:
                strikes += 1
                if strikes > max_throttles:
                    break
                print(f"  GDELT throttled at {c} ({str(e)[:80]}); waiting {cooldown}s ...")
                time.sleep(cooldown)
            except Exception as e:  # noqa: BLE001
                print(f"  GDELT tone: {c} failed: {e}")
                break
        if strikes > max_throttles:
            missing += [x for x in countries[i:] if x not in out]
            if os.path.exists(path) and c not in out:
                out[c] = _read_cached(path)  # fall back to an older/partial cache rather than nothing
            for x in countries[i + 1:]:
                px = os.path.join(data_dir, "gdelt", f"timelinetone__tone_{GDELT_COUNTRIES[x]}.csv")
                if os.path.exists(px):
                    out[x] = _read_cached(px)
            print(f"  GDELT keeps throttling; stopped. Missing: {missing}.\n"
                  "  -> Wait 30-60 min, then rerun, or run `python fetch_gdelt.py` in a terminal "
                  "(it downloads slowly and resumes from the cache).")
            break
    return pd.DataFrame(out).sort_index()


# --------------------------------------------------------------------------
# Political series (approval, ballot, prediction markets)
# --------------------------------------------------------------------------
def load_political_series(data_dir="data") -> dict:
    """Every CSV in data/political/ with columns (date, value) becomes a daily series named after the file."""
    out = {}
    for f in sorted(glob.glob(os.path.join(data_dir, "political", "*.csv"))):
        df = pd.read_csv(f)
        dates = pd.to_datetime(df.iloc[:, 0], utc=True).dt.tz_convert(None).dt.normalize()
        s = pd.Series(pd.to_numeric(df.iloc[:, 1], errors="coerce").values, index=pd.DatetimeIndex(dates))
        s = s.groupby(level=0).last().sort_index()
        out[os.path.splitext(os.path.basename(f))[0]] = s
    return out


def polymarket_history(query, name, outcome="Yes", data_dir="data", refresh=False, keyword=None) -> pd.Series | None:
    """EXPERIMENTAL. Search Polymarket for `query`, take the highest-volume event (optionally the market whose
    question contains `keyword`), and save the daily probability of `outcome` to data/political/<name>.csv.
    Prints the market question so you can check it is the right one."""
    d = os.path.join(data_dir, "political")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{_safe(name)}.csv")
    if os.path.exists(path) and not refresh:
        return load_political_series(data_dir).get(_safe(name))
    try:
        js = _get("https://gamma-api.polymarket.com/public-search?q=" + urllib.parse.quote(query))
        events = sorted(js.get("events", []), key=lambda e: -float(e.get("volume") or 0))
        markets = [m for e in events[:3] for m in e.get("markets", [])]
        if keyword:
            markets = [m for m in markets if keyword.lower() in (m.get("question") or "").lower()] or markets
        markets = sorted(markets, key=lambda m: -float(m.get("volume") or m.get("volumeNum") or 0))
        if not markets:
            print(f"  Polymarket: nothing found for '{query}'")
            return None
        m = markets[0]
        outs = json.loads(m["outcomes"]) if isinstance(m["outcomes"], str) else m["outcomes"]
        toks = json.loads(m["clobTokenIds"]) if isinstance(m["clobTokenIds"], str) else m["clobTokenIds"]
        tok = toks[outs.index(outcome)] if outcome in outs else toks[0]
        print(f"  Polymarket '{name}': {m.get('question')}  [{outcome}]")
        hist = []
        for params in ["interval=max&fidelity=1440", "interval=all&fidelity=1440", "interval=max&fidelity=720"]:
            h = _get(f"https://clob.polymarket.com/prices-history?market={tok}&{params}")
            hist = (h or {}).get("history", [])
            if hist:
                break
        if not hist:
            print("  Polymarket: empty price history")
            return None
        s = pd.Series({pd.to_datetime(int(x["t"]), unit="s").normalize(): float(x["p"]) for x in hist})
        s = s.groupby(level=0).last()
        pd.DataFrame({"date": s.index, "value": s.values}).to_csv(path, index=False)
        return s
    except Exception as e:  # noqa: BLE001
        print(f"  Polymarket fetch failed for '{query}': {e}. You can save a CSV (date,value) to {path} by hand.")
        return None


def polymarket_event(slug, question_contains, name, outcome="Yes", data_dir="data", refresh=False,
                     fidelity_min=1440) -> pd.Series | None:
    """Daily probability of `outcome` for the market inside Polymarket event `slug` whose question contains
    `question_contains` (e.g. slug='which-party-will-win-the-house-in-2026', 'Democratic'). The event slug is
    the last part of the market's URL on polymarket.com. Cached as data/political/<name>.csv."""
    d = os.path.join(data_dir, "political")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{_safe(name)}.csv")
    if os.path.exists(path) and not refresh:
        return load_political_series(data_dir).get(_safe(name))
    try:
        ev = _get("https://gamma-api.polymarket.com/events?slug=" + urllib.parse.quote(slug))
        markets = [m for e in (ev or []) for m in e.get("markets", [])
                   if question_contains.lower() in (m.get("question") or "").lower()]
        if not markets:
            print(f"  Polymarket: no market containing '{question_contains}' in event '{slug}'")
            return None
        m = max(markets, key=lambda m: float(m.get("volume") or m.get("volumeNum") or 0))
        outs = json.loads(m["outcomes"]) if isinstance(m["outcomes"], str) else m["outcomes"]
        toks = json.loads(m["clobTokenIds"]) if isinstance(m["clobTokenIds"], str) else m["clobTokenIds"]
        tok = toks[outs.index(outcome)] if outcome in outs else toks[0]
        h = _get(f"https://clob.polymarket.com/prices-history?market={tok}&interval=max&fidelity={fidelity_min}")
        hist = (h or {}).get("history", [])
        if not hist:
            print(f"  Polymarket: empty history for '{m.get('question')}'")
            return None
        s = pd.Series({pd.to_datetime(int(x["t"]), unit="s").normalize(): float(x["p"]) for x in hist})
        s = s.groupby(level=0).last()
        pd.DataFrame({"date": s.index, "value": s.values}).to_csv(path, index=False)
        print(f"  Polymarket '{name}': {m.get('question')} [{outcome}] {s.index.min():%Y-%m-%d}..{s.index.max():%Y-%m-%d}")
        return s
    except Exception as e:  # noqa: BLE001
        print(f"  Polymarket fetch failed for '{slug}': {e}")
        return None
