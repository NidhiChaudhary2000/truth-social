"""Bilateral US <-> country relations from the GDELT Event Database (bulk files, no API, no throttling).

Why not the DOC API? It is rate-limited per IP, and an 'artlist' query returns at most 250 articles
with no tone field, so a daily series would need *more* requests. The Event Database instead comes as
one static zip per day (https://data.gdeltproject.org/events/YYYYMMDD.export.CSV.zip, about 4-8 MB).
We stream each day, keep only events between the USA and the countries of interest, aggregate them, and
discard the raw file. About 1,700 days for 2022-2026 is roughly 10-12 GB of downloads, but nothing is
stored except a few KB per day, and the download resumes where it stopped.

Each event (CAMEO-coded) has:
  GoldsteinScale  -10 (most conflictual, e.g. military attack) ... +10 (most cooperative)
  QuadClass       1 verbal cooperation, 2 material cooperation, 3 verbal conflict, 4 material conflict
  AvgTone         tone of the articles reporting it
  NumMentions     how often it was mentioned (used as weight)
Directions: 'US2C' = the USA acts towards country C (Actor1=USA, Actor2=C); 'C2US' = C acts towards the USA.

Alternative (one query, if you have a Google account): run bigquery_sql() in the BigQuery console
(free sandbox tier) and save the result as data/gdelt_events_daily.csv. load_events() prefers that file.
"""
from __future__ import annotations

import io
import os
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

# CAMEO actor country codes (ISO-3 style) for the countries used in Part F
CAMEO = {
    "China": "CHN", "Mexico": "MEX", "Canada": "CAN", "India": "IND", "Japan": "JPN", "Russia": "RUS",
    "Ukraine": "UKR", "Israel": "ISR", "Iran": "IRN", "Germany": "DEU", "France": "FRA", "United Kingdom": "GBR",
    "Brazil": "BRA", "South Korea": "KOR", "North Korea": "PRK", "Denmark": "DNK", "Greenland": "GRL",
    "Venezuela": "VEN", "Turkey": "TUR", "Saudi Arabia": "SAU", "Australia": "AUS", "South Africa": "ZAF",
    "Panama": "PAN", "Qatar": "QAT", "Syria": "SYR", "Argentina": "ARG", "Colombia": "COL", "Hungary": "HUN",
    "Poland": "POL", "Cuba": "CUB", "Afghanistan": "AFG", "Taiwan": "TWN", "Vietnam": "VNM",
    "Switzerland": "CHE", "Gaza": "PSE",
}
_CODE2C = {v: k for k, v in CAMEO.items()}
# 0-based column positions; identical in GDELT 1.0 daily (58 cols) and 2.0 (61 cols) exports
_COLS = {7: "a1", 17: "a2", 28: "root", 29: "quad", 30: "gold", 31: "ment", 34: "tone"}
# CAMEO root-code groups -> extra outcome shares (mention-weighted)
ROOT_GROUPS = {"threaten_share": [13], "coerce_share": [16, 17], "consult_share": [3, 4, 5]}
URL = "https://data.gdeltproject.org/events/{:%Y%m%d}.export.CSV.zip"
UA = "TruthSocialComplexity/1.0 (university project)"


def _aggregate(df, day):
    df = df[((df.a1 == "USA") & df.a2.isin(_CODE2C)) | ((df.a2 == "USA") & df.a1.isin(_CODE2C))].copy()
    if not len(df):
        return pd.DataFrame(columns=["day", "country", "direction", "n_events", "mentions", "goldstein", "tone",
                                     "conflict_share", "material_conflict_share", *ROOT_GROUPS])
    for c in ["root", "quad", "gold", "ment", "tone"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["quad", "gold", "ment", "tone"])
    df = df[df.gold.between(-10, 10) & df.quad.isin([1, 2, 3, 4])]  # schema sanity check
    df["direction"] = np.where(df.a1 == "USA", "US2C", "C2US")
    df["country"] = np.where(df.a1 == "USA", df.a2, df.a1)
    df["country"] = df["country"].map(_CODE2C)
    w = df.ment.clip(lower=1)
    df = df.assign(w=w, wg=w * df.gold, wt=w * df.tone, wc=w * df.quad.isin([3, 4]), wm=w * (df.quad == 4),
                   **{f"w_{k}": w * df.root.isin(v) for k, v in ROOT_GROUPS.items()})
    g = df.groupby(["country", "direction"])
    out = pd.DataFrame({"n_events": g.size(), "mentions": g.w.sum(), "goldstein": g.wg.sum() / g.w.sum(),
                        "tone": g.wt.sum() / g.w.sum(), "conflict_share": g.wc.sum() / g.w.sum(),
                        "material_conflict_share": g.wm.sum() / g.w.sum(),
                        **{k: g[f"w_{k}"].sum() / g.w.sum() for k in ROOT_GROUPS}}).reset_index()
    out.insert(0, "day", f"{day:%Y-%m-%d}")
    return out


def _process_day(day, out_dir, retries=3):
    path = os.path.join(out_dir, f"{day:%Y%m%d}.csv")
    if os.path.exists(path) or os.path.exists(path + ".missing"):
        return "cached"
    for a in range(retries):
        try:
            req = urllib.request.Request(URL.format(day), headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=180) as r:
                raw = r.read()
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                open(path + ".missing", "w").write("404\n")
                return "missing"
            time.sleep(5 * (a + 1))
        except Exception:  # noqa: BLE001
            time.sleep(5 * (a + 1))
    else:
        return "failed"
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        with z.open(z.namelist()[0]) as f:
            df = pd.read_csv(f, sep="\t", header=None, usecols=list(_COLS), dtype=str, quoting=3,
                             on_bad_lines="skip", engine="c")
    df = df.rename(columns=_COLS)
    agg = _aggregate(df, day)
    agg.to_csv(path + ".tmp", index=False)
    os.replace(path + ".tmp", path)  # atomic: an interrupted day is redone next time
    return "ok"


def download_events(start="2022-02-14", end=None, data_dir="data", workers=4, verbose=True) -> dict:
    """Download + aggregate every day in [start, end]. Resumable; safe to interrupt."""
    out_dir = os.path.join(data_dir, "gdelt_events", "daily")
    os.makedirs(out_dir, exist_ok=True)
    end = pd.Timestamp(end) if end else pd.Timestamp.today().normalize() - pd.Timedelta("1D")
    days = list(pd.date_range(start, end, freq="D"))
    todo = [d for d in days if not (os.path.exists(os.path.join(out_dir, f"{d:%Y%m%d}.csv")) or
                                    os.path.exists(os.path.join(out_dir, f"{d:%Y%m%d}.csv.missing")))]
    if verbose:
        print(f"GDELT events: {len(days)} days in range, {len(days) - len(todo)} already done, {len(todo)} to fetch "
              f"(~{7 * len(todo) / 1000:.1f} GB to stream)")
    status, t0 = {"ok": 0, "missing": 0, "failed": 0, "cached": len(days) - len(todo)}, time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_process_day, d, out_dir): d for d in todo}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                status[f.result()] += 1
            except Exception as e:  # noqa: BLE001
                status["failed"] += 1
                print(f"  {futs[f]:%Y-%m-%d}: {e}")
            if verbose and (i % 25 == 0 or i == len(todo)):
                rate = i / (time.time() - t0)
                print(f"  {i}/{len(todo)} days  ({rate * 60:.0f} days/min, ~{(len(todo) - i) / max(rate, 1e-9) / 60:.0f} min left)  {status}")
    return status


def load_events(data_dir="data") -> pd.DataFrame:
    """Long table (day, country, direction, n_events, mentions, goldstein, tone, conflict_share, ...).
    Uses data/gdelt_events_daily.csv (BigQuery export or demo data) if present, else the per-day files."""
    import glob
    bq = sorted(glob.glob(os.path.join(data_dir, "gdelt_events_daily*.csv")))  # one file, or shards of an export
    if bq:
        df = pd.concat([pd.read_csv(f) for f in bq], ignore_index=True)
        df.columns = [c.strip().lower() if c.strip().lower() in ("day", "country", "direction") else c.strip()
                      for c in df.columns]
        df["country"] = df["country"].map(lambda v: _CODE2C.get(v, v))  # BigQuery export uses CAMEO codes
        df = df.drop_duplicates(["day", "country", "direction"])
    else:
        files = sorted(f for f in os.listdir(os.path.join(data_dir, "gdelt_events", "daily")) if f.endswith(".csv")) \
            if os.path.isdir(os.path.join(data_dir, "gdelt_events", "daily")) else []
        if not files:
            return pd.DataFrame()
        d = os.path.join(data_dir, "gdelt_events", "daily")
        cons = os.path.join(data_dir, "gdelt_events", f"consolidated_{len(files)}_{files[0][:8]}_{files[-1][:8]}.csv")
        if os.path.exists(cons):
            df = pd.read_csv(cons)
        else:
            df = pd.concat([pd.read_csv(os.path.join(d, f)) for f in files], ignore_index=True)
            for old_c in [f for f in os.listdir(os.path.dirname(cons)) if f.startswith("consolidated_")]:
                os.remove(os.path.join(os.path.dirname(cons), old_c))
            df.to_csv(cons, index=False)
    df["day"] = pd.to_datetime(df["day"].astype(str), format="mixed").dt.normalize()
    return df.dropna(subset=["country"])


def events_panel(ev: pd.DataFrame, metric="goldstein", direction="C2US", min_days=0.5) -> pd.DataFrame:
    """date x country panel of a metric for one direction; keeps countries observed on >= min_days share of days."""
    sub = ev[ev.direction == direction]
    P = sub.pivot_table(index="day", columns="country", values=metric, aggfunc="mean").asfreq("D")
    keep = P.notna().mean() >= min_days
    return P.loc[:, keep]


def bigquery_sql(start="2022-02-14", end=None, partitioned=True) -> str:
    """SQL for the BigQuery console (public GDELT 2.0 dataset). Save the result as data/gdelt_events_daily.csv.

    partitioned=True reads `gdelt-bq.gdeltv2.events_partitioned` and prunes partitions by date, so it scans
    only the 2022-2026 slice (check the 'This query will process ...' estimate before running; the free
    sandbox allows 1 TB per month). If that table is unavailable, use partitioned=False (full-table scan
    of only the 9 columns used)."""
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    codes = ", ".join(f"'{c}'" for c in CAMEO.values())
    roots = ",\n".join(f"  ROUND(SUM(IF(SAFE_CAST(EventRootCode AS INT64) IN ({', '.join(map(str, v))}), NumMentions, 0))"
                       f" / SUM(NumMentions), 4) AS {k}" for k, v in ROOT_GROUPS.items())
    table = "gdelt-bq.gdeltv2.events_partitioned" if partitioned else "gdelt-bq.gdeltv2.events"
    when = (f"_PARTITIONTIME BETWEEN TIMESTAMP('{start}') AND TIMESTAMP('{end}')" if partitioned else
            f"DATEADDED BETWEEN {pd.Timestamp(start):%Y%m%d}000000 AND {pd.Timestamp(end):%Y%m%d}235959")
    return f"""-- GDELT 2.0 events: daily US <-> country relations, {start} .. {end}
-- Save the result as data/gdelt_events_daily.csv (columns must keep these names).
SELECT
  FORMAT_DATE('%F', PARSE_DATE('%Y%m%d', SUBSTR(CAST(DATEADDED AS STRING), 1, 8))) AS day,  -- publication day (UTC)
  IF(Actor1CountryCode = 'USA', Actor2CountryCode, Actor1CountryCode) AS country,
  IF(Actor1CountryCode = 'USA', 'US2C', 'C2US') AS direction,
  COUNT(*) AS n_events,
  SUM(NumMentions) AS mentions,
  ROUND(SUM(GoldsteinScale * NumMentions) / SUM(NumMentions), 4) AS goldstein,
  ROUND(SUM(AvgTone * NumMentions) / SUM(NumMentions), 4) AS tone,
  ROUND(SUM(IF(QuadClass IN (3, 4), NumMentions, 0)) / SUM(NumMentions), 4) AS conflict_share,
  ROUND(SUM(IF(QuadClass = 4, NumMentions, 0)) / SUM(NumMentions), 4) AS material_conflict_share,
{roots}
FROM `{table}`
WHERE {when}
  AND ((Actor1CountryCode = 'USA' AND Actor2CountryCode IN ({codes}))
    OR (Actor2CountryCode = 'USA' AND Actor1CountryCode IN ({codes})))
  AND NumMentions > 0
GROUP BY day, country, direction
ORDER BY day, country, direction"""


def _bq_client(project):
    """BigQuery client. Uses existing Google credentials if present (gcloud ADC, Colab); otherwise opens a
    one-time browser login (pydata-google-auth) and caches the token in your user folder."""
    from google.cloud import bigquery  # type: ignore
    try:
        import google.colab  # type: ignore  # noqa: F401
        from google.colab import auth  # type: ignore
        auth.authenticate_user()
        return bigquery.Client(project=project)
    except ImportError:
        pass
    try:
        import google.auth  # type: ignore
        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/bigquery"])
        return bigquery.Client(project=project, credentials=creds)
    except Exception:  # noqa: BLE001  (no default credentials) -> interactive login
        import pydata_google_auth  # type: ignore
        creds = pydata_google_auth.get_user_credentials(["https://www.googleapis.com/auth/bigquery"],
                                                        use_local_webserver=True)
        return bigquery.Client(project=project, credentials=creds)


def fetch_bigquery(project, start="2022-02-14", end=None, data_dir="data", partitioned=True, max_gb=200,
                   refresh=False) -> pd.DataFrame:
    """Run the GDELT query directly from Python/Jupyter and cache it as data/gdelt_events_daily.csv.

    Setup once:  pip install google-cloud-bigquery pydata-google-auth
    `project` = your Google Cloud project id (a free BigQuery sandbox project is fine). The first call opens
    a browser window to log in. A dry run reports the bytes to be scanned and refuses above `max_gb`.
    If `events_partitioned` is not available it retries with the unpartitioned table."""
    path = os.path.join(data_dir, "gdelt_events_daily.csv")
    if os.path.exists(path) and not refresh:
        print(f"using cached {path}")
        return pd.read_csv(path)
    from google.cloud import bigquery  # type: ignore
    client = _bq_client(project)
    for part in ([True, False] if partitioned else [False]):
        sql = bigquery_sql(start, end, part)
        try:
            dry = client.query(sql, job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False))
            break
        except Exception as e:  # noqa: BLE001
            print(f"  {'partitioned' if part else 'full'} table not usable: {str(e)[:150]}")
    else:
        raise RuntimeError("BigQuery dry run failed for both tables")
    gb = dry.total_bytes_processed / 1e9
    print(f"BigQuery will scan {gb:.1f} GB (free tier: 1,000 GB/month)")
    if gb > max_gb:
        raise RuntimeError(f"{gb:.0f} GB > max_gb={max_gb}; raise max_gb if you are sure")
    rows = client.query(sql).result()
    df = pd.DataFrame([dict(r) for r in rows])
    os.makedirs(data_dir, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"saved {len(df):,} rows to {path}")
    return df
