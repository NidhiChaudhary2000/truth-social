"""Entity / topic tagging and sentiment.

Entities are matched with a curated alias dictionary (people, countries,
institutions, companies) plus the @-mentions in each post. Topics are keyword
lexicons. Sentiment is computed per *sentence*, so a post such as
"Great Republicans ... the Radical Left Democrats are destroying our Country"
gives the two entities opposite signs (used for the signed network).
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd

# canonical name: (category, [regex aliases])   -- regexes are matched with word boundaries
ENTITIES: dict[str, tuple[str, list[str]]] = {
    # --- US politics: people
    "Joe Biden": ("person", [r"Biden", r"Sleepy Joe", r"Crooked Joe"]),
    "Kamala Harris": ("person", [r"Kamala", r"Harris"]),
    "Barack Obama": ("person", [r"Obama"]),
    "Hillary Clinton": ("person", [r"Hillary", r"Clinton"]),
    "Nancy Pelosi": ("person", [r"Pelosi"]),
    "Chuck Schumer": ("person", [r"Schumer"]),
    "Mitch McConnell": ("person", [r"McConnell", r"Mitch"]),
    "Mike Johnson": ("person", [r"Speaker Johnson", r"Mike Johnson"]),
    "JD Vance": ("person", [r"J\.?D\.? Vance", r"Vance"]),
    "Ron DeSantis": ("person", [r"DeSantis", r"DeSanctimonious", r"Ron DeSanctus"]),
    "Nikki Haley": ("person", [r"Haley", r"Birdbrain"]),
    "Gavin Newsom": ("person", [r"Newsom", r"Newscum"]),
    "Liz Cheney": ("person", [r"Liz Cheney", r"Cheney"]),
    "Tim Walz": ("person", [r"Walz"]),
    "AOC": ("person", [r"AOC", r"Ocasio-Cortez"]),
    "Zohran Mamdani": ("person", [r"Mamdani"]),
    "Elon Musk": ("person", [r"Elon", r"Musk"]),
    "Jerome Powell": ("person", [r"Powell"]),
    "Jack Smith": ("person", [r"Jack Smith", r"Deranged Jack"]),
    "Alvin Bragg": ("person", [r"Bragg"]),
    "Letitia James": ("person", [r"Letitia James", r"Peekaboo"]),
    "Fani Willis": ("person", [r"Fani", r"Willis"]),
    "Merrick Garland": ("person", [r"Garland"]),
    "James Comey": ("person", [r"Comey"]),
    "Jeffrey Epstein": ("person", [r"Epstein"]),
    # --- world leaders
    "Vladimir Putin": ("person", [r"Putin"]),
    "Volodymyr Zelensky": ("person", [r"Zelensk[yi]+"]),
    "Xi Jinping": ("person", [r"President Xi", r"Xi Jinping", r"Xi"]),
    "Benjamin Netanyahu": ("person", [r"Netanyahu", r"Bibi"]),
    "Kim Jong Un": ("person", [r"Kim Jong", r"Kim Jong Un"]),
    "Narendra Modi": ("person", [r"Modi"]),
    "Justin Trudeau": ("person", [r"Trudeau"]),
    "Mark Carney": ("person", [r"Carney"]),
    "Claudia Sheinbaum": ("person", [r"Sheinbaum"]),
    "Emmanuel Macron": ("person", [r"Macron"]),
    "Keir Starmer": ("person", [r"Starmer"]),
    "Nicolas Maduro": ("person", [r"Maduro"]),
    "Lula": ("person", [r"Lula"]),
    "Recep Erdogan": ("person", [r"Erdogan", r"Erdo\u011fan"]),
    "Javier Milei": ("person", [r"Milei"]),
    "Viktor Orban": ("person", [r"Orban", r"Orb\u00e1n"]),
    "Cyril Ramaphosa": ("person", [r"Ramaphosa"]),
    "Gustavo Petro": ("person", [r"Petro"]),
    "Mohammed bin Salman": ("person", [r"bin Salman", r"MBS"]),
    "Shigeru Ishiba": ("person", [r"Ishiba"]),
    "Sanae Takaichi": ("person", [r"Takaichi"]),
    "Ali Khamenei": ("person", [r"Khamenei", r"Ayatollah"]),
    # --- countries / regions
    "China": ("country", [r"China", r"Chinese", r"Beijing"]),
    "Mexico": ("country", [r"Mexico", r"Mexican"]),
    "Canada": ("country", [r"Canada", r"Canadian"]),
    "European Union": ("country", [r"European Union", r"E\.U\.", r"EU"]),
    "Japan": ("country", [r"Japan", r"Japanese"]),
    "India": ("country", [r"India", r"Indian"]),
    "Russia": ("country", [r"Russia", r"Russian", r"Moscow"]),
    "Ukraine": ("country", [r"Ukraine", r"Ukrainian", r"Kyiv"]),
    "Iran": ("country", [r"Iran", r"Iranian", r"Tehran"]),
    "Israel": ("country", [r"Israel", r"Israeli"]),
    "Gaza": ("country", [r"Gaza", r"Hamas"]),
    "Venezuela": ("country", [r"Venezuela", r"Venezuelan"]),
    "Brazil": ("country", [r"Brazil", r"Brazilian"]),
    "South Korea": ("country", [r"South Korea", r"(?<!North )(?<!N\. )Korea"]),
    "Taiwan": ("country", [r"Taiwan"]),
    "Vietnam": ("country", [r"Vietnam"]),
    "Germany": ("country", [r"Germany", r"German"]),
    "United Kingdom": ("country", [r"United Kingdom", r"U\.K\.", r"Britain", r"British"]),
    "Switzerland": ("country", [r"Switzerland", r"Swiss"]),
    "Greenland": ("country", [r"Greenland"]),
    "Denmark": ("country", [r"Denmark", r"Danish"]),
    "Panama": ("country", [r"Panama"]),
    "France": ("country", [r"France", r"French"]),
    "North Korea": ("country", [r"North Korea", r"N\. Korea"]),
    "Saudi Arabia": ("country", [r"Saudi Arabia", r"Saudi"]),
    "Qatar": ("country", [r"Qatar"]),
    "Syria": ("country", [r"Syria", r"Syrian"]),
    "Turkey": ("country", [r"Turkey", r"T\u00fcrkiye", r"Turkish"]),
    "Argentina": ("country", [r"Argentina"]),
    "Colombia": ("country", [r"Colombia"]),
    "South Africa": ("country", [r"South Africa"]),
    "Australia": ("country", [r"Australia"]),
    "Hungary": ("country", [r"Hungary"]),
    "Poland": ("country", [r"Poland"]),
    "Cuba": ("country", [r"Cuba"]),
    "Afghanistan": ("country", [r"Afghanistan"]),
    # --- institutions / groups
    "Democrats": ("group", [r"Democrats?", r"Dems", r"Radical Left", r"Democrat Party"]),
    "Republicans": ("group", [r"Republicans?", r"GOP", r"RINOS?"]),
    "MAGA": ("group", [r"MAGA", r"Make America Great Again"]),
    "Federal Reserve": ("institution", [r"Federal Reserve", r"the Fed", r"Fed"]),
    "FBI": ("institution", [r"FBI"]),
    "DOJ": ("institution", [r"DOJ", r"Department of Justice", r"Justice Department"]),
    "Supreme Court": ("institution", [r"Supreme Court", r"SCOTUS"]),
    "Congress": ("institution", [r"Congress", r"the Senate", r"the House"]),
    "NATO": ("institution", [r"NATO"]),
    "Fake News Media": ("media", [r"Fake News", r"Lamestream", r"Mainstream Media"]),
    "CNN": ("media", [r"CNN"]),
    "MSNBC": ("media", [r"MSNBC", r"MSDNC"]),
    "New York Times": ("media", [r"New York Times", r"NY Times", r"NYT"]),
    "Washington Post": ("media", [r"Washington Post"]),
    "Fox News": ("media", [r"Fox News", r"FoxNews", r"Fox"]),
    "ABC": ("media", [r"ABC"]),
    "NBC": ("media", [r"NBC"]),
    "CBS": ("media", [r"CBS", r"60 Minutes"]),
    "Harvard": ("institution", [r"Harvard"]),
    "DOGE": ("institution", [r"DOGE"]),
    # --- companies
    "Apple": ("company", [r"Apple", r"Tim Cook"]),
    "Tesla": ("company", [r"Tesla"]),
    "Amazon": ("company", [r"Amazon", r"Bezos"]),
    "Intel": ("company", [r"Intel"]),
    "Nvidia": ("company", [r"Nvidia", r"NVIDIA"]),
    "Boeing": ("company", [r"Boeing"]),
    "Walmart": ("company", [r"Walmart"]),
    "Ford": ("company", [r"Ford Motor"]),
    "General Motors": ("company", [r"General Motors"]),
    "Meta": ("company", [r"Facebook", r"Zuckerberg", r"Meta"]),
    "Google": ("company", [r"Google", r"Alphabet"]),
    "Microsoft": ("company", [r"Microsoft"]),
    "Disney": ("company", [r"Disney"]),
    "Comcast": ("company", [r"Comcast", r"Concast"]),
    "Lockheed Martin": ("company", [r"Lockheed"]),
    "Mattel": ("company", [r"Mattel"]),
    "Harley-Davidson": ("company", [r"Harley"]),
    "US Steel": ("company", [r"U\.?S\.? Steel", r"Nippon Steel"]),
    "Coca-Cola": ("company", [r"Coca-Cola", r"Coke"]),
    "Pfizer": ("company", [r"Pfizer"]),
    "Exxon": ("company", [r"Exxon"]),
    "Paramount": ("company", [r"Paramount"]),
    "Truth Social": ("company", [r"Truth Social", r"TRUTH Social"]),
}

# Mostly lower-case keyword lexicons (matched case-insensitively)
TOPICS: dict[str, list[str]] = {
    "tariffs_trade": [r"tariffs?", r"trade deal", r"trade deficit", r"reciprocal", r"duties", r"trade war",
                      r"liberation day", r"import tax", r"customs"],
    "markets_economy": [r"stock market", r"dow", r"s&p", r"nasdaq", r"economy", r"inflation", r"jobs",
                        r"gdp", r"great time to buy", r"401k", r"markets?", r"prices", r"gas prices", r"recession"],
    "fed_rates": [r"federal reserve", r"\bfed\b", r"powell", r"interest rates?", r"rate cuts?", r"cut rates"],
    "immigration": [r"border", r"illegal aliens?", r"migrants?", r"deport\w*", r"\bice\b", r"invasion",
                    r"illegals", r"sanctuary"],
    "foreign_war": [r"ukraine", r"russia", r"putin", r"iran", r"israel", r"gaza", r"hamas", r"war\b",
                    r"ceasefire", r"nato", r"nuclear", r"missiles?", r"strikes?"],
    "legal": [r"witch hunt", r"indict\w*", r"trial", r"judge", r"prosecutor", r"hoax", r"lawfare",
              r"jack smith", r"bragg", r"weaponi[sz]\w*", r"court"],
    "elections": [r"election", r"rigged", r"ballots?", r"voter?s?\b", r"polls?", r"midterms?", r"primary",
                  r"endorsement", r"swing state", r"mail-in", r"campaign"],
    "endorsement": [r"complete and total endorsement", r"total endorsement", r"my endorsement",
                    r"endorse"],
    "media_attack": [r"fake news", r"lamestream", r"corrupt media", r"enemy of the people", r"failing new york times"],
    "crypto": [r"crypto\w*", r"bitcoin", r"\$trump", r"stablecoin", r"blockchain", r"digital assets?"],
    "energy": [r"drill", r"\boil\b", r"energy", r"gas\b", r"lng", r"windmills?"],
}

# Geopolitical framing of a post (used for the multiplex war/trade/diplomacy network)
FRAMES: dict[str, list[str]] = {
    "conflict": [r"war\b", r"wars", r"military", r"missiles?", r"strikes?", r"bomb\w*", r"nuclear", r"troops",
                 r"attack\w*", r"invasion", r"invade\w*", r"hostages?", r"terror\w*", r"weapons?", r"army",
                 r"navy", r"drones?", r"killing", r"soldiers", r"obliterat\w*", r"blockade", r"cartels?"],
    "trade": [r"tariffs?", r"trade", r"trading", r"deficit", r"imports?", r"exports?", r"duties", r"sanctions?",
              r"reciprocal", r"ripp\w* (?:us|the united states|america) off", r"tax(?:es)? on", r"steel",
              r"aluminum", r"soybeans?", r"farmers", r"oil"],
    "diplomacy": [r"peace", r"ceasefire", r"cease-fire", r"negotiat\w*", r"talks", r"summit", r"meeting with",
                  r"great call", r"spoke with", r"phone call", r"agreement", r"deal\b", r"treaty", r"friendship",
                  r"relationship", r"ambassador", r"visit\w*"],
}
_frame_regex = {name: re.compile(r"(?i)(?<![\w])(?:" + "|".join(al) + r")") for name, al in FRAMES.items()}


def find_frames(text: str) -> list[str]:
    return [name for name, rx in _frame_regex.items() if rx.search(text)]


# Domain-tuned lexicon used when VADER is not installed, and blended with VADER otherwise.
POS_WORDS = set("""great tremendous beautiful strong winning win wins won incredible amazing fantastic
wonderful best success successful thank thanks congratulations congrats endorsement endorse proud honor
honored love loved perfect happy record historic magnificent patriot patriots brave fighter warrior
tough smart brilliant highly respected safe secure booming boom rich peace deal""".split())
NEG_WORDS = set("""crooked corrupt failed failing fake disaster disastrous weak rigged radical lunatic loser
losers sleepy terrible horrible worst scam hoax witch hunt deranged dumb stupid incompetent sick
pathetic evil dangerous destroy destroying destroyed disgrace disgraceful fraud illegal criminal criminals
low iq nasty treason traitor traitors hate liar liars lies lying ugly bad sad unfair rinos rino
communist marxist fascist enemy weaponized persecution election interference invasion death kill
killed killing badly hostile aggressive threat threatening ripping cheating cheated abuse abused""".split())

TAG_VERSION = 4  # bump when tagging rules change (invalidates the cached tagged posts)
_ent_regex = {name: re.compile(r"(?<![\w])(?:" + "|".join(al) + r")(?![\w])") for name, (_, al) in ENTITIES.items()}
# ALL-CAPS posts are common ("CROOKED JOE BIDEN"): for those we match case-insensitively
_ent_regex_ci = {name: re.compile(rx.pattern, re.I) for name, rx in _ent_regex.items()}
_topic_regex = {name: re.compile(r"(?i)(?<![\w])(?:" + "|".join(al) + r")") for name, al in TOPICS.items()}
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")

try:  # optional, better general-purpose sentiment
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
    _VADER = SentimentIntensityAnalyzer()
except Exception:  # noqa: BLE001
    _VADER = None


def entity_category(name: str) -> str:
    return ENTITIES[name][0] if name in ENTITIES else "handle"


def lexicon_score(s: str) -> float:
    w = re.findall(r"[a-z']+", s.lower())
    if not w:
        return 0.0
    p = sum(x in POS_WORDS for x in w)
    n = sum(x in NEG_WORDS for x in w)
    return 0.0 if p + n == 0 else (p - n) / (p + n)  # in [-1, 1]


def sentiment(s: str) -> float:
    """Sentence/post sentiment in [-1, 1]. VADER compound if available, else lexicon."""
    if _VADER is not None:
        v = _VADER.polarity_scores(s)["compound"]
        lx = lexicon_score(s)
        return float(np.clip(0.7 * v + 0.3 * lx, -1, 1))
    return lexicon_score(s)


def is_shouting(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    return len(letters) > 0 and sum(c.isupper() for c in letters) / len(letters) > 0.6


def find_entities(text: str) -> list[str]:
    rxs = _ent_regex_ci if is_shouting(text) else _ent_regex
    return [name for name, rx in rxs.items() if rx.search(text)]


def find_topics(text: str) -> list[str]:
    return [name for name, rx in _topic_regex.items() if rx.search(text)]


def tag_posts(df: pd.DataFrame, include_handles: bool = True, handle_min_count: int = 5) -> pd.DataFrame:
    """Adds: entities (list), topics (list), sent (post sentiment), ent_sent (dict entity->sentiment)."""
    df = df.copy()
    texts = df["text_nourl"].tolist()
    ents, tops, sents, ent_sent, ent_sid = [], [], [], [], []
    handle_counts = Counter(h.lower() for hs in df["mentions"] for h in hs)
    keep_handles = {h for h, c in handle_counts.items() if c >= handle_min_count}
    for t, hs in zip(texts, df["mentions"]):
        e = find_entities(t)
        if include_handles:
            e += [f"@{h.lower()}" for h in hs if h.lower() in keep_handles and h.lower() != "realdonaldtrump"]
        e = list(dict.fromkeys(e))
        ents.append(e)
        tops.append(find_topics(t))
        sents.append(sentiment(t) if t else 0.0)
        es, sid = {}, {}
        if e:
            sentences = [s for s in _SENT_SPLIT.split(t) if s.strip()]
            scores = [sentiment(s) for s in sentences]
            rxs = _ent_regex_ci if is_shouting(t) else _ent_regex
            for name in e:
                rx = rxs.get(name)
                pat = rx if rx is not None else re.compile(re.escape(name), re.I)  # None -> @handle
                idx = [i for i, s_ in enumerate(sentences) if pat.search(s_)]
                v = float(np.mean([scores[i] for i in idx])) if idx else sents[-1]
                if idx and abs(v) < 0.05:  # neutral naming sentence ("China is calling.") -> use the next sentence too
                    ctx = sorted({j for i in idx for j in (i, i + 1) if j < len(scores)})
                    v = float(np.mean([scores[j] for j in ctx]))
                es[name] = v
                sid[name] = frozenset(idx)
        ent_sent.append(es)
        ent_sid.append(sid)
    df["entities"], df["topics"], df["sent"], df["ent_sent"] = ents, tops, sents, ent_sent
    df["ent_sid"] = ent_sid  # sentence indices in which each entity occurs (used by the signed network)
    df["frames"] = [find_frames(t) for t in texts]
    for k in FRAMES:
        df[f"frame_{k}"] = [k in f for f in df["frames"]]
    for k in TOPICS:
        df[f"topic_{k}"] = [k in t for t in tops]
    return df


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())
