"""Network constructions and analyses.

1. Entity co-mention network (weighted, undirected): who/what is talked about together.
   - strength/degree distributions, clustering, assortativity
   - small-world coefficient vs degree-preserving nulls
   - rich-club coefficient vs nulls
   - disparity-filter backbone (Serrano, Boguna & Vespignani 2009, PNAS 106:6483)
   - Louvain communities, modularity significance vs configuration-model nulls
2. Signed "alignment" network for structural-balance analysis (Heider 1946;
   Cartwright & Harary 1956; Estrada & Benzi 2014 walk-based balance).
3. Temporal network: windowed snapshots, edge persistence, partition similarity.
"""
from __future__ import annotations

import itertools
from collections import Counter, defaultdict

import networkx as nx
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------
def comention_graph(entity_lists, min_node_count=5, min_edge_weight=2, weighting="count") -> nx.Graph:
    """Nodes: entities mentioned >= min_node_count times. Edge weight = number of posts in which
    both appear (or 'jaccard' / 'pmi' normalisation)."""
    node_c = Counter(e for es in entity_lists for e in set(es))
    keep = {e for e, c in node_c.items() if c >= min_node_count}
    edge_c = Counter()
    for es in entity_lists:
        es = sorted(set(e for e in es if e in keep))
        for a, b in itertools.combinations(es, 2):
            edge_c[(a, b)] += 1
    G = nx.Graph()
    for e in sorted(keep):  # sorted: set order depends on Python's hash seed -> non-reproducible Louvain
        G.add_node(e, count=node_c[e])
    N = len(entity_lists)
    for (a, b), w in sorted(edge_c.items()):
        if w < min_edge_weight:
            continue
        if weighting == "jaccard":
            ww = w / (node_c[a] + node_c[b] - w)
        elif weighting == "pmi":
            ww = np.log(w * N / (node_c[a] * node_c[b]))
            if ww <= 0:
                continue
        else:
            ww = w
        G.add_edge(a, b, weight=float(ww), count=w)
    G.remove_nodes_from([n for n in list(G) if G.degree(n) == 0])
    return G


def giant_component(G):
    return G.subgraph(max(nx.connected_components(G), key=len)).copy()


# --------------------------------------------------------------------------
# Structure
# --------------------------------------------------------------------------
def basic_stats(G) -> dict:
    Gc = giant_component(G)
    return dict(nodes=G.number_of_nodes(), edges=G.number_of_edges(), density=nx.density(G),
                gcc_frac=Gc.number_of_nodes() / G.number_of_nodes(),
                mean_degree=2 * G.number_of_edges() / G.number_of_nodes(),
                clustering=nx.average_clustering(G), weighted_clustering=nx.average_clustering(G, weight="weight"),
                transitivity=nx.transitivity(G), assortativity=nx.degree_assortativity_coefficient(G),
                avg_path_gcc=nx.average_shortest_path_length(Gc), diameter_gcc=nx.diameter(Gc))


def _rewired(G, seed, nswap_factor=10):
    H = nx.Graph(G)
    m = H.number_of_edges()
    try:
        nx.double_edge_swap(H, nswap=nswap_factor * m, max_tries=nswap_factor * m * 20, seed=seed)
    except nx.NetworkXAlgorithmError:
        pass
    w = [d["weight"] for _, _, d in G.edges(data=True)]
    rng = np.random.default_rng(seed)
    rng.shuffle(w)
    for (u, v), ww in zip(H.edges(), w):
        H[u][v]["weight"] = ww
    return H


def small_world(G, n_null=10, seed=0) -> dict:
    """sigma = (C/C_r)/(L/L_r), omega-like comparison using degree-preserving rewired nulls."""
    Gc = giant_component(G)
    C, L = nx.average_clustering(Gc), nx.average_shortest_path_length(Gc)
    Cr, Lr = [], []
    for i in range(n_null):
        H = giant_component(_rewired(Gc, seed + i))
        Cr.append(nx.average_clustering(H))
        Lr.append(nx.average_shortest_path_length(H))
    Cr, Lr = np.mean(Cr), np.mean(Lr)
    return dict(C=C, C_null=Cr, L=L, L_null=Lr, sigma=(C / Cr) / (L / Lr))


def rich_club(G, n_null=10, seed=0) -> pd.DataFrame:
    """Normalised rich-club coefficient rho(k) = phi(k)/phi_null(k) (Colizza et al. 2006)."""
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    phi = nx.rich_club_coefficient(G, normalized=False)
    nulls = []
    for i in range(n_null):
        nulls.append(nx.rich_club_coefficient(_rewired(G, seed + i), normalized=False))
    ks = sorted(phi)
    null_mean = np.array([np.mean([nn.get(k, np.nan) for nn in nulls]) for k in ks])
    df = pd.DataFrame({"k": ks, "phi": [phi[k] for k in ks], "phi_null": null_mean})
    df["rho"] = df["phi"] / df["phi_null"]
    return df


def disparity_filter(G, alpha=0.05, weight="weight") -> nx.Graph:
    """Keep edges that are statistically significant for at least one endpoint."""
    B = nx.Graph()
    B.add_nodes_from(G.nodes(data=True))
    for u in G:
        k = G.degree(u)
        if k < 2:
            continue
        s = G.degree(u, weight=weight)
        for v in G[u]:
            p = G[u][v][weight] / s
            a_uv = (1 - p) ** (k - 1)
            if a_uv < alpha:
                B.add_edge(u, v, **G[u][v], alpha=a_uv)
    B.remove_nodes_from([n for n in list(B) if B.degree(n) == 0])
    return B


def centralities(G, top=15) -> pd.DataFrame:
    dist = {(u, v): 1 / d["weight"] for u, v, d in G.edges(data=True)}
    nx.set_edge_attributes(G, dist, "dist")
    df = pd.DataFrame({
        "degree": dict(G.degree()),
        "strength": dict(G.degree(weight="weight")),
        "betweenness": nx.betweenness_centrality(G, weight="dist"),
        "eigenvector": pd.Series(nx.eigenvector_centrality_numpy(giant_component(G), weight="weight")).reindex(list(G)).fillna(0).to_dict(),
        "pagerank": nx.pagerank(G, weight="weight"),
        "core_number": nx.core_number(G),
    })
    return df.sort_values("pagerank", ascending=False)


# --------------------------------------------------------------------------
# Communities
# --------------------------------------------------------------------------
def louvain(G, seed=0, resolution=1.0):
    comms = nx.community.louvain_communities(G, weight="weight", seed=seed, resolution=resolution)
    comms = sorted(comms, key=len, reverse=True)
    Q = nx.community.modularity(G, comms, weight="weight")
    part = {n: i for i, c in enumerate(comms) for n in c}
    return comms, part, Q


def modularity_significance(G, n_null=20, seed=0, weighted=False) -> dict:
    """Louvain Q of the observed graph vs Louvain Q on degree-preserving rewired nulls.

    Default (weighted=False) tests the *topology*. With weighted=True weights are shuffled across
    edges, which is a poor null for hub-dominated co-mention graphs: a few very heavy edges placed at
    random create spurious 'communities' and inflate the null Q (observed on the real data).
    """
    H = G if weighted else nx.Graph(G.edges())
    if not weighted:
        nx.set_edge_attributes(H, 1.0, "weight")
    _, _, Q = louvain(H, seed=seed)
    Qn = [louvain(_rewired(H, seed + 100 + i), seed=seed)[2] for i in range(n_null)]
    return dict(Q=Q, Q_null_mean=float(np.mean(Qn)), Q_null_sd=float(np.std(Qn)),
                z=float((Q - np.mean(Qn)) / (np.std(Qn) + 1e-12)), Q_null=Qn, weighted=weighted)


def partition_stability(G, n_runs=20) -> float:
    """Mean pairwise NMI between Louvain runs with different seeds (robustness of communities)."""
    from sklearn.metrics import normalized_mutual_info_score as nmi
    nodes = list(G)
    parts = [louvain(G, seed=s)[1] for s in range(n_runs)]
    labs = [[p[n] for n in nodes] for p in parts]
    vals = [nmi(a, b) for a, b in itertools.combinations(labs, 2)]
    return float(np.mean(vals))


def compare_algorithms(G) -> pd.DataFrame:
    from sklearn.metrics import normalized_mutual_info_score as nmi
    nodes = list(G)
    algs = {
        "louvain": louvain(G)[0],
        "greedy_modularity": list(nx.community.greedy_modularity_communities(G, weight="weight")),
        "label_propagation": list(nx.community.asyn_lpa_communities(G, weight="weight", seed=0)),
    }
    lab = {k: {n: i for i, c in enumerate(v) for n in c} for k, v in algs.items()}
    rows = []
    for k, v in algs.items():
        rows.append(dict(algorithm=k, n_comms=len(v), Q=nx.community.modularity(G, v, weight="weight"),
                         NMI_vs_louvain=nmi([lab["louvain"][n] for n in nodes], [lab[k][n] for n in nodes])))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Signed network + structural balance
# --------------------------------------------------------------------------
def entity_valence(df_tagged, min_count=5) -> pd.DataFrame:
    """Mean sentence-level sentiment towards each entity (Trump -> entity sign)."""
    vals = defaultdict(list)
    for d in df_tagged["ent_sent"]:
        for e, s in d.items():
            vals[e].append(s)
    rows = [dict(entity=e, n=len(v), valence=np.mean(v), frac_neg=np.mean(np.array(v) < -0.05),
                 frac_pos=np.mean(np.array(v) > 0.05)) for e, v in vals.items() if len(v) >= min_count]
    return pd.DataFrame(rows).set_index("entity").sort_values("valence")


def signed_alignment_graph(df_tagged, min_node_count=8, min_edge_posts=3, thr=0.05,
                           distinct_sentences=True) -> nx.Graph:
    """Edge (i,j) sign = sign of sum over co-mention posts of s_i * s_j, where s is the sentence-level
    valence towards each entity in that post. + : framed on the same side (both praised or both attacked);
    - : framed as opponents (one praised, the other attacked). Weight = |sum|.

    distinct_sentences=True only uses posts where i and j are named in *different* sentences. If they
    share a sentence, both inherit that sentence's score, which mechanically creates + ties (a bias seen
    on the real data: 88% positive ties).
    """
    node_c = Counter(e for d in df_tagged["ent_sent"] for e in d)
    keep = {e for e, c in node_c.items() if c >= min_node_count}
    acc, cnt = defaultdict(float), Counter()
    sids = df_tagged["ent_sid"] if ("ent_sid" in df_tagged and distinct_sentences) else [None] * len(df_tagged)
    for d, sid in zip(df_tagged["ent_sent"], sids):
        items = [(e, s) for e, s in d.items() if e in keep and abs(s) > thr]
        for (a, sa), (b, sb) in itertools.combinations(sorted(items), 2):
            if sid is not None and (sid.get(a, frozenset()) & sid.get(b, frozenset())):
                continue
            acc[(a, b)] += np.sign(sa) * np.sign(sb)
            cnt[(a, b)] += 1
    G = nx.Graph()
    for (a, b), v in acc.items():
        if cnt[(a, b)] >= min_edge_posts and v != 0:
            G.add_edge(a, b, sign=int(np.sign(v)), weight=abs(v), n=cnt[(a, b)])
    return G


def triangle_balance(G, n_null=500, seed=0) -> dict:
    """Fraction of balanced triangles (product of signs > 0) vs a sign-shuffling null."""
    order = {n: i for i, n in enumerate(G)}
    tris = [(a, b, c) for a in G for b in G[a] if order[b] > order[a]
            for c in G[b] if order[c] > order[b] and G.has_edge(a, c)]
    edges = list(G.edges())
    idx = {frozenset(e): i for i, e in enumerate(edges)}
    signs = np.array([G[u][v]["sign"] for u, v in edges])
    T = np.array([[idx[frozenset((a, b))], idx[frozenset((b, c))], idx[frozenset((a, c))]] for a, b, c in tris])
    if len(T) == 0:
        return dict(n_triangles=0)

    def frac(sg):
        p = sg[T].prod(axis=1)
        return np.mean(p > 0), np.bincount((sg[T] < 0).sum(axis=1), minlength=4) / len(T)

    f, types = frac(signs)
    rng = np.random.default_rng(seed)
    null = np.array([frac(rng.permutation(signs))[0] for _ in range(n_null)])
    return dict(n_triangles=len(T), frac_balanced=f, null_mean=null.mean(), null_sd=null.std(),
                z=(f - null.mean()) / (null.std() + 1e-12), p=(np.sum(null >= f) + 1) / (n_null + 1),
                frac_pos_edges=np.mean(signs > 0),
                triad_types={"+++": types[0], "++-": types[1], "+--": types[2], "---": types[3]})


def walk_balance_null(G, beta=1.0, n_null=100, seed=0) -> tuple[float, float]:
    """Mean and sd of K under random permutation of edge signs (same topology)."""
    rng = np.random.default_rng(seed)
    signs = np.array([d["sign"] for *_, d in G.edges(data=True)])
    out = []
    for _ in range(n_null):
        H = G.copy()
        for (u, v), sg in zip(H.edges(), rng.permutation(signs)):
            H[u][v]["sign"] = int(sg)
        out.append(walk_balance(H, beta))
    return float(np.mean(out)), float(np.std(out))


def walk_balance(G, beta=1.0) -> float:
    """Estrada & Benzi (2014): K = tr exp(beta*A_signed) / tr exp(beta*|A|) in (0,1]; 1 = perfectly balanced."""
    nodes = list(G)
    A = nx.to_numpy_array(G, nodelist=nodes, weight=None)
    S = nx.to_numpy_array(G, nodelist=nodes, weight="sign")
    from scipy.special import logsumexp
    lS, lA = np.linalg.eigvalsh(beta * S), np.linalg.eigvalsh(beta * A)
    return float(np.exp(logsumexp(lS) - logsumexp(lA)))


def two_camps(G) -> tuple[dict, float]:
    """Spectral split using the leading eigenvector of the signed adjacency, then greedy local
    improvement to minimise frustration (number of + edges across camps and - edges within)."""
    nodes = list(G)
    ix = {n: i for i, n in enumerate(nodes)}
    E = np.array([(ix[u], ix[v], d["sign"], d["weight"]) for u, v, d in G.edges(data=True)], float)
    S = np.zeros((len(nodes), len(nodes)))
    S[E[:, 0].astype(int), E[:, 1].astype(int)] = E[:, 2] * E[:, 3]
    S = S + S.T
    w, V = np.linalg.eigh(S)
    x = np.sign(V[:, -1])
    x[x == 0] = 1

    ei, ej = E[:, 0].astype(int), E[:, 1].astype(int)

    def frustration(x):
        same = x[ei] == x[ej]
        bad = ((E[:, 2] > 0) & ~same) | ((E[:, 2] < 0) & same)
        return float(np.sum(E[:, 3] * bad))

    best = frustration(x)
    improved = True
    while improved:
        improved = False
        for i in range(len(nodes)):
            x[i] *= -1
            f = frustration(x)
            if f < best - 1e-12:
                best, improved = f, True
            else:
                x[i] *= -1
    total = E[:, 3].sum()
    return {n: int(x[i]) for i, n in enumerate(nodes)}, best / total


# --------------------------------------------------------------------------
# Temporal network
# --------------------------------------------------------------------------
def temporal_snapshots(df_tagged, freq="QS", min_node_count=3, min_edge_weight=1):
    snaps = {}
    for period, g in df_tagged.groupby(pd.Grouper(key="t_et", freq=freq)):
        if len(g) < 50:
            continue
        G = comention_graph(g["entities"].tolist(), min_node_count, min_edge_weight)
        if G.number_of_edges() > 0:
            snaps[period.tz_localize(None) if period.tzinfo else period] = G
    return snaps


def temporal_metrics(snaps) -> pd.DataFrame:
    from sklearn.metrics import normalized_mutual_info_score as nmi
    rows, prev = [], None
    for t, G in snaps.items():
        comms, part, Q = louvain(G)
        r = dict(period=t, nodes=G.number_of_nodes(), edges=G.number_of_edges(), density=nx.density(G),
                 clustering=nx.average_clustering(G), Q=Q, n_comms=len(comms))
        if prev is not None:
            Gp, pp = prev
            Ep = {frozenset(e) for e in Gp.edges()}
            Ec = {frozenset(e) for e in G.edges()}
            r["edge_jaccard"] = len(Ep & Ec) / len(Ep | Ec)
            r["node_jaccard"] = len(set(Gp) & set(G)) / len(set(Gp) | set(G))
            shared = sorted(set(pp) & set(part))
            r["NMI_prev"] = nmi([pp[n] for n in shared], [part[n] for n in shared]) if len(shared) > 5 else np.nan
        rows.append(r)
        prev = (G, part)
    return pd.DataFrame(rows).set_index("period")


def centrality_trajectories(snaps, measure="pagerank", top=12) -> pd.DataFrame:
    rec = {}
    for t, G in snaps.items():
        if measure == "pagerank":
            rec[t] = nx.pagerank(G, weight="weight")
        else:
            rec[t] = dict(G.degree(weight="weight"))
    df = pd.DataFrame(rec).fillna(0)
    return df.loc[df.mean(axis=1).sort_values(ascending=False).index[:top]]
