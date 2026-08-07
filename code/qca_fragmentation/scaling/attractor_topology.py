"""
What the terminal SCCs of the exponential-exponential classes actually look like.

R9's F1 right panel places every rule at (a_att, b_att): the base of the number
of monitored attractors against the base of the largest one.  Most rules sit at
a_att > 1 with b_att = 1 -- exponentially many attractors, each of bounded size.
Four classes sit away from that line with BOTH coordinates above 1, so they have
exponentially many attractors AND the largest one grows exponentially:

    (108, 201)                     a = rho^2 = 1.7549   b = phi    = 1.6180
    (156, 198)                     a = phi   = 1.6180   b = 4^0.2  = 1.3195
    (73, 109)                      a = psi   = 1.4656   b = phi    = 1.6180
    (28, 29, 70, 71, 157, 199)     a = rho   = 1.3247   b = 2^(1/3)= 1.2599

with phi the golden ratio (x^2 = x+1), psi the supergolden (x^3 = x^2+1) and rho
the plastic number (x^3 = x+1); rho^2 = 1.75488 is the first class's constant.

The question this module answers is whether those large attractors are just long
CYCLES -- which is all a V-free rule can offer, its map being a function -- or
something with genuine two-dimensional structure.  The answer is emphatic, and it
goes the other way from what "attractor" suggests: they are not cycles, and they
are not extended either.  They are dense and COMPACT.

  * NONE of them is a cycle.  Every one carries a self-loop on every node, has
    density of order 1/2, and has DIAMETER 2 at nineteen of the twenty sizes
    measured (the twentieth is a 3), up to 2584 nodes.  The state count grows
    like phi^N; the graph distance does not grow.

  * The plastic class is the extreme case: every terminal SCC of all six rules,
    at every N tested, is a COMPLETE digraph K_n with all n^2 edges including
    loops.  Its transition matrix is uniform, so its second eigenvalue is
    exactly 0 -- the monitored dynamics forgets its state in ONE step.  Its size
    is 2^floor((N+1)/3) exactly, which is where b = 2^(1/3) comes from.

  * The (108, 201) class has a spectral gap that does not close: |lambda_2| of
    the row-normalised adjacency converges to 0.299 as the attractor grows from
    21 to 2584 nodes.  Mixing inside the attractor is O(1) while the attractor
    itself is Theta(phi^N).

  * THE UNIFYING FACT.  There are not four structures here, there are TWO.  The
    twelve rules are two coherent reflection pairs -- W108/W201 and W156/W198 --
    together with their eight dissipative children, and for every one of those
    eight the terminal SCCs are Krylov sectors of the parent with the SAME
    vertex set and the SAME induced edges.  The reset never fires on its own
    attractor; it acts purely as a selection rule, keeping some parent sectors
    and discarding the rest.  That is what separates the classes:

        108, 201            keep everything                a = rho^2, b = phi
        73, 109             keep a psi-sized subfamily      a = psi,   b = phi
        156, 198            keep everything                a = phi,   b = 4^0.2
        28,29,70,71,157,199 keep the COMPLETE sectors       a = rho,   b = 2^(1/3)

    The last row is exact for the pure-D children (W28's attractors are exactly
    W156's complete sectors, 36 / 64 / 113 of them at N = 10 / 12 / 14) and a
    strict subset for the ones carrying an E (W199 keeps 16 / 28 / 49 of
    W198's).  Both still count at rho per site.

  * The V-free contrast is total.  All 81 V-free rules have out-degree 1, so
    their terminal SCCs are cycles by construction; at N = 12, 75 of the 81 have
    nothing but fixed points and the longest cycle anywhere is 63 (W90, W165).

On "are they too small for graph measures": the graphs are not too small, but
two of the standard measures are uninformative, and the reason is the density,
not the size.  n runs to 2584 with 2.9M edges, which is ample.  However:

  * CLUSTERING says nothing here.  At density ~0.5 any graph, structured or
    random, has transitivity ~0.5; the measured 0.805 against a density-matched
    null of 0.723 at N = 16 is a small effect on a saturated scale.  It is
    reported next to its null for exactly that reason, never on its own.

  * VERTEX AND EDGE CONNECTIVITY say nothing either: in every case computed they
    equal the minimum degree, which is the generic value for a dense graph and
    tells us about the sparsest vertex rather than about global structure.

The measures that DO separate these graphs from a null are the density scaling
(m/n^2 falls like 0.948^N while the mean degree grows like 1.53^N), reciprocity
(0.61 down to 0.32 as n goes 21 -> 2584), the constant diameter, and the
spectral gap.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .. import results_io
from ..core.cycle import succ
from ..core.rules import coherent_parent, has_V, wolfram_to_tuple
from ..graph import scc
from . import sectors

BC = "obc0"
OUT_JSON = os.path.join(sectors.ANALYTICS, "attractor_topology_{bc}.json")

#: The four exponential-exponential classes of R9 F1's right panel, with the
#: constants R9 reports for them.  `parent` records the coherent rule whose
#: sectors the class reproduces, where there is one (R15's idle-reset story).
CLASSES: List[Dict] = [
    {"key": "rho2_phi", "rules": (108, 201), "a": 1.754878, "b": 1.618034,
     "a_name": r"$\rho^2$", "b_name": r"$\varphi$", "parent": None},
    {"key": "phi_4fifth", "rules": (156, 198), "a": 1.618034, "b": 1.319508,
     "a_name": r"$\varphi$", "b_name": r"$4^{1/5}$", "parent": None},
    {"key": "psi_phi", "rules": (73, 109), "a": 1.465571, "b": 1.618034,
     "a_name": r"$\psi$", "b_name": r"$\varphi$", "parent": {73: 201, 109: 108}},
    {"key": "rho_cbrt2", "rules": (28, 29, 70, 71, 157, 199), "a": 1.324718,
     "b": 1.259921, "a_name": r"$\rho$", "b_name": r"$2^{1/3}$", "parent": None},
]

#: Sizes at which the full topology is computed.  N = 16 puts the largest
#: attractor at 2584 nodes and 2.9M edges, which every measure here handles;
#: past that the dense spectral step is the binding constraint, not Tarjan.
NS = (8, 10, 12, 14, 16)

#: Caps, so that a measure is either computed honestly or reported as absent --
#: never silently approximated.
PATH_CAP = 4000        # matrix-power distances, transitivity, spectra
SPECTRAL_CAP = 3000    # eigenvalues of the row-normalised adjacency
#: Connectivity is the one measure with no matrix shortcut -- it is n max-flow
#: solves, and on a graph with 76k edges that is two minutes at n = 377 alone.
#: It is also the least informative of them here, because on a graph this dense
#: it simply equals the minimum degree; `min_undirected_degree` is reported at
#: every size so the reader can see that for themselves.
CONN_CAP = 200


# --- 1. the graphs -----------------------------------------------------------

def attractor_graph(rule: int, N: int, members: Sequence[int],
                    bc: str = BC) -> nx.DiGraph:
    """The induced subgraph of the transition graph on one terminal SCC.

    Edges leaving the member set are dropped, which is a no-op for a TERMINAL
    SCC -- by definition nothing leaves -- and the assertion below says so.
    """
    t = wolfram_to_tuple(rule)
    S = set(members)
    G = nx.DiGraph()
    G.add_nodes_from(members)
    for x in members:
        ys = succ(x, N, t, bc)
        assert all(y in S for y in ys), (rule, N, x)   # terminal, so no leaks
        for y in ys:
            G.add_edge(x, y)
    return G


def terminal_graphs(rule: int, N: int, bc: str = BC) -> List[nx.DiGraph]:
    t = wolfram_to_tuple(rule)
    return [attractor_graph(rule, N, c, bc)
            for c in scc.recurrent_classes(rule, N, bc, t)]


def largest_terminal(rule: int, N: int, bc: str = BC) -> nx.DiGraph:
    t = wolfram_to_tuple(rule)
    big = max(scc.recurrent_classes(rule, N, bc, t), key=len)
    return attractor_graph(rule, N, big, bc)


# --- 2. the measures ---------------------------------------------------------

def _distances(A: np.ndarray) -> Tuple[int, float, bool]:
    """Diameter, mean distance over ordered pairs u != v, and whether every such
    pair is reachable -- by boolean matrix powers rather than BFS.

    BFS from every node is the obvious route and it is the wrong one here: these
    graphs have n ~ 2600 and m ~ 3M, so networkx's O(n*m) pass is ~10^10 Python
    steps.  What makes the matrix route cheap is the very property being
    measured -- the diameter is 2, so the loop runs twice, and each step is one
    BLAS matmul on an n x n float32 array (26 MB at n = 2584).  The trade is only
    right because the diameter is tiny; a long-diameter graph would want BFS.
    """
    n = A.shape[0]
    D = np.zeros((n, n), dtype=np.float32)
    reach = A > 0
    D[reach] = 1.0
    frontier, k = A, 1
    while not reach.all() and k < n:
        k += 1
        frontier = ((frontier @ A) > 0).astype(np.float32)
        new = (frontier > 0) & ~reach
        if not new.any():
            break                       # nothing further is reachable at all
        D[new] = k
        reach |= new
    off = ~np.eye(n, dtype=bool)
    sel = off & reach
    ds = D[sel]
    return (int(ds.max()) if ds.size else 0,
            float(ds.mean()) if ds.size else 0.0,
            bool(ds.size == n * (n - 1)))


def _transitivity(A: np.ndarray) -> Tuple[float, float]:
    """Undirected transitivity of the loopless simple graph, and the density of
    that same graph -- which is what an Erdos-Renyi null would predict for it.

    Counted as trace(B^3) / sum_i d_i(d_i-1) instead of by enumerating triangles:
    networkx's routine costs O(sum d^2) ~ 3*10^9 here, and we already pay for
    matmuls of this size in `_distances`.
    """
    n = A.shape[0]
    B = ((A > 0) | (A.T > 0)).astype(np.float32)
    np.fill_diagonal(B, 0.0)
    deg = B.sum(axis=1)
    denom = float((deg * (deg - 1)).sum())
    if denom == 0:
        return 0.0, 0.0
    tri = float(np.einsum("ij,ji->", B @ B, B))          # = trace(B^3)
    p = float(deg.sum() / (n * (n - 1))) if n > 1 else 0.0
    return tri / denom, p


def _spectra(A: np.ndarray) -> Tuple[float, float]:
    """|lambda_2| of the row-normalised adjacency, and the Perron root of A.

    P is the transition matrix of the monitored walk with equal weight on each
    Kraus branch -- the classical process whose support a terminal SCC is -- so
    |lambda_2| is its mixing rate.  Sparse Arnoldi for the big cases: a dense
    non-symmetric eigendecomposition at n = 2584 costs minutes, and we need two
    eigenvalues, not 2584 of them.
    """
    n = A.shape[0]
    P = A / A.sum(axis=1, keepdims=True)
    if n <= 700:
        ev = np.sort(np.abs(np.linalg.eigvals(P)))[::-1]
        per = float(np.sort(np.abs(np.linalg.eigvals(A)))[::-1][0])
        return float(ev[1]), per
    import scipy.sparse as sp
    import scipy.sparse.linalg as spl
    evs = np.abs(spl.eigs(sp.csr_matrix(P.astype(np.float64)), k=3,
                          which="LM", return_eigenvectors=False))
    per = float(np.abs(spl.eigs(sp.csr_matrix(A.astype(np.float64)), k=1,
                                which="LM", return_eigenvectors=False))[0])
    return float(np.sort(evs)[::-1][1]), per


def measure(G: nx.DiGraph) -> Dict:
    """
    Graph-theoretic descriptors of one attractor.

    Every entry that a cap suppresses is written as None rather than omitted, so
    a table can distinguish "not computed at this size" from "zero".
    """
    n, m = G.number_of_nodes(), G.number_of_edges()
    od = np.array([d for _, d in G.out_degree()], dtype=float)
    idg = np.array([d for _, d in G.in_degree()], dtype=float)
    loops = nx.number_of_selfloops(G)
    out: Dict = {
        "n": n, "m": m,
        "density": m / (n * n),
        "mean_out_degree": float(od.mean()),
        "out_min": int(od.min()), "out_max": int(od.max()),
        "in_min": int(idg.min()), "in_max": int(idg.max()),
        "out_cv": float(od.std() / od.mean()) if od.mean() else 0.0,
        "in_cv": float(idg.std() / idg.mean()) if idg.mean() else 0.0,
        "n_distinct_out_degrees": len(set(od.tolist())),
        "self_loops": loops,
        "every_node_has_a_loop": bool(loops == n),
        # the three shapes worth naming
        "is_cycle": bool(m == n and od.max() == 1 and idg.max() == 1),
        "is_functional": bool(od.max() == 1),
        "is_complete_with_loops": bool(m == n * n),
        "reciprocity": float(nx.reciprocity(G)) if m else 0.0,
        "aperiodic": bool(nx.is_aperiodic(G)),
    }

    #: one dense adjacency, reused by every matrix-based measure below
    A = ((nx.to_numpy_array(G, nodelist=sorted(G.nodes())) > 0)
         .astype(np.float32)) if n <= PATH_CAP else None

    # Distances.  A self-loop makes every node reach itself in one step, so we
    # exclude the u == v pair; otherwise "mean distance" would be diluted by n
    # zeros and would shrink towards 1 for reasons of bookkeeping.
    if A is not None:
        diam, mean_d, all_reach = _distances(A)
        out["diameter"], out["mean_distance"] = diam, mean_d
        out["reaches_all"] = all_reach
    else:
        out["diameter"] = out["mean_distance"] = out["reaches_all"] = None

    # Clustering, reported WITH the null it has to be read against.  On a graph
    # this dense the two are close by construction and the measure carries
    # little information; see the module docstring.
    if A is not None and n > 1:
        out["transitivity"], out["transitivity_er_null"] = _transitivity(A)
    else:
        out["transitivity"] = out["transitivity_er_null"] = 0.0

    U = nx.Graph((u, v) for u, v in G.edges() if u != v)
    U.add_nodes_from(G)
    out["min_undirected_degree"] = int(min(d for _, d in U.degree())) if n else 0
    if 1 < n <= CONN_CAP:
        out["edge_connectivity"] = int(nx.edge_connectivity(U))
        out["node_connectivity"] = int(nx.node_connectivity(U))
        out["connectivity_equals_min_degree"] = bool(
            out["edge_connectivity"] == out["min_undirected_degree"])
    else:
        out["edge_connectivity"] = out["node_connectivity"] = None
        out["connectivity_equals_min_degree"] = None

    if A is not None and 1 < n <= SPECTRAL_CAP:
        out["slem"], out["perron_adjacency"] = _spectra(A)
    else:
        out["slem"] = out["perron_adjacency"] = None
    return out


# --- 3. per-rule profiles ----------------------------------------------------

def profile(rule: int, Ns: Sequence[int] = NS, bc: str = BC) -> Dict:
    """The largest terminal SCC at each N, measured, plus a census of all of
    them (cheap descriptors only -- there can be thousands)."""
    t = wolfram_to_tuple(rule)
    rows, census = [], []
    for N in Ns:
        classes = scc.recurrent_classes(rule, N, bc, t)
        sizes = sorted((len(c) for c in classes), reverse=True)
        big = max(classes, key=len)
        rows.append(dict(N=N, **measure(attractor_graph(rule, N, big, bc))))
        # cheap pass over ALL terminal SCCs: shape only
        # A single state with a self-loop is a cycle AND a complete digraph, so
        # it gets its own bucket: lumping it in with either makes one of the two
        # columns read as evidence when it is only bookkeeping.
        shapes = Counter()
        for c in classes:
            G = attractor_graph(rule, N, c, bc)
            k, e = len(c), G.number_of_edges()
            if k == 1:
                shapes["singleton"] += 1
            elif e == k * k:
                shapes["complete"] += 1
            elif e == k and max(d for _, d in G.out_degree()) == 1:
                shapes["cycle"] += 1
            else:
                shapes["other"] += 1
        nontrivial = len(classes) - shapes.get("singleton", 0)
        census.append({"N": N, "n_attractors": len(classes),
                       "sizes_top": sizes[:6],
                       "size_histogram": dict(sorted(Counter(sizes).items())),
                       "shapes": dict(shapes),
                       "n_nontrivial": nontrivial,
                       "all_complete": bool(
                           nontrivial and shapes.get("complete", 0) == nontrivial),
                       "any_cycle": bool(shapes.get("cycle", 0))})
    return {"rule": rule, "tuple": "".join(t), "largest": rows,
            "census": census}


def growth(rows: Sequence[Dict], key: str) -> Optional[Dict]:
    """Per-site base of a quantity sampled on an even-N ladder.

    The ladder steps by 2 because the brick-wall unit cell is two sites, so the
    ratio between consecutive rows is a base PER TWO SITES and the square root
    is the per-site figure R9 reports.
    """
    vals = [(r["N"], r[key]) for r in rows if r.get(key)]
    if len(vals) < 3:
        return None
    ratios = [vals[i + 1][1] / vals[i][1] for i in range(len(vals) - 1)]
    return {"ratios_per_two_sites": [float(x) for x in ratios],
            "base_per_site": float(math.sqrt(ratios[-1])),
            "stable": bool(abs(ratios[-1] / ratios[-2] - 1) < 0.02)}


# --- 4. the two controls -----------------------------------------------------

def vfree_control(N: int = 12, bc: str = BC) -> Dict:
    """
    Every V-free rule has a DETERMINISTIC map, so out-degree is 1 everywhere and
    a terminal SCC is a cycle -- not by observation but by definition.  We check
    the out-degree rather than assume it, and record how long those cycles get,
    because "a cycle of length 63" and "a fixed point" are very different
    answers to "is the attractor extended?" and the data says it is the latter
    almost everywhere.
    """
    rows = []
    for r in range(256):
        t = wolfram_to_tuple(r)
        if has_V(t):
            continue
        classes = scc.recurrent_classes(r, N, bc, t)
        deg1 = all(len(succ(x, N, t, bc)) == 1 for c in classes for x in c)
        rows.append({"rule": r, "tuple": "".join(t),
                     "n_attractors": len(classes),
                     "longest_cycle": max(len(c) for c in classes),
                     "all_out_degree_one": deg1})
    rows.sort(key=lambda x: -x["longest_cycle"])
    return {"N": N, "n_rules": len(rows),
            "all_deterministic": all(x["all_out_degree_one"] for x in rows),
            "longest_cycle": rows[0]["longest_cycle"],
            "fixed_point_only": sum(1 for x in rows if x["longest_cycle"] == 1),
            "histogram": dict(sorted(Counter(
                x["longest_cycle"] for x in rows).items())),
            "top": rows[:6]}


def parent_identity(bc: str = BC, Ns: Sequence[int] = (8, 10, 12)) -> List[Dict]:
    """
    Are the eight dissipative rules here a structure of their own, or their
    coherent parent's Krylov sectors seen through a reset that never fires?

    Set equality is the weak version of the question and it is not enough: the
    child could inherit the vertex set and still have different edges, since a
    reset that fires merges branches.  So the induced EDGE sets are compared as
    well, on the same vertex set, and `graph_identical` is the claim that
    matters.  For the complete-digraph class we additionally ask whether the
    attractors are exactly those parent sectors that are already complete --
    they are for the pure-D children and a strict subset for the ones carrying
    an E, which is worth recording rather than glossing.
    """
    out = []
    for child in (73, 109, 28, 29, 70, 71, 157, 199):
        parent = coherent_parent(child)
        tc, tp = wolfram_to_tuple(child), wolfram_to_tuple(parent)
        rows = []
        for N in Ns:
            att = [frozenset(c) for c in scc.recurrent_classes(child, N, bc, tc)]
            sec = [frozenset(c) for c in scc.recurrent_classes(parent, N, bc, tp)]
            sa, ss = set(att), set(sec)
            # same vertex sets AND same induced edges, attractor by attractor
            same_graph = True
            for c in att:
                gc = attractor_graph(child, N, sorted(c), bc)
                gp = nx.DiGraph()
                gp.add_nodes_from(c)
                for x in c:
                    for y in succ(x, N, tp, bc):
                        if y in c:
                            gp.add_edge(x, y)
                if set(gc.edges()) != set(gp.edges()):
                    same_graph = False
                    break
            complete_sec = {c for c in sec
                            if attractor_graph(parent, N, sorted(c), bc)
                            .number_of_edges() == len(c) ** 2}
            rows.append({
                "N": N,
                "child_attractors": len(sa), "parent_sectors": len(ss),
                "child_sets_are_parent_sectors": bool(sa <= ss),
                "graph_identical": same_graph,
                "largest_identical": bool(max(sa, key=len) == max(ss, key=len)),
                "parent_complete_sectors": len(complete_sec),
                "child_inside_complete_sectors": bool(sa <= complete_sec),
                "child_is_all_complete_sectors": bool(sa == complete_sec),
            })
        out.append({"child": child, "parent": parent,
                    "child_tuple": "".join(tc), "parent_tuple": "".join(tp),
                    "rows": rows})
    return out


# --- 5. assemble -------------------------------------------------------------

def build(bc: str = BC, Ns: Sequence[int] = NS) -> Dict:
    data: Dict = {"bc": bc, "Ns": list(Ns), "classes": []}
    for cls in CLASSES:
        entry = {k: v for k, v in cls.items() if k != "parent"}
        entry["parent"] = cls["parent"]
        entry["profiles"] = [profile(r, Ns, bc) for r in cls["rules"]]
        rep = entry["profiles"][-1]["largest"]
        entry["growth_nodes"] = growth(rep, "n")
        entry["growth_edges"] = growth(rep, "m")
        entry["growth_perron"] = growth(rep, "perron_adjacency")
        data["classes"].append(entry)
    data["vfree_control"] = vfree_control(12, bc)
    data["parent_identity"] = parent_identity(bc)
    with open(OUT_JSON.format(bc=bc), "w") as f:
        json.dump(data, f, indent=1)
    return data


def load(bc: str = BC) -> Optional[Dict]:
    p = OUT_JSON.format(bc=bc)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


# --- 6. figures --------------------------------------------------------------

_COL = {"rho2_phi": "#1f4e9c", "phi_4fifth": "#c62828",
        "psi_phi": "#7b3fa0", "rho_cbrt2": "#d98324"}
_MARK = {"rho2_phi": "o", "phi_4fifth": "s", "psi_phi": "^", "rho_cbrt2": "D"}


def _style(ax):
    ax.grid(True, color="#e9e9e6", linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _rep(cls: Dict) -> Dict:
    """One representative per class: rules inside a class share their series
    exactly (they are reflections or parent/child pairs), so plotting all of
    them thickens a line and repeats a legend entry."""
    return cls["profiles"][-1]


def fig_topology(bc: str, out: str, d: Optional[Dict] = None):
    d = d or load(bc) or build(bc)
    fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.2))

    # (a) size against density -- the "not a cycle" panel
    ax = axes[0]
    _style(ax)
    for cls in d["classes"]:
        rows = _rep(cls)["largest"]
        ax.plot([r["n"] for r in rows], [r["density"] for r in rows],
                marker=_MARK[cls["key"]], ms=5, lw=1.4, color=_COL[cls["key"]],
                label=f"$W_{{{_rep(cls)['rule']}}}$: {cls['a_name']}, {cls['b_name']}")
    ax.axhline(1.0, color="#444444", ls=":", lw=0.9)
    ax.annotate("complete digraph", (0.03, 1.0),
                xycoords=("axes fraction", "data"), fontsize=7.4,
                va="bottom", color="#444444")
    # a cycle on n nodes has density 1/n -- the comparison the question asks for
    nn = np.array([4, 3000], float)
    ax.plot(nn, 1.0 / nn, color="#6f6f6f", ls="--", lw=1.3)
    ax.annotate("a cycle: $1/n$", (700, 1 / 700), fontsize=7.4, color="#6f6f6f",
                xytext=(-6, 8), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$n$, states in the largest attractor")
    ax.set_ylabel("edge density $m/n^2$")
    ax.set_title("(a)  none of them is a cycle", fontsize=9.5)
    ax.legend(fontsize=6.9, frameon=False, loc="lower left")

    # (b) diameter
    ax = axes[1]
    _style(ax)
    for cls in d["classes"]:
        rows = _rep(cls)["largest"]
        ax.plot([r["n"] for r in rows], [r["diameter"] for r in rows],
                marker=_MARK[cls["key"]], ms=5, lw=1.4, color=_COL[cls["key"]])
    ax.plot(nn, (nn - 1) / 2, color="#6f6f6f", ls="--", lw=1.3)
    ax.annotate("a cycle: $\\sim n/2$", (300, 150), fontsize=7.4,
                color="#6f6f6f", rotation=32)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.7, 3000)
    ax.set_xlabel("$n$, states in the largest attractor")
    ax.set_ylabel("directed diameter")
    ax.set_title("(b)  the diameter does not grow", fontsize=9.5)

    # (c) spectral gap
    ax = axes[2]
    _style(ax)
    for cls in d["classes"]:
        rows = [r for r in _rep(cls)["largest"] if r["slem"] is not None]
        ax.plot([r["n"] for r in rows], [r["slem"] for r in rows],
                marker=_MARK[cls["key"]], ms=5, lw=1.4, color=_COL[cls["key"]])
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.02)
    ax.set_xlabel("$n$, states in the largest attractor")
    ax.set_ylabel(r"$|\lambda_2|$ of the row-normalised adjacency")
    ax.set_title("(c)  the gap does not close", fontsize=9.5)

    # (d) out-degree distribution at the largest N computed
    ax = axes[3]
    _style(ax)
    for cls in d["classes"]:
        p = _rep(cls)
        r = p["largest"][-1]
        G = largest_terminal(p["rule"], r["N"], bc)
        od = np.array(sorted((v for _, v in G.out_degree()), reverse=True),
                      dtype=float)
        ax.plot(np.arange(1, len(od) + 1) / len(od), od / len(od),
                marker=_MARK[cls["key"]], ms=3.2, lw=1.2,
                color=_COL[cls["key"]],
                label=f"$W_{{{p['rule']}}}$, $N={r['N']}$, $n={r['n']}$")
    ax.set_xlabel("rank / $n$")
    ax.set_ylabel("out-degree / $n$")
    ax.set_ylim(0, 1.05)
    ax.set_title("(d)  degree profile, not a constant", fontsize=9.5)
    ax.legend(fontsize=6.9, frameon=False, loc="lower left")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


# --- 7. tables ---------------------------------------------------------------

def write_tables(bc: str, outdir: str, d: Optional[Dict] = None):
    d = d or load(bc) or build(bc)
    os.makedirs(outdir, exist_ok=True)

    rows = []
    for cls in d["classes"]:
        p = _rep(cls)
        r = p["largest"][-1]
        gap = "---" if r["slem"] is None else f"{r['slem']:.4f}"
        rows.append(
            f"{cls['a_name']}, {cls['b_name']} & $W_{{{p['rule']}}}$ & "
            f"\\rt{{{p['tuple']}}} & {r['N']} & {r['n']} & {r['m']} & "
            f"{r['density']:.3f} & {r['diameter']} & {r['mean_distance']:.3f} & "
            f"{r['reciprocity']:.3f} & {gap} & "
            f"{'yes' if r['is_complete_with_loops'] else 'no'} \\\\")
    tab = ("\\begin{tabular}{lllrrrrrrrrc}\n\\toprule\n"
           "class $(a_\\att,b_\\att)$ & rule & tuple & $N$ & $n$ & $m$ & "
           "$m/n^2$ & diam & $\\bar d$ & recip. & $|\\lambda_2|$ & complete \\\\\n"
           "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    with open(os.path.join(outdir, f"tab_r17_topology_{bc}.tex"), "w") as f:
        f.write(tab)

    # the shape census: what fraction of ALL terminal SCCs is complete/cycle
    crows = []
    for cls in d["classes"]:
        for p in cls["profiles"]:
            c = p["census"][-1]
            sh = c["shapes"]
            crows.append(
                f"$W_{{{p['rule']}}}$ & \\rt{{{p['tuple']}}} & {c['N']} & "
                f"{c['n_attractors']} & {sh.get('singleton', 0)} & "
                f"{c['n_nontrivial']} & {sh.get('complete', 0)} & "
                f"{sh.get('cycle', 0)} & {sh.get('other', 0)} & "
                f"{'yes' if c['all_complete'] else 'no'} \\\\")
    ctab = ("\\begin{tabular}{llrrrrrrrc}\n\\toprule\n"
            "rule & tuple & $N$ & attractors & fixed pts & $>1$ state & "
            "complete & cycles & other & all complete \\\\\n\\midrule\n"
            + "\n".join(crows) + "\n\\bottomrule\n\\end{tabular}\n")
    with open(os.path.join(outdir, f"tab_r17_shapes_{bc}.tex"), "w") as f:
        f.write(ctab)

    prows = []
    for pi in d["parent_identity"]:
        r = pi["rows"][-1]
        prows.append(
            f"$W_{{{pi['child']}}}$ & \\rt{{{pi['child_tuple']}}} & "
            f"$W_{{{pi['parent']}}}$ & \\rt{{{pi['parent_tuple']}}} & {r['N']} & "
            f"{r['child_attractors']} & {r['parent_sectors']} & "
            f"{'yes' if r['child_sets_are_parent_sectors'] else 'no'} & "
            f"{'yes' if r['graph_identical'] else 'no'} & "
            f"{r['parent_complete_sectors']} & "
            f"{'all' if r['child_is_all_complete_sectors'] else ('subset' if r['child_inside_complete_sectors'] else '---')} \\\\")
    ptab = ("\\begin{tabular}{llllrrrccrc}\n\\toprule\n"
            "child & & parent & & $N$ & attractors & sectors & same sets & "
            "same edges & parent complete & child vs those \\\\\n\\midrule\n"
            + "\n".join(prows) + "\n\\bottomrule\n\\end{tabular}\n")
    with open(os.path.join(outdir, f"tab_r17_parents_{bc}.tex"), "w") as f:
        f.write(ptab)

    v = d["vfree_control"]
    hist = ", ".join(f"{k}: {n}" for k, n in sorted(v["histogram"].items(),
                                                    key=lambda x: int(x[0])))
    vt = ("\\begin{tabular}{lr}\n\\toprule\n"
          "V-free control at $N=" + str(v["N"]) + "$ & value \\\\\n\\midrule\n"
          f"rules & {v['n_rules']} \\\\\n"
          f"out-degree $1$ everywhere & "
          f"{'all' if v['all_deterministic'] else 'NOT all'} \\\\\n"
          f"longest cycle anywhere & {v['longest_cycle']} \\\\\n"
          f"rules with fixed points only & {v['fixed_point_only']} \\\\\n"
          f"cycle-length histogram & {hist} \\\\\n"
          "\\bottomrule\n\\end{tabular}\n")
    with open(os.path.join(outdir, f"tab_r17_vfree_{bc}.tex"), "w") as f:
        f.write(vt)
    return tab


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(
        description="topology of the exponential-exponential terminal SCCs")
    ap.add_argument("--bc", default=BC)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--n-max", type=int, default=max(NS))
    args = ap.parse_args(argv)
    Ns = tuple(n for n in NS if n <= args.n_max)
    d = build(args.bc, Ns) if args.rebuild else (load(args.bc) or
                                                 build(args.bc, Ns))
    for cls in d["classes"]:
        p = _rep(cls)
        r = p["largest"][-1]
        gap = "--" if r["slem"] is None else f"{r['slem']:.4f}"
        print(f"[{cls['key']:11s}] W{p['rule']:<4d} {p['tuple']}  N={r['N']} "
              f"n={r['n']:5d} m={r['m']:8d} dens={r['density']:.4f} "
              f"diam={r['diameter']} recip={r['reciprocity']:.3f} "
              f"slem={gap} complete={r['is_complete_with_loops']} "
              f"cycle={r['is_cycle']}")
    v = d["vfree_control"]
    print(f"V-free control: {v['n_rules']} rules, all deterministic="
          f"{v['all_deterministic']}, longest cycle={v['longest_cycle']}, "
          f"fixed-point-only={v['fixed_point_only']}")
    for pi in d["parent_identity"]:
        last = pi["rows"][-1]
        print(f"W{pi['child']:<4d} -> W{pi['parent']}: attractors ARE parent "
              f"sectors={last['child_sets_are_parent_sectors']} "
              f"same edges={last['graph_identical']} "
              f"({last['child_attractors']} of {last['parent_sectors']}); "
              f"inside parent's complete sectors="
              f"{last['child_inside_complete_sectors']} "
              f"(all of them={last['child_is_all_complete_sectors']})")
    figdir = os.path.join(results_io.REPO_ROOT, "figures")
    os.makedirs(figdir, exist_ok=True)
    fig_topology(args.bc, os.path.join(figdir,
                                       f"fig_attractor_topology_{args.bc}"), d)
    write_tables(args.bc, os.path.join(results_io.REPO_ROOT, "reports", "tex"), d)
    print("figure + tables written")


if __name__ == "__main__":
    main()
