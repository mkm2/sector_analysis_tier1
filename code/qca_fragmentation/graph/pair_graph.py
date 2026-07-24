"""
Tier 1d: the doubled-space transition graph ("pair graph" G2).

Context: task_tier1d_pair_graph.md.  The Tier-1a graph M is the exact dynamics of
the MONITORED QCA (dephase in the computational basis every cycle).  For the
UNMONITORED channel Phi the population attractor need not be basis-spanned; the
pair graph closes that gap.

Nodes of G2 are ordered pairs (x, y) of basis states.  There is an edge
(x, y) -> (x', y') iff some global Kraus operator K_b has BOTH
<x'|K_b|x> != 0 and <y'|K_b|y> != 0.  The global Kraus operator is named by the
frozenset of sites at which the A1 (jump) branch fired -- exactly the label that
``core.cycle.one_cycle_branches_labeled`` already attaches to each surviving
branch.  So the pair join is a join on IDENTICAL Kraus labels: a label present
for x but absent for y contributes no edge (that is precisely the same-mu
requirement of the superoperator sum_b K_b (x) conj(K_b)).

Key structural facts (proved in R7, asserted here):

  * The diagonal subgraph {(x, x)} of G2 is exactly M.
  * (T1) span(S) is Phi-invariant iff S is closed under M-edges, so the weakly
    connected components (WCCs) of M are EXACT sectors of the unmonitored
    channel.  What does NOT transfer is the recurrent/transient split.
  * (T3) every peripheral eigenoperator A of Phi (|lambda|=1, incl. fixed
    points) has entrywise support inside P_rec := forward closure of the cyclic
    pairs of G2.  Hence dim Fix(Phi) <= |P_rec|.
  * A cyclic pair (self-loop or member of a G2-SCC of size > 1) projects to an
    M-cycle on each coordinate, so BOTH coordinates are M-recurrent; and a
    terminal SCC is M-closed, so the forward closure never leaves
    recurrent x recurrent.  Therefore P_rec is contained in Drec x Drec, where
    Drec is the set of M-recurrent states.  We build G2 on that small block.
  * (T4) if the off-diagonal part of P_rec is empty and its diagonal part equals
    the M-recurrent set, every peripheral eigenoperator is basis-diagonal and
    the Tier-1a attractor data is EXACT for the unmonitored channel:
    ``certified = True``.

By default we compute only the WCC-DIAGONAL blocks (x, y in the same sector);
these carry the per-sector attractor bounds and the certificate.  The
off-diagonal blocks (``cross_sector=True``) bound noiseless pairing of two
distinct sectors -- e.g. a stationary coherence between two frozen states in
different sectors -- and are needed for the FULL dim Fix(Phi) upper bound used by
the validation oracles at small N.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from ..core.cycle import one_cycle_branches_labeled, succ as cycle_succ
from ..core.rules import Tuple4, is_unitary
from .scc import tarjan, recurrent_classes


@dataclass
class PairResult:
    N: int
    rule: int
    bc: str
    cross_sector: bool
    n_recurrent_states: int          # |Drec|
    pair_rec_size: int               # |P_rec|
    pair_offdiag: int                # # (x,y) in P_rec with x != y
    pair_diag_extra: int             # # (x,x) in P_rec with x not M-recurrent
    fix_upper: int                   # |P_rec|  (upper bound on dim Fix Phi)
    certified: Optional[bool]        # T4 boolean (None if bounded-only)
    bounded_only: bool = False       # True iff the node budget was hit
    n_pair_nodes: int = 0            # size of the block actually enumerated
    runtime: float = 0.0
    # optional payloads (filled on request; not serialised by default)
    offdiag_pairs: Optional[List[Tuple[int, int]]] = field(default=None, repr=False)
    prec: Optional[Set[Tuple[int, int]]] = field(default=None, repr=False)


# --------------------------------------------------------------------------- #
# Labelled single-cycle supports: {Kraus label -> frozenset of output states}. #
# Exact: amplitudes are accumulated as integer (a, b) pairs of Z[1/sqrt2] and a #
# state is in the support iff its accumulated amplitude is non-zero.            #
# --------------------------------------------------------------------------- #

def labeled_supports(x: int, N: int, t: Tuple4, bc: str) -> Dict[frozenset, frozenset]:
    acc: Dict[frozenset, Dict[int, Tuple[int, int]]] = {}
    for amps, _m, lab in one_cycle_branches_labeled(x, N, t, bc):
        d = acc.get(lab)
        if d is None:
            acc[lab] = d = {}
        for y, (a, b) in amps.items():
            p = d.get(y)
            d[y] = (a, b) if p is None else (p[0] + a, p[1] + b)
    out: Dict[frozenset, frozenset] = {}
    for lab, d in acc.items():
        s = frozenset(y for y, (a, b) in d.items() if a or b)
        if s:
            out[lab] = s
    return out


def succ2_from(sx: Dict[frozenset, frozenset],
               sy: Dict[frozenset, frozenset]):
    """Successor pairs of (x, y) given their labelled supports (join on label)."""
    for lab, fx in sx.items():
        fy = sy.get(lab)
        if not fy:
            continue
        for xp in fx:
            for yp in fy:
                yield xp, yp


# --------------------------------------------------------------------------- #
# Weakly connected components of M (the exact sectors, T1).                    #
# --------------------------------------------------------------------------- #

def wcc_labels(N: int, t: Tuple4, bc: str) -> List[int]:
    """Union-find over M-edges treated as undirected: WCC root per node."""
    from array import array
    total = 1 << N
    parent = array("q", range(total))

    def find(a: int) -> int:
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:
            parent[a], a = root, parent[a]
        return root

    for x in range(total):
        for y in cycle_succ(x, N, t, bc):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx
    return [find(i) for i in range(total)]


# --------------------------------------------------------------------------- #
# Generic iterative Tarjan over a materialised adjacency list (compact ids).   #
# --------------------------------------------------------------------------- #

def _tarjan_adj(adj: List[List[int]]):
    n = len(adj)
    index = [0] * n
    low = [0] * n
    onstack = bytearray(n)
    comp = [-1] * n
    comp_sizes: List[int] = []
    stack: List[int] = []
    counter = 1
    for root in range(n):
        if index[root]:
            continue
        work = [[root, 0]]
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        onstack[root] = 1
        while work:
            node, ptr = work[-1]
            neigh = adj[node]
            advanced = False
            while ptr < len(neigh):
                w = neigh[ptr]
                ptr += 1
                if not index[w]:
                    work[-1][1] = ptr
                    index[w] = low[w] = counter
                    counter += 1
                    stack.append(w)
                    onstack[w] = 1
                    work.append([w, 0])
                    advanced = True
                    break
                elif onstack[w] and index[w] < low[node]:
                    low[node] = index[w]
            if advanced:
                continue
            work[-1][1] = ptr
            if low[node] == index[node]:
                cid = len(comp_sizes)
                size = 0
                while True:
                    v = stack.pop()
                    onstack[v] = 0
                    comp[v] = cid
                    size += 1
                    if v == node:
                        break
                comp_sizes.append(size)
            work.pop()
            if work:
                parent = work[-1][0]
                if low[node] < low[parent]:
                    low[parent] = low[node]
    return comp, comp_sizes


# --------------------------------------------------------------------------- #
# Main analysis.                                                              #
# --------------------------------------------------------------------------- #

def analyze_pair(rule: int, N: int, bc: str, t: Tuple4, *,
                 cross_sector: bool = False,
                 node_budget: int = 6_000_000,
                 return_support: bool = False) -> PairResult:
    """Tier-1d pair-graph analysis of one (rule, N, bc) unit.

    ``cross_sector=False`` (default): the WCC-DIAGONAL blocks S x S (x, y in the
    same sector, transient states INCLUDED) -> per-sector attractor bounds and
    the T4 certificate.  ``cross_sector=True``: all pairs over 0..2^N-1 -> the
    full dim Fix(Phi) upper bound (used by the small-N oracles).

    P_rec is built on the full block, not just the recurrent states: a cyclic
    pair projects to an M-cycle on each coordinate, but that cycle can be a
    NON-terminal SCC, and the forward closure then flows downstream through
    M-transient states (the "immortal transient support", e.g. rule 22).  These
    show up as ``pair_diag_extra`` diagonal pairs (x, x) with x not M-recurrent.
    """
    t0 = time.time()
    if is_unitary(t):
        # Single Kraus label: G2 = M x M; nothing to certify beyond Tier 1a/R2.
        return PairResult(N=N, rule=rule, bc=bc, cross_sector=cross_sector,
                          n_recurrent_states=0, pair_rec_size=0, pair_offdiag=0,
                          pair_diag_extra=0, fix_upper=0, certified=True,
                          runtime=time.time() - t0)

    # M-recurrent states (union of terminal SCCs) -- for the certificate.
    classes = recurrent_classes(rule, N, bc, t)
    m_rec = set(x for cl in classes for x in cl)
    nrec = len(m_rec)

    total = 1 << N

    # Node block: pairs within a WCC (diagonal) or all pairs (cross-sector).
    if cross_sector:
        pairs = [(x, y) for x in range(total) for y in range(total)]
    else:
        wcc = wcc_labels(N, t, bc)
        groups: Dict[int, List[int]] = {}
        for x in range(total):
            groups.setdefault(wcc[x], []).append(x)
        pairs = [(x, y) for g in groups.values() for x in g for y in g]

    if len(pairs) > node_budget:
        return PairResult(N=N, rule=rule, bc=bc, cross_sector=cross_sector,
                          n_recurrent_states=nrec, pair_rec_size=-1,
                          pair_offdiag=-1, pair_diag_extra=-1, fix_upper=-1,
                          certified=None, bounded_only=True,
                          n_pair_nodes=len(pairs), runtime=time.time() - t0)

    pos = {p: i for i, p in enumerate(pairs)}
    firsts = set(x for x, _ in pairs)
    sup = {x: labeled_supports(x, N, t, bc) for x in firsts}

    # Materialise adjacency over the block (successors stay inside by T1).
    adj: List[List[int]] = [[] for _ in pairs]
    for i, (x, y) in enumerate(pairs):
        seen = set()
        for xp, yp in succ2_from(sup[x], sup[y]):
            j = pos.get((xp, yp))
            if j is not None and j not in seen:
                seen.add(j)
                adj[i].append(j)

    comp, comp_sizes = _tarjan_adj(adj)

    # Cyclic pairs: in an SCC of size > 1, OR carrying a self-loop.
    cyclic: List[int] = []
    for i in range(len(pairs)):
        if comp_sizes[comp[i]] > 1 or i in adj[i]:
            cyclic.append(i)

    # Forward closure of the cyclic pairs.
    reached = bytearray(len(pairs))
    stack = list(cyclic)
    for i in cyclic:
        reached[i] = 1
    while stack:
        i = stack.pop()
        for j in adj[i]:
            if not reached[j]:
                reached[j] = 1
                stack.append(j)

    prec = {pairs[i] for i in range(len(pairs)) if reached[i]}
    diag = {x for (x, y) in prec if x == y}
    offdiag = [(x, y) for (x, y) in prec if x != y]
    pair_diag_extra = len(diag - m_rec)

    certified = (len(offdiag) == 0 and diag == m_rec) if not cross_sector else None

    res = PairResult(
        N=N, rule=rule, bc=bc, cross_sector=cross_sector,
        n_recurrent_states=nrec, pair_rec_size=len(prec),
        pair_offdiag=len(offdiag), pair_diag_extra=pair_diag_extra,
        fix_upper=len(prec), certified=certified,
        n_pair_nodes=len(pairs), runtime=time.time() - t0,
    )
    if return_support:
        res.offdiag_pairs = offdiag
        res.prec = prec
    return res
