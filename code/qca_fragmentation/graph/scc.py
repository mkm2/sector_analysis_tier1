"""
Directed-graph analysis of the one-cycle transition graph.

Context Tier 1 sec.4.  Nodes are bitmasks 0..2^N-1; edges are implicit via
succ(x) (streamed, never materialised).  We run ITERATIVE Tarjan (no recursion)
to obtain strongly connected components (Krylov sectors for unitary rules;
recurrent classes / attractors for dissipative rules), the condensation DAG,
terminal SCCs, basins, and transient depth.

The graph is DIRECTED and is never symmetrised by hand: for real-orthogonal
Hadamard circuits <y|U|x> != 0 does not imply <x|U|y> != 0.  SCC on the directed
graph is the correct treatment; for unitary (doubly stochastic) rules every SCC
is terminal, which we assert as a runtime check.

Ergodic early-exit: if any SCC exceeds a fraction f_erg (default 0.5) of 2^N the
rule is flagged "effectively ergodic at this N"; enumeration stops.  This is a
classification outcome, not a failure.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from ..core.cycle import succ as cycle_succ
from ..core.rules import Tuple4, is_unitary


@dataclass
class GraphResult:
    N: int
    rule: int
    bc: str
    ergodic: bool
    # component structure (present when not ergodic)
    n_scc: int = 0
    n_recurrent: int = 0
    sizes_scc: List[int] = field(default_factory=list)          # all SCCs, desc
    sizes_recurrent: List[int] = field(default_factory=list)    # terminal, desc
    sizes_basins: List[int] = field(default_factory=list)       # exclusive, desc
    shared_basin_size: int = 0
    transient_depth: int = 0
    n_transient_scc: int = 0
    # bookkeeping
    ergodic_bound: int = 0        # size of the SCC that triggered early-exit
    f_erg: float = 0.5
    comp_id: Optional[array] = None   # per-node SCC id (kept for Tier 1b reuse)


def _make_succ(rule: int, N: int, bc: str, t: Tuple4):
    def f(x: int) -> List[int]:
        return cycle_succ(x, N, t, bc)
    return f


def tarjan(
    N: int,
    succ_fn: Callable[[int], List[int]],
    *,
    f_erg: float = 0.5,
    detect_ergodic: bool = True,
    node_budget: Optional[int] = None,
) -> Tuple[Optional[array], List[int], bool, int]:
    """
    Iterative Tarjan over the implicit directed graph on 0..2^N-1.

    Returns (comp_id, comp_sizes, ergodic, ergodic_bound).  comp_id[node] is the
    SCC index; components are numbered in the order Tarjan finalises them
    (reverse-topological: a component is finalised before its predecessors).
    If ergodic early-exit triggers (an SCC exceeds f_erg*2^N, or the number of
    explored nodes exceeds node_budget) returns (None, [], True, bound).
    """
    total = 1 << N
    erg_thresh = int(f_erg * total)

    index = array("q", bytes(8 * total))     # 0 => unvisited (we use idx+1)
    lowlink = array("q", bytes(8 * total))
    onstack = bytearray(total)
    comp_id = array("q", [-1]) * total
    comp_sizes: List[int] = []

    tarjan_stack: List[int] = []
    counter = 1  # 1-based so that index==0 means unvisited

    for root in range(total):
        if index[root]:
            continue
        # explicit DFS stack: entries are [node, neighbors, ptr]
        work: List[list] = [[root, None, 0]]
        while work:
            frame = work[-1]
            node = frame[0]
            if frame[1] is None:
                # first visit
                index[node] = counter
                lowlink[node] = counter
                counter += 1
                if node_budget is not None and counter > node_budget:
                    return None, [], True, node_budget
                tarjan_stack.append(node)
                onstack[node] = 1
                frame[1] = succ_fn(node)
            neighbors = frame[1]
            ptr = frame[2]
            advanced = False
            while ptr < len(neighbors):
                w = neighbors[ptr]
                ptr += 1
                if not index[w]:
                    frame[2] = ptr
                    work.append([w, None, 0])
                    advanced = True
                    break
                elif onstack[w]:
                    if index[w] < lowlink[node]:
                        lowlink[node] = index[w]
            if advanced:
                continue
            frame[2] = ptr
            # done exploring node's neighbors
            if lowlink[node] == index[node]:
                # node is an SCC root -> pop component
                cid = len(comp_sizes)
                size = 0
                while True:
                    w = tarjan_stack.pop()
                    onstack[w] = 0
                    comp_id[w] = cid
                    size += 1
                    if w == node:
                        break
                comp_sizes.append(size)
                if detect_ergodic and size > erg_thresh:
                    return None, [], True, size
            work.pop()
            if work:
                parent = work[-1][0]
                if lowlink[node] < lowlink[parent]:
                    lowlink[parent] = lowlink[node]

    return comp_id, comp_sizes, False, 0


def sectors_union_find(
    N: int,
    succ_fn: Callable[[int], List[int]],
    *,
    f_erg: float = 0.5,
    detect_ergodic: bool = True,
    node_budget: Optional[int] = None,
):
    """
    Sector sizes for a UNITARY rule via union-find over streamed edges.

    Valid because a unitary (doubly stochastic) channel makes every state
    recurrent, so each weakly connected component is a single SCC; the sector
    partition equals the connected components of the undirected graph.  This is
    the memory-light path (O(2^N) ints, no stored adjacency, no DFS stack) used
    for the overnight sweep and mirrors the HSF oracle's method.

    Returns (sizes_desc, ergodic, max_size).  Early-exits when a component
    exceeds f_erg*2^N (returns (None, True, size)).
    """
    total = 1 << N
    parent = array("q", range(total))
    size = array("q", [1]) * total
    erg_thresh = int(f_erg * total)
    max_size = 1

    def find(a: int) -> int:
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:
            parent[a], a = root, parent[a]
        return root

    for x in range(total):
        rx = find(x)
        for y in succ_fn(x):
            ry = find(y)
            if rx != ry:
                if size[rx] < size[ry]:
                    rx, ry = ry, rx
                parent[ry] = rx
                size[rx] += size[ry]
                if size[rx] > max_size:
                    max_size = size[rx]
        if detect_ergodic and max_size > erg_thresh:
            return None, True, max_size

    sizes = [size[r] for r in range(total) if parent[r] == r]
    sizes.sort(reverse=True)
    return sizes, False, max_size


def _condensation(
    N: int, succ_fn: Callable[[int], List[int]], comp_id: array, n_comp: int
):
    """Second streaming pass: build condensation adjacency (successors only)."""
    succ_comps: List[set] = [set() for _ in range(n_comp)]
    total = 1 << N
    for x in range(total):
        cu = comp_id[x]
        for y in succ_fn(x):
            cv = comp_id[y]
            if cv != cu:
                succ_comps[cu].add(cv)
    return succ_comps


def _analyze_condensation(comp_sizes: List[int], succ_comps: List[set]):
    """
    Terminal SCCs, transient depth (longest path), and basin classification.

    Basins: reverse-topological DP tracking, per component, a single reachable
    terminal id and a 'multiple' flag (a component whose forward set reaches >1
    terminal is 'shared').  Basin size of terminal t = sum of |c| over components
    reaching exactly t.
    """
    n = len(comp_sizes)
    terminal = [len(succ_comps[c]) == 0 for c in range(n)]

    # topological order via Kahn on the condensation DAG
    indeg = [0] * n
    for c in range(n):
        for d in succ_comps[c]:
            indeg[d] += 1
    from collections import deque
    q = deque([c for c in range(n) if indeg[c] == 0])
    topo: List[int] = []
    while q:
        c = q.popleft()
        topo.append(c)
        for d in succ_comps[c]:
            indeg[d] -= 1
            if indeg[d] == 0:
                q.append(d)
    assert len(topo) == n, "condensation is not a DAG (bug)"

    # longest path (edges) = transient depth
    dist = [0] * n
    for c in reversed(topo):  # reverse topo so successors processed first
        best = 0
        for d in succ_comps[c]:
            if dist[d] + 1 > best:
                best = dist[d] + 1
        dist[c] = best
    transient_depth = max(dist) if dist else 0

    # basin classification: (rep_terminal, multiple_flag) per component
    rep = [-1] * n
    multiple = [False] * n
    for c in reversed(topo):
        if terminal[c]:
            rep[c] = c
            multiple[c] = False
            continue
        r = -1
        mult = False
        for d in succ_comps[c]:
            if multiple[d]:
                mult = True
            elif rep[d] != -1:
                if r == -1:
                    r = rep[d]
                elif r != rep[d]:
                    mult = True
        rep[c] = r
        multiple[c] = mult

    basin_of_terminal: Dict[int, int] = {c: 0 for c in range(n) if terminal[c]}
    shared = 0
    for c in range(n):
        if multiple[c]:
            shared += comp_sizes[c]
        else:
            basin_of_terminal[rep[c]] += comp_sizes[c]

    return terminal, transient_depth, basin_of_terminal, shared


def recurrent_classes(rule: int, N: int, bc: str, t: Tuple4) -> List[List[int]]:
    """
    Return the member state lists of the recurrent classes (terminal SCCs) of
    the directed transition graph.  Used by Tier 1b (attractor classification).
    """
    succ_fn = _make_succ(rule, N, bc, t)
    comp_id, comp_sizes, ergodic, _ = tarjan(N, succ_fn, detect_ergodic=False)
    n_comp = len(comp_sizes)
    succ_comps = _condensation(N, succ_fn, comp_id, n_comp)
    terminal = [len(succ_comps[c]) == 0 for c in range(n_comp)]
    members: List[List[int]] = [[] for _ in range(n_comp)]
    for x in range(1 << N):
        members[comp_id[x]].append(x)
    return [members[c] for c in range(n_comp) if terminal[c]]


def analyze(
    rule: int,
    N: int,
    bc: str,
    t: Tuple4,
    *,
    f_erg: float = 0.5,
    detect_ergodic: bool = True,
    node_budget: Optional[int] = None,
    keep_comp_id: bool = False,
) -> GraphResult:
    """Full Tier-1a analysis of one (rule, N, bc) unit."""
    succ_fn = _make_succ(rule, N, bc, t)
    unitary = is_unitary(t)

    if unitary and not keep_comp_id:
        # Doubly stochastic => SCC == weakly connected component.  Use the
        # memory-light union-find path (O(2^N) ints, one streamed succ pass;
        # mirrors the HSF oracle).  All sectors terminal, transient depth 0.
        sizes, ergodic, max_size = sectors_union_find(
            N, succ_fn, f_erg=f_erg, detect_ergodic=detect_ergodic,
            node_budget=node_budget,
        )
        if ergodic:
            return GraphResult(N=N, rule=rule, bc=bc, ergodic=True,
                               ergodic_bound=max_size, f_erg=f_erg)
        n_comp = len(sizes)
        return GraphResult(
            N=N, rule=rule, bc=bc, ergodic=False,
            n_scc=n_comp, n_recurrent=n_comp,
            sizes_scc=sizes, sizes_recurrent=list(sizes),
            sizes_basins=list(sizes), shared_basin_size=0,
            transient_depth=0, n_transient_scc=0, f_erg=f_erg,
        )

    comp_id, comp_sizes, ergodic, bound = tarjan(
        N, succ_fn, f_erg=f_erg, detect_ergodic=detect_ergodic,
        node_budget=node_budget,
    )
    if ergodic:
        return GraphResult(N=N, rule=rule, bc=bc, ergodic=True,
                           ergodic_bound=bound, f_erg=f_erg)

    n_comp = len(comp_sizes)

    if unitary:
        # keep_comp_id path: Tarjan ran; all SCCs terminal (asserted elsewhere).
        sizes_scc = sorted(comp_sizes, reverse=True)
        res = GraphResult(
            N=N, rule=rule, bc=bc, ergodic=False,
            n_scc=n_comp, n_recurrent=n_comp,
            sizes_scc=sizes_scc, sizes_recurrent=list(sizes_scc),
            sizes_basins=list(sizes_scc), shared_basin_size=0,
            transient_depth=0, n_transient_scc=0, f_erg=f_erg,
        )
        res.comp_id = comp_id
        return res

    succ_comps = _condensation(N, succ_fn, comp_id, n_comp)
    terminal, depth, basin_of_terminal, shared = _analyze_condensation(
        comp_sizes, succ_comps
    )

    sizes_scc = sorted(comp_sizes, reverse=True)
    rec_sizes = sorted((comp_sizes[c] for c in range(n_comp) if terminal[c]),
                       reverse=True)
    basin_sizes = sorted(basin_of_terminal.values(), reverse=True)

    res = GraphResult(
        N=N, rule=rule, bc=bc, ergodic=False,
        n_scc=n_comp,
        n_recurrent=sum(terminal),
        sizes_scc=sizes_scc,
        sizes_recurrent=rec_sizes,
        sizes_basins=basin_sizes,
        shared_basin_size=shared,
        transient_depth=depth,
        n_transient_scc=n_comp - sum(terminal),
        f_erg=f_erg,
    )
    if keep_comp_id:
        res.comp_id = comp_id
    return res
