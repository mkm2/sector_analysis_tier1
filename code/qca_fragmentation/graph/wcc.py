"""
Weakly connected components of the one-cycle transition graph (Tier 1e).

WHY THIS IS THE SECTOR-LEVEL OBJECT.  The directed analysis in scc.py reports
terminal SCCs ("recurrent classes").  Those are the right object for the
MONITORED circuit, but they are the wrong object for a sector axis:

  (P1) terminal SCCs do not partition the basis -- transient states belong to no
       recurrent class, so sum(sizes_recurrent) << 2^N and no sum rule holds;
  (P2) they are monitored-relative -- the transient/recurrent split is a
       property of the measured chain, and the unmonitored channel can keep
       immortal weight on transient states (R5 rule 22, R7);
  (P3) the Tier-2 wall grammar predicts SECTORS, not attractors.

A weakly connected component fixes all three.  It is the finest decomposition of
the basis into sets closed under the one-cycle support relation in BOTH
directions, so span(WCC) is an ENCLOSURE of the channel -- invariant under every
Kraus operator, hence exact for the monitored and the unmonitored dynamics
alike.  WCCs partition the basis, they are the true analogue of the unitary
Krylov sectors, and they are the minimal projectors of the DIAGONAL part of the
commutant of the Kraus set (cf. R8 sec.10.3: a diagonal D commutes with the
dynamics iff it is constant on the components of the support graph).

Vocabulary, used consistently in code, records and figures:

    sector              = weakly connected component      (both experiments)
    monitored attractor = terminal SCC                    (measured circuit)
    channel attractor   = block of the fixed-point algebra (Tier 1b / 1d)

Never call a terminal SCC a "sector" and never call a WCC an "attractor".

ALGORITHM.  One streaming pass over succ(x) for x = 0..2^N-1, union-find with
union-by-size and path compression, every edge treated as UNDIRECTED (union(x,y)
for every y in succ(x)).  No adjacency storage, no DFS stack.  This is the same
code path that scc.sectors_union_find already used for unitary rules; it lives
here now and both callers share it.

Union-find is far cheaper in memory than the directed Tarjan (which is what caps
the dissipative sweep at N = 13..17), so the sector axis for dissipative rules
can reach the unitary ceiling.  The Tarjan node_budget is deliberately NOT
inherited here.
"""

from __future__ import annotations

import time
from array import array
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from ..core import rules as rules_mod
from ..core.cycle import succ as cycle_succ
from ..core.rules import Tuple4


@dataclass
class WCCResult:
    N: int
    rule: int
    bc: str
    #: one component holds more than f_erg * 2^N.  This is a CLASSIFICATION,
    #: not a failure: the decomposition below is still complete unless
    #: `aborted` is set.  The sweep uses it to stop enlarging N.
    ergodic: bool
    n_wcc: int = 0
    sizes_wcc: List[int] = field(default_factory=list)   # descending, complete
    d_max_wcc: int = 0
    n_frozen: int = 0            # number of size-1 sectors (frozen states)
    ergodic_bound: int = 0
    #: True only when the scan was cut short, so the partition is INCOMPLETE
    #: and the sum rule A1 must not be applied.  Off by default: the Tier-1e
    #: pass wants exact n_wcc and d_max for the hyperbola anchors (V2), and
    #: union-find is cheap enough to always finish the pass.
    aborted: bool = False
    f_erg: float = 0.5
    runtime: float = 0.0


def union_find_components(
    N: int,
    succ_fn: Callable[[int], List[int]],
    *,
    f_erg: float = 0.5,
    detect_ergodic: bool = True,
) -> Tuple[Optional[List[int]], bool, int]:
    """
    Undirected connected components of the streamed successor graph.

    Returns (sizes_desc, ergodic, max_size).  On the ergodic early exit the
    sizes are None and max_size is the component that tripped the bound -- the
    partition is then incomplete on purpose, so callers must not apply the sum
    rule to it.

    Every edge is symmetrised by construction: we union(x, y) for each
    y in succ(x), and union is a symmetric operation.  Do NOT be tempted to
    reuse the directed successor sets without this.
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


def make_succ(rule: int, N: int, bc: str, t: Optional[Tuple4] = None):
    if t is None:
        t = rules_mod.wolfram_to_tuple(rule)

    def f(x: int) -> List[int]:
        return cycle_succ(x, N, t, bc)
    return f


def weak_components(rule: int, N: int, bc: str, t: Optional[Tuple4] = None, *,
                    f_erg: float = 0.5,
                    detect_ergodic: bool = False) -> WCCResult:
    """
    The Tier-1e sector decomposition of one (rule, N, bc) unit.

    `detect_ergodic` defaults to FALSE here, unlike the Tier-1a Tarjan path.
    Aborting the scan would leave an incomplete partition, which would break the
    A1 sum rule and would cost us the exact anchors the hyperbola validation
    needs (V2: rule 51 must come out as exactly one sector of size 2^N, not as
    "some component exceeded half the space").  Union-find finishes the pass
    cheaply, so we always do, and report `ergodic` as a classification derived
    from d_max.  The exponential saving lives in the sweep instead: once a rule
    is ergodic at some N, larger N are skipped.
    """
    if t is None:
        t = rules_mod.wolfram_to_tuple(rule)
    t0 = time.time()
    sizes, aborted, max_size = union_find_components(
        N, make_succ(rule, N, bc, t), f_erg=f_erg,
        detect_ergodic=detect_ergodic)
    dt = time.time() - t0
    if aborted:
        return WCCResult(N=N, rule=rule, bc=bc, ergodic=True, aborted=True,
                         ergodic_bound=max_size, f_erg=f_erg, runtime=dt)
    # A1: the sum rule.  A violation means the edge stream dropped states.
    assert sum(sizes) == (1 << N), (
        f"A1 violated for W{rule} N={N} {bc}: sum(sizes_wcc)={sum(sizes)} "
        f"!= 2^N={1 << N}")
    d_max = sizes[0]
    return WCCResult(
        N=N, rule=rule, bc=bc, ergodic=bool(d_max > f_erg * (1 << N)),
        n_wcc=len(sizes), sizes_wcc=sizes, d_max_wcc=d_max,
        n_frozen=sum(1 for s in sizes if s == 1),
        ergodic_bound=d_max, f_erg=f_erg, runtime=dt)


# --- cross-tier consistency assertions (task sec.2) --------------------------

def check_against_scc(res: WCCResult, scc_rec: dict,
                      *, unitary: Optional[bool] = None) -> Dict[str, object]:
    """
    A2/A3/A4 against an existing Tier-1a record.

      A2  unitary rules: n_wcc == n_recurrent == n_scc and the size multisets
          agree elementwise (doubly stochastic => weak, strong and
          forward-closure partitions coincide).  A mismatch is an engine bug.
      A3  n_wcc <= n_scc.  (Containment of each terminal SCC in exactly one WCC
          is guaranteed structurally: an SCC is connected in the directed graph,
          hence connected in the undirected one.)
      A4  NEITHER n_recurrent >= n_wcc nor n_recurrent <= n_wcc holds in
          general.  Assert only that every WCC contains at least one terminal
          SCC -- which, since the condensation of a finite digraph always has a
          terminal node reachable from every node, reduces to
          n_recurrent >= 1 whenever the space is non-empty, and is checked
          quantitatively as n_recurrent >= n_wcc only for the unitary case.

    Returns a dict of the checks performed; raises AssertionError on failure.
    """
    if unitary is None:
        unitary = rules_mod.is_unitary(rules_mod.wolfram_to_tuple(res.rule))
    out: Dict[str, object] = {"a2": None, "a3": None, "a4": None}
    if res.aborted or scc_rec.get("ergodic_flag"):
        return out            # incomplete partition on one side: nothing to compare
    n_scc = scc_rec.get("n_scc")
    n_rec = scc_rec.get("n_recurrent")
    if n_scc is not None:
        assert res.n_wcc <= n_scc, (
            f"A3 violated for W{res.rule} N={res.N} {res.bc}: "
            f"n_wcc={res.n_wcc} > n_scc={n_scc}")
        out["a3"] = True
    if n_rec is not None:
        # A4: a WCC always holds >= 1 terminal SCC, so there are at least as
        # many terminal SCCs as WCCs.  (The converse bound is false: one WCC
        # may hold several terminal SCCs.)
        assert n_rec >= res.n_wcc, (
            f"A4 violated for W{res.rule} N={res.N} {res.bc}: "
            f"n_recurrent={n_rec} < n_wcc={res.n_wcc}, so some WCC contains "
            f"no terminal SCC")
        out["a4"] = True
    if unitary and n_scc is not None and n_rec is not None:
        from ..results_io import sizes_from_record
        assert res.n_wcc == n_rec == n_scc, (
            f"A2 violated for unitary W{res.rule} N={res.N} {res.bc}: "
            f"n_wcc={res.n_wcc}, n_recurrent={n_rec}, n_scc={n_scc}")
        ref = sizes_from_record(scc_rec, "sizes_recurrent")
        if ref:
            assert list(res.sizes_wcc) == list(ref), (
                f"A2 violated for unitary W{res.rule} N={res.N} {res.bc}: "
                f"sector size multisets differ")
        out["a2"] = True
    return out
