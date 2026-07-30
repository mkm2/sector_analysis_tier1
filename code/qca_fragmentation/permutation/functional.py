"""
Functional-graph analysis: WCC, cycles (terminal SCCs) and basins in one pass.

The X-gate circuits have out-degree exactly 1, and that collapses the whole
Tier-1e apparatus into a single linear-time sweep.  Three standard facts about a
functional graph do the work:

  (F1) Every weakly connected component contains exactly ONE cycle.  Hence
       n_recurrent = n_wcc: one monitored attractor per sector, always.
  (F2) The basin of that cycle is its entire component, so basin sizes ARE
       sector sizes and the shared basin is empty.
  (F3) Every non-cyclic node has a well-defined distance to the cycle, so the
       transient depth is just the deepest tree.

None of these hold for the Hadamard circuits, where succ(x) is set-valued: R9's
rule 22 has three attractors sitting inside two sectors, and A4 had to be
asserted in one direction only.  Here the two tiers coincide by construction,
which is exactly what makes this family a useful control.

The cycle-finding is the usual three-colour walk (0 unvisited / 1 on the current
path / 2 finished).  Each node is pushed on a path at most once, so the whole
sweep is O(2^N) time and O(2^N) memory in int32 arrays.
"""

from __future__ import annotations

import time
from array import array
from collections import Counter
from typing import Dict, List, Optional, Sequence


def analyze_map(f: Sequence[int], N: int) -> Dict:
    """
    Full decomposition of the functional graph x -> f[x] on 2^N nodes.

    Returns sector (== basin) sizes, cycle lengths, transient depth and the
    derived fractions.  `f` must be a sequence of length 2^N.
    """
    n = 1 << N
    assert len(f) == n, (len(f), n)
    colour = bytearray(n)                 # 0 unvisited, 1 on path, 2 finished
    comp = array("i", [-1]) * n           # which cycle the node flows to
    depth = array("i", [-1]) * n          # distance to that cycle
    cycle_len: List[int] = []

    for s in range(n):
        if colour[s]:
            continue
        path: List[int] = []
        x = s
        while colour[x] == 0:
            colour[x] = 1
            path.append(x)
            x = f[x]
        if colour[x] == 1:
            # a new cycle: the suffix of `path` starting at x
            k = path.index(x)
            cid = len(cycle_len)
            cyc = path[k:]
            cycle_len.append(len(cyc))
            for v in cyc:
                comp[v] = cid
                depth[v] = 0
                colour[v] = 2
            tail = path[:k]
        else:
            tail = path                    # ran into already-finished territory
        for v in reversed(tail):
            nx = f[v]
            comp[v] = comp[nx]
            depth[v] = depth[nx] + 1
            colour[v] = 2

    try:                                   # bincount is much faster at large N
        import numpy as np
        counts = np.bincount(np.frombuffer(comp, dtype=np.int32),
                             minlength=len(cycle_len))
        sector_sizes = sorted((int(c) for c in counts if c), reverse=True)
    except Exception:
        sizes = Counter(comp[x] for x in range(n))
        sector_sizes = sorted(sizes.values(), reverse=True)
    cyc_sorted = sorted(cycle_len, reverse=True)
    n_cyclic = sum(cycle_len)
    try:
        import numpy as np
        max_depth = int(np.frombuffer(depth, dtype=np.int32).max()) if n else 0
    except Exception:
        max_depth = max(depth[x] for x in range(n)) if n else 0

    assert sum(sector_sizes) == n, "sector sizes must partition the basis"
    assert len(sector_sizes) == len(cycle_len), \
        "functional graph: one cycle per weak component (F1)"

    return {
        "N": N,
        "n_wcc": len(sector_sizes),
        "sizes_wcc": sector_sizes,
        "d_max_wcc": sector_sizes[0],
        "n_frozen": sum(1 for s in sector_sizes if s == 1),
        "n_recurrent": len(cyc_sorted),
        "sizes_recurrent": cyc_sorted,
        "d_max_recurrent": cyc_sorted[0],
        "n_fixed_points": sum(1 for c in cycle_len if c == 1),
        "transient_depth": int(max_depth),
        "transient_fraction": 1.0 - n_cyclic / n,
        "d_max_ratio": sector_sizes[0] / n,
        "att_per_sector": len(cyc_sorted) / len(sector_sizes),
        "reversible": n_cyclic == n,
    }


def analyze(rule: int, N: int, bc: str, *, vectorised: bool = True) -> Dict:
    """
    Analyse one X-gate rule.

    The map itself is built with numpy (xca.step_table_np) and then handed to the
    walk as a plain list: scalar indexing in the three-colour walk is the hot
    path, and a list beats both array('i') and a numpy array there by a wide
    margin.  `vectorised=False` falls back to the scalar builder, which the tests
    use to check the two agree.
    """
    from .xca import step_table, step_table_np
    t0 = time.time()
    f = (step_table_np(rule, N, bc).tolist() if vectorised
         else list(step_table(rule, N, bc)))
    out = analyze_map(f, N)
    out.update({"rule": rule, "bc": bc, "runtime": time.time() - t0})
    return out


def cross_check_wcc(rule: int, N: int, bc: str, res: Optional[Dict] = None) -> Dict:
    """
    Independent check of (F1)/(F2): recompute the weak components with the
    Tier-1e union-find, which knows nothing about functional graphs, and require
    the same partition.
    """
    from ..graph.wcc import union_find_components
    from .xca import succ_fn
    if res is None:
        res = analyze(rule, N, bc)
    sizes, aborted, _ = union_find_components(N, succ_fn(rule, N, bc),
                                              detect_ergodic=False)
    assert not aborted
    ok = list(sizes) == list(res["sizes_wcc"])
    assert ok, (rule, N, bc, sizes[:8], res["sizes_wcc"][:8])
    return {"rule": rule, "N": N, "bc": bc, "agrees": ok,
            "n_wcc": len(sizes)}
