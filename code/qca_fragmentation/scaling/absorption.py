"""
Absorption probabilities of the dissipative rules  (Report R22).

The question R22 answers is whether the attractor a state ends in is DETERMINED
by the initial condition or merely WEIGHTED by it.  That is the operational
content of "strong symmetry" (Buca-Prosen) versus plain multistability, and by
R22 Prop. 2 it is decided by whether the conserved quantities

    J_i = sum_x phi_i(x) |x><x|,   phi_i(x) = P[absorbed in terminal class i | x]

are idempotent, i.e. whether every phi_i(x) is 0 or 1.  That in turn holds iff
every weakly connected component contains exactly one terminal component.

Unlike the rest of the project this module needs the transition PROBABILITIES,
not just the support graph: `graph/scc.py` works with the unweighted successor
relation, which is all the component structure needs.  Here the weights matter,
so the one-cycle Markov matrix is rebuilt from the gate semantics

    I -> keep,   V = H -> 1/2 to each branch,   D -> reset 0,   E -> reset 1

with the project's even-first brick-wall convention.  Cost is O(4^N); this is a
small-N diagnostic, not a sweep.  N <= 12 is comfortable, N = 14 is the edge.

CLI
    python -m qca_fragmentation.scaling.absorption --rules 203,73 --N 10
    python -m qca_fragmentation.scaling.absorption --table          # the R22 table
"""

from __future__ import annotations

import argparse
import collections
import json
import os
from typing import Dict, List, Tuple

import numpy as np

from ..core import rules
from ..graph import scc
from .. import results_io

BC_DEFAULT = "obc0"

# The rules R22 tabulates: two whose weak components proliferate (so the fate is
# forced) and two whose do not (so it is not).
R22_ROWS: Tuple[Tuple[int, int], ...] = (
    (73, 10), (109, 10), (36, 10), (203, 10), (203, 12),
)


def transition_matrix(rule: int, N: int, bc: str = BC_DEFAULT) -> np.ndarray:
    """P[y, x] = probability that one full brick-wall cycle takes x to y."""
    t = rules.wolfram_to_tuple(rule)
    dim = 1 << N

    def layer(vec: Dict[int, float], offset: int) -> Dict[int, float]:
        out: Dict[int, float] = collections.defaultdict(float)
        for x, p in vec.items():
            branches: List[Tuple[int, float]] = [(x, p)]
            for i in range(offset, N, 2):
                if i - 1 >= 0:
                    left = (x >> (i - 1)) & 1
                else:
                    left = 0 if bc == "obc0" else (x >> (N - 1)) & 1
                if i + 1 < N:
                    right = (x >> (i + 1)) & 1
                else:
                    right = 0 if bc == "obc0" else x & 1
                g = t[2 * left + right]
                nxt: List[Tuple[int, float]] = []
                for y, q in branches:
                    if g == "I":
                        nxt.append((y, q))
                    elif g == "V":
                        nxt.append((y & ~(1 << i), q * 0.5))
                        nxt.append((y | (1 << i), q * 0.5))
                    elif g == "D":
                        nxt.append((y & ~(1 << i), q))
                    else:                      # "E"
                        nxt.append((y | (1 << i), q))
                branches = nxt
            for y, q in branches:
                out[y] += q
        return out

    P = np.zeros((dim, dim))
    for x in range(dim):
        v = layer(layer({x: 1.0}, 0), 1)
        for y, q in v.items():
            P[y, x] += q
    return P


def absorption(rule: int, N: int, bc: str = BC_DEFAULT):
    """(phi, terminal_ids) with phi[x, i] = P[absorbed in terminal class i | x]."""
    t = rules.wolfram_to_tuple(rule)
    g = scc.analyze(rule, N, bc, t, detect_ergodic=False, keep_comp_id=True)
    comp = np.asarray(g.comp_id)
    P = transition_matrix(rule, N, bc)
    if not np.allclose(P.sum(axis=0), 1.0):
        raise AssertionError("columns of P must be probability distributions")

    dim = 1 << N
    leaves: Dict[int, set] = collections.defaultdict(set)
    for x in range(dim):
        for y in np.nonzero(P[:, x])[0]:
            if comp[y] != comp[x]:
                leaves[int(comp[x])].add(int(comp[y]))
    terminal = sorted({int(c) for c in comp} - set(leaves))
    tmap = {c: i for i, c in enumerate(terminal)}

    is_term = np.array([int(comp[x]) in tmap for x in range(dim)])
    phi = np.zeros((dim, len(terminal)))
    for x in range(dim):
        if is_term[x]:
            phi[x, tmap[int(comp[x])]] = 1.0

    trans = np.nonzero(~is_term)[0]
    if len(trans):
        A = np.eye(len(trans)) - P[np.ix_(trans, trans)].T
        B = np.zeros((len(trans), len(terminal)))
        for i, x in enumerate(trans):
            for y in np.nonzero(P[:, x])[0]:
                if is_term[y]:
                    B[i, tmap[int(comp[y])]] += P[y, x]
        phi[trans] = np.linalg.solve(A, B)
    return phi, terminal


def summary(rule: int, N: int, bc: str = BC_DEFAULT) -> dict:
    """The row R22 quotes: how often the initial state fixes the fate."""
    phi, terminal = absorption(rule, N, bc)
    if not np.allclose(phi.sum(axis=1), 1.0):
        raise AssertionError("absorption probabilities must sum to 1")
    certain = float(np.isclose(phi.max(axis=1), 1.0).mean())
    pr = 1.0 / (phi ** 2).sum(axis=1)
    w = results_io.load_wcc_results(rule, bc).get(N, {})
    return {
        "rule": rule, "N": N, "bc": bc,
        "tuple": "".join(rules.wolfram_to_tuple(rule)),
        "n_wcc": w.get("n_wcc"),
        "n_terminal": len(terminal),
        "certain_fraction": certain,
        "idempotent": bool(np.isclose(certain, 1.0)),
        "eff_attractors_mean": float(pr.mean()),
        "eff_attractors_max": float(pr.max()),
    }


def strong_symmetry_holds(rule: int, N: int, bc: str = BC_DEFAULT) -> bool:
    """R22 Prop. 2: the conserved quantities are projectors iff n_rec == n_wcc."""
    return summary(rule, N, bc)["idempotent"]


def build(rows=R22_ROWS, bc: str = BC_DEFAULT) -> List[dict]:
    return [summary(r, N, bc) for r, N in rows]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="R22 absorption-probability diagnostic")
    ap.add_argument("--rules", type=str, default=None, help="comma-separated")
    ap.add_argument("--N", type=int, default=10)
    ap.add_argument("--bc", type=str, default=BC_DEFAULT)
    ap.add_argument("--table", action="store_true", help="reproduce the R22 table")
    ap.add_argument("--json", type=str, default=None)
    a = ap.parse_args(argv)

    if a.table:
        out = build(bc=a.bc)
    else:
        rs = [int(s) for s in (a.rules or "203").split(",")]
        out = [summary(r, a.N, a.bc) for r in rs]

    print(f"{'rule':>6} {'tuple':<6} {'N':>3} {'n_wcc':>6} {'n_term':>7} "
          f"{'certain':>8} {'eff mean':>9} {'eff max':>8}")
    for d in out:
        print(f"W{d['rule']:<5} {d['tuple']:<6} {d['N']:>3} {str(d['n_wcc']):>6} "
              f"{d['n_terminal']:>7} {100 * d['certain_fraction']:>7.1f}% "
              f"{d['eff_attractors_mean']:>9.2f} {d['eff_attractors_max']:>8.2f}")
    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        json.dump(out, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
