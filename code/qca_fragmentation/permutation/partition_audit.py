"""Partition-level audit of R11's gate comparison  (R11 rev.2, sec.3).

R11 verifies the refinement theorem by union-find over the two graphs, but the
published check ran at N = 8, 9 and obc0 only -- exhaustive over RULES, not over
sizes.  This module widens it and adds three things the original did not have.

WHAT IS CHECKED, AND WHY EACH CAN FAIL.

  refinement   Every X component sits inside one H component, tested state by
               state.  The H side is the support of the EXACT Z[1/sqrt2]
               amplitudes (core.cycle.succ), not an activity predicate, so this
               is an operator-level test: if two Hadamard branches cancelled
               exactly, succ_H would lose a state and the X image could fall
               outside it.  (It cannot -- see `branch_multiplicity` -- but the
               test does not assume that.)

  equality     The FULL partition, not a count of components.  Cor. 4 of R11
               says the 81 V-free rules must give equality because their two
               circuits are literally the same map; that is the theorem floor.
               Anything above 81 is accidental and N-dependent.

  branch_multiplicity
               The number of branches reaching each final state.  Counting
               cancellations in the accumulated amplitude dictionary CANNOT see
               this: the dictionary sums contributions into one key before any
               count is taken, so two branches meeting with non-cancelling
               amplitudes are invisible.  Multiplicity must be counted on
               explicitly enumerated (state, flip-set) branches.  Multiplicity 1
               everywhere is the content of the branch/flip-set bijection: each
               site is visited once per cycle, so the flip set is recoverable as
               (final XOR start) and no two branches can meet.  Hence no
               destructive interference within one cycle, for channels in {I,V}.
               (Stated and proved as Prop. 3 of the companion commutant-algebra
               report; verified independently here.)
"""
from __future__ import annotations

import sys
from typing import Dict, List, Optional, Sequence

from ..core import rules as rules_mod
from ..core.cycle import _compile, succ
from ..graph import wcc as _wcc
from . import xca as _xca


def _find(p: List[int], a: int) -> int:
    while p[a] != a:
        p[a] = p[p[a]]
        a = p[a]
    return a


def _union(p: List[int], a: int, b: int) -> None:
    ra, rb = _find(p, a), _find(p, b)
    if ra != rb:
        p[ra] = rb


def partitions(rule: int, N: int, bc: str):
    """(H labels, X labels) as flat arrays of component roots over 2^N states."""
    n = 1 << N
    s = _wcc.make_succ(rule, N, bc)
    T = _xca.step_table(rule, N, bc)
    hp, xp = list(range(n)), list(range(n))
    for x in range(n):
        for y in s(x):
            _union(hp, x, y)
        _union(xp, x, int(T[x]))
    return [_find(hp, x) for x in range(n)], [_find(xp, x) for x in range(n)]


def same_partition(a: Sequence[int], b: Sequence[int]) -> bool:
    fwd: Dict[int, int] = {}
    rev: Dict[int, int] = {}
    for u, v in zip(a, b):
        if fwd.setdefault(u, v) != v or rev.setdefault(v, u) != u:
            return False
    return True


def refines(a: Sequence[int], b: Sequence[int]) -> bool:
    """Does partition `a` refine partition `b`?  (each a-block inside a b-block)"""
    seen: Dict[int, int] = {}
    for u, v in zip(a, b):
        if seen.setdefault(u, v) != v:
            return False
    return True


def is_v_free(rule: int) -> bool:
    """R11 Cor. 4: no V means the gate is never invoked, so the two circuits are
    the same map.  There are 3^4 = 81 such rules."""
    return "V" not in rules_mod.wolfram_to_tuple(rule)


def branch_multiplicity(rule: int, N: int, bc: str) -> Dict[str, object]:
    """Enumerate branches explicitly and count how many reach each final state.

    Returns the worst multiplicity, whether (final XOR start) always equals the
    chosen flip set, and the per-site visit counts.  Unitary rules only.
    """
    t = rules_mod.wolfram_to_tuple(rule)
    if not rules_mod.is_unitary(t):
        raise ValueError(f"rule {rule} has reset channels")
    steps = _compile(N, t, bc)
    visits: Dict[int, int] = {}
    for (i, _l, _r, _t) in steps:
        visits[i] = visits.get(i, 0) + 1
    worst, flipset_ok = 0, True
    for x0 in range(1 << N):
        live = [(x0, 0)]
        for (i, lpos, rpos, tt) in steps:
            nxt = []
            for (y, fl) in live:
                m = (y >> lpos) & 1 if lpos >= 0 else 0
                n = (y >> rpos) & 1 if rpos >= 0 else 0
                if tt[2 * m + n] == "V":
                    nxt.append((y, fl))
                    nxt.append((y ^ (1 << i), fl | (1 << i)))
                else:
                    nxt.append((y, fl))
            live = nxt
        counts: Dict[int, int] = {}
        for (y, fl) in live:
            counts[y] = counts.get(y, 0) + 1
            if (y ^ x0) != fl:
                flipset_ok = False
        worst = max(worst, max(counts.values()))
    return {"max_multiplicity": worst, "flipset_is_xor": flipset_ok,
            "visits_per_site": sorted(set(visits.values()))}


def audit(N: int, bc: str, rules_to_scan: Optional[Sequence[int]] = None) -> Dict:
    """Refinement, equality and the V-free decomposition at one (N, bc)."""
    rs = list(rules_to_scan) if rules_to_scan is not None else list(range(256))
    bad_refine, equal = [], []
    for rule in rs:
        h, x = partitions(rule, N, bc)
        if not refines(x, h):
            bad_refine.append(rule)
        if same_partition(h, x):
            equal.append(rule)
    vfree_eq = [r for r in equal if is_v_free(r)]
    return {"N": N, "bc": bc, "rules": len(rs),
            "refine_failures": bad_refine,
            "refines_ok": len(rs) - len(bad_refine),
            "n_equal": len(equal),
            "n_equal_v_free": len(vfree_eq),
            "n_equal_accidental": len(equal) - len(vfree_eq),
            "v_free_total": sum(1 for r in rs if is_v_free(r))}


def main(argv: List[str]) -> int:
    print("R11 rev.2 partition audit -- refinement, equality, multiplicity\n")
    print(f"{'bc':>6} {'N':>3} {'refines':>10} {'equal':>7} "
          f"{'= V-free':>9} {'+ accidental':>13}")
    for bc in ("obc0", "pbc"):
        for N in (7, 8, 9):
            a = audit(N, bc)
            print(f"{bc:>6} {N:>3} {a['refines_ok']:>7}/256 {a['n_equal']:>7} "
                  f"{a['n_equal_v_free']:>9} {a['n_equal_accidental']:>13}",
                  flush=True)
    print("\nbranch multiplicity (unitary rules, the test that can fail):")
    uni = [r for r in range(256) if rules_mod.is_unitary(
        rules_mod.wolfram_to_tuple(r))]
    for bc in ("obc0", "pbc"):
        worst = max(branch_multiplicity(r, 7, bc)["max_multiplicity"]
                    for r in uni)
        print(f"  bc={bc:<5} max over {len(uni)} rules at N=7: {worst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
