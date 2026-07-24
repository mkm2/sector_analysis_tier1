"""
Tier 1d: diagonal strong/weak conserved charges via the pair graph.

Context task_tier1d_pair_graph.md Sec.3 and the visibility theorem.  A diagonal
charge is a function  s : basis -> Q  built from a range-r local density with an
even/odd sublattice split (the brickwork breaks 1-site translation but keeps
2-site translation):

    s(x) = sum_i  c[ i mod 2, window_r(x, i) ]        (features = 2 * 2^r)

The visibility theorem separates two kinds of diagonal charge:

  * STRONG charge -- conserved along every single-branch M-edge, s(x') = s(x);
    equivalently s is constant on each weakly connected component (sector).
    These are exactly the diagonal part of the strong commutant {K_mu}'.
  * WEAK charge -- imposes NO constraint on M (diagonal unitaries act trivially
    on diagonal states) but is a G2 DIFFERENCE-conservation law:
        s(x) - s(y)  is constant along every G2 edge (x, y) -> (x', y'),
    i.e.  s(x) - s(y) - s(x') + s(y') = 0  for every off-diagonal G2 edge.

Because a strong charge trivially satisfies the G2 difference law, the charge
spaces nest  C  subset  S  subset  Z  where

    Z = { c : G2 difference law holds }        (all diagonal charges)
    S = { c : s is constant on every WCC }      (strong charges)
    C = { c : s is globally constant }          (trivial)

and the reported counts are

    n_charges = dim Z - dim C,
    n_strong  = dim S - dim C,
    n_weak    = dim Z - dim S.

All ranks are computed exactly over the rationals (Fraction), so the counts are
certified, not numerical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

from ..core.cycle import succ as cycle_succ
from ..core.rules import Tuple4, is_unitary
from ..graph.pair_graph import labeled_supports, succ2_from, wcc_labels
from ..graph.scc import recurrent_classes

# Rules whose wall-Hadamard core hosts the protected coherence (R5 open item i).
WALL_HADAMARD_CORE = [18, 19, 22, 23, 50, 55, 146, 151, 178, 179]


@dataclass
class ChargeResult:
    N: int
    rule: int
    bc: str
    r: int
    n_charges: int
    n_strong: int
    n_weak: int
    weak_grades_coherence: Optional[bool] = None   # wall-core diagnostic
    bounded_only: bool = False
    runtime: float = 0.0
    d_values_on_coherence: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Local-density feature map.                                                   #
# --------------------------------------------------------------------------- #

def window_positions(N: int, r: int, bc: str) -> List[int]:
    if bc == "pbc":
        return list(range(N))
    return list(range(0, N - r + 1))          # obc0: windows fully inside chain


def feature_vector(x: int, N: int, r: int, bc: str) -> Tuple[int, ...]:
    """Counts of each (parity, r-bit window) pattern in x.  Length 2 * 2^r."""
    d = 2 * (1 << r)
    vec = [0] * d
    for i in window_positions(N, r, bc):
        w = 0
        for k in range(r):
            w |= ((x >> ((i + k) % N)) & 1) << k
        vec[(i & 1) * (1 << r) + w] += 1
    return tuple(vec)


# --------------------------------------------------------------------------- #
# Exact rank over the rationals.                                              #
# --------------------------------------------------------------------------- #

def _rank(rows: List[List[Fraction]], ncols: int) -> int:
    mat = [list(row) for row in rows]
    rank = 0
    pivot_row = 0
    for col in range(ncols):
        piv = None
        for rr in range(pivot_row, len(mat)):
            if mat[rr][col]:
                piv = rr
                break
        if piv is None:
            continue
        mat[pivot_row], mat[piv] = mat[piv], mat[pivot_row]
        pv = mat[pivot_row][col]
        prow = mat[pivot_row]
        for rr in range(len(mat)):
            if rr != pivot_row and mat[rr][col]:
                f = mat[rr][col] / pv
                mat[rr] = [a - f * b for a, b in zip(mat[rr], prow)]
        rank += 1
        pivot_row += 1
        if pivot_row >= len(mat):
            break
    return rank


def _nullity(diff_rows, d):
    if not diff_rows:
        return d
    return d - _rank([[Fraction(v) for v in row] for row in diff_rows], d)


# --------------------------------------------------------------------------- #
# Main analysis.                                                              #
# --------------------------------------------------------------------------- #

def analyze_charges(rule: int, N: int, bc: str, t: Tuple4, *, r: int = 2,
                    node_budget: int = 4_000_000) -> ChargeResult:
    import time
    t0 = time.time()
    d = 2 * (1 << r)
    if is_unitary(t):
        return ChargeResult(N=N, rule=rule, bc=bc, r=r, n_charges=0, n_strong=0,
                            n_weak=0, runtime=time.time() - t0)

    total = 1 << N
    feat = {x: feature_vector(x, N, r, bc) for x in range(total)}
    f0 = feat[0]

    # C: charges producing a globally constant s  ->  differences Phi(x)-Phi(0).
    c_rows = [[feat[x][k] - f0[k] for k in range(d)] for x in range(1, total)]
    dimC = _nullity(c_rows, d)

    # S: strong charges  ->  s constant along every single-branch M-edge.
    m_rows = []
    seen_m = set()
    for x in range(total):
        fx = feat[x]
        for xp in cycle_succ(x, N, t, bc):
            if xp == x:
                continue
            row = tuple(fx[k] - feat[xp][k] for k in range(d))
            if row not in seen_m:
                seen_m.add(row)
                m_rows.append(list(row))
    dimS = _nullity(m_rows, d)

    # Z: G2 difference law over off-diagonal edges of the WCC-diagonal blocks.
    wcc = wcc_labels(N, t, bc)
    groups: Dict[int, List[int]] = {}
    for x in range(total):
        groups.setdefault(wcc[x], []).append(x)
    npairs = sum(len(g) * len(g) for g in groups.values())
    if npairs > node_budget:
        return ChargeResult(N=N, rule=rule, bc=bc, r=r, n_charges=-1,
                            n_strong=-1, n_weak=-1, bounded_only=True,
                            runtime=time.time() - t0)

    n_strong = dimS - dimC

    # WEAK charges grade the PROTECTED coherence: they are gradings D(x,y) =
    # s(x) - s(y) of the off-diagonal part of P_rec that are conserved along the
    # internal G2 edges of that coherence support.  Imposing the difference law
    # on ALL off-diagonal pairs (task Sec.3, literal) over-counts vacuously when
    # coherences decay in one step (e.g. rule 0: no surviving off-diagonal edge
    # leaves the law unconstrained), so we restrict to P_rec's off-diagonal part.
    from ..graph.pair_graph import analyze_pair
    pair = analyze_pair(rule, N, bc, t, cross_sector=False, return_support=True)
    poff = pair.offdiag_pairs or []
    poffset = set(poff)
    n_weak = 0
    if poff:
        sup = {x: labeled_supports(x, N, t, bc) for x in set(x for p in poff for x in p)}
        # psi(x,y) = feat(x) - feat(y);  D-constant constraints psi(p) - psi(p0).
        def psi(p):
            x, y = p
            return [feat[x][k] - feat[y][k] for k in range(d)]
        p0 = psi(poff[0])
        dconst_rows = [[psi(p)[k] - p0[k] for k in range(d)] for p in poff[1:]]
        # conservation along internal G2 edges:  psi(p) - psi(p') = 0.
        edge_rows = []
        seen_e = set()
        for p in poff:
            x, y = p
            pp = psi(p)
            for xp, yp in succ2_from(sup.get(x, {}), sup.get(y, {})):
                if (xp, yp) in poffset:
                    q = psi((xp, yp))
                    row = tuple(pp[k] - q[k] for k in range(d))
                    if any(row) and row not in seen_e:
                        seen_e.add(row)
                        edge_rows.append(list(row))
        n_weak = _nullity(edge_rows, d) - _nullity(dconst_rows, d)
        n_weak = max(n_weak, 0)

    res = ChargeResult(N=N, rule=rule, bc=bc, r=r,
                       n_charges=max(n_strong, 0) + n_weak,
                       n_strong=max(n_strong, 0),
                       n_weak=n_weak,
                       runtime=time.time() - t0)

    if n_weak > 0:
        vals = set()
        # report the D-value spectrum of a representative grading.
        # a charge conserved along edges (null of edge_rows) that is not
        # D-constant on P_rec (not annihilated by dconst_rows) -> a grading.
        c = _explicit_weak_charge(dconst_rows, edge_rows, d)
        if c is not None:
            for (x, y) in poff:
                vals.add(sum(c[k] * (feat[x][k] - feat[y][k]) for k in range(d)))
        res.weak_grades_coherence = len(vals) >= 2
        res.d_values_on_coherence = sorted(str(v) for v in vals)[:12]
    return res


def _explicit_weak_charge(m_rows, z_rows, d):
    """One coefficient vector in Z that is NOT strong (Fraction), or None."""
    # Nullspace basis of Z (the off-diagonal G2 difference law) via RREF.
    rows = [[Fraction(v) for v in row] for row in z_rows]
    ncols = d
    mat = [list(row) for row in rows]
    pivots = []
    pr = 0
    for col in range(ncols):
        piv = None
        for rr in range(pr, len(mat)):
            if mat[rr][col]:
                piv = rr
                break
        if piv is None:
            continue
        mat[pr], mat[piv] = mat[piv], mat[pr]
        pv = mat[pr][col]
        mat[pr] = [v / pv for v in mat[pr]]
        for rr in range(len(mat)):
            if rr != pr and mat[rr][col]:
                f = mat[rr][col]
                mat[rr] = [a - f * b for a, b in zip(mat[rr], mat[pr])]
        pivots.append(col)
        pr += 1
        if pr >= len(mat):
            break
    free = [c for c in range(ncols) if c not in pivots]
    basis = []
    for fc in free:
        vec = [Fraction(0)] * ncols
        vec[fc] = Fraction(1)
        for i, pc in enumerate(pivots):
            vec[pc] = -mat[i][fc]
        basis.append(vec)
    # m_rows selects a charge NOT annihilated by them (e.g. not D-constant).
    for vec in basis:
        if any(sum(Fraction(row[k]) * vec[k] for k in range(d)) != 0
               for row in m_rows):
            return vec
    return basis[0] if basis else None


# --------------------------------------------------------------------------- #
# Census / CLI.                                                               #
# --------------------------------------------------------------------------- #

def core_census(N: int = 6, bc: str = "pbc", r: int = 3,
                rules_iter=None) -> Dict:
    """Weak/strong diagonal-charge census; wall-Hadamard core by default."""
    from ..core.rules import wolfram_to_tuple
    rows = {}
    for rule in (WALL_HADAMARD_CORE if rules_iter is None else rules_iter):
        t = wolfram_to_tuple(rule)
        res = analyze_charges(rule, N, bc, t, r=r)
        rows[str(rule)] = {
            "tuple": "".join(t), "n_strong": res.n_strong,
            "n_weak": res.n_weak, "weak_grades_coherence": res.weak_grades_coherence,
            "d_values": res.d_values_on_coherence}
    return {"N": N, "bc": bc, "r": r, "rules": rows}


def main(argv=None):
    import argparse
    import json
    import os
    from .. import results_io, ENGINE_VERSION
    ap = argparse.ArgumentParser(description="Tier 1d diagonal weak/strong charges")
    ap.add_argument("--core", action="store_true",
                    help="wall-Hadamard core census")
    ap.add_argument("--rules", type=int, nargs="+", default=None)
    ap.add_argument("--N", type=int, default=6)
    ap.add_argument("--bc", default="pbc")
    ap.add_argument("--r", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(
        results_io.REPO_ROOT, "analytics", "weak_charges.json"))
    args = ap.parse_args(argv)
    rules_iter = args.rules if args.rules else (None if args.core else range(256))
    doc = core_census(args.N, args.bc, args.r, rules_iter)
    doc["engine_version"] = ENGINE_VERSION
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
    for rid, row in sorted(doc["rules"].items(), key=lambda kv: int(kv[0])):
        print(f"W{rid:>3} {row['tuple']}: strong={row['n_strong']} "
              f"weak={row['n_weak']} grades={row['weak_grades_coherence']} "
              f"{row['d_values']}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
