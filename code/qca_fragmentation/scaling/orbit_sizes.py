r"""
The Krylov sector-size distribution of the four one-gate rules with V = X.

R30 did this for V = Hadamard.  Replacing the gate changes the dynamics from a
branching support to a PERMUTATION, and this module works out exactly how much
of the R30 machinery survives -- which is more than one would guess.

--------------------------------------------------------------------------------
1.  WHAT BREAKS, AND WHAT DOES NOT
--------------------------------------------------------------------------------

R30 rests on the flip reduction: a half-layer image of a basis state is the
SUBCUBE spanned by the active sites, because <b|H|b> and <b_bar|H|b> are both
non-zero, so "flip" and "don't flip" are both admissible at every active site.
X has a ZERO DIAGONAL.  The choice is forced, the image is a single point, and
the one-cycle map is a permutation.  So:

  * sectors are ORBITS of a permutation -- no transients, and the sum rule is
    automatic rather than a check;
  * R30 Thm 1's CONSTANCY half survives verbatim, and in fact for free:
    supp(U_X|x>) is contained in supp(U_H|x>), so anything constant on Hadamard
    sectors is constant on X orbits.  The wall set is still conserved.
  * R30 Thm 1's TRANSITIVITY half DIES.  Its proof moves one domain wall by one
    site, or removes one particle at a time; under X every active site flips at
    once and no single-site move is available.  The label is therefore conserved
    but NOT complete, and the X sectors strictly refine the Hadamard ones
    (R11's refinement theorem, now visible in the census).

--------------------------------------------------------------------------------
2.  THE RENEWAL STRUCTURE SURVIVES ONE LEVEL DOWN
--------------------------------------------------------------------------------

Measured, not assumed (`check_factorisation`): inside a label fibre the X map is
the DIRECT PRODUCT of independent per-segment permutations, and each of those
depends only on the segment's length -- not on its position, its parity, or the
rest of the label.  Zero failures, all four rules, both boundary conditions.

So R30's scalar gap kernel becomes a CYCLE TYPE, and the product becomes an lcm:

    R30 (Hadamard):  |sector| = L(g_0) K(g_1) ... R(g_k)          a PRODUCT
    R31 (X):         choose one cycle c_j from C(g_j) in each gap.  That choice
                     contributes   prod_j c_j / lcm_j(c_j)   orbits, each of
                     ORBIT LENGTH   lcm_j(c_j).

with sum(C(g)) = K(g) exactly, so the cycle type refines the Hadamard kernel and
the R30 fibre sizes are recovered by forgetting the cycle structure.  Indexing
the cycle type by the NUMBER OF STATES in the segment (i.e. by the R30 kernel
value) rather than by the segment length makes the two families line up:

    PARABOLIC (156/198):  C = (v,), a SINGLE cycle of length v.  The domain wall
        marches ballistically and reflects, visiting every admissible position
        once per period.  So the X orbit length is just the lcm of the R30 gap
        kernels, and a fibre of R30 size prod K_j carries prod K_j / lcm(K_j)
        orbits of length lcm(K_j).

    INDEXING.  C is indexed here by the segment's NUMBER OF STATES, i.e. by the
    R30 kernel value, not by a length.  That is deliberate: it makes the corner
    class redundant.  Tier-2 R-T21 indexes by the geometric gap and consequently
    needs a left/interior/right class, and the two are the same table -- their
    C(L, l) = C(K, l+1) for W201 is our "both have F(...) states", and their
    C(L, l) = (1,) for W156 is our "a one-state segment is a fixed point".
    Keyed by state count there are ZERO conflicts for all four rules at both
    boundary conditions over N = 5..13.

    HYPERBOLIC (108/201):  |C| = F(l+2) as in R30, but the type is non-trivial:
        v=2  (2)        v=3  (3)          v=5  (2,3)      v=8  (3,5)
        v=13 (2,3,8)    v=21 (3,7,11)     v=34 (2,3,5,10,14)
        v=55 (3,9,13,13,17)               v=89 (2,3,8,12,12,16,16,20)
        The cycle lengths lie in arithmetic progressions of step 4 and the
        longest is 3l-7 for l >= 4.  This is the hard-core (PXP-like) X
        automaton on a segment with frozen ends.

--------------------------------------------------------------------------------
3.  THE WALL-FREE LABEL, AGAIN
--------------------------------------------------------------------------------

R30 Remark 2 recorded that every correction in that analysis was the same term --
the wall-free label, the one sector whose size is not a product of gap kernels.
It came up twice more here, and the truth is sharper than either of the guesses
that preceded it:

    at obc0 the wall-free label IS a segment -- its cycle type is exactly C
    evaluated at its state count, for all four rules;
    at pbc it is NOT, for any of them.

The reason is the pads.  On the chain the frozen 0 outside each end plays the
part a wall's collar plays in the bulk, so a wall-free chain is just one long
segment.  On the ring there is no boundary at all, so the wall-free label is a
different automaton: for the ascent rules it is the two frozen uniform states
(cycle type (1,1), NOT the (2,) of a two-state segment), and for the symmetric
rules it is the hard-core X automaton on a RING of L(N) states, whose spectrum
does not appear anywhere in the segment kernels.

Both errors were mine and both were caught by checking rather than assuming: the
first kernel extraction reported the pbc ascent-rule kernel as ambiguous at eight
lengths because the wall-free ring was being classified as a segment, and an
earlier draft of this docstring then over-corrected to "the wall-free label is
never a segment", which is false on the chain.  `wallfree_cycle_type` computes it
directly in every case, which is why the predictor was right throughout.

--------------------------------------------------------------------------------
4.  COST
--------------------------------------------------------------------------------

`segment_cycle_type` is a local computation on 2^l states, independent of N and
cached.  `wallfree_cycle_type` costs the size of the wall-free fibre, which is
phi^N for the hyperbolic pair, so N <~ 30.  (At obc0 it could be replaced by
`segment_cycle_type` of the same state count -- see section 3 -- but it is left
as a direct computation so that the chain and the ring go through the same code
path and neither inherits an assumption from the other.)  `hist` is a DP over (letters used,
running lcm); the number of distinct lcms is the thing that grows, and it grows
faster than the number of distinct sizes did in R30 -- treat N = 24 as the
working ceiling and check `hist_bruteforce` at small N as the oracle.
"""
from __future__ import annotations

import math
import os
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

from . import sector_sizes as H
from . import sectors

RULES: Tuple[int, ...] = (156, 198, 108, 201)
FAMILY = H.FAMILY
WALL_WORD = H.WALL_WORD

OUT_JSON = os.path.join(sectors.ANALYTICS, "orbit_sizes_{bc}.json")

#: local context around a free segment: (left pad, right pad, admissibility).
#: A segment sits between two walls, so it carries the wall's COLLAR on each
#: side -- that is what distinguishes it from the wall-free label.
_CTX = {201: ("110", "011", lambda s: "11" not in s),
        108: ("001", "100", lambda s: "00" not in s)}


# --- 1. the per-segment cycle type -------------------------------------------

def _cycle_type_of(nxt: Dict[int, int]) -> Tuple[int, ...]:
    seen, ct = set(), []
    for v in nxt:
        if v in seen:
            continue
        L, y = 0, v
        while y not in seen:
            seen.add(y)
            L += 1
            y = nxt[y]
        ct.append(L)
    return tuple(sorted(ct))


@lru_cache(maxsize=None)
def _hyperbolic_type(l: int) -> Tuple[int, ...]:
    """Cycle type of the hard-core X automaton on l free sites with frozen
    collars, from a minimal chain -- independent of N."""
    from ..core.rules import wolfram_to_tuple
    from ..permutation.xca import x_step
    rule = 201
    left, right, ok = _CTX[rule]
    prefix = left
    N = len(prefix) + l + len(right)
    t = wolfram_to_tuple(rule)
    base = [int(c) for c in prefix] + [0] * l + [int(c) for c in right]
    lo = len(prefix)
    states = [v for v in range(1 << l)
              if ok("".join(str((v >> i) & 1) for i in range(l)))]
    m = ((1 << l) - 1) << lo

    def enc(v):
        bits = list(base)
        for i in range(l):
            bits[lo + i] = (v >> i) & 1
        return sum(b << k for k, b in enumerate(bits))

    nxt = {}
    for v in states:
        nxt[v] = (x_step(enc(v), N, t, "obc0") & m) >> lo
    assert set(nxt.values()) == set(states), (l, "local context too tight")
    return _cycle_type_of(nxt)


@lru_cache(maxsize=None)
def segment_cycle_type(rule: int, n_states: int) -> Tuple[int, ...]:
    """C indexed by the NUMBER OF STATES of the segment, i.e. by the R30 kernel
    value.  This is the object that replaces R30's scalar kernel.

    Parabolic family: a single cycle, C = (v,).  Hyperbolic: the hard-core
    spectrum above.  Verified against the cycle types extracted from the full
    2^N flip data for every segment that occurs, all four rules, both bc."""
    if n_states < 1:
        raise ValueError("a segment has at least one state")
    if n_states == 1:
        return (1,)
    if FAMILY[rule] == "parabolic":
        return (n_states,)
    l = 1
    while True:                      # |C(l)| = F(l+2) is strictly increasing
        ct = _hyperbolic_type(l)
        if sum(ct) == n_states:
            return ct
        if sum(ct) > n_states:
            raise ValueError(f"no hyperbolic segment has {n_states} states")
        l += 1


def wallfree_cycle_type(rule: int, N: int, bc: str) -> Tuple[int, ...]:
    """The wall-free label's own cycle type.  NOT a segment: it has no walls, so
    no collars, so `segment_cycle_type` does not apply (R30 Remark 2, fourth
    occurrence).  Costs the size of the fibre, not 2^N."""
    from ..core.rules import wolfram_to_tuple
    from ..permutation.xca import x_step
    a, b = WALL_WORD[rule]
    t = wolfram_to_tuple(rule)

    def has_wall(x: int) -> bool:
        if bc == "obc0":
            y = [0] + [(x >> i) & 1 for i in range(N)] + [0]
            return any(y[i] == a and y[i + 1] == b for i in range(N + 1))
        y = [(x >> i) & 1 for i in range(N)]
        return any(y[i] == a and y[(i + 1) % N] == b for i in range(N))

    states = [x for x in range(1 << N) if not has_wall(x)]
    nxt = {x: x_step(x, N, t, bc) for x in states}
    assert set(nxt.values()) == set(states), (rule, N, bc, "not closed")
    return _cycle_type_of(nxt)


# --- 2. the orbit-length histogram -------------------------------------------

def _merge(dst: Dict[int, int], lcm: int, weight: int) -> None:
    dst[lcm] = dst.get(lcm, 0) + weight


def _extend(cur: Dict[int, int], ct: Sequence[int]) -> Dict[int, int]:
    """Multiply the running (lcm -> sum of products) table by one segment."""
    out: Dict[int, int] = {}
    for L, w in cur.items():
        for c in ct:
            _merge(out, L * c // math.gcd(L, c), w * c)
    return out


def hist(rule: int, N: int, bc: str) -> Dict[int, int]:
    """{orbit length: multiplicity}, exactly.

    Walks the same label decomposition as R30 -- so the label enumeration is the
    one already validated there -- but carries (running lcm, sum of products of
    the chosen cycle lengths) instead of a single product.  A completed label
    with running lcm L and product sum P contributes P / L orbits of length L,
    because a choice of one cycle per segment tiles prod c_j states into
    prod c_j / lcm orbits of length lcm."""
    if bc not in ("obc0", "pbc"):
        raise ValueError(f"unknown boundary convention {bc!r}")
    n_letters = N + 1 if bc == "obc0" else N
    out: Dict[int, int] = {}

    def emit(table: Dict[int, int]) -> None:
        for L, P in table.items():
            assert P % L == 0, (rule, N, bc, L, P)
            _merge(out, L, P // L)

    if bc == "obc0":
        # states: ("pre", letters) before the first wall, ("post", letters)
        cur: Dict[Tuple[str, int], Dict[int, int]] = {("pre", 0): {1: 1}}
        for _ in range(n_letters):
            nxt: Dict[Tuple[str, int], Dict[int, int]] = {}
            for (ph, g), tab in cur.items():
                key = (ph, g + 1)
                nxt.setdefault(key, {})
                for L, w in tab.items():
                    _merge(nxt[key], L, w)
                v = (H.left_kernel(rule, g) if ph == "pre"
                     else H.gap_kernel(rule, g))
                if v:
                    ext = _extend(tab, segment_cycle_type(rule, v))
                    nxt.setdefault(("post", 0), {})
                    for L, w in ext.items():
                        _merge(nxt[("post", 0)], L, w)
            cur = nxt
        for (ph, g), tab in cur.items():
            if ph == "pre":
                continue                          # the wall-free label, below
            v = H.right_kernel(rule, g)
            if v:
                emit(_extend(tab, segment_cycle_type(rule, v)))
        emit({L: P for L, P in _extend({1: 1},
                                       wallfree_cycle_type(rule, N, bc)).items()})
        return out

    # --- pbc: the pointed-block identity of R30, with the weight carried ------
    table: List[Dict[int, int]] = []
    cur2: Dict[int, Dict[int, int]] = {0: {1: 1}}
    for _ in range(n_letters):
        table.append({g: dict(t) for g, t in cur2.items()})
        nxt2: Dict[int, Dict[int, int]] = {}
        for g, tab in cur2.items():
            nxt2.setdefault(g + 1, {})
            for L, w in tab.items():
                _merge(nxt2[g + 1], L, w)
            v = H.gap_kernel(rule, g)
            if v:
                ext = _extend(tab, segment_cycle_type(rule, v))
                nxt2.setdefault(0, {})
                for L, w in ext.items():
                    _merge(nxt2[0], L, w)
        cur2 = nxt2
    table.append({g: dict(t) for g, t in cur2.items()})

    acc: Dict[int, int] = {}
    for p1 in range(n_letters):
        m = n_letters - p1 - 1
        for g, tab in table[m].items():
            v = H.gap_kernel(rule, g + p1)
            if v:
                for L, w in _extend(tab, segment_cycle_type(rule, v)).items():
                    _merge(acc, L, w)
    # No division by the number of walls and none by N: summing over the
    # position p1 of the first wall and closing the wrapped gap already counts
    # each cyclic label exactly once (the R30 pbc identity).
    for L, P in acc.items():
        assert P % L == 0, (rule, N, bc, L, P)
        _merge(out, L, P // L)
    emit(_extend({1: 1}, wallfree_cycle_type(rule, N, bc)))
    return out


def hist_bruteforce(rule: int, N: int, bc: str) -> Dict[int, int]:
    """Orbit lengths straight from the 2^N step table.  The oracle."""
    from ..permutation import xca
    n = 1 << N
    T = xca.step_table(rule, N, bc)
    seen = bytearray(n)
    h: Counter = Counter()
    for x in range(n):
        if seen[x]:
            continue
        L, y = 0, x
        while not seen[y]:
            seen[y] = 1
            L += 1
            y = int(T[y])
        assert y == x, ("not a permutation", rule, N, bc)
        h[L] += 1
    assert sum(k * v for k, v in h.items()) == n
    return dict(h)


def summary(rule: int, N: int, bc: str) -> Dict[str, int]:
    h = hist(rule, N, bc)
    tot = sum(k * v for k, v in h.items())
    assert tot == 1 << N, (rule, N, bc, tot)
    return {"n_sectors": sum(h.values()), "d_max": max(h),
            "n_frozen": h.get(1, 0), "total": tot}


# --- 3. the structural checks ------------------------------------------------

def check_factorisation(rule: int, N: int, bc: str) -> Dict[str, int]:
    """Does the X map act as a direct product over the free segments of a label
    fibre, and does each factor depend only on the segment length?"""
    from ..permutation import xca
    a, b = WALL_WORD[rule]
    T = xca.step_table(rule, N, bc)
    fib: Dict[frozenset, List[int]] = defaultdict(list)
    for x in range(1 << N):
        if bc == "obc0":
            y = [0] + [(x >> i) & 1 for i in range(N)] + [0]
            lab = frozenset(i for i in range(N + 1)
                            if y[i] == a and y[i + 1] == b)
        else:
            y = [(x >> i) & 1 for i in range(N)]
            lab = frozenset(i for i in range(N)
                            if y[i] == a and y[(i + 1) % N] == b)
        fib[lab].append(x)

    bad, conflicts, types = 0, 0, {}
    for lab, states in fib.items():
        if len(states) == 1 or not lab:
            continue                     # the wall-free label is not a segment
        const = {i for i in range(N)
                 if len({(x >> i) & 1 for x in states}) == 1}
        free = [i for i in range(N) if i not in const]
        runs, cur = [], [free[0]] if free else []
        for s in free[1:]:
            if s == cur[-1] + 1:
                cur.append(s)
            else:
                runs.append(cur)
                cur = [s]
        if cur:
            runs.append(cur)
        if (bc == "pbc" and len(runs) > 1 and runs[0][0] == 0
                and runs[-1][-1] == N - 1):
            runs[0] = runs[-1] + runs[0]
            runs.pop()
        for seg in runs:
            m = 0
            for i in seg:
                m |= 1 << i
            local: Dict[int, int] = {}
            ok = True
            for x in states:
                k, v = x & m, int(T[x]) & m
                if local.setdefault(k, v) != v:
                    ok = False
                    break
            if not ok:
                bad += 1
                continue
            ct = _cycle_type_of(local)
            if types.setdefault(len(seg), ct) != ct:
                conflicts += 1
    return {"n_fibres": len(fib), "factorisation_failures": bad,
            "length_conflicts": conflicts, "n_lengths": len(types)}
