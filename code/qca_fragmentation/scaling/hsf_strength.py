"""
Strong or weak fragmentation?  Rules 108/201 and 156/198 -- an addendum to R18.

R9 and R19 report D_max / 2^N -> 0 for all four rules, which is necessary for
fragmentation but is not the criterion anybody uses.  The literature (Sala et
al., PRX 10, 011047 (2020); Khemani, Hermele, Nandkishore, PRB 101, 174204
(2020); Moudgalya, Bernevig, Regnault, Rep. Prog. Phys. 85, 086501 (2022))
grades fragmentation by the largest Krylov sector RELATIVE TO THE SYMMETRY
SECTOR THAT CONTAINS IT:

    strong :  D_max(s) / dim(s) -> 0  in the thermodynamic limit
    weak   :  D_max(s) / dim(s) -> const > 0,

where s is the symmetry sector of the conserved charges -- the "available"
Hilbert space.  Weak fragmentation still has one thermal Krylov sector filling
its symmetry sector, with the frozen and small sectors a vanishing remainder;
strong fragmentation has no such sector.  Dividing by 2^N instead of by dim(s)
is the same test only when the symmetry is trivial, and for these four rules it
is not.

TWO ANSWERS, AND THE SECOND IS THE ONE THAT MATTERS.

  crude:  D_max / 2^N  -> 0 for all four, as R9 already reported.
  proper: D_max(s) / dim(s) -> 0 EXPONENTIALLY for all four, so all four are
          STRONGLY fragmented and the conclusion is not close.

The exponential rate is the point.  The symmetry cannot rescue the ratio because
it only removes a POLYNOMIAL factor: the charges have O(N^2) joint level sets
(R18), so the largest symmetry sector still holds Theta(2^N / N) states, while
D_max grows like phi^N (108/201) or 4^(N/5) (156/198).  Hence

    D_max(s*) / dim(s*)  =  Theta( N * (phi/2)^N )   =  Theta( N * 0.809^N )
    D_max(s*) / dim(s*)  =  Theta( N * (4^{1/5}/2)^N ) = Theta( N * 0.660^N ).

A weakly fragmented model would have to keep that ratio bounded away from zero;
these are two of the cleanest strong-fragmentation signatures one could ask for.

WHAT COUNTS AS "THE SYMMETRY" -- the one thing that has to be got right.

Strong-vs-weak is only defined relative to a declared symmetry algebra.  Taken
literally with ALL conserved quantities the question is empty: the wall set is
itself conserved and is a complete invariant (R18), so the symmetry sectors
would BE the Krylov sectors and the ratio would be 1 by construction.  The
grading is therefore always relative to the LOCAL charges one would write down
for the model, and the honest procedure is to take as many of them as exist.

So this module does not assume R18's pair.  It DETECTS the full space of
conserved charges of the form

    Q(x) = sum_i q(x_i, x_{i+1}, ..., x_{i+r-1})

by exact rational elimination -- the null space of the count-vector differences
inside each Krylov sector -- for r = 1, 2, 3, 4, and uses the joint level sets
of a basis as the symmetry sectors.  Larger r can only make the symmetry sectors
smaller and the ratio larger, so pushing r is what makes a "strong" verdict
safe rather than convenient.

The detector reproduces R18 independently: at r = 2 it finds exactly two
non-constant charges for each of the four rules, and their level sets are the
sublattice-resolved wall counts.  It also answers the question R18 left open,
namely whether a longer-range charge would have done better: r = 3 and r = 4 add
nothing at all for 156/198, and what they add for 108/201 is a bounded number of
extra level sets, still polynomial.  The verdict does not move.

THE CONTROLS.  A diagnostic that returns "strong" for everything says nothing,
so the same pipeline is run on the polynomial corner of the unitary sweep:

    W_60, W_102   D_max = 2^(N-1) -- HALF the Hilbert space in one sector.
                  The ratio is 1/2 and stays there.  This is what weak looks
                  like, and it is a rule of this project, not a foreign model.
    W_150, W_105  D_max ~ 2^N / sqrt(N).  The ratio vanishes, but only
                  polynomially -- the marginal case, and worth having on the
                  same axes so that "exponentially" is visibly a different
                  statement from "eventually".

Cost.  Krylov labels come from the flip reduction (graph.flip_graph), so N = 20
is seconds and the statistics are numpy reductions over 2^N entries.
"""

from __future__ import annotations

import json
import os
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .. import results_io
from ..core.rules import is_unitary, wolfram_to_tuple
from ..graph.flip_graph import flip_components_np
from . import sectors

BC_DEFAULT = "obc0"
OUT_JSON = os.path.join(sectors.ANALYTICS, "hsf_strength_{bc}.json")

#: The four rules R18 is about.
RULES: Tuple[int, ...] = (108, 201, 156, 198)

#: The polynomial corner, run through the same pipeline as a contrast.
CONTROLS: Tuple[int, ...] = (60, 102, 150, 105)

#: Ranges the charge detector is pushed to.  r = 2 is R18's; 3 and 4 are the
#: check that a longer-range charge would not have rescued the symmetry story.
RANGES: Tuple[int, ...] = (1, 2, 3, 4)

#: N grid.  Detection is done at the smaller sizes (the basis stabilises early
#: and the count matrix is the memory-hungry object); the ratios are measured
#: as far out as the flip reduction is comfortable.
#:
#: Detection is PARITY-RESOLVED and this is not optional.  With a two-site unit
#: cell on an open chain, which sublattice each end of the chain belongs to
#: depends on N mod 2, so a basis found at N = 12 is in general not conserved at
#: N = 13.  An earlier pass of this module transferred one basis to every N and
#: silently produced charges that were not conserved at odd N; `strength`
#: reports `krylov_inside_sym`, and that flag is what caught it.
DETECT_NS_EVEN: Tuple[int, ...] = (10, 12, 14)
DETECT_NS_ODD: Tuple[int, ...] = (11, 13, 15)
NS_DEFAULT: Tuple[int, ...] = tuple(range(6, 21))

#: Growth bases the closing estimate uses, from R18/R19 (derived, not fitted).
PHI = 1.618033988749895
QUARTIC = 4.0 ** 0.2


# --- 1. windows, counts, charges ---------------------------------------------

def _bit_getter(N: int, bc: str):
    """s[j] for the padded string the dynamics actually sees, as a function of
    the padded index j, vectorised over all 2^N states.

    obc0 pads a frozen 0 outside each end, so s has length N + 2 and
    s[j] = bit j-1 of x for 1 <= j <= N.  pbc has length N and wraps.
    """
    X = np.arange(1 << N, dtype=np.uint32 if N <= 31 else np.uint64)
    if bc == "obc0":
        L = N + 2

        def bit(j: int) -> np.ndarray:
            if 1 <= j <= N:
                return ((X >> np.uint32(j - 1)) & 1).astype(np.uint8)
            return np.zeros(1 << N, dtype=np.uint8)
    elif bc == "pbc":
        L = N

        def bit(j: int) -> np.ndarray:
            return ((X >> np.uint32(j % N)) & 1).astype(np.uint8)
    else:
        raise ValueError(f"unknown boundary convention {bc!r}")
    return bit, L


def windows(N: int, bc: str, r: int):
    """Yield (position, r-bit window value) at each admissible position, over
    all 2^N states.  The window at i encodes (s_i, ..., s_{i+r-1}) with s_i most
    significant -- the same reading order as wall_charges.wall_set, and the
    position is the PADDED index, so its parity is the one R18's sublattice
    split uses."""
    bit, L = _bit_getter(N, bc)
    starts = range(L - r + 1) if bc == "obc0" else range(L)
    for i in starts:
        w = np.zeros(1 << N, dtype=np.uint8)
        for k in range(r):
            j = (i + k) % L if bc == "pbc" else i + k
            w = (w << np.uint8(1)) | bit(j)
        yield i, w


def n_params(r: int, p: int) -> int:
    return p * (1 << r)


def count_matrix(N: int, bc: str, r: int, p: int = 2) -> np.ndarray:
    """(2^N, p * 2^r) matrix of r-window pattern counts, resolved by the
    position's residue mod p.

    p = 2 is the physically right unit cell: a brick-wall cycle is invariant
    under two-site translations only, so a conserved density may live on one
    sublattice.  That is what lets the staggered wall number S and R18's pair
    (|W_even|, |W_odd|) appear at all; with p = 1 they are invisible.
    """
    m = np.zeros((1 << N, n_params(r, p)), dtype=np.int16)
    idx = np.arange(1 << N)
    for i, w in windows(N, bc, r):
        m[idx, (i % p) * (1 << r) + w] += 1
    return m


def charge_values(q: Sequence[int], N: int, bc: str, r: int,
                  p: int = 2) -> np.ndarray:
    """Q(x) = sum_i q[(i mod p), window_i(x)], without the count matrix."""
    qa = np.asarray(q, dtype=np.int64)
    out = np.zeros(1 << N, dtype=np.int64)
    for i, w in windows(N, bc, r):
        out += qa[(i % p) * (1 << r) + w]
    return out


# --- 2. Krylov labels ---------------------------------------------------------

_KL_CACHE: Dict[Tuple[int, int, str], np.ndarray] = {}


def krylov_labels(rule: int, N: int, bc: str) -> np.ndarray:
    """Compressed sector id per state, from the flip reduction.

    Cached for one key only: the four ranges are measured at the same (rule, N)
    one after another and the reduction is the expensive part, but a full cache
    would hold several arrays of 2^N entries at once.
    """
    key = (rule, N, bc)
    if key in _KL_CACHE:
        return _KL_CACHE[key]
    t = wolfram_to_tuple(rule)
    if not is_unitary(t):
        raise ValueError("this module is for unitary rules "
                         "(the flip reduction is proved only there)")
    _, lab = flip_components_np(N, t, bc, return_labels=True)
    _, inv = np.unique(lab, return_inverse=True)
    out = inv.astype(np.int64)
    _KL_CACHE.clear()
    _KL_CACHE[key] = out
    return out


def matches_r18(rule: int, N: int, bc: str = BC_DEFAULT) -> Dict:
    """Do the detected range-2 charges have R18's level sets?

    R18 names the two independent charges as the sublattice-resolved wall
    counts (|W_even|, |W_odd|) of the rule's minimal wall word, found by Tier-2's
    null-space detector.  This module found two charges at range 2 by a
    completely different route -- exact elimination on count-vector differences
    inside the Krylov sectors, with no notion of a wall anywhere.  If the two
    partitions of the 2^N states coincide, the two accounts agree.
    """
    from . import wall_charges as WC
    word = WC.wall_words(rule)[0]
    det = conserved_charges(rule, N, bc, 2)
    mine = sym_labels(det["nonconstant"], N, bc, 2)
    theirs = []
    for x in range(1 << N):
        w = WC.wall_set(x, N, word, bc)
        theirs.append((sum(1 for i in w if i % 2 == 0),
                       sum(1 for i in w if i % 2)))
    lut = {v: i for i, v in enumerate(sorted(set(theirs)))}
    theirs_arr = np.array([lut[v] for v in theirs], dtype=np.int64)
    # same partition <=> the pair (mine, theirs) has as many distinct values as
    # either coordinate alone
    both = len({(int(a), int(b)) for a, b in zip(mine, theirs_arr)})
    return {"rule": rule, "N": N, "bc": bc, "word": word,
            "n_mine": int(mine.max()) + 1, "n_r18": len(lut),
            "n_joint": both,
            "same_partition": both == int(mine.max()) + 1 == len(lut),
            "n_charges": det["n_independent"]}


# --- 3. the charge detector ---------------------------------------------------

def _nullspace_int(rows: np.ndarray, width: int) -> List[List[Fraction]]:
    """Exact null space of the row space, by Fraction elimination.

    The rows are integer count differences and there are at most `width` of
    them that matter, so the echelon basis saturates almost immediately; the
    loop then rejects the rest in a few operations each.
    """
    piv: Dict[int, List[Fraction]] = {}
    for row in rows:
        v = [Fraction(int(a)) for a in row]
        for c in sorted(piv):
            if v[c]:
                f = v[c]
                v = [a - f * b for a, b in zip(v, piv[c])]
        lead = next((c for c in range(width) if v[c]), None)
        if lead is None:
            continue
        v = [a / v[lead] for a in v]
        for c in list(piv):
            if piv[c][lead]:
                f = piv[c][lead]
                piv[c] = [a - f * b for a, b in zip(piv[c], v)]
        piv[lead] = v
        if len(piv) == width:
            break
    free = [c for c in range(width) if c not in piv]
    basis = []
    for f in free:
        v = [Fraction(0)] * width
        v[f] = Fraction(1)
        for c, prow in piv.items():
            v[c] = -prow[f]
        basis.append(v)
    return basis


def _integerise(v: Sequence[Fraction]) -> List[int]:
    from math import gcd
    den = 1
    for a in v:
        den = den * a.denominator // gcd(den, a.denominator)
    w = [int(a * den) for a in v]
    g = 0
    for a in w:
        g = gcd(g, abs(a))
    if g > 1:
        w = [a // g for a in w]
    if next((a for a in w if a), 0) < 0:
        w = [-a for a in w]
    return w


def conserved_charges(rule: int, N: int, bc: str, r: int,
                      p: int = 2) -> Dict:
    """Every range-r site-basis charge that is constant on every Krylov sector.

    Returns the integer basis of the null space, the subset that is not
    identically constant, and the rank of the certified charge VALUES including
    the constant -- the quantity R18 quotes as 3 for r = 2.
    """
    kl = krylov_labels(rule, N, bc)
    c = count_matrix(N, bc, r, p).astype(np.int32)
    # difference from a representative of each sector
    first = np.full(int(kl.max()) + 1, -1, dtype=np.int64)
    order = np.argsort(kl, kind="stable")
    seen = np.r_[True, kl[order][1:] != kl[order][:-1]]
    first[kl[order][seen]] = order[seen]
    d = c - c[first[kl]]
    d = np.unique(d, axis=0)
    d = d[np.any(d != 0, axis=1)]

    width = n_params(r, p)
    basis_f = _nullspace_int(d, width)
    basis = [_integerise(v) for v in basis_f]

    vals, nonconst = [], []
    for q in basis:
        Q = charge_values(q, N, bc, r, p)
        # exact check: constant on every sector
        assert np.all(Q[first[kl]] == Q), (rule, N, bc, r, p, q)
        if int(Q.min()) != int(Q.max()):
            nonconst.append(q)
            vals.append(Q)
    if vals:
        M = np.stack(vals + [np.ones(1 << N, dtype=np.int64)], axis=1)
        rank = int(np.linalg.matrix_rank(M.astype(np.float64)))
        # keep an independent subset, so that the level sets are computed from
        # rank-many charges rather than from a redundant spanning set
        keep, cur = [], [np.ones(1 << N, dtype=np.int64)]
        for q, Q in zip(nonconst, vals):
            trial = cur + [Q]
            if int(np.linalg.matrix_rank(
                    np.stack(trial, axis=1).astype(np.float64))) > len(cur):
                cur = trial
                keep.append(q)
        nonconst = keep
    else:
        rank = 1
    return {"rule": rule, "N": N, "bc": bc, "r": r, "p": p,
            "basis": basis, "nonconstant": nonconst,
            "value_rank_with_constant": rank,
            "n_independent": rank - 1}


def stable_basis(rule: int, bc: str, r: int, parity: int,
                 Ns: Optional[Sequence[int]] = None, p: int = 2) -> Dict:
    """Detect at several N of one parity and require the answer not to move.

    `parity` is N mod 2; see DETECT_NS_EVEN for why the split is mandatory.
    """
    Ns = Ns if Ns is not None else (DETECT_NS_EVEN if parity == 0
                                    else DETECT_NS_ODD)
    got = [conserved_charges(rule, N, bc, r, p) for N in Ns]
    ranks = {g["n_independent"] for g in got}
    return {"rule": rule, "bc": bc, "r": r, "p": p, "parity": parity,
            "Ns": list(Ns),
            "n_independent": got[-1]["n_independent"],
            "stable": len(ranks) == 1,
            "ranks": sorted(ranks),
            "nonconstant": got[-1]["nonconstant"]}


# --- 4. symmetry sectors and the ratios ---------------------------------------

def sym_labels(qs: Sequence[Sequence[int]], N: int, bc: str,
               r: int, p: int = 2) -> np.ndarray:
    """Compressed joint level-set id of the given charges."""
    if not qs:
        return np.zeros(1 << N, dtype=np.int64)
    key = np.zeros(1 << N, dtype=np.int64)
    for q in qs:
        Q = charge_values(q, N, bc, r, p)
        lo, hi = int(Q.min()), int(Q.max())
        key = key * (hi - lo + 1) + (Q - lo)
        # re-compress after every charge: the product of the ranges would
        # otherwise overflow int64 once there are ten of them
        _, key = np.unique(key, return_inverse=True)
        key = key.astype(np.int64)
    return key


def strength(rule: int, N: int, bc: str, qs: Sequence[Sequence[int]],
             r: int, p: int = 2) -> Dict:
    """The strong-vs-weak numbers for one (rule, N).

    ratio_dominant is the criterion: the largest Krylov sector inside the
    LARGEST symmetry sector, divided by that symmetry sector's dimension.
    ratio_worst takes the maximum over all symmetry sectors, which is always 1
    because the extremal charge sectors are single frozen states -- it is
    reported so that the reader can see why the dominant sector is the
    meaningful one.  explored_fraction is the average over computational basis
    states of D_K(x) / dim(sym(x)): the fraction of its own symmetry sector a
    random product state can reach.
    """
    kl = krylov_labels(rule, N, bc)
    sl = sym_labels(qs, N, bc, r, p)
    K = int(kl.max()) + 1
    S = int(sl.max()) + 1

    ksz = np.bincount(kl, minlength=K).astype(np.int64)
    ssz = np.bincount(sl, minlength=S).astype(np.int64)

    # each Krylov sector must lie inside a single symmetry sector
    s1 = np.bincount(kl, weights=sl.astype(np.float64), minlength=K)
    s2 = np.bincount(kl, weights=(sl.astype(np.float64)) ** 2, minlength=K)
    constant = bool(np.all(np.abs(s2 * ksz - s1 ** 2) < 1e-6))
    sym_of_k = np.rint(s1 / ksz).astype(np.int64)

    dmax_by_s = np.zeros(S, dtype=np.int64)
    np.maximum.at(dmax_by_s, sym_of_k, ksz)
    sumsq_by_s = np.bincount(sym_of_k, weights=ksz.astype(np.float64) ** 2,
                             minlength=S)

    big = int(np.argmax(ssz))
    total = float(1 << N)
    ratios = dmax_by_s / ssz
    explored = float((sumsq_by_s / ssz).sum() / total)

    gmax = int(ksz.max())
    gsym = int(sym_of_k[int(np.argmax(ksz))])

    return {
        "rule": rule, "N": N, "bc": bc, "r": r, "p": p,
        "n_charges": len(qs),
        "n_krylov": K, "n_sym": S,
        "krylov_inside_sym": constant,
        "dim_sym_max": int(ssz[big]),
        "d_max_in_dominant": int(dmax_by_s[big]),
        "ratio_dominant": float(dmax_by_s[big] / ssz[big]),
        "ratio_worst": float(ratios.max()),
        "explored_fraction": explored,
        "d_max_global": gmax,
        "crude_ratio": gmax / total,
        "ratio_of_global_max": float(gmax / ssz[gsym]),
        "sym_share_of_space": float(ssz[big] / total),
        "dim_sym_max_times_N_over_2N": float(ssz[big] * N / total),
    }


# --- 4b. the two unfragmented rules, in closed form ---------------------------
#
# W150 = IVVI fires when the two neighbours DIFFER; W105 = VIIV fires when they
# AGREE.  In the bond variables of the padded chain,
#
#     u_i = s_i XOR s_{i+1},   i = 0 .. N        (N+1 bonds)
#
# the map x -> u is a bijection from the 2^N states onto the EVEN-WEIGHT vectors
# of {0,1}^{N+1}, because obc0 pins s_0 = s_{N+1} = 0.  Flipping site i toggles
# u_{i-1} and u_i, so:
#
#   W150  toggles (u_{i-1}, u_i) exactly when they DIFFER -- a wall HOPS, and
#         weight(u) = D is conserved;
#   W105  toggles (u_{i-1}, u_i) exactly when they AGREE  -- a wall PAIR is
#         created or destroyed, and D is NOT conserved.  But in the STAGGERED
#         bond variable v_i = u_i XOR (i mod 2) the condition u_{i-1} = u_i
#         reads v_{i-1} != v_i, so W105 is the same hopping model in v, and
#         weight(v) is conserved.
#
# Both are therefore the identical model -- adjacent transpositions on a chain
# of N+1 bonds -- which is why in each case the level sets of the single charge
# ARE the Krylov sectors, and why the sector sizes are binomial coefficients.
# The only difference is WHICH residue class of weights the constraint allows:
#
#     weight(u) even   <=>   weight(v) = ceil(N/2)  (mod 2).
#
# Everything below is obc0 only; on the ring the bijection and the constraint
# both change, and the level sets stop being the sectors (see the pbc table).

def _eps(N: int) -> int:
    """The residue class weight(v) is confined to: ceil(N/2) mod 2."""
    return (-(-N // 2)) % 2


def sector_distribution(rule: int, N: int) -> Dict[int, int]:
    """{charge value: sector size} in closed form, obc0.

    W150 is labelled by the domain-wall number D = weight(u), which is even:

        size(D) = C(N+1, D),      D = 0, 2, 4, ..., <= N+1.

    W105 is labelled by the staggered wall number S = weight(v) - ceil(N/2),
    which is also always even:

        size(S) = C(N+1, S + ceil(N/2)),
        S + ceil(N/2) = 0 .. N+1  in steps of 2.

    So both distributions are a single parity class of row N+1 of Pascal's
    triangle; they differ only in which class.
    """
    from math import comb
    if rule == 150:
        return {w: comb(N + 1, w) for w in range(0, N + 2, 2)}
    if rule == 105:
        c = -(-N // 2)
        return {k - c: comb(N + 1, k) for k in range(_eps(N), N + 2, 2)}
    raise ValueError("closed form is for W150 and W105 only")


def measured_distribution(rule: int, N: int) -> Dict[int, int]:
    """The same thing from the engine: charge value -> Krylov sector size."""
    from . import wall_charges as WC
    charge = WC.charge_total if rule == 150 else WC.charge_staggered
    kl = krylov_labels(rule, N, BC_DEFAULT)
    q = np.array([charge(x, N, BC_DEFAULT) for x in range(1 << N)])
    out: Dict[int, int] = {}
    for k in range(int(kl.max()) + 1):
        vals = q[kl == k]
        assert int(vals.min()) == int(vals.max()), (rule, N, "charge not constant")
        out[int(vals[0])] = int(vals.size)
    return out


def n_sectors_closed(rule: int, N: int) -> int:
    """W150: floor((N+1)/2) + 1.  W105: the count of its parity class."""
    if rule == 150:
        return (N + 1) // 2 + 1
    if rule == 105:
        return (N + 1 - _eps(N)) // 2 + 1
    raise ValueError("closed form is for W150 and W105 only")


def d_max_closed(rule: int, N: int) -> int:
    """The largest sector -- the largest binomial in the allowed parity class.

    W105 always reaches the CENTRAL binomial, because floor((N+1)/2) =
    ceil(N/2) is exactly the residue its weights are confined to.  W150 reaches
    it too whenever N+1 is odd (the peak value then sits at two indices of
    opposite parity, so both classes get it) or N+1 = 0 mod 4.  It misses it
    exactly at N = 1 (mod 4), which is why the two part company at N = 9 (210
    against 252) and N = 13 (3003 against 3432) -- the divergence R20 observed
    between A214282 and the central binomials A001405.
    """
    return max(sector_distribution(rule, N).values())


def n_frozen_closed(rule: int, N: int) -> int:
    """Size-1 sectors: the ends of the row, w = 0 and w = N+1, when the parity
    class admits them.  Reproduces R19's period-2 and period-4 laws."""
    return sum(1 for v in sector_distribution(rule, N).values() if v == 1)


def distributions_agree(rule: int, N: int) -> bool:
    return measured_distribution(rule, N) == sector_distribution(rule, N)


def same_multiset(N: int) -> bool:
    """W150 and W105 have the same sector-SIZE multiset unless N = 1 (mod 4).

    The reflection w -> N+1-w is a symmetry of row N+1, and it swaps the two
    parity classes exactly when N+1 is odd.  So the multisets differ only when
    the classes differ (eps = 1, i.e. N = 1, 2 mod 4) AND N is odd -- i.e. at
    N = 1 (mod 4), which is precisely the residue at which R19 found W105 to
    have NO frozen states.  It is the same fact: there the class excludes both
    w = 0 and w = N+1.
    """
    a = sorted(sector_distribution(150, N).values())
    b = sorted(sector_distribution(105, N).values())
    return a == b


def write_distribution_table(outdir: str, Ns: Sequence[int] = (8, 9, 10, 11)):
    """The two distributions side by side, as the report prints them."""
    os.makedirs(outdir, exist_ok=True)
    rows = []
    for N in Ns:
        for rule in (150, 105):
            d = sector_distribution(rule, N)
            sizes = ", ".join(str(d[k]) for k in sorted(d))
            lab = "$D$" if rule == 150 else "$S$"
            keys = ", ".join(str(k) for k in sorted(d))
            rows.append(f"$W_{{{rule}}}$ & {N} & {lab} & {keys} & {sizes} \\\\")
        rows.append("\\addlinespace")
    tab = ("\\setlength{\\tabcolsep}{4pt}\n"
           "\\begin{tabular}{llll l}\n\\toprule\n"
           "rule & $N$ & label & values & sector sizes \\\\\n\\midrule\n"
           + "\n".join(rows[:-1]) + "\n\\bottomrule\n\\end{tabular}\n")
    p = os.path.join(outdir, "tab_r21_distribution.tex")
    with open(p, "w") as f:
        f.write(tab)
    return p


# --- 5. sweep -----------------------------------------------------------------

def _decay(Ns: Sequence[int], ys: Sequence[float]) -> Optional[float]:
    """Geometric decay rate from the last few terms: y(N+1)/y(N)."""
    pts = [(n, y) for n, y in zip(Ns, ys) if y > 0]
    if len(pts) < 4:
        return None
    tail = pts[-4:]
    rs = [tail[i + 1][1] / tail[i][1] for i in range(len(tail) - 1)]
    return float(sum(rs) / len(rs))


def _decay2(Ns: Sequence[int], ys: Sequence[float]) -> Optional[float]:
    """Decay rate PER UNIT N, read off a grid of step 2.

    The parity split makes the N -> N+1 ratio zig-zag, so the rate is taken on
    one sublattice and square-rooted back to a per-site figure.
    """
    pts = [(n, y) for n, y in zip(Ns, ys) if y > 0]
    if len(pts) < 3:
        return None
    tail = pts[-3:]
    rs = [(tail[i + 1][1] / tail[i][1]) ** (1.0 / (tail[i + 1][0] - tail[i][0]))
          for i in range(len(tail) - 1)]
    return float(sum(rs) / len(rs))


def analyse(rule: int, bc: str = BC_DEFAULT,
            Ns: Sequence[int] = NS_DEFAULT,
            r: int = 2, p: int = 2) -> Dict:
    """Detect the charges at every range in RANGES and measure the ratios at
    each of them separately.

    Reporting one range would beg the question.  The strong/weak grading is
    defined at a FIXED finite range taken to N -> infinity, and a longer range
    always resolves at least as finely, so the honest presentation is the whole
    ladder r = 1, 2, 3, 4 with a verdict at each.
    """
    det = {(rr, par): stable_basis(rule, bc, rr, par, p=p)
           for rr in RANGES for par in (0, 1)}
    by_r = {}
    for rr in RANGES:
        rows = [strength(rule, N, bc, det[(rr, N % 2)]["nonconstant"], rr, p)
                for N in Ns]
        assert all(x["krylov_inside_sym"] for x in rows), (rule, rr,
                                                           "charge not conserved")
        ev = [x for x in rows if x["N"] % 2 == 0]
        by_r[str(rr)] = {
            "n_independent": [det[(rr, 0)]["n_independent"],
                              det[(rr, 1)]["n_independent"]],
            "stable": det[(rr, 0)]["stable"] and det[(rr, 1)]["stable"],
            "nonconstant_even": det[(rr, 0)]["nonconstant"],
            "nonconstant_odd": det[(rr, 1)]["nonconstant"],
            "rows": rows,
            # the parity split makes the raw N -> N+1 ratio zig-zag, so the
            # decay is read off the even sublattice and reported per step of 1
            "decay_dominant": _decay2([x["N"] for x in ev],
                                      [x["ratio_dominant"] for x in ev]),
            "decay_explored": _decay2([x["N"] for x in ev],
                                      [x["explored_fraction"] for x in ev]),
            "sym_growth": _decay2([x["N"] for x in ev],
                                  [float(x["n_sym"]) for x in ev]),
        }
    #: the richest symmetry available -- the version of the test that is
    #: hardest to pass.
    best = max(RANGES, key=lambda rr: max(by_r[str(rr)]["n_independent"]))
    rows = by_r[str(r)]["rows"]
    ev = [x for x in rows if x["N"] % 2 == 0]
    return {
        "rule": rule, "bc": bc, "r": r, "p": p, "r_richest": best,
        "tuple": "".join(wolfram_to_tuple(rule)),
        "by_range": by_r,
        "rows": rows,
        "decay_dominant": by_r[str(r)]["decay_dominant"],
        "decay_explored": by_r[str(r)]["decay_explored"],
        "decay_crude": _decay2([x["N"] for x in ev],
                               [x["crude_ratio"] for x in ev]),
    }


def verdict(entry: Dict, r: Optional[int] = None, *,
            floor: float = 0.95) -> str:
    """strong / weak / marginal, from the decay of the dominant-sector ratio.

    A geometric decay rate comfortably below 1 is strong fragmentation; a ratio
    that does not decay at all is weak; a rate creeping towards 1 is the
    polynomial (marginal) case, which is neither.

    The numerical rate is a finite-N reading and NOT the whole argument -- see
    `theorem_bound`, which settles the same question asymptotically and does not
    depend on the window N <= 20.
    """
    b = entry["by_range"][str(r if r is not None else entry["r"])]
    d = b["decay_dominant"]
    row = b["rows"][-1]
    last = row["ratio_dominant"]
    if row["n_krylov"] == row["n_sym"]:
        # every Krylov sector is a symmetry sector: there is no fragmentation
        # here at all, only a conventional conservation law.  Calling this
        # "weak" would be wrong -- weak fragmentation still HAS small sectors
        # beside the thermal one.
        return "unfragmented"
    if d is None:
        return "undecided"
    if d >= 0.995 and last > 0.05:
        return "weak"
    if d < floor:
        return "strong"
    return "marginal (polynomial)"


def theorem_bound(rule: int, base: float, m: int) -> Dict:
    """The asymptotic argument, which no finite-N fit can supply.

    Fix the range r and the unit cell p.  Every charge in the ansatz is a sum of
    at most L = N + 2 bounded terms, so each takes O(N) distinct values, and m
    of them have at most O(N^m) joint level sets -- POLYNOMIAL in N, for any
    fixed range.  Hence the largest symmetry sector holds at least
    2^N / O(N^m) states and

        D_max(s*) / dim(s*)  <=  O( N^m * D_max / 2^N ).

    So whenever D_max grows like base^N with base < 2 the ratio vanishes
    exponentially and the fragmentation is STRONG -- the symmetry can only ever
    remove a polynomial factor, never an exponential one.  Conversely a rule
    whose D_max is Theta(2^N / poly) can never be strong by this route, which is
    exactly what separates the controls.
    """
    return {"rule": rule, "d_max_base": base, "n_charges": m,
            "ratio_base": base / 2.0,
            "strong": base < 2.0,
            "statement": (f"D_max(s*)/dim(s*) = O(N^{m} * "
                          f"({base:.6f}/2)^N) = O(N^{m} * "
                          f"{base / 2.0:.6f}^N)")}


#: The D_max growth base of each rule in play, derived (R18/R19, and confirmed
#: against the OEIS entries in R20) rather than fitted.
DMAX_BASE = {108: PHI, 201: PHI, 156: QUARTIC, 198: QUARTIC,
             60: 2.0, 102: 2.0, 150: 2.0, 105: 2.0}


def annotate(d: Dict) -> Dict:
    """Attach the verdicts and the asymptotic bound.  Kept separate from
    `build` so that a cached run can be re-graded without recomputing it."""
    r = d["r"]
    for e in d["rules"]:
        rule = int(e["rule"])
        e["verdict"] = verdict(e)
        e["verdict_by_range"] = {str(rr): verdict(e, rr) for rr in RANGES}
        e["theorem"] = theorem_bound(
            rule, DMAX_BASE[rule], max(e["by_range"][str(r)]["n_independent"]))
    return d


def build(bc: str = BC_DEFAULT, Ns: Sequence[int] = NS_DEFAULT,
          rules: Sequence[int] = RULES + CONTROLS, r: int = 2,
          p: int = 2) -> Dict:
    d = {"bc": bc, "Ns": list(Ns), "r": r, "p": p,
         "rules": [analyse(rule, bc, Ns, r, p) for rule in rules]}
    annotate(d)
    os.makedirs(sectors.ANALYTICS, exist_ok=True)
    with open(OUT_JSON.format(bc=bc), "w") as f:
        json.dump(d, f, indent=1, default=str)
    return d


def load(bc: str = BC_DEFAULT) -> Optional[Dict]:
    p = OUT_JSON.format(bc=bc)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


# --- 6. figure ----------------------------------------------------------------

_COL = {108: "#eda100", 201: "#006400", 156: "#e34948", 198: "#e34948",
        60: "#2a78d6", 102: "#2a78d6", 150: "#4a3aa7", 105: "#4a3aa7"}
_MK = {108: "s", 201: "^", 156: "o", 198: "o",
       60: "D", 102: "D", 150: "v", 105: "v"}
_LS = {108: "-", 201: "-", 156: "-", 198: (0, (4, 2)),
       60: "--", 102: (0, (1, 1)), 150: "--", 105: (0, (1, 1))}


def _style(ax):
    ax.grid(True, color="#e9e9e6", linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_strength(bc: str, out: str, d: Optional[Dict] = None):
    d = d or load(bc) or build(bc)
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.4))

    for ax, key, title, ylab in (
            (axes[0], "crude_ratio",
             "(a)  the crude ratio $D_{\\max}/2^N$", "$D_{\\max}/2^N$"),
            (axes[1], "ratio_dominant",
             "(b)  the criterion: $D_{\\max}(s^*)/\\dim s^*$",
             "largest Krylov / largest symmetry sector"),
            (axes[2], "explored_fraction",
             "(c)  fraction of its symmetry sector a random state reaches",
             r"$\langle D_K(x)/\dim s(x)\rangle$")):
        _style(ax)
        for e in d["rules"]:
            rule = int(e["rule"])
            Ns = [x["N"] for x in e["rows"]]
            ys = [x[key] for x in e["rows"]]
            ax.plot(Ns, ys, marker=_MK[rule], ms=4.5, lw=1.6,
                    ls=_LS[rule], color=_COL[rule],
                    label=f"$W_{{{rule}}}$")
        ax.set_yscale("log")
        ax.set_xlabel("$N$")
        ax.set_ylabel(ylab, fontsize=8.5)
        ax.set_title(title, fontsize=9.5)
    for ax in axes[1:]:
        ax.axhline(1.0, color="#444444", ls=":", lw=1.0)
        ax.axhline(0.5, color="#444444", ls=":", lw=1.0)
    axes[1].annotate("$1$: $W_{150}$, $W_{105}$ --- Krylov $=$ symmetry, "
                     "unfragmented", (0.03, 1.0),
                     xycoords=("axes fraction", "data"), fontsize=7.0,
                     va="top", color="#444444")
    axes[1].annotate("$1/2$: $W_{60}$, $W_{102}$ --- weak", (0.03, 0.5),
                     xycoords=("axes fraction", "data"), fontsize=7.0,
                     va="bottom", color="#444444")

    # the rates the asymptotic bound predicts, anchored at the right-hand end,
    # so that the measured slopes can be read against them directly
    Ns = [x["N"] for x in d["rules"][0]["rows"]]
    for ax in (axes[1], axes[2]):
        for base, lab, col in ((PHI, r"$(\varphi/2)^N$", "#eda100"),
                               (QUARTIC, r"$(4^{1/5}/2)^N$", "#e34948")):
            src = next(e for e in d["rules"]
                       if DMAX_BASE[int(e["rule"])] == base)
            key = ("ratio_dominant" if ax is axes[1] else "explored_fraction")
            anchor = src["rows"][-1][key]
            ys = [anchor * (base / 2.0) ** (n - Ns[-1]) for n in Ns]
            ax.plot(Ns, ys, ls=(0, (5, 3)), lw=1.0, color=col, alpha=0.55,
                    zorder=0)
        ax.annotate(r"dashed guides: $(\varphi/2)^N$ and $(4^{1/5}/2)^N$",
                    (0.03, 0.04), xycoords="axes fraction", ha="left",
                    fontsize=6.9, color="#555555")
    handles, labels = axes[0].get_legend_handles_labels()
    seen, h2, l2 = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            h2.append(h)
            l2.append(l)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend(h2, l2, frameon=False, fontsize=8.5, ncols=8,
               loc="lower center", bbox_to_anchor=(0.5, -0.01),
               handlelength=2.4, columnspacing=1.4)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


# --- 7. tables ----------------------------------------------------------------

def write_tables(bc: str, outdir: str, d: Optional[Dict] = None) -> List[str]:
    d = d or load(bc) or build(bc)
    os.makedirs(outdir, exist_ok=True)
    paths = []

    rows = []
    for e in d["rules"]:
        last = e["rows"][-1]
        m = " / ".join(str(max(e["by_range"][str(rr)]["n_independent"]))
                       for rr in RANGES)
        rows.append(
            f"$W_{{{e['rule']}}}$ & \\rt{{{e['tuple']}}} & "
            f"{DMAX_BASE[int(e['rule'])]:.4f} & {m} & "
            f"{last['n_sym']} & {last['n_krylov']} & "
            f"{last['ratio_dominant']:.2e} & "
            f"{e['decay_dominant']:.3f} & "
            f"{DMAX_BASE[int(e['rule'])] / 2:.3f} & {e['verdict']} \\\\")
    nmax = d["Ns"][-1]
    tab = ("\\setlength{\\tabcolsep}{4pt}\n"
           "\\begin{tabular}{llrlrrrrrl}\n\\toprule\n"
           " & & $D_{\\max}$ & charges & "
           "\\multicolumn{2}{c}{sectors at $N=" + str(nmax) + "$}"
           " & & & & \\\\\n"
           "\\cmidrule(lr){5-6}\n"
           "rule & tuple & base $\\lambda$ & $r=1/2/3/4$ & sym. & Krylov & "
           "$D_{\\max}(s^*)/\\dim s^*$ & decay & $\\lambda/2$ & "
           "verdict \\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    p = os.path.join(outdir, f"tab_r21_strength_{bc}.tex")
    with open(p, "w") as f:
        f.write(tab)
    paths.append(p)

    rows = []
    for e in d["rules"]:
        if int(e["rule"]) not in RULES:
            continue
        for x in e["rows"]:
            if x["N"] % 4 != 0:
                continue
            rows.append(
                f"$W_{{{e['rule']}}}$ & {x['N']} & {x['n_sym']} & "
                f"{x['dim_sym_max']} & {x['d_max_in_dominant']} & "
                f"{x['ratio_dominant']:.4f} & {x['crude_ratio']:.4f} & "
                f"{x['explored_fraction']:.4f} \\\\")
    tab2 = ("\\begin{tabular}{lrrrrrrr}\n\\toprule\n"
            "rule & $N$ & sym.\\ sectors & $\\dim s^*$ & $D_{\\max}(s^*)$ & "
            "$D_{\\max}(s^*)/\\dim s^*$ & $D_{\\max}/2^N$ & explored \\\\\n"
            "\\midrule\n" + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}\n")
    p = os.path.join(outdir, f"tab_r21_ratios_{bc}.tex")
    with open(p, "w") as f:
        f.write(tab2)
    paths.append(p)

    rows = []
    for e in d["rules"]:
        for rr in RANGES:
            b = e["by_range"][str(rr)]
            dd = b["decay_dominant"]
            rows.append(
                f"$W_{{{e['rule']}}}$ & {rr} & "
                f"{max(b['n_independent'])} & {b['rows'][-1]['n_sym']} & "
                f"{b['rows'][-1]['ratio_dominant']:.4f} & "
                f"{'---' if dd is None else f'{dd:.3f}'} & "
                f"{e['verdict_by_range'][str(rr)]} \\\\")
    tab3 = ("\\begin{tabular}{lrrrrrl}\n\\toprule\n"
            "rule & range $r$ & charges & sym.\\ sectors & "
            "$D_{\\max}(s^*)/\\dim s^*$ & decay & verdict \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    p = os.path.join(outdir, f"tab_r21_ranges_{bc}.tex")
    with open(p, "w") as f:
        f.write(tab3)
    paths.append(p)
    return paths


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(
        description="strong vs weak fragmentation for 108/201/156/198")
    ap.add_argument("--bc", default=BC_DEFAULT)
    ap.add_argument("--n-max", type=int, default=max(NS_DEFAULT))
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--r", type=int, default=2)
    args = ap.parse_args(argv)
    Ns = tuple(n for n in NS_DEFAULT if n <= args.n_max)
    d = (build(args.bc, Ns, r=args.r) if args.rebuild
         else annotate(load(args.bc) or build(args.bc, Ns, r=args.r)))

    for e in d["rules"]:
        last = e["rows"][-1]
        m = [max(e["by_range"][str(rr)]["n_independent"]) for rr in RANGES]
        print(f"W{e['rule']:<4d} {e['tuple']}  charges r=1..4: {m}  "
              f"N={last['N']}  sym={last['n_sym']:5d} "
              f"krylov={last['n_krylov']:7d}  "
              f"Dmax(s*)/dim(s*)={last['ratio_dominant']:.3e}  "
              f"decay={e['decay_dominant']:.3f} "
              f"(base/2 = {DMAX_BASE[int(e['rule'])] / 2:.3f})  ->  "
              f"{e['verdict']}")
        print("      " + e["theorem"]["statement"]
              + "   by range: "
              + ", ".join(f"r={rr}: {e['verdict_by_range'][str(rr)]}"
                          for rr in RANGES))
    figdir = os.path.join(results_io.REPO_ROOT, "figures")
    os.makedirs(figdir, exist_ok=True)
    fig_strength(args.bc, os.path.join(figdir, f"fig_hsf_strength_{args.bc}"), d)
    texdir = os.path.join(results_io.REPO_ROOT, "reports", "tex")
    for p in write_tables(args.bc, texdir, d):
        print("wrote", p)
    if args.bc == BC_DEFAULT:
        print("wrote", write_distribution_table(texdir))


if __name__ == "__main__":
    main()
