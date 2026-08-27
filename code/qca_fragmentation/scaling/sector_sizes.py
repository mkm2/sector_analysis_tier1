r"""
The sector-size DISTRIBUTION of the four one-Hadamard rules, exactly and
asymptotically, at obc0 and at pbc.

R9 gave the sector COUNT and the largest sector; R18 showed both are consequences
of the wall grammar.  This module gives the whole distribution -- how many Krylov
sectors have each size -- and shows that a single 2x2 matrix decides everything.

--------------------------------------------------------------------------------
1.  THE WALL WORD IS A LENGTH-2 WORD, SO ITS INDICATOR MATRIX HAS RANK ONE
--------------------------------------------------------------------------------

Each rule freezes one two-letter word (R18):

    W_108: `00`     W_201: `11`     W_156: `01`     W_198: `10`

and the sector of a state is exactly the set of positions at which that word
occurs -- on the PADDED chain at obc0 (y_0 = y_{N+1} = 0), on the ring at pbc.
Because the word reads the pair (y_i, y_{i+1}), the label letter at i is a
function of one BOND, so the preimage of a label word w is the set of paths of a
two-node graph with position-dependent edge sets:

    M_w[u][v] = 1  iff  [ (u,v) == word ] == w ,        w in {0,1}

    |sector(w)|  =  <l| M_{w_0} M_{w_1} ... M_{w_N} |r>          (obc0)
    |sector(w)|  =  tr( M_{w_0} ... M_{w_{N-1}} )                (pbc)

with l = r = e_0 at obc0.  Writing the word as (a,b),

    M_1 = |a><b|   is RANK ONE,        M_0 = J - |a><b| ,

J the all-ones matrix.  Rank one is the whole point: every wall CUTS the matrix
product, so the size factorises over the gaps between consecutive walls,

    |sector| = L(g_0) * K(g_1) * ... * K(g_{k-1}) * R(g_k)        (obc0)
    |sector| = K(g_1) * ... * K(g_k)   (cyclically)               (pbc, k >= 1)

    K(g) = (M_0^g)_{b,a},   L(g) = (M_0^g)_{0,a},   R(g) = (M_0^g)_{b,0},
    T(m) = (M_0^m)_{0,0}  or  tr(M_0^m)   for the wall-free label.

g_j is the NUMBER OF NON-WALL LETTERS between two walls, so g = 0 means two
adjacent walls.  Everything below is a statement about powers of one 2x2 matrix.

--------------------------------------------------------------------------------
2.  THE TWO FAMILIES ARE THE TWO JORDAN TYPES OF  J - |a><b|
--------------------------------------------------------------------------------

    W_201  word 11:  M_0 = [[1,1],[1,0]]     eigenvalues  phi, -1/phi   HYPERBOLIC
    W_108  word 00:  M_0 = [[0,1],[1,1]]     eigenvalues  phi, -1/phi   HYPERBOLIC
    W_156  word 01:  M_0 = [[1,0],[1,1]]     eigenvalue   1 (defective) PARABOLIC
    W_198  word 10:  M_0 = [[1,1],[0,1]]     eigenvalue   1 (defective) PARABOLIC

The word is DIAGONAL (a == b) exactly for 108/201 and OFF-DIAGONAL for 156/198,
and that single bit is the fork:

    diagonal word     ->  M_0 is a Fibonacci matrix, K(g) = F(g-1)   EXPONENTIAL
    off-diagonal word ->  M_0 is unipotent,          K(g) = g        LINEAR

Every difference between the two families follows.  The gap kernel is the
transfer weight of a wall-free stretch, so:

  * D_max.  108/201: the wall-free label is one huge sector, F(N+2) at obc0
    (F(N) for 108, whose padding costs two sites) and L(N) -- Lucas -- at pbc,
    i.e. phi^N.  156/198: a wall-free stretch is worth only its LENGTH, so the
    biggest sector is the best product of gap lengths at one site of cost per
    gap: max prod g_j with sum (g_j + 1) = N, which is 4^{N/5} = 1.31951^N.
  * n_frozen (size-1 sectors).  K(g) = 1 for g in {0,2,3} (F(-1)=F(1)=F(2)=1)
    for 108/201 -- an exponential, phi^N, family of frozen states -- but only
    for g = 1 for 156/198, which forces a rigid alternating wall pattern and
    leaves LINEARLY many frozen sectors (exactly 4 at even N, 2 at odd N, on the
    ring).
  * the sector count itself: lambda_0 = phi for 156/198 (gaps >= 1, i.e. no two
    adjacent walls -> Fibonacci) and rho^2 = 1.754878 for 108/201 (gaps != 1,
    i.e. binary words avoiding `101` -- R18's a(N) = 2a(N-1) - a(N-2) + a(N-3)).

--------------------------------------------------------------------------------
3.  THE GENERATING FUNCTION, AND THE DISTRIBUTION AT LARGE N
--------------------------------------------------------------------------------

Renewal decomposition with one variable per label letter, so a wall plus the gap
after it costs x^{g+1}:

    B_s(x) = sum_{g >= 0}  K(g)^s x^{g+1}          (bulk: wall + following gap)
    A_s(x) = sum_{g >= 0}  L(g)^s x^g              (before the first wall)
    C_s(x) = sum_{g >= 0}  R(g)^s x^{g+1}          (last wall + tail)
    E_s(x) = sum_{m >= 0}  T(m)^s x^m              (no wall at all)

    sum_N x^{N+1} Z_N^{obc0}(s) = E_s + A_s C_s / (1 - B_s)
    sum_N x^N     Z_N^{pbc}(s)  = x B_s'(x) / (1 - B_s)  +  sum_N tr(M_0^N)^s x^N

where Z_N(s) = sum over sectors of |sector|^s.  s = 1 must give 2^N and does:
for the parabolic family B_1 = x^2/(1-x)^2, whose root is x = 1/2 on the nose.

Consequences, all from ONE scalar equation B_s(1/lambda_s) = 1:

  * Z_N(s) ~ const * lambda_s^N, so tau(s) = ln lambda_s is the free energy of
    the size distribution.  tau(0) = ln n_sectors per site, tau(1) = ln 2 always.
  * ln|sector| over sectors is asymptotically NORMAL: the label is a renewal
    chain and ln|sector| = sum_j ln K(g_j) is a renewal-reward sum.  Hence

        |sector| = exp( alpha_0 N + sigma sqrt(N) * xi ),   xi ~ N(0,1),

    a LOG-NORMAL law with alpha_0 = tau'(0) and sigma^2 = tau''(0).  Both are
    bulk quantities, so they are the SAME at obc0 and pbc; the boundary moves
    only prefactors and the extremes.
  * the full large-deviation form, #{sectors of size e^{alpha N}} = e^{N g(alpha)}
    with g the Legendre transform g(alpha) = inf_s (tau(s) - s alpha).  Its right
    edge is alpha_max = D_max exponent (g = 0, one sector); its left edge is
    alpha = 0 with g(0) = ln lambda_{-inf} = the frozen-sector growth rate.
  * the sector containing a UNIFORMLY RANDOM STATE has typical size
    exp(tau'(1) N) -- the size-weighted, not the sector-weighted, typical value.
    tau'(1) < ln 2 strictly is exactly strong fragmentation (R21).

--------------------------------------------------------------------------------
4.  WHAT IS PROVED HERE AND WHAT IS MEASURED
--------------------------------------------------------------------------------

`sector_is_wall_set` proves the label is a complete invariant for these four
rules (constancy: no move can create or destroy an occurrence of the word;
transitivity: the free stretches are exactly the independent-set / kink-position
spaces, which single flips connect).  R18 had it as an exhaustive check to
N = 14.  Given that, everything above is a derivation, and `hist()` is exact for
any N without touching the 2^N space; `hist_bruteforce()` re-derives the same
multiset from the flip graph and is the regression anchor.

COST, measured, not assumed.  `hist` carries one DP state per distinct partial
matrix product, and that count grows fast -- 1.0e4 at N = 40, 6.4e5 at N = 80,
1.3e7 at N = 120 for W156 at obc0, about 5.8 GB.  The number of distinct SIZES
is polynomial (540 at N = 32), the number of distinct partial products is not.
Treat N = 100 as the working ceiling and use `moment_Z` -- the s-fold tensor
transfer matrix, exact and O(4^s log N) -- for anything larger.
"""
from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from . import sectors

#: rule -> the minimal wall word, as the bond pattern (y_i, y_{i+1}).
WALL_WORD: Dict[int, Tuple[int, int]] = {
    108: (0, 0), 201: (1, 1), 156: (0, 1), 198: (1, 0),
}
RULES: Tuple[int, ...] = (156, 198, 108, 201)

#: which family a rule belongs to, i.e. the Jordan type of M_0.
FAMILY = {156: "parabolic", 198: "parabolic", 108: "hyperbolic",
          201: "hyperbolic"}

OUT_JSON = os.path.join(sectors.ANALYTICS, "sector_sizes_{bc}.json")

Mat = Tuple[Tuple[int, int], Tuple[int, int]]


# --- 1. the two transfer matrices --------------------------------------------

def wall_matrices(rule: int) -> Tuple[Mat, Mat]:
    """(M_0, M_1) for the wall word of `rule`.  M_1 is a matrix unit."""
    a, b = WALL_WORD[rule]
    m1 = [[0, 0], [0, 0]]
    m1[a][b] = 1
    m0 = [[1, 1], [1, 1]]
    m0[a][b] = 0
    return (tuple(map(tuple, m0)), tuple(map(tuple, m1)))


def _mul(A: Mat, B: Mat) -> Mat:
    return ((A[0][0] * B[0][0] + A[0][1] * B[1][0],
             A[0][0] * B[0][1] + A[0][1] * B[1][1]),
            (A[1][0] * B[0][0] + A[1][1] * B[1][0],
             A[1][0] * B[0][1] + A[1][1] * B[1][1]))


def m0_power(rule: int, g: int) -> Mat:
    """M_0^g by repeated squaring (exact integers)."""
    if g < 0:
        raise ValueError("g must be >= 0")
    m0, _ = wall_matrices(rule)
    out: Mat = ((1, 0), (0, 1))
    while g:
        if g & 1:
            out = _mul(out, m0)
        m0 = _mul(m0, m0)
        g >>= 1
    return out


def gap_kernel(rule: int, g: int) -> int:
    """K(g) = (M_0^g)_{b,a}: the weight of a stretch of g non-wall letters
    between two walls.  K(g) = F(g-1) for a diagonal word, K(g) = g for an
    off-diagonal one."""
    a, b = WALL_WORD[rule]
    return m0_power(rule, g)[b][a]


def left_kernel(rule: int, g: int) -> int:
    """L(g) = (M_0^g)_{0,a}: from the frozen 0 at the left pad to the first wall."""
    a, _b = WALL_WORD[rule]
    return m0_power(rule, g)[0][a]


def right_kernel(rule: int, g: int) -> int:
    """R(g) = (M_0^g)_{b,0}: from the last wall to the frozen 0 at the right pad."""
    _a, b = WALL_WORD[rule]
    return m0_power(rule, g)[b][0]


def free_kernel(rule: int, m: int, bc: str) -> int:
    """T(m): the wall-free label -- (M_0^m)_{0,0} at obc0, tr(M_0^m) at pbc."""
    P = m0_power(rule, m)
    return P[0][0] if bc == "obc0" else P[0][0] + P[1][1]


# --- 2. the size of one sector, from its label -------------------------------

def size_from_label(rule: int, N: int, bc: str, walls: Sequence[int]) -> int:
    """|sector| from the renewal product.  `walls` are the wall positions:
    0..N at obc0 (padded chain, N+1 letters), 0..N-1 at pbc.

    Returns 0 for a label that no state realises.  At pbc the empty label of
    156/198 returns 2, which is the R18 ring defect: two frozen states sharing
    one label, i.e. TWO sectors of size 1 (see `hist_pbc`)."""
    w = sorted(walls)
    n_letters = N + 1 if bc == "obc0" else N
    if any(not 0 <= p < n_letters for p in w) or len(set(w)) != len(w):
        raise ValueError(f"wall positions out of range for {bc} N={N}: {walls}")
    if not w:
        return free_kernel(rule, n_letters, bc)
    if bc == "obc0":
        out = left_kernel(rule, w[0])
        for j in range(len(w) - 1):
            out *= gap_kernel(rule, w[j + 1] - w[j] - 1)
        return out * right_kernel(rule, N - w[-1])
    out = 1
    for j in range(len(w)):
        nxt = w[(j + 1) % len(w)] + (N if j == len(w) - 1 else 0)
        out *= gap_kernel(rule, nxt - w[j] - 1)
    return out


def enumerate_sectors(rule: int, N: int, bc: str):
    """(label, size) for every realisable label.  Exponential in N -- this is
    the direct reading of the renewal formula, used to pin `hist`."""
    n_letters = N + 1 if bc == "obc0" else N
    for mask in range(1 << n_letters):
        walls = [i for i in range(n_letters) if (mask >> i) & 1]
        size = size_from_label(rule, N, bc, walls)
        if size:
            yield walls, size


# --- 3. the exact histogram in polynomial time -------------------------------

def hist_obc0(rule: int, N: int) -> Dict[int, int]:
    """{size: multiplicity} at obc0.  DP over the running row vector
    e_0^T prod M_w; label words that die (zero vector) are dropped, which is the
    subset construction that a moment-only method cannot do."""
    m0, m1 = wall_matrices(rule)
    cur: Dict[Tuple[int, int], int] = {(1, 0): 1}
    for _ in range(N + 1):
        nxt: Dict[Tuple[int, int], int] = defaultdict(int)
        for v, mult in cur.items():
            for M in (m0, m1):
                u = (v[0] * M[0][0] + v[1] * M[1][0],
                     v[0] * M[0][1] + v[1] * M[1][1])
                if u != (0, 0):
                    nxt[u] += mult
        cur = nxt
    out: Counter = Counter()
    for v, mult in cur.items():
        if v[0]:
            out[v[0]] += mult
    return dict(out)


def hist_pbc(rule: int, N: int) -> Dict[int, int]:
    """{size: multiplicity} at pbc, by the same DP on the full 2x2 product.

    THE RING DEFECT (R18).  For 156/198 the wall-free label has trace 2: the two
    uniform states 0^N and 1^N.  Neither contains the word, both are frozen, and
    they are NOT connected -- so that label carries two sectors of size 1, not
    one of size 2.  This is the only place where the label fails to separate
    sectors, and it is corrected here explicitly rather than silently."""
    m0, m1 = wall_matrices(rule)
    cur: Dict[Mat, int] = {((1, 0), (0, 1)): 1}
    for _ in range(N):
        nxt: Dict[Mat, int] = defaultdict(int)
        for A, mult in cur.items():
            for M in (m0, m1):
                B = _mul(A, M)
                if B != ((0, 0), (0, 0)):
                    nxt[B] += mult
        cur = nxt
    out: Counter = Counter()
    for A, mult in cur.items():
        tr = A[0][0] + A[1][1]
        if tr:
            out[tr] += mult
    if rule in (156, 198):
        assert out.get(2, 0) >= 1, "ring defect missing -- check the word"
        out[2] -= 1
        if not out[2]:
            del out[2]
        out[1] += 2
    return dict(out)


def hist(rule: int, N: int, bc: str) -> Dict[int, int]:
    """The exact sector-size histogram.  Polynomial in N."""
    if bc == "obc0":
        return hist_obc0(rule, N)
    if bc == "pbc":
        return hist_pbc(rule, N)
    raise ValueError(f"unknown boundary convention {bc!r}")


def hist_bruteforce(rule: int, N: int, bc: str) -> Dict[int, int]:
    """The same multiset from the 2^N flip graph.  Regression anchor only."""
    from ..core.rules import wolfram_to_tuple
    from ..graph.flip_graph import flip_components_np
    sizes = flip_components_np(N, wolfram_to_tuple(rule), bc)
    assert sum(sizes) == 1 << N
    return dict(Counter(sizes))


def moment_Z(rule: int, N: int, bc: str, s: int) -> int:
    """Z_N(s) = sum over sectors of |sector|^s, exactly, for INTEGER s >= 1.

    The s-th power of a scalar matrix element is a matrix element of the s-fold
    tensor power, and the sum over the label letter factorises, so

        Z_N(s) = <l|^{ox s} (M_0^{ox s} + M_1^{ox s})^{n_letters} |r>^{ox s}
                 (obc0)  /  tr of the same operator (pbc),

    a 2^s x 2^s integer matrix power -- independent of N in cost, up to the
    log N of the exponentiation.  Since M_0 + M_1 = J for every rule, s = 1
    gives J^{n_letters} and hence 2^N: the sum rule is a THEOREM of the
    construction, not a check on it.

    Caveat that makes this complementary rather than a replacement for `hist`:
    the tensor trick sums over all label WORDS.  Non-realisable words contribute
    0^s = 0 for s >= 1 and are harmless, but 0^0 = 1, so s = 0 (the sector
    COUNT) needs the subset construction in `hist` instead.  At pbc the two
    ascent rules also need the ring-defect correction (2 - 2^s), because the
    empty label carries two sectors of size 1 rather than one of size 2."""
    if s < 1:
        raise ValueError("moment_Z is for integer s >= 1; s = 0 needs hist()")
    import numpy as np
    m0, m1 = (np.array(M, dtype=object) for M in wall_matrices(rule))
    Ts = None
    for M in (m0, m1):
        P = np.array([[1]], dtype=object)
        for _ in range(s):
            P = np.kron(P, M)
        Ts = P if Ts is None else Ts + P
    n_letters = N + 1 if bc == "obc0" else N
    out = np.eye(Ts.shape[0], dtype=object)
    B, e = Ts, n_letters
    while e:
        if e & 1:
            out = out.dot(B)
        B = B.dot(B)
        e >>= 1
    if bc == "obc0":
        idx = 0                                   # e_0^{ox s}
        val = int(out[idx, idx])
    else:
        val = int(sum(out[i, i] for i in range(out.shape[0])))
        if rule in (156, 198):
            val += 2 - 2 ** s                     # the ring defect
    return val


def summary(rule: int, N: int, bc: str) -> Dict[str, int]:
    h = hist(rule, N, bc)
    tot = sum(k * v for k, v in h.items())
    assert tot == 1 << N, (rule, N, bc, tot)
    return {"n_sectors": sum(h.values()), "d_max": max(h),
            "n_frozen": h.get(1, 0), "total": tot}


# --- 4. closed forms ---------------------------------------------------------

def fib(n: int) -> int:
    """F(n) with F(0)=0, F(1)=1, extended to n = -1 by F(-1) = 1."""
    if n < 0:
        if n == -1:
            return 1
        raise ValueError("only F(-1) is needed")
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def lucas(n: int) -> int:
    a, b = 2, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def d_max_closed(rule: int, N: int, bc: str) -> int:
    """Largest sector, in closed form.

    HYPERBOLIC family: the wall-free label wins, because a wall costs a letter
    and cuts the Fibonacci growth (F(m) <= phi^{m-1}, so k walls lose a factor
    phi^{2k}).  Its size is T(n_letters).

    PARABOLIC family: a wall-free stretch is worth only its length, so the best
    label is the best product of gap lengths at a cost of one letter per gap,
    D(n) = max_g g * D(n - g - 1), maximised by parts of 4 -> 4^{N/5}."""
    if FAMILY[rule] == "hyperbolic":
        return free_kernel(rule, N + 1 if bc == "obc0" else N, bc)
    n_letters = N + 1 if bc == "obc0" else N
    if bc == "obc0":
        # size = prod_{j=1..k} g_j with p_1 + sum (g_j + 1) = N+1, p_1 >= 0
        best = [1] * (n_letters + 1)
        for n in range(1, n_letters + 1):
            best[n] = max([best[n - 1]] +
                          [g * best[n - g - 1]
                           for g in range(1, n) if n - g - 1 >= 0])
        return best[n_letters]
    # pbc: a cyclic sequence of k >= 1 gaps with sum (g_j + 1) = N
    best = 1
    for k in range(1, N // 2 + 1):
        rem, q, r = N - k, (N - k) // k, (N - k) % k
        best = max(best, (q + 1) ** r * q ** (k - r))
    return max(best, 1)


def n_frozen_closed(rule: int, N: int, bc: str) -> int:
    """Number of size-1 sectors = number of frozen states.

    A sector has size 1 iff every renewal factor is 1.  K(g) = 1 happens for
    g in {0, 2, 3} in the hyperbolic family (F(-1) = F(1) = F(2) = 1) -- an
    exponential, phi^N, family -- and only for g = 1 in the parabolic one, which
    pins the walls to a rigid alternating pattern."""
    if FAMILY[rule] == "parabolic":
        if bc == "obc0":
            return (N - 1) // 2 + 2
        return 4 if N % 2 == 0 else 2
    return multiplicity(rule, N, bc, 1)


def n_sectors_closed(rule: int, N: int, bc: str) -> int:
    """Sector count from the s = 0 renewal count (realisable labels only).

    Parabolic + obc0: labels are the independent sets of a path on N vertices,
    F(N+2).  Parabolic + pbc: independent sets of a cycle, L(N), plus 1 for the
    ring defect.  Hyperbolic: binary label words avoiding `101` (gap 1 is the
    only forbidden gap, since F(0) = 0), with rule-dependent end conditions."""
    if FAMILY[rule] == "parabolic":
        return fib(N + 2) if bc == "obc0" else lucas(N) + 1
    # a(N) = 2a(N-1) - a(N-2) + a(N-3), dominant root rho^2 = 1.754878 (R18),
    # at BOTH boundary conditions; only the seed differs.
    seed = {("108", "obc0"): (7, 12, 21), ("201", "obc0"): (4, 7, 12),
            ("108", "pbc"): (5, 10, 17), ("201", "pbc"): (5, 10, 17)}[
        (str(rule), bc)]                       # values at N = 3, 4, 5
    n0 = 3
    if N < n0:
        raise ValueError("closed form seeded from N = 3")
    a, b, c = seed
    for _ in range(N - n0):
        a, b, c = b, c, 2 * c - b + a
    return a


# --- 4b. n_N(sigma): how many sectors have exactly this size -----------------

def _divisors(n: int) -> List[int]:
    ds, d = [], 1
    while d * d <= n:
        if n % d == 0:
            ds.append(d)
            if d != n // d:
                ds.append(n // d)
        d += 1
    return sorted(ds)


def gap_menu(rule: int, cap: int) -> List[Tuple[int, int]]:
    """(gap length g, factor K(g)) for every gap with 1 <= K(g) <= cap.

    Finite in both families, and this is where the two part company:

      parabolic  K(g) = g       -> one gap per factor value, g = K, cost K+1
      hyperbolic K(g) = F(g-1)  -> factors are FIBONACCI numbers only, and the
                                   factor 1 is realised THREE ways, g in {0,2,3}
                                   (F(-1) = F(1) = F(2) = 1), at costs 1, 3, 4.

    So the set of achievable sector sizes is the multiplicative semigroup
    generated by the factor values -- ALL integers for 156/198, and products of
    Fibonacci numbers >= 2 for 108/201 -- PLUS the wall-free label, which is the
    one term that is not a product of kernels.  At obc0 that label is a corner of
    M_0^M and so Fibonacci too; at pbc it is a TRACE, hence the Lucas number
    L(N), which is generally NOT a Fibonacci product.  So on the ring 7 = L(4),
    11 = L(5), 29 = L(7), 199 = L(11) ... each occur, once, at that one N.
    Excluded at both boundaries are the integers that are neither Fibonacci
    products nor Lucas numbers: 14, 17, 19, 22, 23, 28, 31, 33, 35, 37, 38, ...
    (Tier-2 R-T20 rev.3 falsified the earlier, Fibonacci-only claim; the code was
    already right, the claim about it was not.)"""
    out, g = [], 0
    while True:
        k = gap_kernel(rule, g)
        if k > cap:
            break
        if k >= 1:
            out.append((g, k))
        g += 1
        if g > 4 * cap + 8:
            break
    return out


def multiplicity(rule: int, N: int, bc: str, sigma: int) -> int:
    """n_N(sigma): the number of Krylov sectors of size EXACTLY sigma.

    Exact at any N, in O(N d(sigma)).  The renewal factorisation says a sector
    of size sigma is an ordered factorisation of sigma into gap kernels, padded
    with kernel-1 gaps to fill the chain, so the DP only ever needs to carry a
    DIVISOR of sigma -- which is why this costs nothing while `hist`, which
    carries every partial product, blows up past N ~ 100.

    obc0: gaps g_0 .. g_k with sum g_j + k = N+1, weight L(g_0) K(g_1) ...
    K(g_{k-1}) R(g_k), plus the wall-free label of size T(N+1).
    pbc: the pointed-block identity.  Count triples (label, marked wall, offset
    inside that wall's block).  Summed one way that is N times the number of
    labels, because the blocks of a label tile the ring; summed the other way it
    is N times the sum over LINEAR block sequences of total cost N weighted by
    the cost of the first block.  The N cancels, so

        number of labels = sum over linear sequences of cost(first block),

    with no division by the number of walls -- which is what makes the cyclic
    count the same linear DP."""
    if sigma < 1:
        return 0
    divs = _divisors(sigma)
    pos = {d: i for i, d in enumerate(divs)}
    menu = gap_menu(rule, sigma)

    def blank():
        return [[0] * len(divs) for _ in range(N + 5)]

    if bc == "obc0":
        total = N + 1
        # start: the pre-first-wall stretch, weight L(g_0), cost g_0
        cur = blank()
        for g in range(total + 1):
            v = left_kernel(rule, g)
            if v and sigma % v == 0:
                cur[g][pos[v]] += 1
        # any number of interior (wall + gap) blocks, weight K(g), cost g+1
        run = [row[:] for row in cur]
        for t in range(total + 1):
            for i, d in enumerate(divs):
                n = run[t][i]
                if not n:
                    continue
                for g, k in menu:
                    nd, nt = d * k, t + g + 1
                    if nt <= total and sigma % nd == 0:
                        run[nt][pos[nd]] += n
        # close with one (wall + gap) block of weight R(g), cost g+1
        out = 0
        for t in range(total + 1):
            for i, d in enumerate(divs):
                n = run[t][i]
                if not n:
                    continue
                g = total - t - 1
                if g < 0:
                    continue
                v = right_kernel(rule, g)
                if v and d * v == sigma:
                    out += n
        if free_kernel(rule, total, bc) == sigma:      # the wall-free label
            out += 1
        return out

    # --- pbc -----------------------------------------------------------------
    total = N
    first = blank()
    for g, k in menu:                                  # the pointed first block
        if g + 1 <= total and sigma % k == 0:
            first[g + 1][pos[k]] += g + 1              # weight = its cost
    for t in range(total + 1):
        for i, d in enumerate(divs):
            n = first[t][i]
            if not n:
                continue
            for g, k in menu:
                nd, nt = d * k, t + g + 1
                if nt <= total and sigma % nd == 0:
                    first[nt][pos[nd]] += n
    out = first[total][pos[sigma]]
    if rule in (156, 198):
        if sigma == 1:
            out += 2                                   # the ring defect
    elif free_kernel(rule, total, bc) == sigma:
        out += 1
    return out


def omega(n: int) -> int:
    """Number of prime factors of n with multiplicity."""
    c, d = 0, 2
    while d * d <= n:
        while n % d == 0:
            c += 1
            n //= d
        d += 1
    return c + (1 if n > 1 else 0)


def prime_orderings(n: int) -> int:
    """Distinct orderings of the prime multiset of n: Omega(n)! / prod a_i!."""
    f: Dict[int, int] = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            f[d] = f.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        f[n] = f.get(n, 0) + 1
    out = math.factorial(sum(f.values()))
    for a in f.values():
        out //= math.factorial(a)
    return out


def sopfr(n: int) -> int:
    """Sum of prime factors with multiplicity (A001414)."""
    s, d = 0, 2
    while d * d <= n:
        while n % d == 0:
            s += d
            n //= d
        d += 1
    return s + (n if n > 1 else 0)


def multiplicity_asymptotic(rule: int, N: int, bc: str, sigma: int) -> float:
    """The large-N law for n_N(sigma) at FIXED sigma.

    PARABOLIC (156/198).  A sector of size sigma is an ordered factorisation of
    sigma into parts >= 2 padded with 1-gaps (each costing 2 letters), so the
    longest factorisations -- into PRIMES -- dominate and the count is a
    POLYNOMIAL in N whose degree is set by number theory:

        obc0   n_N(sigma) ~ P(sigma) (N/2)^{Omega+1} / (Omega+1)!
        pbc    n_N(sigma) ~ 2 P(sigma) (N/2)^{Omega} / Omega!,  and only when
               N = sopfr(sigma) + Omega(sigma)  (mod 2); the other parity is one
               degree lower.  The two frozen states of the ring defect are added
               to sigma = 1 at every N.

    P(sigma) = Omega! / prod a_i! is the number of orderings of the prime
    multiset.  So n_N(4) and n_N(6) both grow like N^3 at obc0 but n_N(6) is
    twice as common, and n_N(p) for a prime p grows only like N^2: the sector
    census of these automata reads off the prime factorisation.

    HYPERBOLIC (108/201).  The 1-gap comes in three lengths (0, 2, 3) with
    costs 1, 3, 4, and x + x^3 + x^4 = 1 at x = 1/phi, so padding is
    EXPONENTIALLY degenerate and n_N(sigma) ~ A(sigma) N^{m} phi^N with m the
    largest number of Fibonacci factors.  Only the exponent and the degree are
    returned in closed form here (A is a residue that the exact `multiplicity`
    gives for free); the point is the contrast: polynomial vs phi^N."""
    if FAMILY[rule] != "parabolic":
        raise NotImplementedError(
            "closed form implemented for the parabolic family; for 108/201 the "
            "growth is A(sigma) N^m phi^N -- use `multiplicity` for the value "
            "and `multiplicity_degree` for m")
    w, P = omega(sigma), prime_orderings(sigma)
    if bc == "obc0":
        return P * (N / 2.0) ** (w + 1) / math.factorial(w + 1)
    #: the ring defect contributes two extra frozen sectors at every N
    defect = 2.0 if sigma == 1 else 0.0
    if (N - sopfr(sigma) - w) % 2:
        return defect
    return defect + 2.0 * P * (N / 2.0) ** w / math.factorial(w)


def multiplicity_degree(rule: int, sigma: int) -> int:
    """The polynomial degree in the law above: Omega(sigma) for the parabolic
    family, and for the hyperbolic one the largest number of factors in an
    ordered factorisation of sigma into Fibonacci numbers >= 2 (or -1 when
    sigma is not such a product, i.e. never a sector size)."""
    if FAMILY[rule] == "parabolic":
        return omega(sigma)
    fibs = [fib(j) for j in range(3, 60) if fib(j) <= sigma]

    def best(v: int) -> int:
        if v == 1:
            return 0
        out = -1
        for f in fibs:
            if v % f == 0:
                sub = best(v // f)
                if sub >= 0:
                    out = max(out, sub + 1)
        return out
    return best(sigma)


# --- 5. the free energy tau(s) and the large-N law ---------------------------

def bulk_series(rule: int, s: float, x, dps: int = 30, cut: int = 90):
    """B_s(x) = sum_{g>=0} K(g)^s x^{g+1}, to `dps` digits.

    Parabolic family: K(g) = g, so B_s(x) = x * Li_{-s}(x) in closed form.

    Hyperbolic family: K(g) = F(g-1).  The first `cut` terms are summed exactly
    and the tail in CLOSED FORM, using F(m) = phi^m/sqrt5 * (1 + O(phi^{-2m})):
    beyond g = 90 that approximation is exact to ~40 digits, so the tail is a
    geometric series.  Naive summation would need ~1/(1 - phi^s x) terms and is
    unusable near the radius of convergence, which is exactly where the root of
    B_s = 1 must be bracketed from."""
    from mpmath import mp, mpf, polylog, power, sqrt
    mp.dps = dps
    x = mpf(x)
    if FAMILY[rule] == "parabolic":
        return x * polylog(-s, x)
    phi = (1 + sqrt(5)) / 2
    s = mpf(s)
    tot = x                                   # g = 0: K(0) = F(-1) = 1
    a, b = 1, 1                               # K(g) = F(g-1), from g = 2
    for g in range(2, cut):
        tot += power(a, s) * x ** (g + 1)
        a, b = b, a + b
    ratio = power(phi, s) * x
    if ratio >= 1:
        return mp.inf
    tail = (power(5, -s / 2) * power(phi, -s) * x
            * ratio ** cut / (1 - ratio))
    return tot + tail


def bulk_radius(rule: int, s: float, dps: int = 30):
    """Radius of convergence of B_s: 1 for the parabolic family, phi^{-s}
    (capped at 1, since K(g)^s <= 1 for s <= 0) for the hyperbolic one."""
    from mpmath import mp, mpf
    mp.dps = dps
    if FAMILY[rule] == "parabolic":
        return mpf(1)
    phi = (1 + mp.sqrt(5)) / 2
    return min(mpf(1), phi ** (-mpf(s)))


def lambda_s(rule: int, s: float, dps: int = 30) -> float:
    """The growth rate of Z_N(s) = sum_sectors |sector|^s: the unique
    lambda_s = 1/x_s with B_s(x_s) = 1.

    B_s is strictly increasing on (0, radius) and exceeds 1 before the radius in
    every case here, so the root exists, is unique, and controls the singularity
    of BOTH the obc0 and the pbc generating function -- which is why the
    exponential shape of the size distribution is boundary independent."""
    from mpmath import mp, mpf
    mp.dps = dps
    lo = mpf(10) ** -12
    rad = bulk_radius(rule, s, dps)
    hi = None
    for k in range(6, dps - 4):                  # walk in towards the radius
        cand = rad * (1 - mpf(10) ** -k)
        if bulk_series(rule, s, cand, dps) >= 1:
            hi = cand
            break
    if hi is None:
        raise RuntimeError(
            f"cannot bracket the root for rule {rule}, s={s}: x_s is within "
            f"1e-{dps - 5} of the radius of convergence.  For the hyperbolic "
            f"family x_s approaches phi^{{-s}} like 5^{{-s/2}}phi^{{-2s}}, so "
            f"|s| beyond about 6 needs more precision, not a different method.")
    for _ in range(4 * dps + 60):
        mid = (lo + hi) / 2
        if bulk_series(rule, s, mid, dps) < 1:
            lo = mid
        else:
            hi = mid
    return float(1 / ((lo + hi) / 2))


def tau(rule: int, s: float, dps: int = 30) -> float:
    """The free energy of the size distribution, tau(s) = ln lambda_s."""
    return math.log(lambda_s(rule, s, dps))


def tau_derivatives(rule: int, s: float = 0.0, h: float = 1e-3,
                    dps: int = 40) -> Tuple[float, float]:
    """(tau'(s), tau''(s)) by a central difference on the high-precision tau."""
    t_m, t_0, t_p = (tau(rule, s - h, dps), tau(rule, s, dps),
                     tau(rule, s + h, dps))
    return ((t_p - t_m) / (2 * h), (t_p - 2 * t_0 + t_m) / (h * h))


def lognormal_params(rule: int) -> Dict[str, float]:
    """alpha_0 = tau'(0) and sigma^2 = tau''(0): ln|sector| over sectors is
    asymptotically N(alpha_0 N, sigma^2 N).  Bulk quantities, so the same at
    obc0 and pbc."""
    a0, v0 = tau_derivatives(rule, 0.0)
    a1, _ = tau_derivatives(rule, 1.0)
    return {"alpha_typ": a0, "sigma2": v0, "alpha_state": a1,
            "lambda_0": lambda_s(rule, 0.0), "alpha_max": alpha_max(rule)}


def alpha_max(rule: int) -> float:
    """ln D_max per site: ln phi for the hyperbolic family, (ln 4)/5 for the
    parabolic one."""
    if FAMILY[rule] == "hyperbolic":
        return math.log((1 + math.sqrt(5)) / 2)
    return math.log(4) / 5


def spectrum(rule: int, ss: Sequence[float] = None) -> List[Dict[str, float]]:
    """The large-deviation spectrum, parametrically: for each s,
    alpha = tau'(s) and g(alpha) = tau(s) - s alpha, so that the number of
    sectors of size e^{alpha N} is e^{N g(alpha)}."""
    if ss is None:
        ss = [-4, -2, -1, -0.5, 0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5]
    out = []
    for s in ss:
        t = tau(rule, s)
        a, _ = tau_derivatives(rule, s)
        out.append({"s": float(s), "tau": t, "alpha": a, "g": t - s * a})
    return out


def cumulants(rule: int, N: int, bc: str) -> Dict[str, float]:
    """(n_sectors, mean, var) of ln|sector| over sectors, exactly, in O(N^2).

    The exponential-state DP in `hist` keeps every distinct partial product; this
    one keeps only what the cumulants need.  A label is a renewal walk whose
    state is the number of letters since the last wall, so the DP carries
    (count, sum ln, sum ln^2) over O(N) states for O(N) steps.  Factors enter
    multiplicatively, i.e. ln-additively:

        S1 <- S1 + n ln f,     S2 <- S2 + 2 ln f S1 + n (ln f)^2 .

    At pbc the smallest wall position p_1 is a canonical marker, so the cyclic
    sum is the same linear DP closed with K(g + p_1) and summed over p_1 -- the
    inner DP does not depend on p_1, which keeps it O(N^2).

    Used to pin the O(1) offsets c1, c2 in mean = alpha_typ N + c1 and
    var = sigma^2 N + c2.  Those offsets are RULE-DEPENDENT at obc0 and vanish
    at pbc (Tier-2 R-T20 rev.2; reproduced here independently)."""
    n_letters = N + 1 if bc == "obc0" else N
    ln = math.log

    def acc(dst, key, n, s1, s2, f):
        """fold (n, s1, s2) into dst[key] after multiplying every size by f."""
        if not n or f == 0:
            return
        lf = ln(f)
        cur = dst.get(key)
        add3 = (n, s1 + n * lf, s2 + 2 * lf * s1 + n * lf * lf)
        dst[key] = add3 if cur is None else (cur[0] + add3[0],
                                             cur[1] + add3[1],
                                             cur[2] + add3[2])

    if bc == "obc0":
        # state: ("pre", g) no wall yet, g letters used; ("post", g) gap since
        # the last wall.  Start: ("pre", 0).
        cur = {("pre", 0): (1, 0.0, 0.0)}
        for _ in range(n_letters):
            nxt: Dict[Tuple[str, int], Tuple[int, float, float]] = {}
            for (ph, g), (n, s1, s2) in cur.items():
                acc(nxt, (ph, g + 1), n, s1, s2, 1)            # a non-wall letter
                f = left_kernel(rule, g) if ph == "pre" else gap_kernel(rule, g)
                acc(nxt, ("post", 0), n, s1, s2, f)            # place a wall
            cur = nxt
        tot: Tuple[int, float, float] = (0, 0.0, 0.0)
        out: Dict[Tuple[str, int], Tuple[int, float, float]] = {}
        for (ph, g), (n, s1, s2) in cur.items():
            if ph == "pre":
                acc(out, ("end", 0), n, s1, s2, free_kernel(rule, g, bc))
            else:
                acc(out, ("end", 0), n, s1, s2, right_kernel(rule, g))
        tot = out.get(("end", 0), (0, 0.0, 0.0))
    else:
        # linear DP for the stretch strictly after the first wall
        table: List[Dict[int, Tuple[int, float, float]]] = []
        cur = {0: (1, 0.0, 0.0)}
        for m in range(n_letters):
            table.append(cur)
            nxt = {}
            for g, (n, s1, s2) in cur.items():
                acc(nxt, g + 1, n, s1, s2, 1)
                acc(nxt, 0, n, s1, s2, gap_kernel(rule, g))
            cur = nxt
        table.append(cur)
        out = {}
        for p1 in range(n_letters):
            m = n_letters - p1 - 1
            for g, (n, s1, s2) in table[m].items():
                acc(out, 0, n, s1, s2, gap_kernel(rule, g + p1))
        f0 = free_kernel(rule, n_letters, bc)          # the empty label
        if rule in (156, 198):
            acc(out, 0, 2, 0.0, 0.0, 1)                # ring defect: 2 singletons
        else:
            acc(out, 0, 1, 0.0, 0.0, f0)
        tot = out.get(0, (0, 0.0, 0.0))
    n, s1, s2 = tot
    mean = s1 / n
    return {"n_sectors": n, "mean": mean, "var": s2 / n - mean * mean}


def offsets(rule: int, bc: str, N: int = 192) -> Dict[str, float]:
    """The O(1) terms c1, c2 in mean = alpha_typ N + c1, var = sigma^2 N + c2."""
    a0, v0 = tau_derivatives(rule, 0.0)
    c = cumulants(rule, N, bc)
    return {"c1": c["mean"] - a0 * N, "c2": c["var"] - v0 * N}


# --- 6. the invariance proof, checked ----------------------------------------

def sector_is_wall_set(rule: int, N: int, bc: str) -> Dict[str, object]:
    """Check the two halves of the completeness claim on the flip graph:
    the label is CONSTANT on each sector and INJECTIVE across sectors (modulo
    the 156/198 ring defect)."""
    from ..core.rules import wolfram_to_tuple
    from ..graph.flip_graph import flip_components_np
    labels = flip_components_np(N, wolfram_to_tuple(rule), bc,
                                return_labels=True)[1]
    a, b = WALL_WORD[rule]
    by_root: Dict[int, set] = defaultdict(set)
    for x in range(1 << N):
        if bc == "obc0":
            y = [0] + [(x >> i) & 1 for i in range(N)] + [0]
            w = frozenset(i for i in range(N + 1)
                          if y[i] == a and y[i + 1] == b)
        else:
            y = [(x >> i) & 1 for i in range(N)]
            w = frozenset(i for i in range(N)
                          if y[i] == a and y[(i + 1) % N] == b)
        by_root[int(labels[x])].add(w)
    constant = all(len(v) == 1 for v in by_root.values())
    seen: Dict[frozenset, int] = {}
    collisions = 0
    for root, v in by_root.items():
        w = next(iter(v))
        if w in seen:
            collisions += 1
        seen[w] = root
    return {"constant": constant, "n_sectors": len(by_root),
            "collisions": collisions}


# --- 7. report driver --------------------------------------------------------

def build(bc: str, ns: Sequence[int] = tuple(range(3, 25)),
          check_to: int = 16) -> Dict:
    rec: Dict[str, object] = {"bc": bc, "rules": {}}
    for rule in RULES:
        per_n = {}
        for N in ns:
            h = hist(rule, N, bc)
            s = summary(rule, N, bc)
            if N <= check_to:
                assert h == hist_bruteforce(rule, N, bc), (rule, N, bc)
            assert s["d_max"] == d_max_closed(rule, N, bc), ("d_max", rule, N, bc)
            assert s["n_frozen"] == n_frozen_closed(rule, N, bc), \
                ("n_frozen", rule, N, bc)
            assert s["n_sectors"] == n_sectors_closed(rule, N, bc), \
                ("n_sectors", rule, N, bc)
            per_n[str(N)] = {**s,
                             "hist": {str(k): v for k, v in sorted(h.items())}}
        rec["rules"][str(rule)] = {
            "word": "".join(str(c) for c in WALL_WORD[rule]),
            "family": FAMILY[rule],
            "m0": wall_matrices(rule)[0],
            "asymptotics": lognormal_params(rule),
            "spectrum": spectrum(rule),
            "by_N": per_n,
        }
    return rec


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--bc", default="obc0", choices=("obc0", "pbc"))
    p.add_argument("--nmax", type=int, default=24)
    p.add_argument("--check-to", type=int, default=16)
    args = p.parse_args(argv)
    rec = build(args.bc, tuple(range(3, args.nmax + 1)), args.check_to)
    path = OUT_JSON.format(bc=args.bc)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(rec, fh)
    for rule in RULES:
        r = rec["rules"][str(rule)]
        a = r["asymptotics"]
        print(f"W{rule} `{r['word']}` {r['family']:11s} "
              f"lambda_0={a['lambda_0']:.6f} alpha_typ={a['alpha_typ']:.6f} "
              f"sigma2={a['sigma2']:.6f} alpha_state={a['alpha_state']:.6f} "
              f"alpha_max={a['alpha_max']:.6f}")
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
