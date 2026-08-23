"""R27 rev.2 -- the exact operator Schmidt rank of U(t) on the kink module.

Upgrades the O(t) upper bound of R27 rev.1 to the two-sided law

    2t  <=  chi_module(t)  <=  2t + 2 ,        measured value  2t + 1 ,

for W156 and W198 on their maximal one-wall (kink) module, obc0.

Three ingredients, each checked here:

  (i)   REDUCTION.  Every kink state |k> = 1^k 0^{l-k} is a product across any
        cut, and k <-> (min(k,x), max(k,x)) is a bijection, so the reshuffle of
        U(t)|_module is a re-indexing of the coefficient matrix A^t.  Its four
        index sectors are a rank-<=1 column (both indices left), a rank-<=1 row
        (both right) and the two crossing blocks, which sit in disjoint rows and
        columns.  Hence
            rank(C_LR) + rank(C_RL)  <=  chi  <=  2 + rank(C_LR) + rank(C_RL).

  (ii)  UPPER BOUND (transfer-rank lemma).  For any A and any splitting
        P_L + P_R = 1, decomposing paths by the LAST time in L gives the exact
        identity
            P_R A^t P_L = sum_{s=0}^{t-1} (P_R A P_R)^{t-1-s} (P_R A P_L)
                                          (P_L A^s P_L),
        so  rank(P_R A^t P_L) <= t * rank(P_R A P_L).
        On the kink module rank(P_R A P_L) = 1: in the cell basis
        p_j = (|2j-1>+|2j>)/sqrt2, m_j = (|2j-1>-|2j>)/sqrt2 the propagator is a
        two-band walk whose inter-cell transfer operators
            T_+ = (1/2)|p><p - m| ,   T_- = (1/2)|m><p + m|
        are both RANK ONE.  So rank(C_LR) <= t.

  (iii) LOWER BOUND.  With P_R A P_L = a b^T rank one, the identity above reads
        P_R A^t P_L = sum_{s=1}^{t} u_s v_s^T with
            u_s = (P_R A P_R)^{t-s} a ,   v_s = (P_L A^{s-1} P_L)^T b .
        The light cone makes u_s reach exactly t-s cells right of the cut and
        v_s exactly s-1 cells left of it, with a nonzero leading amplitude
        2^{-(t-s)} resp. 2^{-(s-1)}, so {u_s} and {v_s} are each linearly
        independent and the rank is exactly t.

Run:  python -m qca_fragmentation.quantum.kink_schmidt
      python -m qca_fragmentation.quantum.kink_schmidt --full N
      python -m qca_fragmentation.quantum.kink_schmidt --write
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

from ..core import rules as rules_mod
from ..core.cycle import one_cycle_branches

BC = "obc0"
CACHE = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                     "analytics", "r27_schmidt_growth.json")
SQ2 = np.sqrt(2.0)
TOL = 1e-9
KINK_RULES = (156, 198)


# --- the kink module ---------------------------------------------------------

def kink_state(k: int, N: int, rule: int) -> int:
    """|k>, k = 1..N, of the maximal one-wall sector.  Site s is bit s.

    W156 = IIVI fires at (left, right) = (1, 0), so its descending family is
    1^k 0^{N-k}; W198 = IVII is the mirror and its family is 0^{N-k} 1^k.
    """
    if rule == 156:
        return sum(1 << s for s in range(k))
    if rule == 198:
        return sum(1 << s for s in range(N - k, N))
    raise ValueError(f"no kink module registered for rule {rule}")


def _amps(x: int, N: int, word) -> Dict[int, float]:
    (amps, m), = one_cycle_branches(x, N, word, BC)
    return {y: (a + b * SQ2) / (2.0 ** m) for y, (a, b) in amps.items()
            if (a, b) != (0, 0)}


def propagator(rule: int, N: int) -> Tuple[np.ndarray, List[int], float]:
    """Exact single-particle propagator A on the kink module, and the leak.

    `leak` is the largest amplitude that escapes the module in one period; it is
    exactly 0.0 whenever the module is a Krylov sector, which is asserted by the
    caller.
    """
    word = rules_mod.wolfram_to_tuple(rule)
    ks = list(range(1, N + 1))
    idx = {kink_state(k, N, rule): j for j, k in enumerate(ks)}
    A = np.zeros((len(ks), len(ks)))
    leak = 0.0
    for j, k in enumerate(ks):
        for y, v in _amps(kink_state(k, N, rule), N, word).items():
            if y in idx:
                A[idx[y], j] = v
            else:
                leak = max(leak, abs(v))
    return A, ks, leak


# Which pairing of kink positions tiles the chain into cells.  W198 is the
# left-right mirror of W156, and reflection maps k -> l+1-k, so its cell tiling
# is the shifted one.
CELL_OFFSET = {156: 0, 198: 1}


def cell_basis(N: int, offset: int = 0) -> np.ndarray:
    """Columns p_1, m_1, p_2, m_2, ... in the |k> basis (k = 1..N at index k-1).

    Cell j is the pair of kink positions {2j-1+offset, 2j+offset}, and
        p_j = (|2j-1+off> + |2j+off>)/sqrt2,
        m_j = (|2j-1+off> - |2j+off>)/sqrt2.
    Positions left over at the ends keep their own basis vector, appended last,
    so the result is always orthogonal.
    """
    B = np.zeros((N, N))
    c, used = 0, set()
    j = 1
    while 2 * j + offset <= N:
        a, b = 2 * j - 2 + offset, 2 * j - 1 + offset      # 0-based indices
        B[a, c] = B[b, c] = 1 / SQ2                        # p_j
        B[a, c + 1], B[b, c + 1] = 1 / SQ2, -1 / SQ2       # m_j
        used |= {a, b}
        c += 2
        j += 1
    for i in range(N):
        if i not in used:
            B[i, c] = 1.0
            c += 1
    return B


# --- the three ingredients ---------------------------------------------------

def _rank(M: np.ndarray, tol: float = TOL) -> int:
    if min(M.shape) == 0:
        return 0
    s = np.linalg.svd(M, compute_uv=False)
    return int(np.sum(s > tol * max(s[0], 1e-30)))


def schmidt_rank_module(At: np.ndarray, ks: List[int], x: int) -> int:
    """Exact operator Schmidt rank of sum_kk' At_kk' |k><k'| across bond x.

    |k> = |L_a> (x) |R_b> with a = min(k, x), b = max(k, x); the pair (a, b)
    determines k, so the reshuffle is a plain re-indexing of At.
    """
    A_idx = sorted({min(k, x) for k in ks})
    B_idx = sorted({max(k, x) for k in ks})
    ai = {a: i for i, a in enumerate(A_idx)}
    bi = {b: i for i, b in enumerate(B_idx)}
    na, nb = len(A_idx), len(B_idx)
    M = np.zeros((na * na, nb * nb))
    for p, k in enumerate(ks):
        for q, kp in enumerate(ks):
            M[ai[min(k, x)] * na + ai[min(kp, x)],
              bi[max(k, x)] * nb + bi[max(kp, x)]] += At[p, q]
    return _rank(M)


def crossing_ranks(At: np.ndarray, ks: List[int], x: int) -> Tuple[int, int]:
    """(rank of the block k <= x < k', rank of the block k' <= x < k)."""
    lo = [p for p, k in enumerate(ks) if k <= x]
    hi = [p for p, k in enumerate(ks) if k > x]
    if not lo or not hi:
        return 0, 0
    return _rank(At[np.ix_(lo, hi)]), _rank(At[np.ix_(hi, lo)])


def last_passage_residual(A: np.ndarray, xcut: int, t: int) -> float:
    """max |P_R A^t P_L - sum_s (P_R A P_R)^{t-1-s} (P_R A P_L) (P_L A^s P_L)|."""
    n = A.shape[0]
    PL = np.diag([1.0 if i < xcut else 0.0 for i in range(n)])
    PR = np.eye(n) - PL
    ARR, ARL = PR @ A @ PR, PR @ A @ PL
    rhs = np.zeros_like(A)
    for s in range(t):
        rhs += (np.linalg.matrix_power(ARR, t - 1 - s) @ ARL
                @ (PL @ np.linalg.matrix_power(A, s) @ PL))
    return float(np.abs(PR @ np.linalg.matrix_power(A, t) @ PL - rhs).max())


def rank_one_factorisation(A: np.ndarray, xcut: int, t: int):
    """u_s, v_s of ingredient (iii); returns (dim{u_s}, dim{v_s}, rank, residual)."""
    n = A.shape[0]
    PL = np.diag([1.0 if i < xcut else 0.0 for i in range(n)])
    PR = np.eye(n) - PL
    ARR, ARL = PR @ A @ PR, PR @ A @ PL
    u0, s0, vh0 = np.linalg.svd(ARL)
    if int(np.sum(s0 > 1e-10 * s0[0])) != 1:
        raise AssertionError("cross transfer is not rank one")
    a, b = u0[:, 0] * s0[0], vh0[0].conj()
    us = [np.linalg.matrix_power(ARR, t - s) @ a for s in range(1, t + 1)]
    vs = [(PL @ np.linalg.matrix_power(A, s - 1) @ PL).T @ b
          for s in range(1, t + 1)]
    U, V = np.array(us).T, np.array(vs).T
    res = float(np.abs(U @ V.T - PR @ np.linalg.matrix_power(A, t) @ PL).max())
    return _rank(U, 1e-10), _rank(V, 1e-10), _rank(U @ V.T, 1e-10), res


# --- the full chain ----------------------------------------------------------

def full_schmidt_rank(rule: int, N: int, x: int, tmax: int) -> List[int]:
    """Brute-force chi_x(t) of the FULL U(t) on 2^N.  O(8^N) per step."""
    from .asymptotic import kraus_operators
    K = kraus_operators(rule, N, BC)
    if len(K) != 1:
        raise ValueError(f"rule {rule} is not unitary")
    U = K[0]
    dl, dr = 1 << x, 1 << (N - x)
    out, Ut = [], np.eye(1 << N)
    for t in range(tmax + 1):
        if t:
            Ut = U @ Ut
        M = Ut.reshape(dr, dl, dr, dl).transpose(1, 3, 0, 2).reshape(dl * dl, dr * dr)
        out.append(_rank(M))
    return out


def full_schmidt_rank_sparse(rule: int, N: int, x: int, tmax: int,
                             probes: int = 160, seed: int = 0) -> List[int]:
    """chi_x(t) of the full U(t) with a sparse propagator and a randomised rank.

    The propagator is assembled column by column from the exact branch
    machinery, so nothing is truncated.  The reshuffle is a COO index remap, and
    the rank of a matrix of rank < `probes` is recovered exactly by projecting
    onto `probes` Gaussian directions.  Raises if the cap is ever reached.
    """
    import scipy.sparse as sp
    word = rules_mod.wolfram_to_tuple(rule)
    rows, cols, vals = [], [], []
    for y in range(1 << N):
        for z, v in _amps(y, N, word).items():
            rows.append(z); cols.append(y); vals.append(v)
    U = sp.csr_matrix((vals, (rows, cols)), shape=(1 << N, 1 << N))
    dl, dr = 1 << x, 1 << (N - x)
    rng = np.random.default_rng(seed)
    out, Ut = [], sp.identity(1 << N, format="csr")
    for t in range(tmax + 1):
        if t:
            Ut = (U @ Ut).tocsr()
            Ut.eliminate_zeros()
        C = Ut.tocoo()
        M = sp.coo_matrix((C.data, ((C.row % dl) * dl + (C.col % dl),
                                    (C.row // dl) * dr + (C.col // dl))),
                          shape=(dl * dl, dr * dr)).tocsr()
        s = np.linalg.svd(M @ rng.normal(size=(M.shape[1], probes)),
                          compute_uv=False)
        r = int(np.sum(s > TOL * max(s[0], 1e-30)))
        if r >= probes:
            raise RuntimeError(f"rank reached the {probes} probe cap")
        out.append(r)
    return out


# --- report ------------------------------------------------------------------

def write_cache(path: str = CACHE) -> dict:
    """Rebuild analytics/r27_schmidt_growth.json (the numbers quoted in R27)."""
    data: dict = {"bc": BC, "module": {}, "full": {}}
    for rule in KINK_RULES:
        for N in (16, 20, 24, 30):
            A, ks, leak = propagator(rule, N)
            assert leak == 0.0
            for x in (3, N // 2, N // 2 + 1):
                row, At = [], np.eye(len(ks))
                for t in range(11):
                    if t:
                        At = A @ At
                    row.append(schmidt_rank_module(At, ks, x))
                data["module"][f"{rule}/{N}/{x}"] = row
        for N in (8, 10, 12, 14, 16):
            data["full"][f"{rule}/{N}"] = full_schmidt_rank_sparse(
                rule, N, N // 2, min(10, N // 2 + 2))
    with open(path, "w") as fh:
        json.dump(data, fh, indent=1, sort_keys=True)
    return data


def main(argv: List[str]) -> int:
    if len(argv) > 1 and argv[1] == "--write":
        d = write_cache()
        print(f"wrote {os.path.normpath(CACHE)}: "
              f"{len(d['module'])} module series, {len(d['full'])} full series")
        return 0
    if len(argv) > 1 and argv[1] == "--full":
        N = int(argv[2]) if len(argv) > 2 else 12
        for rule in KINK_RULES:
            series = full_schmidt_rank(rule, N, N // 2, min(12, N))
            print(f"W{rule} N={N} x={N//2}: chi_full(t) = {series}")
            print("        6t-7      = "
                  f"{[max(1, 6 * t - 7) for t in range(len(series))]}")
        return 0

    print("(i)+(ii)+(iii)  the kink module of W156 / W198, obc0\n")
    for rule in KINK_RULES:
        A, ks, leak = propagator(rule, 24)
        assert leak == 0.0, f"module of W{rule} is not closed (leak {leak})"
        uerr = float(np.abs(A.T @ A - np.eye(len(ks))).max())
        B = cell_basis(24, CELL_OFFSET[rule])
        Ac = B.T @ A @ B
        j = 6
        Tp = Ac[2 * j:2 * j + 2, 2 * (j - 1):2 * j]
        Tm = Ac[2 * (j - 2):2 * (j - 1), 2 * (j - 1):2 * j]
        print(f"W{rule}: N=24, module dim {len(ks)}, leak {leak:.1e}, "
              f"unitarity {uerr:.1e}; cell-basis transfer ranks "
              f"T_+ {_rank(Tp, 1e-10)}, T_- {_rank(Tm, 1e-10)}")
        x = 12
        print(f"{'t':>4} {'chi':>5} {'C_LR':>6} {'C_RL':>6} {'2t+1':>6} "
              f"{'lastpass':>10} {'dim u':>6} {'dim v':>6} {'fact err':>10}")
        At = np.eye(len(ks))
        for t in range(1, 11):
            At = A @ At
            lr, rl = crossing_ranks(At, ks, x)
            du, dv, rk, res = rank_one_factorisation(A, x, t)
            print(f"{t:>4} {schmidt_rank_module(At, ks, x):>5} {lr:>6} {rl:>6} "
                  f"{2 * t + 1:>6} {last_passage_residual(A, x, t):>10.1e} "
                  f"{du:>6} {dv:>6} {res:>10.1e}")
        print()

    print("chi_module(t) over N and cut (should be min(2t+1, 2*min(x, l-x)))")
    print(f"{'rule':>5} {'N':>4} {'x':>4}  chi(t), t = 0..10")
    for rule in KINK_RULES:
        for N in (16, 20, 24, 30):
            A, ks, leak = propagator(rule, N)
            assert leak == 0.0
            for x in (3, N // 2, N // 2 + 1):
                row, At = [], np.eye(len(ks))
                for t in range(11):
                    if t:
                        At = A @ At
                    row.append(schmidt_rank_module(At, ks, x))
                print(f"{rule:>5} {N:>4} {x:>4}  {row}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
