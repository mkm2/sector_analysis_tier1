"""Operator Schmidt ranks with no cutoff anywhere  (Report R27 rev.3, sec.6).

Every amplitude in this project lies in Z[1/sqrt2]: it is (a + b sqrt2)/2^m with
integer a, b.  The operator Schmidt rank of U(t) is therefore an ALGEBRAIC
quantity and need never be read off a singular-value spectrum with a threshold.

Why this module exists.  Three independent pipelines in this project each
manufactured a false result by holding an SVD cutoff fixed while the spectrum
moved underneath it, and they failed in BOTH directions:

  * the kink module at 1e-9   -- genuine tail decays ~5.6x per period,
                                 so a fixed cutoff UNDERCOUNTS: it invented a
                                 departure from chi = 2t+1 starting at t = 14,
                                 N-independently, which looked like a real
                                 discovery until the exact rank killed it;
  * the Hadamard MPO at 1e-10 -- same mechanism, invented a dip at t = 12;
  * the X-gate MPO at 1e-14   -- genuine tail is FLAT at 1e-4 with eight orders
                                 of clear space below, so a tight cutoff
                                 OVERCOUNTS: 161 phantom channels against 104.

Two rules came out of that, and this module implements the second:
  (i)  the certificate is sigma_r / sigma_{r+1} at the claimed rank, never the
       value of a cutoff and never a sweep of one;
  (ii) where the object is integral or lies in a number ring, take the algebraic
       rank and the question does not arise.

METHOD.  Pick a prime p = +-1 (mod 8), so 2 is a quadratic residue and sqrt2
exists in F_p, and map the whole computation -- U, its powers, the reshuffle, the
elimination -- into F_p.  Keeping 2^N (p-1)^2 < 2^53 makes float64 matmul EXACT,
so the powers still run at BLAS speed.  rank_{F_p} <= rank_{Q(sqrt2)} always,
with equality for all but finitely many p, so agreement across two primes is the
usual certificate; `chi_exact` checks two primes by default.

Under V = X the circuit is a permutation of basis states, so U(t) is a
permutation matrix, the reshuffle is 0/1, and no prime is needed at all -- see
`chi_permutation`.
"""
from __future__ import annotations

import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core import rules as rules_mod
from ..core.cycle import one_cycle_branches

BC = "obc0"


# --- arithmetic in F_p -------------------------------------------------------

def is_prime(n: int) -> bool:
    if n < 2:
        return False
    small = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for q in small:
        if n % q == 0:
            return n == q
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in small:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def sqrt2_mod(p: int) -> Optional[int]:
    """Tonelli-Shanks; returns None when 2 is not a residue mod p."""
    n = 2 % p
    if pow(n, (p - 1) // 2, p) != 1:
        return None
    if p % 4 == 3:
        return pow(n, (p + 1) // 4, p)
    q, s = p - 1, 0
    while q % 2 == 0:
        q //= 2
        s += 1
    z = 2
    while pow(z, (p - 1) // 2, p) != p - 1:
        z += 1
    m, c, t, r = s, pow(z, q, p), pow(n, q, p), pow(n, (q + 1) // 2, p)
    while t != 1:
        i, t2 = 0, t
        while t2 != 1:
            t2 = t2 * t2 % p
            i += 1
        b = pow(c, 1 << (m - i - 1), p)
        m, c = i, b * b % p
        t, r = t * c % p, r * b % p
    return r


def primes_for(N: int, count: int = 2) -> List[int]:
    """Largest primes p = +-1 mod 8 with 2^N (p-1)^2 < 2^52.

    The bound is what keeps float64 matmul exact: a matrix product sums 2^N
    terms each below (p-1)^2.  2^52 rather than 2^53 leaves a factor two of
    headroom, which costs nothing and removes a very tight edge case at N = 13.
    """
    cap = int(((2 ** 52) / (1 << N)) ** 0.5)
    out, n = [], min(cap, 1 << 20) - 1
    while len(out) < count and n > 1000:
        if n % 8 in (1, 7) and is_prime(n):
            out.append(n)
        n -= 1
    if len(out) < count:
        raise ValueError(f"no suitable primes for N={N}")
    return out


def rank_mod(M: np.ndarray, p: int, *, inplace: bool = False) -> int:
    """Row-echelon rank of an F_p matrix held exactly in float64."""
    A = M if inplace else M.copy()
    rows, cols = A.shape
    rank = 0
    for c in range(cols):
        if rank == rows:
            break
        nz = np.nonzero(A[rank:, c])[0]
        if nz.size == 0:
            continue
        piv = rank + int(nz[0])
        if piv != rank:
            A[[rank, piv]] = A[[piv, rank]]
        A[rank] = np.mod(A[rank] * pow(int(A[rank, c]), p - 2, p), p)
        below = np.nonzero(A[rank + 1:, c])[0]
        if below.size:
            idx = rank + 1 + below
            A[idx] = np.mod(A[idx] - np.outer(A[idx, c], A[rank]), p)
        rank += 1
    return rank


# --- the propagator and its reshuffle ----------------------------------------

def build_U_mod(rule: int, N: int, p: int, s: Optional[int] = None) -> np.ndarray:
    """One brick-wall cycle as a dense matrix over F_p (unitary rules only)."""
    if s is None:
        s = sqrt2_mod(p)
        if s is None:
            raise ValueError(f"2 is not a quadratic residue mod {p}")
    word = rules_mod.wolfram_to_tuple(rule)
    if not rules_mod.is_unitary(word):
        raise ValueError(f"rule {rule} is not unitary")
    inv2 = pow(2, p - 2, p)
    U = np.zeros((1 << N, 1 << N))
    for x in range(1 << N):
        (amps, m), = one_cycle_branches(x, N, word, BC)
        scale = pow(inv2, m, p)
        for y, (a, b) in amps.items():
            if (a, b) != (0, 0):
                U[y, x] = ((a + b * s) % p) * scale % p
    return U


def reshuffle(Ut: np.ndarray, N: int, x: int) -> np.ndarray:
    """Operator reshuffle across bond x.  Site s is bit s, so the LEFT block is
    the low bits and the row index is (l_out, l_in)."""
    dl, dr = 1 << x, 1 << (N - x)
    return np.ascontiguousarray(
        Ut.reshape(dr, dl, dr, dl).transpose(1, 3, 0, 2).reshape(dl * dl, dr * dr))


def chi_exact(rule: int, N: int, t: int, x: Optional[int] = None,
              primes: Optional[Sequence[int]] = None) -> int:
    """Exact operator Schmidt rank of U(t) across bond x (default: max over
    bonds).  Raises if two primes disagree, which is the certificate failing."""
    ps = list(primes) if primes else primes_for(N, 2)
    vals = []
    for p in ps:
        U = build_U_mod(rule, N, p)
        Ut = U.copy()
        for _ in range(t - 1):
            Ut = np.mod(Ut @ U, p)
        bonds = [x] if x is not None else range(1, N)
        vals.append(max(rank_mod(reshuffle(Ut, N, b), p, inplace=True)
                        for b in bonds))
    if len(set(vals)) != 1:
        raise RuntimeError(f"primes disagree for rule {rule} N={N} t={t}: {vals}")
    return vals[0]


# --- the permutation (V = X) case, where no prime is needed ------------------

def chi_permutation(rule: int, N: int, t: int, x: Optional[int] = None,
                    p: int = 1000003) -> int:
    """Exact rank for V = X, where U(t) is a permutation matrix and the
    reshuffle is 0/1.  Sparse incremental elimination; p is only a convenience
    for the field arithmetic, the answer does not depend on it."""
    from ..permutation.orbits import make_bitstep
    step = make_bitstep(rule, N, BC)
    perm = [step(v) for v in range(1 << N)]
    cur = list(range(1 << N))
    for _ in range(t):
        cur = [perm[y] for y in cur]
    bonds = [x] if x is not None else range(1, N)
    best = 0
    for b in bonds:
        dl, dr = 1 << b, 1 << (N - b)
        rows: Dict[int, Dict[int, int]] = {}
        for xin in range(1 << N):
            y = cur[xin]
            rows.setdefault((y % dl) * dl + (xin % dl), {})[
                (y // dl) * dr + (xin // dl)] = 1
        basis: Dict[int, Dict[int, int]] = {}
        rank = 0
        for row in rows.values():
            r = dict(row)
            while r:
                c = min(r)
                if c not in basis:
                    inv = pow(r[c], p - 2, p)
                    basis[c] = {k: v * inv % p for k, v in r.items()}
                    rank += 1
                    break
                f, br = r[c], basis[c]
                for k, v in br.items():
                    nv = (r.get(k, 0) - f * v) % p
                    if nv:
                        r[k] = nv
                    elif k in r:
                        del r[k]
        best = max(best, rank)
    return best


# --- closed forms -------------------------------------------------------------

def fib(n: int) -> int:
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def PLATEAU_TIME(rule: int, N: int) -> int:
    """A time comfortably past saturation: the kink slot (156/198) needs about
    2N periods, the hard-core slot (201/108) saturates in fewer than N."""
    return 2 * N if rule in (156, 198) else N


def ceiling_closed_form(rule: int, N: int) -> Optional[int]:
    """The measured plateau of chi, maximised over bonds, at odd N (R27 sec.4)."""
    if N % 2 == 0:
        return None
    k = (N + 1) // 2
    if rule in (156, 198):
        return (N * N + 15) // 4
    if rule == 108:
        return fib(k) * fib(k + 1)
    if rule == 201:
        return fib(k) * (fib(k) + 3 * fib(k - 1))
    return None


def main(argv: List[str]) -> int:
    print("exact ranks over Q(sqrt2) via F_p -- no cutoff anywhere\n")
    print(f"{'rule':>5} {'N':>4} {'plateau':>9} {'closed form':>12}  primes")
    for rule in (156, 198, 108, 201):
        for N in (7, 9, 11):
            # a time comfortably past saturation: the kink slot needs ~2N, the
            # hard-core slot saturates in under N.
            t = PLATEAU_TIME(rule, N)
            ps = primes_for(N, 2)
            got = chi_exact(rule, N, t, primes=ps)
            pred = ceiling_closed_form(rule, N)
            ok = "OK" if got == pred else "MISMATCH"
            print(f"{rule:>5} {N:>4} {got:>9} {str(pred):>12}  "
                  f"{'/'.join(map(str, ps))}  {ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
