"""Regression tests for R27 rev.3 sec.4 and sec.6: ranks with no cutoff.

The point of the module under test is that the operator Schmidt rank is an
algebraic quantity, so these tests assert exact integers and never a tolerance.
"""

import numpy as np
import pytest

from qca_fragmentation.quantum import exact_rank as er
from qca_fragmentation.quantum import kink_schmidt as ks

KINK = (156, 198)
HARD = (108, 201)


# --- the F_p arithmetic ------------------------------------------------------

@pytest.mark.parametrize("N", [7, 9, 11, 13, 16])
def test_primes_keep_float64_matmul_exact(N):
    """2^N (p-1)^2 must stay under 2^53 or the 'exact' matmul silently is not."""
    for p in er.primes_for(N, 2):
        assert p % 8 in (1, 7)
        assert er.is_prime(p)
        assert (1 << N) * (p - 1) ** 2 < 2 ** 53


@pytest.mark.parametrize("p", [1048559, 1048447, 741431, 741409])
def test_sqrt2_mod_is_a_square_root(p):
    s = er.sqrt2_mod(p)
    assert s is not None
    assert s * s % p == 2 % p


def test_sqrt2_absent_when_two_is_not_a_residue():
    # p = 3, 5 mod 8 -> 2 is a non-residue
    assert er.sqrt2_mod(11) is None
    assert er.sqrt2_mod(13) is None


def test_rank_mod_matches_numpy_on_random_integer_matrices():
    p = er.primes_for(8, 1)[0]
    rng = np.random.default_rng(0)
    for r in (1, 3, 7):
        A = rng.integers(0, 50, size=(12, r)).astype(float)
        B = rng.integers(0, 50, size=(r, 15)).astype(float)
        M = np.mod(A @ B, p)
        assert er.rank_mod(M, p) == min(r, np.linalg.matrix_rank(A @ B))


def test_rank_mod_inplace_flag_leaves_input_alone_when_false():
    p = er.primes_for(6, 1)[0]
    M = np.array([[1.0, 2.0], [2.0, 4.0]])
    before = M.copy()
    er.rank_mod(M, p)
    assert np.array_equal(M, before)


# --- the propagator ----------------------------------------------------------

@pytest.mark.parametrize("rule", KINK + HARD)
def test_build_U_mod_is_a_permutation_of_the_float_support(rule):
    """The F_p propagator must have exactly the support of the real one."""
    N, p = 7, er.primes_for(7, 1)[0]
    Um = er.build_U_mod(rule, N, p)
    from qca_fragmentation.quantum.asymptotic import kraus_operators
    Uf = kraus_operators(rule, N, "obc0")[0]
    assert np.array_equal(Um != 0, np.abs(Uf) > 1e-12)


def test_build_U_mod_rejects_dissipative_rules():
    with pytest.raises(ValueError):
        er.build_U_mod(232, 6, er.primes_for(6, 1)[0])   # DIIE, has resets


# --- the ceiling, exactly ----------------------------------------------------

@pytest.mark.parametrize("rule", KINK + HARD)
@pytest.mark.parametrize("N", [7, 9])
def test_plateau_matches_the_closed_form(rule, N):
    got = er.chi_exact(rule, N, er.PLATEAU_TIME(rule, N))
    assert got == er.ceiling_closed_form(rule, N), (rule, N, got)


def test_closed_forms_reproduce_the_published_values():
    assert [er.ceiling_closed_form(156, N) for N in (7, 9, 11, 13)] == [16, 24, 34, 46]
    assert [er.ceiling_closed_form(108, N) for N in (7, 9, 11, 13)] == [15, 40, 104, 273]
    assert [er.ceiling_closed_form(201, N) for N in (7, 9, 11, 13)] == [27, 70, 184, 481]
    assert er.ceiling_closed_form(156, 8) is None       # odd N only


@pytest.mark.parametrize("N", [7, 9])
def test_the_mirror_pair_shares_the_ceiling_but_not_the_argmax(N):
    """156 and 198 are exact mirrors: same max over bonds, argmax reflected."""
    t = er.PLATEAU_TIME(156, N)
    p = er.primes_for(N, 1)[0]
    prof = {}
    for rule in KINK:
        U = er.build_U_mod(rule, N, p)
        Ut = U.copy()
        for _ in range(t - 1):
            Ut = np.mod(Ut @ U, p)
        prof[rule] = [er.rank_mod(er.reshuffle(Ut, N, x), p, inplace=True)
                      for x in range(1, N)]
    assert max(prof[156]) == max(prof[198]) == er.ceiling_closed_form(156, N)
    assert prof[156] == prof[198][::-1]                       # mirror images
    assert prof[156].index(max(prof[156])) + 1 == (N + 1) // 2
    assert prof[198].index(max(prof[198])) + 1 == (N - 1) // 2


# --- the permutation case, where no prime is needed --------------------------

@pytest.mark.parametrize("rule", KINK)
@pytest.mark.parametrize("N", [9, 11])
def test_xgate_reaches_the_same_ceiling(rule, N):
    """max over t of chi under V=X equals the Hadamard ceiling (R27 sec.4.3)."""
    best = max(er.chi_permutation(rule, N, t) for t in range(1, 40))
    assert best == er.ceiling_closed_form(rule, N), (rule, N, best)


def test_xgate_rank_is_prime_independent():
    """It is an integer rank; the field is a convenience, not an input."""
    a = er.chi_permutation(156, 9, 11, p=1000003)
    b = er.chi_permutation(156, 9, 11, p=2000003)
    assert a == b == 24


# --- the trap this module exists to avoid ------------------------------------

def test_exact_module_rank_is_t_where_a_1e9_cutoff_says_otherwise():
    """The false departure: float64 at 1e-9 reports rank < t from t=14 at N=60,
    while the algebraic rank is exactly t (R27 sec.6.2)."""
    A, kk, leak = ks.propagator(156, 60)
    assert leak == 0.0
    x = 30
    At = np.eye(60)
    saw_short = False
    for t in range(1, 19):
        At = A @ At
        lo = [i for i, k in enumerate(kk) if k <= x]
        hi = [i for i, k in enumerate(kk) if k > x]
        block = At[np.ix_(lo, hi)]
        s = np.linalg.svd(block, compute_uv=False)
        s = s / s[0]
        cut = int(np.sum(s > 1e-9))
        # the true null space stays many orders below the smallest live value
        assert s[t - 1] > 1e2 * max(s[t], 1e-300), (t, s[t - 1], s[t])
        if cut < t:
            saw_short = True
    assert saw_short, "expected the 1e-9 read to undercount somewhere by t=18"
