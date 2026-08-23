"""Regression tests for R27 rev.2: the operator Schmidt rank of U(t).

Pins the three ingredients of Theorem 1 (the last-passage identity, the rank-one
cross transfer, the matching rank-t factorisation), the exact module law
chi = min(2t+1, chi_sat), and the full-chain law chi = 6t - 7.
"""

import json
import os

import numpy as np
import pytest

from qca_fragmentation.quantum import kink_schmidt as ks

RULES = (156, 198)
CACHE = os.path.normpath(ks.CACHE)


# --- the module and its propagator -------------------------------------------

@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("N", [10, 16, 24])
def test_module_is_closed_and_unitary(rule, N):
    A, kk, leak = ks.propagator(rule, N)
    assert leak == 0.0, (rule, N, leak)          # exact: no tolerance
    assert len(kk) == N
    assert np.abs(A.T @ A - np.eye(N)).max() < 1e-14


@pytest.mark.parametrize("rule", RULES)
def test_propagator_hops_at_most_two_kink_positions(rule):
    A, kk, _ = ks.propagator(rule, 20)
    nz = np.argwhere(np.abs(A) > 1e-14)
    assert int(np.max(np.abs(nz[:, 0] - nz[:, 1]))) == 2


# --- ingredient (ii): last passage and the rank-one transfer ------------------

@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("t", [1, 2, 5, 9])
def test_last_passage_identity(rule, t):
    A, _, _ = ks.propagator(rule, 24)
    assert ks.last_passage_residual(A, 12, t) < 1e-14


def test_last_passage_holds_for_a_generic_unitary():
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.normal(size=(10, 10)) + 1j * rng.normal(size=(10, 10)))
    for t in (1, 3, 5):
        assert ks.last_passage_residual(q, 5, t) < 1e-13


@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("N", [12, 20, 24])
def test_cross_transfer_is_rank_one_at_every_cut(rule, N):
    """rank(P_R A P_L) = 1 -- the static quantity the linear law hangs on."""
    A, _, _ = ks.propagator(rule, N)
    for x in range(1, N):
        PL = np.diag([1.0 if i < x else 0.0 for i in range(N)])
        assert ks._rank((np.eye(N) - PL) @ A @ PL, 1e-10) == 1, (rule, N, x)


@pytest.mark.parametrize("rule", RULES)
def test_cell_basis_transfer_operators_are_rank_one(rule):
    """T_+ = |p><p - m|/2 and T_- = |m><p + m|/2, both rank one (eq. 8 of R27)."""
    N = 24
    A, _, _ = ks.propagator(rule, N)
    B = ks.cell_basis(N, ks.CELL_OFFSET[rule])
    Ac = B.T @ A @ B
    for j in range(3, N // 2 - 2):
        col = slice(2 * (j - 1), 2 * j)
        Tp = Ac[2 * j:2 * j + 2, col]
        Tm = Ac[2 * (j - 2):2 * (j - 1), col]
        assert ks._rank(Tp, 1e-10) == 1, (rule, j)
        assert ks._rank(Tm, 1e-10) == 1, (rule, j)
        # each transfer lands on a SINGLE coin state -- one row is exactly zero.
        # Which one it is depends on the orientation: W198 is the mirror of
        # W156, so its right-moving coin is the other member of the pair.
        assert sum(np.abs(Tp[r]).max() < 1e-14 for r in (0, 1)) == 1, (rule, j)
        assert sum(np.abs(Tm[r]).max() < 1e-14 for r in (0, 1)) == 1, (rule, j)
        # and the two chiralities use different coins
        assert (np.abs(Tp[0]).max() < 1e-14) != (np.abs(Tm[0]).max() < 1e-14)


# --- ingredient (iii): the matching lower bound -------------------------------

@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("t", [1, 2, 4, 7, 9])
def test_rank_t_factorisation(rule, t):
    A, _, _ = ks.propagator(rule, 30)
    du, dv, rk, res = ks.rank_one_factorisation(A, 15, t)
    assert res < 1e-14
    assert du == dv == rk == t


# --- the laws -----------------------------------------------------------------

@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("N", [12, 16, 20])
def test_module_law_at_every_cut(rule, N):
    """chi(t) = min(2t+1, chi_sat) for t >= 1, and 2 at t = 0, at every cut.

    chi_sat is a t-independent, cut-dependent ceiling (a finite-chain edge
    effect); the content of the test is that the approach to it is a plain
    truncation of 2t+1, with no intermediate regime.
    """
    A, kk, leak = ks.propagator(rule, N)
    assert leak == 0.0
    for x in range(2, N - 1):
        At, vals = np.eye(N), []
        for t in range(3 * N):
            if t:
                At = A @ At
            vals.append(ks.schmidt_rank_module(At, kk, x))
        assert vals[0] == 2, (rule, N, x)
        sat = max(vals)
        assert vals[1:] == [min(2 * t + 1, sat) for t in range(1, len(vals))], \
            (rule, N, x, vals)
        # Theorem 1's window: strict growth while the light cone is interior
        for t in range(1, min(x, N - x)):
            assert vals[t] == 2 * t + 1, (rule, N, x, t)


@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("N", [16, 24])
def test_crossing_ranks_are_exactly_t(rule, N):
    A, kk, _ = ks.propagator(rule, N)
    At = np.eye(N)
    for t in range(1, N // 2):
        At = A @ At
        lr, rl = ks.crossing_ranks(At, kk, N // 2)
        assert lr == rl == t, (rule, N, t, lr, rl)


@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("N", [8, 10])
def test_full_chain_law_bruteforce(rule, N):
    """chi(t) = 6t - 7 for 4 <= t <= N/2, computed on the full 2^N space."""
    series = ks.full_schmidt_rank(rule, N, N // 2, N // 2)
    assert series[:4] == [1, 4, 8, 12]
    for t in range(4, N // 2 + 1):
        assert series[t] == 6 * t - 7, (rule, N, t, series)


@pytest.mark.parametrize("rule", RULES)
def test_sparse_and_dense_full_chain_agree(rule):
    assert (ks.full_schmidt_rank_sparse(rule, 10, 5, 6)
            == ks.full_schmidt_rank(rule, 10, 5, 6))


# --- the cached numbers quoted in R27 -----------------------------------------

def _cache():
    if not os.path.exists(CACHE):
        pytest.skip("analytics/r27_schmidt_growth.json missing; run --write")
    with open(CACHE) as fh:
        return json.load(fh)


def test_cached_full_series_obey_the_law():
    d = _cache()
    for rule in RULES:
        for N in (8, 10, 12, 14, 16):
            s = d["full"][f"{rule}/{N}"]
            assert s[:4] == [1, 4, 8, 12], (rule, N, s)
            for t in range(4, N // 2 + 1):
                assert s[t] == 6 * t - 7, (rule, N, t, s)


def test_the_two_rules_have_identical_full_series():
    d = _cache()
    for N in (8, 10, 12, 14, 16):
        assert d["full"][f"156/{N}"] == d["full"][f"198/{N}"], N


def test_cached_module_series_obey_the_law():
    d = _cache()
    for key, row in d["module"].items():
        assert row[0] == 2, key
        sat = max(row)
        assert row[1:] == [min(2 * t + 1, sat) for t in range(1, len(row))], key
