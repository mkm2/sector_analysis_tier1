"""
Tier 1a graph regressions — the context Tier 1 sec.8 ground-truth table.
All items here MUST pass.
"""
from math import comb

import pytest

from qca_fragmentation.core import rules
from qca_fragmentation.graph import scc


def A(rule, N, bc, **kw):
    return scc.analyze(rule, N, bc, rules.wolfram_to_tuple(rule),
                       detect_ergodic=False, **kw)


def lucas(n):
    a, b = 2, 1
    for _ in range(n):
        a, b = b, a + b
    return a


@pytest.mark.parametrize("bc", ["pbc", "obc0"])
@pytest.mark.parametrize("N", [4, 6, 8])
def test_rule204_all_singletons(bc, N):
    r = A(204, N, bc)
    assert r.n_scc == (1 << N)
    assert set(r.sizes_scc) == {1}
    assert r.n_recurrent == (1 << N)


@pytest.mark.parametrize("bc", ["pbc", "obc0"])
@pytest.mark.parametrize("N", [4, 6, 8, 10])
def test_rule51_single_full_sector(bc, N):
    r = A(51, N, bc)
    assert r.n_scc == 1 and r.sizes_scc == [1 << N]


@pytest.mark.parametrize("N,Dmax,nsec", [
    (8, 126, 5), (9, 210, 6), (10, 462, 6), (11, 924, 7),
    (12, 1716, 7), (13, 3003, 8), (14, 6435, 8),
])
def test_rule150_obc0_reference(N, Dmax, nsec):
    r = A(150, N, "obc0")
    assert r.sizes_recurrent[0] == Dmax
    assert r.n_recurrent == nsec
    # every sector size is a binomial C(N+1, w) for even w
    binoms = {comb(N + 1, w) for w in range(0, N + 2, 2)}
    assert all(s in binoms for s in r.sizes_recurrent)


def test_rule150_obc0_N17_exact_sizes():
    r = A(150, 17, "obc0")
    assert r.sizes_recurrent == [43758, 43758, 18564, 18564, 3060, 3060,
                                 153, 153, 1, 1]


@pytest.mark.parametrize("N", [6, 8, 10, 12])
def test_rule54_frozen_plus_giant(N):
    r = A(54, N, "pbc")
    assert 1 in r.sizes_scc
    assert max(r.sizes_scc) == (1 << N) - 1


def test_rule156_pbc_lucas():
    r3 = A(156, 3, "pbc")
    assert sorted(r3.sizes_scc) == [1, 1, 2, 2, 2]
    for N in range(3, 12):
        assert A(156, N, "pbc").n_scc == lucas(N) + 1


@pytest.mark.parametrize("N", [4, 6, 8, 10])
def test_rule22_pbc_three_recurrent(N):
    r = A(22, N, "pbc")
    assert r.n_recurrent == 3
    assert r.sizes_recurrent == [1, 1, 1]


@pytest.mark.parametrize("bc", ["pbc", "obc0"])
def test_reflection_preserves_sector_sizes_unitary(bc):
    # For UNITARY rules, left-right reflection (swap r01,r10) preserves the
    # multiset of SCC sizes at every N and both boundary conventions.  This is
    # the property the reflection reduction of the unitary sweep relies on;
    # it is cross-checked against HSF (rules 156 and 198 share sector data).
    #
    # NOTE: this does NOT extend to dissipative rules under the even-first
    # brickwork convention, because reflection swaps the even/odd layers whose
    # order matters for the directed dissipative transition graph.  The
    # dissipative sweep therefore does not reduce by reflection.
    for r in rules.UNITARY_RULES:
        rr = rules.reflect_wolfram(r)
        for N in (4, 5, 6, 7):
            assert A(r, N, bc).sizes_scc == A(rr, N, bc).sizes_scc, (r, rr, N, bc)


def test_spinflip_partners_201_vs_108():
    # 201=(V,I,I,I) and 108=(I,I,I,V) are 0<->1 spin-flip (complement) partners.
    # Spin-flip = bit-flip X^{tensor N}, a basis permutation; XHX preserves gate
    # magnitudes, so it preserves the support/transition graph -> sector sizes
    # match under pbc (no boundary).  Under obc0 the fixed vacuum-0 padding maps
    # to vacuum-1, breaking the symmetry -> sizes differ (Fibonacci for 201 vs
    # shifted-Fibonacci for 108, matching HSF rules 1 and 8).
    for N in (8, 9, 10, 11, 12):
        assert A(201, N, "pbc").sizes_scc == A(108, N, "pbc").sizes_scc, N
        assert A(201, N, "obc0").sizes_scc != A(108, N, "obc0").sizes_scc, N
    # obc0 largest sectors follow the two Fibonacci offsets
    fib = {55: 8, 89: 9, 144: 10, 233: 11, 377: 12}
    for D, N in fib.items():
        assert A(201, N, "obc0").sizes_recurrent[0] == D
    shifted = {21: 8, 34: 9, 55: 10, 89: 11, 144: 12}
    for D, N in shifted.items():
        assert A(108, N, "obc0").sizes_recurrent[0] == D


def test_reflection_size_variance_is_unitary_only():
    # Guard the documented fact: at least one dissipative reflection pair has
    # DIFFERENT sector sizes under the even-first convention (here 187<->243).
    rr = rules.reflect_wolfram(187)
    assert rr == 243
    assert A(187, 6, "obc0").sizes_scc != A(243, 6, "obc0").sizes_scc


@pytest.mark.parametrize("bc", ["pbc", "obc0"])
@pytest.mark.parametrize("N", [6, 8, 10])
def test_union_find_matches_tarjan_unitary(bc, N):
    # For unitary rules the memory-light union-find path (default) must agree
    # with the directed Tarjan path (keep_comp_id=True) on the sector sizes.
    for r in rules.UNITARY_RULES:
        t = rules.wolfram_to_tuple(r)
        uf = scc.analyze(r, N, bc, t, detect_ergodic=False)
        tj = scc.analyze(r, N, bc, t, detect_ergodic=False, keep_comp_id=True)
        assert uf.sizes_scc == tj.sizes_scc, (r, bc, N)


def test_ergodic_early_exit_flags():
    # rule 51 is a single full sector (size 2^N > f_erg*2^N): must trip early-exit
    r = scc.analyze(51, 8, "pbc", rules.wolfram_to_tuple(51),
                    f_erg=0.5, detect_ergodic=True)
    assert r.ergodic and r.ergodic_bound == (1 << 8)
    # rule 204 (all singletons) must NOT be flagged ergodic
    r2 = scc.analyze(204, 8, "pbc", rules.wolfram_to_tuple(204),
                     f_erg=0.5, detect_ergodic=True)
    assert not r2.ergodic
