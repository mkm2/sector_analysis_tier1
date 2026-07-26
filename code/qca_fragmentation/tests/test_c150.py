"""
Rule 150 (C150): the closed-form sector structure and the quantum layer of R8.
"""

from math import comb, ceil

import numpy as np
import pytest

from qca_fragmentation import c150
from qca_fragmentation.core import cycle, rules
from qca_fragmentation.graph import scc
from qca_fragmentation import results_io


# --- the wall (bond) representation -----------------------------------------

@pytest.mark.parametrize("N", range(2, 12))
def test_bond_map_is_a_bijection_onto_even_weight_strings(N):
    """obc0: x -> b is a bijection onto the even-weight (N+1)-strings."""
    seen = set()
    for x in range(1 << N):
        b = c150.wall_string(x, N, "obc0")
        assert bin(b).count("1") % 2 == 0
        assert b >> (N + 1) == 0
        assert c150.spin_from_walls(b, N) == x
        seen.add(b)
    assert seen == {b for b in range(1 << (N + 1))
                   if bin(b).count("1") % 2 == 0}


@pytest.mark.parametrize("N", range(3, 12))
def test_bond_map_on_the_ring_is_two_to_one(N):
    from collections import Counter
    c = Counter(c150.wall_string(x, N, "pbc") for x in range(1 << N))
    assert set(c.values()) == {2}
    for x in range(1 << N):
        flipped = x ^ ((1 << N) - 1)
        assert c150.wall_string(x, N, "pbc") == c150.wall_string(flipped, N, "pbc")


@pytest.mark.parametrize("bc", ["obc0", "pbc"])
@pytest.mark.parametrize("N", range(3, 11))
def test_wall_number_is_conserved_by_every_edge(N, bc):
    """Q is diagonal, so support-level conservation gives [U, Q] = 0 exactly."""
    for x in range(1 << N):
        q = c150.wall_number(x, N, bc)
        for y in cycle.succ(x, N, c150.TUPLE, bc):
            assert c150.wall_number(y, N, bc) == q, (N, bc, x, y)


def test_the_move_is_a_hardcore_hop():
    """A flip is allowed iff exactly one of the two touched bonds carries a
    wall, and it moves that wall to the other bond."""
    N = 7
    for x in range(1 << N):
        b = c150.wall_string(x, N, "obc0")
        for i in range(N):
            left = (x >> (i - 1)) & 1 if i > 0 else 0
            right = (x >> (i + 1)) & 1 if i < N - 1 else 0
            fires = c150.TUPLE[2 * left + right] == "V"
            bi, bip = (b >> i) & 1, (b >> (i + 1)) & 1
            assert fires == (bi != bip)
            if fires:
                b2 = c150.wall_string(x ^ (1 << i), N, "obc0")
                assert b2 == b ^ (1 << i) ^ (1 << (i + 1))


# --- closed forms ------------------------------------------------------------

@pytest.mark.parametrize("bc", ["obc0", "pbc"])
@pytest.mark.parametrize("N", range(3, 13))
def test_closed_form_matches_engine(N, bc):
    got = scc.analyze(150, N, bc, c150.TUPLE, detect_ergodic=False)
    assert got.sizes_recurrent == c150.sector_sizes(N, bc), (N, bc)


@pytest.mark.parametrize("bc", ["obc0", "pbc"])
def test_closed_form_matches_every_stored_unit(bc):
    """The deep Tier-1a dataset (N = 6..20) entry for entry."""
    stored = results_io.load_results(150, bc)
    assert stored, "no stored units for rule 150"
    for N, rec in stored.items():
        sizes = results_io.sizes_from_record(rec)
        assert sizes == c150.sector_sizes(N, bc), (N, bc)


def test_sector_count_and_dmax_formulas():
    for N in range(3, 60):
        cf = c150.sector_sizes_obc0(N)
        assert len(cf) == c150.n_sectors_obc0(N) == (N + 1) // 2 + 1
        assert cf[0] == c150.d_max_obc0(N)
        assert sum(cf) == 1 << N              # the shells tile the whole space
        assert c150.d_max_argmax_obc0(N) % 2 == 0


def test_dmax_base_is_exactly_two_with_a_sqrt_prefactor():
    """D_max/2^N -> 0 like N^{-1/2}: the base is 2, not 2.022."""
    from math import pi, sqrt
    for N in (20, 40, 80, 160):
        r = c150.d_max_ratio_obc0(N)
        asym = 2.0 * sqrt(2.0 / (pi * (N + 1)))
        assert abs(r / asym - 1.0) < 0.05 / (N ** 0.5) + 0.04, (N, r, asym)
    # ratios shrink, so no pure exponential with base > 2 can fit
    assert (c150.d_max_ratio_obc0(40) < c150.d_max_ratio_obc0(20)
            < c150.d_max_ratio_obc0(10))


def test_pbc_frozen_shells_split_in_two():
    """w = 0 and (even N) w = N have no available hop, so each of the two spin
    preimages is its own singleton sector."""
    for N in (6, 8, 10):
        sizes = c150.sector_sizes_pbc(N)
        assert sizes.count(1) == 4, N
    for N in (7, 9, 11):
        assert c150.sector_sizes_pbc(N).count(1) == 2, N


def test_verify_helper_is_clean():
    out = c150.verify_against_engine(max_N=10)
    assert out["mismatches"] == []
    assert out["charge_violations"] == 0
    assert out["units_checked"] > 30


# --- the quantum layer -------------------------------------------------------

@pytest.mark.parametrize("N,w", [(6, 2), (7, 2), (8, 4)])
def test_sector_unitary_is_orthogonal_and_confined(N, w):
    from qca_fragmentation.quantum import rule150_spectra as rs
    U, states = rs.sector_unitary(N, w)
    assert len(states) == comb(N + 1, w)
    assert np.abs(U.T @ U - np.eye(len(states))).max() < 1e-12
    assert rs.sector_states(N, w) == c150.states_with_walls(N, w)


@pytest.mark.parametrize("N", range(4, 11))
def test_fix_dimension_closed_forms(N):
    from qca_fragmentation.quantum import rule150_spectra as rs
    m = ceil(N / 2)
    assert rs.fix_dim_full(N) == 2 ** m
    assert m == len(range(0, N, 2))                  # = number of even sites
    assert rs.fix_is_halflayer_intersection(N)
    for w in range(0, N + 2, 2):
        if comb(N + 1, w) > 400:
            continue
        r = rs.spectral_portrait(N, w)
        assert r["mult_plus_one"] == comb(m, w // 2), (N, w)


@pytest.mark.parametrize("N,w", [(8, 4), (10, 4), (9, 4)])
def test_inner_sectors_are_not_krylov_spaces(N, w):
    from qca_fragmentation.quantum import rule150_spectra as rs
    r = rs.spectral_portrait(N, w)
    assert r["dim"] > N + 1                          # an inner shell
    assert r["n_distinct"] < r["dim"]
    assert max(r["dim_krylov"]) < r["dim"]
    assert not r["sector_is_a_krylov_space"]
    # the long-time state is not uniform on the sector
    assert max(r["max_pop_over_uniform"]) > 1.5


@pytest.mark.parametrize("N", [4, 6, 8, 10])
def test_extremal_shell_is_nondegenerate(N):
    """|S_w| <= N+1 is exactly where the spectrum is nondegenerate."""
    from qca_fragmentation.quantum import rule150_spectra as rs
    r = rs.spectral_portrait(N, N)
    assert r["dim"] == N + 1
    assert r["n_distinct"] == r["dim"]
    assert r["mult_plus_one"] == 1
    assert r["sector_is_a_krylov_space"]


@pytest.mark.parametrize("N", [5, 6, 7])
def test_reflection_acts_as_symmetry_or_time_reversal(N):
    from qca_fragmentation.quantum import rule150_spectra as rs
    r = rs.spectral_portrait(N, 2)
    assert r["reflection"] == ("commutes" if N % 2 else "conjugates U to U^-1")


@pytest.mark.parametrize("N", [5, 6, 7])
def test_bond_representation_reproduces_the_engine(N):
    from qca_fragmentation.quantum import rule150_spectra as rs
    assert rs.bond_matches_engine(N) < 1e-14


@pytest.mark.parametrize("N", [6, 7])
def test_only_the_gaussian_reference_is_free(N):
    """Both sign defects must go before the many-body spectrum becomes additive."""
    from qca_fragmentation.quantum import rule150_spectra as rs
    assert rs.additivity_defect(N, string=False, det_phase=True) < 1e-12
    assert rs.additivity_defect(N, string=True, det_phase=False) > 0.1
    assert rs.additivity_defect(N, string=False, det_phase=False) > 0.1
    assert rs.additivity_defect(N, string=True, det_phase=True) > 0.1


# --- the frontier store ------------------------------------------------------

def test_frontier_records_agree_with_the_closed_form():
    fr = c150.load_frontier()
    if not fr:
        pytest.skip("no frontier units computed yet")
    for key, rec in fr.items():
        assert rec["closed_form_exact"], key
        assert rec["sizes"] == c150.sector_sizes(rec["N"], rec["bc"]), key
        assert rec["n_sectors"] == rec["n_sectors_closed_form"], key
