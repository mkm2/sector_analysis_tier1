"""
R11: the two gates on one rule space.

The tests here pin the refinement theorem, which is what makes the comparison
meaningful at all, and the corollary that the V-free rules are the same circuit
under both gates -- a cross-implementation check between two engines that share
no code below core/cycle.py::_compile.
"""

import pytest

from qca_fragmentation.core import rules
from qca_fragmentation.graph import wcc
from qca_fragmentation.permutation import compare, functional as fnl, xca


@pytest.mark.parametrize("N", [7, 8])
@pytest.mark.parametrize("bc", ["obc0", "pbc"])
def test_every_X_edge_is_a_Hadamard_edge(N, bc):
    """R11 Thm 1: flipping the target bit is one of the Hadamard outcomes."""
    for rule in range(256):
        succ = wcc.make_succ(rule, N, bc)
        T = xca.step_table(rule, N, bc)
        for x in range(1 << N):
            assert T[x] in succ(x), (rule, N, bc, x)


@pytest.mark.parametrize("N", [8])
def test_X_sectors_refine_hadamard_sectors(N):
    rc = compare.refinement_check(N, "obc0")
    assert rc["edge_failures"] == []
    assert rc["refine_failures"] == []


@pytest.mark.parametrize("N", [8, 9])
def test_the_two_inequalities(N):
    """n_wcc can only go up and D_max only down when the gate becomes X."""
    for rule in range(256):
        h = wcc.weak_components(rule, N, "obc0")
        x = fnl.analyze(rule, N, "obc0")
        assert x["n_wcc"] >= h.n_wcc, (rule, N, h.n_wcc, x["n_wcc"])
        assert x["d_max_wcc"] <= h.d_max_wcc, (rule, N)


@pytest.mark.parametrize("N", [8, 9])
def test_v_free_rules_are_the_same_circuit(N):
    """81 rules never invoke the gate, so both engines must agree exactly."""
    shared = [r for r in range(256) if not compare.has_V(r)]
    assert len(shared) == 81
    for rule in shared:
        h = wcc.weak_components(rule, N, "obc0")
        x = fnl.analyze(rule, N, "obc0")
        assert sorted(h.sizes_wcc) == sorted(x["sizes_wcc"]), (rule, N)


def test_rule_51_is_the_extreme_case():
    """One sector of 2^N under H; 2^(N-1) two-cycles under X."""
    for N in (8, 9):
        h = wcc.weak_components(51, N, "obc0")
        x = fnl.analyze(51, N, "obc0")
        assert h.n_wcc == 1 and h.d_max_wcc == (1 << N)
        assert x["n_wcc"] == (1 << (N - 1)) and x["d_max_wcc"] == 2


def test_the_common_window_is_actually_common():
    """Every comparison must use the Hadamard window, or the bases are not
    comparable (R11 sec.1)."""
    assert compare.COMMON_N_CAP == 16
    p = compare.x_point(156, "obc0")
    assert p is not None and p["N_max"] <= compare.COMMON_N_CAP


def test_no_rule_loses_its_sector_growth_class():
    """The class cross-tabulation must be upper-triangular."""
    d = compare.load("obc0") or compare.build("obc0")
    order = {"constant": 0, "polynomial": 1, "exponential": 2}
    for r in d["rows"]:
        h, x = r["cls_a_h"], r["cls_a_x"]
        if h in order and x in order:
            assert order[x] >= order[h], (r["rule"], h, x)
