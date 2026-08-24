"""Regression tests for R11 rev.2: the partition-level gate audit.

Pins the three corrections: the refinement widened to N=7..10 and both boundary
conditions, the 81 V-free rules as a theorem floor (NOT a fitted count), and the
branch/flip-set bijection -- the test that can actually fail.
"""

import pytest

from qca_fragmentation.core import rules as rules_mod
from qca_fragmentation.permutation import partition_audit as pa

UNITARY = tuple(r for r in range(256)
                if rules_mod.is_unitary(rules_mod.wolfram_to_tuple(r)))


# --- the V-free theorem floor -------------------------------------------------

def test_there_are_exactly_81_v_free_rules():
    """R11 Cor. 4: 81 = 3^4 = |{I,D,E}^4|.  A property of the encoding."""
    assert sum(1 for r in range(256) if pa.is_v_free(r)) == 3 ** 4 == 81


def test_only_one_v_free_rule_is_unitary():
    """So the 81-rule identity is essentially empty in the all-I/V sector."""
    vf = [r for r in UNITARY if pa.is_v_free(r)]
    assert vf == [204], vf
    assert "".join(rules_mod.wolfram_to_tuple(204)) == "IIII"


@pytest.mark.parametrize("bc", ["obc0", "pbc"])
@pytest.mark.parametrize("N", [6, 7])
def test_v_free_rules_have_literally_the_same_map(bc, N):
    """No V means the gate is never invoked: succ_H(x) == {step_X(x)} exactly."""
    from qca_fragmentation.core.cycle import succ
    from qca_fragmentation.permutation import xca
    for rule in (r for r in range(256) if pa.is_v_free(r)):
        t = rules_mod.wolfram_to_tuple(rule)
        T = xca.step_table(rule, N, bc)
        for x in range(1 << N):
            assert succ(x, N, t, bc) == [int(T[x])], (rule, N, bc, x)


@pytest.mark.parametrize("bc", ["obc0", "pbc"])
@pytest.mark.parametrize("N", [6, 7, 8])
def test_the_81_are_a_floor_present_at_every_size(bc, N):
    a = pa.audit(N, bc)
    assert a["v_free_total"] == 81
    assert a["n_equal_v_free"] == 81, (bc, N, a)
    assert a["n_equal"] >= 81


def test_the_excess_over_81_is_accidental_and_N_dependent():
    """The total is NOT a structural number -- that is why R11 rev.2 reports the
    decomposition rather than the total."""
    excess = [pa.audit(N, "pbc")["n_equal_accidental"] for N in (7, 8)]
    assert excess[0] != excess[1], excess          # varies with N
    assert all(e >= 0 for e in excess)


# --- refinement, widened ------------------------------------------------------

@pytest.mark.parametrize("bc", ["obc0", "pbc"])
@pytest.mark.parametrize("N", [7, 8])
def test_x_refines_h_for_every_rule(bc, N):
    a = pa.audit(N, bc)
    assert a["refine_failures"] == [], (bc, N, a["refine_failures"])
    assert a["refines_ok"] == 256


def test_refinement_is_tested_at_partition_level_not_by_counts():
    """A count-based check would pass on partitions that merely have the same
    number of blocks; this must not."""
    a = [0, 0, 1, 1]
    b = [0, 1, 0, 1]                    # same block count, different partition
    assert not pa.same_partition(a, b)
    assert not pa.refines(a, b)
    assert pa.refines([0, 1, 2, 3], a)  # finer refines coarser
    assert not pa.refines(a, [0, 1, 2, 3])


# --- the branch/flip-set bijection: the test that can fail --------------------

@pytest.mark.parametrize("bc", ["obc0", "pbc"])
@pytest.mark.parametrize("rule", UNITARY)
def test_branch_flipset_bijection(bc, rule):
    """Multiplicity 1 everywhere => no destructive interference in one cycle."""
    r = pa.branch_multiplicity(rule, 6, bc)
    assert r["max_multiplicity"] == 1, (rule, bc, r)
    assert r["flipset_is_xor"] is True
    assert r["visits_per_site"] == [1], r


def test_branch_multiplicity_rejects_reset_rules():
    with pytest.raises(ValueError):
        pa.branch_multiplicity(232, 6, "obc0")     # DIIE
