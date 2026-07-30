"""
X-gate QCA: the 256 symbol tables with V = X instead of the Hadamard.

These are permutation circuits -- every local symbol is a function on bits, so
the transition graph has out-degree 1 and the whole Tier-1e/Tier-1a distinction
collapses.  The tests pin that collapse, because it is the control the family is
for.
"""

from array import array

import pytest

from qca_fragmentation import results_io
from qca_fragmentation.core import rules
from qca_fragmentation.graph import wcc
from qca_fragmentation.permutation import functional as fnl
from qca_fragmentation.permutation import xca


# --- the map itself -----------------------------------------------------------

@pytest.mark.parametrize("N", [7, 8])
@pytest.mark.parametrize("bc", ["obc0", "pbc"])
def test_only_the_I_V_rules_are_reversible(N, bc):
    """I and V are bijective, D and E are not, so reversibility must coincide
    exactly with the 16-rule unitary set."""
    for r in range(256):
        assert xca.is_reversible(r, N, bc) is rules.is_unitary(
            rules.wolfram_to_tuple(r)), (r, N, bc)


def test_step_table_matches_the_reference_step():
    for r in (22, 90, 150, 201):
        for N in (7, 8):
            t = rules.wolfram_to_tuple(r)
            ref = [xca.x_step(x, N, t, "obc0") for x in range(1 << N)]
            assert list(xca.step_table(r, N, "obc0")) == ref


def test_local_action_is_the_symbol_table():
    """
    With a uniform table the neighbour pattern is irrelevant, so one cycle is
    just the local gate applied at every site: I is the identity, V (= X) is the
    global flip, D sends everything to 0 and E to all-ones.
    """
    N = 5
    mask = (1 << N) - 1
    for sym, expect in (("I", lambda x: x),
                        ("V", lambda x: ~x & mask),
                        ("D", lambda x: 0),
                        ("E", lambda x: mask)):
        t = (sym, sym, sym, sym)
        for x in range(1 << N):
            assert xca.x_step(x, N, t, "obc0") == expect(x), (sym, x)


# --- the functional-graph facts ----------------------------------------------

@pytest.mark.parametrize("rule", [0, 22, 28, 51, 90, 105, 150, 156, 201, 204,
                                  232, 255])
@pytest.mark.parametrize("N", [8, 10])
def test_F1_one_cycle_per_weak_component(rule, N):
    """The defining fact: in a functional graph every WCC holds exactly one
    cycle, so n_recurrent == n_wcc.  This is what fails for the Hadamard
    circuits, where R9's rule 22 has 3 attractors inside 2 sectors."""
    d = fnl.analyze(rule, N, "obc0")
    assert d["n_recurrent"] == d["n_wcc"]
    assert d["att_per_sector"] == 1.0
    assert sum(d["sizes_wcc"]) == 2 ** N


@pytest.mark.parametrize("rule", [22, 90, 150, 204, 232])
@pytest.mark.parametrize("N", [8, 10])
def test_F2_union_find_agrees_with_the_basin_decomposition(rule, N):
    """Independent check: the Tier-1e union-find knows nothing about functional
    graphs and must still produce the same partition."""
    out = fnl.cross_check_wcc(rule, N, "obc0")
    assert out["agrees"]


def test_reversible_rules_have_no_transient():
    for rule in rules.UNITARY_RULES:
        d = fnl.analyze(rule, 9, "obc0")
        assert d["reversible"]
        assert d["transient_fraction"] == 0.0
        assert d["transient_depth"] == 0
        # sectors ARE the cycles
        assert d["sizes_wcc"] == d["sizes_recurrent"]


def test_irreversible_rules_have_a_transient():
    for rule in (22, 28, 90, 232):
        d = fnl.analyze(rule, 10, "obc0")
        assert not d["reversible"]
        assert d["transient_fraction"] > 0.0
        assert d["transient_depth"] >= 1


# --- anchors ------------------------------------------------------------------

@pytest.mark.parametrize("N", [6, 8, 10])
def test_rule_204_is_the_identity(N):
    """IIII does nothing, so every state is a fixed point."""
    d = fnl.analyze(204, N, "obc0")
    assert d["n_wcc"] == 2 ** N == d["n_recurrent"] == d["n_fixed_points"]
    assert d["d_max_wcc"] == 1


@pytest.mark.parametrize("N", [6, 8, 10])
def test_rule_51_flips_every_site_every_step(N):
    """VVVV with V = X is the global flip: 2-cycles everywhere, no fixed point
    (unlike the Hadamard case, where 51 gives ONE sector of size 2^N)."""
    d = fnl.analyze(51, N, "obc0")
    assert d["n_wcc"] == 2 ** (N - 1)
    assert d["d_max_wcc"] == 2
    assert d["n_fixed_points"] == 0
    assert d["sizes_recurrent"] == [2] * 2 ** (N - 1)


def test_the_sum_rule_and_the_hyperbola_hold():
    """Sectors still partition, so n_wcc * D_max >= 2^N at finite N."""
    for rule in (0, 22, 51, 150, 156, 201, 204):
        for N in (8, 10, 12):
            d = fnl.analyze(rule, N, "obc0")
            assert sum(d["sizes_wcc"]) == 2 ** N
            assert d["n_wcc"] * d["d_max_wcc"] >= 2 ** N


# --- record schema ------------------------------------------------------------

def test_xgate_record_roundtrip():
    res = fnl.analyze(150, 10, "obc0")
    rec = results_io.record_from_xgate(res)
    assert set(results_io.XGATE_FIELDS) >= set(rec)
    assert results_io.sizes_from_xgate_record(rec, "wcc") == res["sizes_wcc"]
    assert (results_io.sizes_from_xgate_record(rec, "recurrent")
            == res["sizes_recurrent"])


# --- the vectorised map builder ----------------------------------------------

@pytest.mark.parametrize("bc", ["obc0", "pbc"])
@pytest.mark.parametrize("N", [7, 9])
def test_numpy_step_table_matches_the_scalar_one(bc, N):
    """
    step_table_np lifts the loop over x into numpy but keeps the SITE order from
    _compile, so it is a vectorisation and not a simultaneous-update
    approximation.  Checked on every 7th rule, both boundary conventions.
    """
    for rule in range(0, 256, 7):
        assert (xca.step_table_np(rule, N, bc).tolist()
                == list(xca.step_table(rule, N, bc))), (rule, N, bc)


@pytest.mark.parametrize("rule", [22, 90, 150, 201, 232])
def test_vectorised_and_scalar_analyses_agree(rule):
    for N in (9, 11):
        a = fnl.analyze(rule, N, "obc0", vectorised=True)
        b = fnl.analyze(rule, N, "obc0", vectorised=False)
        for k in ("n_wcc", "d_max_wcc", "n_recurrent", "d_max_recurrent",
                  "transient_depth", "sizes_wcc", "sizes_recurrent"):
            assert a[k] == b[k], (rule, N, k)


def test_simultaneous_update_would_be_wrong_at_odd_N_pbc():
    """
    Guard on the reason the step list is taken from _compile rather than
    reimplemented: at odd N with pbc the even sublattice is not an independent
    set, so a simultaneous sublattice update differs from the sequential sweep.
    """
    N, rule = 9, 150
    t = rules.wolfram_to_tuple(rule)
    seq = list(xca.step_table(rule, N, "pbc"))

    def simultaneous(x):
        for layer in (range(0, N, 2), range(1, N, 2)):
            y = x
            for i in layer:
                left = (x >> ((i - 1) % N)) & 1
                right = (x >> ((i + 1) % N)) & 1
                s = t[2 * left + right]
                if s == "V":
                    y ^= 1 << i
                elif s == "D":
                    y &= ~(1 << i)
                elif s == "E":
                    y |= 1 << i
            x = y
        return x

    assert [simultaneous(x) for x in range(1 << N)] != seq


# --- the meaning of "reversible" (R10 sec.1.1) --------------------------------

_REVERSIBLE_16 = {51, 54, 57, 60, 99, 102, 105, 108,
                  147, 150, 153, 156, 195, 198, 201, 204}
_REVERSIBLE_ECA_6 = [15, 51, 85, 170, 204, 240]


@pytest.mark.parametrize("N", [7, 8, 9])
@pytest.mark.parametrize("bc", ["obc0", "pbc"])
def test_reversible_set_is_exactly_the_sixteen(N, bc):
    """Reversible = the one-cycle map is a bijection.  Holds iff every symbol is
    I or V, for both boundary conventions and both parities of N."""
    got = {r for r in range(256) if xca.is_reversible(r, N, bc)}
    assert got == _REVERSIBLE_16


def _eca_step(x, N, r):
    y = 0
    for i in range(N):
        nb = ((((x >> ((i + 1) % N)) & 1) << 2)
              | (((x >> i) & 1) << 1) | ((x >> ((i - 1) % N)) & 1))
        if (r >> nb) & 1:
            y |= 1 << i
    return y


def test_the_sixteen_are_not_the_six_reversible_eca():
    """
    The nearby literature count is SIX -- the reversible elementary CA, one
    layer and simultaneous update.  That is a different construction AND a
    differently quantified statement: an ECA counts as reversible when its map
    is a bijection for EVERY N, and the six are the intersection over N.  At a
    fixed N the bijective set is strictly larger (at N=7 it also contains 45,
    75, 89 and 150), which is why this must not be asserted per N.

    Our 16, by contrast, are reversible at every N separately -- the proof is
    local and never touches the lattice size.
    """
    per_n = {}
    for N in (6, 7, 8, 9):
        per_n[N] = {r for r in range(256)
                    if len({_eca_step(x, N, r) for x in range(1 << N)})
                    == (1 << N)}
    always = set.intersection(*per_n.values())
    assert always == set(_REVERSIBLE_ECA_6)
    assert any(len(per_n[N]) > 6 for N in per_n)      # accidental at fixed N
    assert always & _REVERSIBLE_16 == {51, 204}
    assert len(_REVERSIBLE_16) == 16 and len(always) == 6
