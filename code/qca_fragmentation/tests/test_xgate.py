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


# --- movement analyses (R10 sec.5-8) ------------------------------------------

from qca_fragmentation.permutation import movement as mvmod   # noqa: E402
from qca_fragmentation.scaling import sectors as _sectors      # noqa: E402


def test_children_partition_the_rule_space():
    """Every rule is a descendant of exactly one reversible parent, and a parent
    with k identity slots has 3^k - 1 children."""
    seen = []
    for parent in rules.UNITARY_RULES:
        kids = mvmod.children(parent)
        k = sum(1 for s in rules.wolfram_to_tuple(parent) if s == "I")
        assert len(kids) == 3 ** k - 1, (parent, len(kids))
        seen.extend(kids)
    assert sorted(seen) == sorted(set(range(256)) - set(rules.UNITARY_RULES))
    assert len(mvmod.children(204)) == 80          # the whole V-free family


def test_the_reset_only_rules_are_the_sixteen_boolean_functions():
    assert len(mvmod.RESET_ONLY) == 16
    affine = {r for r in mvmod.RESET_ONLY
              if mvmod._is_affine(tuple(1 if s == "E" else 0
                                        for s in rules.wolfram_to_tuple(r)))}
    # the projections are affine too (degenerate in one argument), so the set
    # is larger than {90, 165}; what is unique to those two is XOR itself
    assert {90, 165} <= affine
    for r in mvmod.RESET_ONLY:
        tt = tuple(1 if s == "E" else 0 for s in rules.wolfram_to_tuple(r))
        xor_like = tt in ((0, 1, 1, 0), (1, 0, 0, 1))
        assert xor_like == (r in (90, 165))


@pytest.mark.parametrize("N", [10, 11])
def test_a_cycle_never_exceeds_its_sector(N):
    """L_k <= |S_k| pointwise, which is what makes movement 1 point downward."""
    for rule in (0, 1, 22, 73, 90, 108, 156, 201, 204):
        res = fnl.analyze(rule, N, "obc0")
        assert res["d_max_recurrent"] <= res["d_max_wcc"], (rule, N)


def test_dissipation_inherits_the_parent_orbits_for_73_and_109():
    """R10 sec.8: for these two the reset acts as the identity on the recurrent
    set, so every attractor of the child is an orbit of the parent."""
    for child, parent in ((73, 201), (109, 108)):
        for N in (10, 12):
            ccyc, T = mvmod._cycles(child, N, "obc0")
            pcyc, Tp = mvmod._cycles(parent, N, "obc0")
            psets = [frozenset(c) for c in pcyc]
            assert all(frozenset(c) in psets for c in ccyc), (child, N)
            assert max(len(c) for c in ccyc) == max(len(c) for c in pcyc)


def test_inheritance_summary_is_stable_across_parity():
    inh = mvmod.inheritance(10, "obc0")
    assert inh["n_children"] == 240
    assert inh["rules"]["73"]["inherits"] is True
    assert inh["rules"]["109"]["inherits"] is True
    # the identity's cluster: a child cannot have a longer cycle than a parent
    # whose orbits are all fixed points, EXCEPT trivially, so 90/165 show up
    assert 90 in inh["longer_than_parent"]


# --- the two fitting guards (R10 sec.3) ---------------------------------------

def test_bounded_series_is_not_read_as_base_two():
    """X45's longest cycle is 1,2,2,1,2,2,... and used to be fitted as 2^N."""
    Ns = list(range(6, 25))
    ys = [(1, 2, 2)[i % 3] for i in range(len(Ns))]
    assert _sectors._volume_fraction_base(Ns, ys) is None
    d = _sectors.series_descriptor(45, "obc0", "x_d_max_recurrent", Ns, ys)
    assert d["cls"] != "exponential", d


def test_arithmetic_progression_is_polynomial_not_exponential():
    """X1's cycle lengths are 44,47,50,... : a straight line, base 1."""
    Ns = list(range(6, 25))
    ys = [2 + 3 * n for n in Ns]
    d = _sectors.series_descriptor(1, "obc0", "x_d_max_recurrent", Ns, ys)
    assert d["cls"] == "polynomial" and abs(d["base"] - 1.0) < 1e-9, d


def test_the_guards_leave_a_genuine_binomial_series_alone():
    """W134's D_max is the central binomials ~ 2^N/sqrt(N): still base 2."""
    from math import comb
    Ns = list(range(6, 17))
    ys = [comb(N + 1, (N + 1) // 2) for N in Ns]
    vf = _sectors._volume_fraction_base(Ns, ys)
    assert vf is not None and abs(vf[0] - 2.0) < 1e-12
    assert abs(vf[1] + 0.5) < 0.25, vf


def test_an_honestly_exponential_cycle_survives_the_guards():
    """X57's longest cycle really does grow like 2^N."""
    from qca_fragmentation.permutation import analysis as _an
    s = _an.load_series(57, "obc0", 24)
    d = _sectors.series_descriptor(57, "obc0", "x_d_max_recurrent",
                                   s["N"], s["d_max_recurrent"])
    assert d["cls"] == "exponential" and d["base"] > 1.9, d


# --- the bitwise stepper (R10 sec.10) -----------------------------------------

from qca_fragmentation.permutation import orbits as orb          # noqa: E402
from qca_fragmentation.permutation import coherence as coh       # noqa: E402


@pytest.mark.parametrize("N", [6, 7, 8, 9])
def test_bitwise_step_equals_the_compiled_step(N):
    """The sublattice identity must hold for the RESET rules too, not just the
    I/V ones: an even site writes an even bit and reads only odd bits, so the
    even half-sweep cannot change any input to another even site whatever the
    local action is."""
    for rule in range(256):
        step = orb.make_bitstep(rule, N, "obc0")
        t = rules.wolfram_to_tuple(rule)
        for x in range(1 << N):
            assert step(x) == xca.x_step(x, N, t, "obc0"), (rule, N, x)


def test_bitwise_step_refuses_pbc():
    """At pbc with odd N the sublattices are not independent sets."""
    with pytest.raises(ValueError):
        orb.make_bitstep(150, 9, "pbc")


def test_sampled_orbit_length_is_a_lower_bound_on_the_exact_maximum():
    """The sampler exhibits an orbit, so it can never exceed L_max; whether it
    attains it depends on how much of the space the longest orbit covers.  For
    73 and 109 it is tight at 400 draws; for 156 it is not always, because its
    longest orbit lives in a rare sector (R11 sec.7.2)."""
    from qca_fragmentation.permutation import analysis as an
    for rule, tight in ((156, False), (73, True), (109, True)):
        s = an.load_series(rule, "obc0", 24)
        if 20 not in s["N"]:
            pytest.skip("sweep not present")
        i = s["N"].index(20)
        g = orb.sample_orbits(rule, 20, samples=400, burn=orb.burn_for(rule, 20),
                              cap=10 ** 7)
        assert g["abandoned"] == 0
        assert g["max"] <= s["d_max_recurrent"][i]
        if tight:
            assert g["max"] == s["d_max_recurrent"][i]


# --- coherence: the correction to R10's first draft ---------------------------

def test_a_functional_graph_still_decoheres():
    """Rule 90's reset labels de-synchronise around its cycles, so no coherence
    survives -- exactly R7's finding, reproduced from the X-gate side."""
    d = coh.dfs_blocks(90, 12, "obc0")
    assert d["dfs_max"] == 1 and not d["no_jump_anywhere"]


def test_rule_232_has_an_exponentially_large_dfs():
    """The type case: every recurrent state is a fixed point on which no reset
    ever jumps, so the whole recurrent set spans a decoherence-free subspace."""
    dims = []
    for N in (8, 10, 12):
        d = coh.dfs_blocks(232, N, "obc0")
        assert d["no_jump_anywhere"] and d["n_blocks"] == 1
        assert d["dfs_max"] == d["n_rec"]
        dims.append(d["dfs_max"])
    assert dims == [44, 117, 305]
    assert (dims[2] / dims[0]) ** 0.25 == pytest.approx(1.618, abs=0.01)


def test_inheritance_is_exactly_the_no_jump_condition():
    """R10 sec.8 measured that the reset acts as the identity on the recurrent
    set for 177 of 240 rules.  That is the same statement as 'no Kraus jump
    fires on a recurrent state', i.e. the DFS criterion."""
    from qca_fragmentation.permutation import movement as mvm
    inh = mvm.inheritance(10, "obc0")
    for r_s, v in inh["rules"].items():
        d = coh.dfs_blocks(int(r_s), 10, "obc0")
        assert v["inherits"] == d["no_jump_anywhere"], (r_s, v, d)


def test_rule_1_dfs_is_the_golden_mean_shift():
    """Its recurrent set -- and hence its DFS -- is exactly the set of strings
    with no two adjacent 1s, the hard-core constrained space."""
    for N in (10, 12):
        rec = {x for c in coh.recurrent_states(1, N, "obc0") for x in c}
        want = {x for x in range(1 << N) if "11" not in format(x, f"0{N}b")}
        assert rec == want
        assert coh.dfs_blocks(1, N, "obc0")["dfs_max"] == len(want)


def test_dfs_blocks_agree_with_the_exact_cesaro_rank():
    """
    The label census is validated against the full superoperator, not just
    against itself.  For a V-free rule the channel is the same in both families,
    so the Tier-1b oracle applies: a single DFS block of dimension d whose states
    are all fixed points must give dim Fix(Phi) = d^2.
    """
    from qca_fragmentation.quantum import dissipative_report as dr
    for rule, N, bc in ((232, 6, "obc0"), (90, 6, "pbc")):
        d = coh.dfs_blocks(rule, N, bc)
        g = dr.cesaro_gap(rule, N, bc)
        assert d["n_blocks"] == 1 and d["cycle_lengths"] == [1], (rule, d)
        assert g["cesaro_rank"] == d["dfs_max"] ** 2, (rule, d, g)
        assert g["classical_dim"] == d["dfs_max"]


def test_rule_10_pbc_has_the_two_cycle_coherence_R7_found():
    """R7: rule 10 (DEDD) has a 2-cycle whose Kraus labels are matched, so the
    coherence between its two states is protected.  Found here from the label
    side, at N = 8 pbc."""
    d = coh.dfs_blocks(10, 8, "pbc")
    assert d["dfs_max"] == 2
