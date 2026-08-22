"""Tests for the R22 sec.6 asymptotic-manifold decomposition."""
import numpy as np
import pytest

from qca_fragmentation.core import rules
from qca_fragmentation.graph import scc
from qca_fragmentation.quantum import asymptotic as AS


# --- the decomposition is internally consistent -----------------------------

@pytest.mark.parametrize("rule", (232, 91, 189, 19, 203, 36, 150, 204, 61, 27))
def test_the_block_data_adds_up(rule):
    z = AS.decompose(rule, 4)
    assert z.consistent, z.describe()
    assert sum(b.support for b in z.blocks) == z.dim_asym
    assert sum(b.n * b.n for b in z.blocks) == z.dim_alg
    assert all(b.n * b.m == b.support for b in z.blocks)


@pytest.mark.parametrize("rule", (232, 91, 189, 19, 203, 36, 150))
def test_the_fixed_points_never_outnumber_the_peripheral_algebra(rule):
    """Fix(Phi-dagger) sits inside the peripheral span; equality iff every U_k = 1."""
    z = AS.decompose(rule, 4)
    assert z.dim_fix <= z.dim_alg


@pytest.mark.parametrize("rule", (232, 91, 189, 36))
def test_the_decomposition_does_not_depend_on_the_random_seed(rule):
    a = AS.decompose(rule, 4, seed=0)
    b = AS.decompose(rule, 4, seed=17)
    assert sorted((x.n, x.m) for x in a.blocks) == sorted((x.n, x.m) for x in b.blocks)


# --- R22 Prop. 1 seen from the manifold side --------------------------------

@pytest.mark.parametrize("rule", (150, 105, 204, 108, 201))
def test_unitary_rules_lose_nothing(rule):
    """Unital => no transient part => the asymptotic space is everything, and the
    dynamics on it is unitary, so there is one block with m_k = 1."""
    assert rules.is_unitary(rules.wolfram_to_tuple(rule))
    z = AS.decompose(rule, 4)
    assert z.dim_asym == z.dim == 16
    assert z.n_blocks == 1
    assert z.blocks[0].n == 16 and z.blocks[0].m == 1
    assert z.is_purely_coherent


# --- the two pure cases, and the mixed one ----------------------------------

def test_the_majority_rule_is_one_decoherence_free_subspace():
    """W232 = DIIE is Wolfram's majority rule; its whole recurrent set is a DFS."""
    z = AS.decompose(232, 4)
    assert z.n_blocks == 1
    assert (z.blocks[0].n, z.blocks[0].m) == (7, 1)
    assert z.dim_alg == 49 == z.dim_fix          # U_k = 1: the full algebra M_7
    assert z.is_purely_coherent
    assert z.logical_qubits == pytest.approx(np.log2(7))


def test_w91_has_a_single_steady_state_and_no_memory():
    z = AS.decompose(91, 4)
    assert z.n_blocks == 1
    assert (z.blocks[0].n, z.blocks[0].m) == (1, 3)
    assert z.is_purely_classical
    assert z.logical_qubits == 0.0


def test_w189_is_bistable_with_no_coherence():
    z = AS.decompose(189, 4)
    assert z.n_blocks == 2
    assert sorted((b.n, b.m) for b in z.blocks) == [(1, 3), (1, 4)]
    assert z.is_purely_classical


def test_w19_carries_both_kinds_of_block():
    """A decoherence-free C^4 alongside a nine-dimensional junk state."""
    z = AS.decompose(19, 4)
    assert sorted((b.n, b.m) for b in z.blocks) == [(1, 9), (4, 1)]
    assert not z.is_purely_classical and not z.is_purely_coherent
    assert z.dim_fix < z.dim_alg                 # U_k on the DFS is not the identity


@pytest.mark.parametrize("rule,expected", (
    (232, [(7, 1)]),
    (91, [(1, 3)]),
    (189, [(1, 3), (1, 4)]),
    (19, [(1, 9), (4, 1)]),
    (203, [(4, 1)]),
    (36, [(6, 1)]),
))
def test_the_blocks_R22_quotes_are_the_blocks_the_module_returns(rule, expected):
    z = AS.decompose(rule, 4)
    assert sorted((b.n, b.m) for b in z.blocks) == expected


# --- the structural observation R22 sec.6 records ---------------------------

@pytest.mark.parametrize("rule", (232, 91, 189, 19, 203, 36, 61, 103, 27, 122,
                                  150, 204, 3, 17, 45, 47, 179, 231))
def test_no_block_mixes_the_two_factors(rule):
    """Observed across all 256 rules at N = 4: every block is either a pure DFS
    (m_k = 1) or a single state (n_k = 1); no genuine noiseless SUBSYSTEM."""
    z = AS.decompose(rule, 4)
    assert all(b.n == 1 or b.m == 1 for b in z.blocks), z.describe()


# --- agreement with the population graph ------------------------------------

@pytest.mark.parametrize("rule", (232, 189, 36))
def test_the_asymptotic_dimension_matches_the_recurrent_state_count(rule):
    """When the attractor is basis-spanned the graph gets the DIMENSION right --
    it is the split into n_k and m_k that it cannot see."""
    z = AS.decompose(rule, 4)
    g = scc.analyze(rule, 4, "obc0", rules.wolfram_to_tuple(rule),
                    detect_ergodic=False)
    assert z.dim_asym == sum(g.sizes_recurrent)


def test_the_majority_rule_reproduces_the_xgate_coherence_result():
    """permutation/coherence.py gets dim A = n_rec^2 for W232 by a completely
    different route (label synchronisation); the superoperator must agree."""
    for N, n_rec in ((4, 7), (5, 11)):
        z = AS.decompose(232, N)
        assert z.dim_alg == n_rec * n_rec, (N, z.dim_alg)


# --- memory / error correction ----------------------------------------------

def test_fates_refuses_a_branching_rule():
    with pytest.raises(ValueError):
        AS.fates(150, 6)


@pytest.mark.parametrize("N,n_att,b0,b1", (
    (4, 7, 7, 2), (6, 17, 17, 5), (8, 44, 41, 12), (10, 117, 99, 29),
))
def test_the_majority_basins_are_what_R22_quotes(N, n_att, b0, b1):
    p = AS.memory_profile(232, N)
    assert (p["n_attractors"], p["basin_zero"], p["basin_one"]) == (n_att, b0, b1)


@pytest.mark.parametrize("N", (5, 6, 8, 10, 12))
def test_the_autonomous_distance_of_majority_voting_is_one_at_every_size(N):
    """The islands of Guedes et al.: a two-block of flipped bits is itself a
    protected pointer state, so the automaton corrects exactly one error however
    long the code -- against a repetition distance growing like N/2."""
    assert AS.autonomous_distance(232, N, 0) == 1
    assert AS.memory_profile(232, N)["repetition_distance"] >= (N - 1) // 2


def test_the_codeword_basin_shrinks_exponentially():
    fracs = [AS.memory_profile(232, N)["code_basin"] for N in (8, 10, 12, 14)]
    assert all(a > b for a, b in zip(fracs, fracs[1:]))
    ratios = [b / a for a, b in zip(fracs, fracs[1:])]
    assert all(0.55 < r < 0.65 for r in ratios), ratios      # ~ 0.777^2 per 2 sites


# --- R24: the eight rules of R23 --------------------------------------------

R23_RULES = (28, 29, 70, 71, 157, 199, 73, 109)


@pytest.mark.parametrize("rule", R23_RULES)
@pytest.mark.parametrize("N", (4, 5))
def test_the_R23_rules_have_a_single_full_matrix_block(rule, N):
    """dim A = D^2 exactly: one block, m_k = 1, nothing erased asymptotically."""
    z = AS.decompose(rule, N)
    assert z.n_blocks == 1
    assert (z.blocks[0].n, z.blocks[0].m) == (z.dim_asym, 1)
    assert z.dim_alg == z.dim_asym ** 2


@pytest.mark.parametrize("rule", R23_RULES)
@pytest.mark.parametrize("N", (4, 5))
def test_the_dissipation_is_transient_only(rule, N):
    """Exactly one Kraus operator survives on the recurrent space, and it is an
    isometry there -- so the asymptotic evolution is closed."""
    W = AS.code_space(rule, N)
    A, err, others = AS.asymptotic_unitary(rule, N, W=W)
    assert others == 0, (rule, N, others)
    assert err < 1e-12, (rule, N, err)


@pytest.mark.parametrize("rule", R23_RULES)
@pytest.mark.parametrize("N", (4, 5))
def test_the_code_space_is_basis_spanned(rule, N):
    spanned, words = AS.basis_spanned(rule, N)
    assert spanned
    assert len(words) == AS.decompose(rule, N).dim_asym


@pytest.mark.parametrize("rule", R23_RULES)
def test_no_R23_rule_can_detect_a_single_pauli_error(rule):
    """Basis-spanned => distance one, and no subcode helps."""
    assert AS.code_distance(rule, 4) == 0


@pytest.mark.parametrize("rule,N,dims", (
    (28, 4, (25, 45)), (70, 4, (32, 45)), (29, 4, (3, 13)), (71, 4, (6, 13)),
    (157, 4, (3, 13)), (199, 4, (5, 13)), (73, 4, (19, 35)), (109, 4, (9, 31)),
))
def test_the_algebra_dimensions_R24_quotes(rule, N, dims):
    z = AS.algebra_report(rule, N)
    assert (z["dim_C"], z["dim_F"]) == dims
    assert z["C_inside_F"] < 1e-10


# --- R24: the corrected chain  n_wcc <= dim(F /\ D) <= n_rec ----------------

@pytest.mark.parametrize("rule,N", (
    (29, 4), (44, 4), (104, 4), (203, 4), (217, 4), (233, 4),
    (36, 5), (44, 5), (100, 5), (203, 5), (73, 5), (28, 5),
))
def test_the_conserved_diagonal_count_is_squeezed(rule, N):
    """R22 Prop. 4 holds for the MONITORED chain; coherently only <= survives."""
    from qca_fragmentation.graph import wcc
    d = 1 << N
    t = rules.wolfram_to_tuple(rule)
    cd = AS.diagonal_dimension(AS.commutant_basis(rule, N), d)
    fd = AS.diagonal_dimension(AS.conserved_basis(rule, N), d)
    g = scc.analyze(rule, N, "obc0", t, detect_ergodic=False)
    w = wcc.weak_components(rule, N, "obc0", t)
    assert cd == w.n_wcc, (rule, N)
    assert cd <= fd <= g.n_recurrent, (rule, N, cd, fd, g.n_recurrent)


def test_rule_29_loses_a_terminal_class_when_the_monitoring_is_switched_off():
    """The fair coin of R23 sec.5 is not a conserved quantity of the channel."""
    d = 1 << 4
    fd = AS.diagonal_dimension(AS.conserved_basis(29, 4), d)
    g = scc.analyze(29, 4, "obc0", rules.wolfram_to_tuple(29), detect_ergodic=False)
    assert g.n_recurrent == 3
    assert fd == 2


def test_the_gap_of_W36_is_real_but_that_of_W203_is_not():
    d5 = 1 << 5
    assert AS.diagonal_dimension(AS.conserved_basis(36, 5), d5) == 9      # > n_wcc = 7
    assert AS.diagonal_dimension(AS.commutant_basis(36, 5), d5) == 7
    assert AS.diagonal_dimension(AS.conserved_basis(203, 5), d5) == 1     # = n_wcc
    assert AS.diagonal_dimension(AS.commutant_basis(203, 5), d5) == 1


# --- R24 sec.6: the dissipative pair is the unitary pair, one letter changed --

def test_the_dissipative_rules_are_one_letter_mutations():
    assert rules.wolfram_to_tuple(201) == ("V", "I", "I", "I")
    assert rules.wolfram_to_tuple(73) == ("V", "I", "I", "D")
    assert rules.wolfram_to_tuple(108) == ("I", "I", "I", "V")
    assert rules.wolfram_to_tuple(109) == ("E", "I", "I", "V")
    assert rules.is_unitary(rules.wolfram_to_tuple(201))
    assert rules.is_unitary(rules.wolfram_to_tuple(108))
    assert not rules.is_unitary(rules.wolfram_to_tuple(73))
    assert not rules.is_unitary(rules.wolfram_to_tuple(109))


@pytest.mark.parametrize("rule", (73, 109))
@pytest.mark.parametrize("N", (4, 5, 6, 7, 8, 9))
def test_the_reset_is_invisible_on_the_constrained_space(rule, N):
    """R24 Prop. 3.  Agreement is EXACT, not to machine precision."""
    z = AS.parent_agreement(rule, N)
    assert z["difference"] == 0.0, (rule, N, z)
    assert z["escaped"] == 0
    assert z["kraus_labels"] == 1          # only the no-jump branch occurs
    assert z["unitarity"] < 1e-14


@pytest.mark.parametrize("N,dim", ((4, 8), (6, 21), (8, 55), (10, 144)))
def test_the_constrained_space_is_fibonacci(N, dim):
    assert len(AS.constrained_states(N, 1)) == dim
    assert len(AS.constrained_states(N, 0)) == dim


@pytest.mark.parametrize("rule", (73, 109))
def test_the_protected_space_grows_at_the_tribonacci_constant(rule):
    """R24 sec.2.3: D obeys a(n) = a(n-1) + a(n-2) + a(n-3)."""
    rows = AS.code_growth(rule, range(4, 15))
    D = [r["D"] for r in rows]
    for i in range(3, len(D)):
        assert D[i] == D[i - 1] + D[i - 2] + D[i - 3], (rule, i, D)
    assert D[-1] / D[-2] == pytest.approx(1.8393, abs=1e-3)


@pytest.mark.parametrize("rule", (73, 109))
def test_these_two_are_fragmented_in_both_senses_at_every_size(rule):
    for r in AS.code_growth(rule, range(4, 16)):
        assert r["n_wcc"] == r["n_rec"], (rule, r)


def test_the_largest_enclosure_is_fibonacci_and_the_total_is_not():
    """d_max grows at phi, D at the tribonacci constant -- the difference is the
    cross-enclosure coherence."""
    rows = AS.code_growth(73, range(6, 15))
    dm = [r["d_max"] for r in rows]
    for i in range(2, len(dm)):
        assert dm[i] == dm[i - 1] + dm[i - 2], dm
    assert rows[-1]["D"] > rows[-1]["d_max"] * 5


@pytest.mark.parametrize("N,sizes", ((5, [13]), (7, [34]), (9, [89]), (11, [233])))
def test_the_constrained_space_is_irreducible_for_W73(N, sizes):
    z = AS.constrained_decomposition(73, N)
    assert z["sizes"] == sizes
    assert z["kraus_labels"] == 1 and z["escaped"] == 0
    assert z["transient_inside"] == 0


@pytest.mark.parametrize("N,sizes", (
    (5, [5, 3, 3, 2]), (7, [13, 8, 8, 5]), (9, [34, 21, 21, 13]),
    (11, [89, 55, 55, 34]),
))
def test_W109_splits_its_constrained_space_four_ways(N, sizes):
    """And NOT because the obc0 boundary lets the reset fire -- it never fires."""
    z = AS.constrained_decomposition(109, N)
    assert z["sizes"] == sizes
    assert z["kraus_labels"] == 1 and z["escaped"] == 0
    assert z["transient_inside"] == 0
    assert sum(sizes) == z["dim"]


@pytest.mark.parametrize("rule,parent,forbid", ((73, 201, 1), (109, 108, 0)))
@pytest.mark.parametrize("N", (5, 7, 9))
def test_the_split_belongs_to_the_parent(rule, parent, forbid, N):
    a = AS.constrained_decomposition(rule, N, forbid=forbid)
    b = AS.constrained_decomposition(parent, N, forbid=forbid)
    assert a["sizes"] == b["sizes"]


# --- R24 sec.2.3: three exponents per rule ----------------------------------

#: D(n) = sum_i c_i D(n-1-i), verified term by term over N = 4..18 (obc0).
D_RECURRENCES = {
    73: (1, 1, 1), 109: (1, 1, 1),          # tribonacci, root 1.839287
    28: (1, 1, 1, -2), 70: (1, 1, 1, -2),   # root 1.521380
    157: (0, 1, 2), 199: (0, 1, 2),         # same root, different recurrence
    29: (0, 1, 2), 71: (0, 1, 2),
}


@pytest.mark.parametrize("rule,coeffs", sorted(D_RECURRENCES.items()))
def test_the_protected_dimension_obeys_an_exact_integer_recurrence(rule, coeffs):
    D = [r["D"] for r in AS.code_growth(rule, range(4, 15))]
    k = len(coeffs)
    for n in range(k, len(D)):
        assert D[n] == sum(c * D[n - 1 - i] for i, c in enumerate(coeffs)), (rule, n, D)


@pytest.mark.parametrize("rules_,expected", (
    ((73, 109), 1.839287),
    ((28, 70, 157, 199, 29, 71), 1.521380),
))
def test_the_dominant_roots_are_what_R24_quotes(rules_, expected):
    for rule in rules_:
        c = D_RECURRENCES[rule]
        root = max(abs(z) for z in np.roots([1] + [-x for x in c]))
        assert root == pytest.approx(expected, abs=1e-6), rule


def test_the_four_short_rules_share_one_D_series():
    """W29 and W71 have fewer weak components than enclosures, yet the same D:
    the multistability repartitions the recurrent space, it does not shrink it."""
    series = {tuple(r["D"] for r in AS.code_growth(rule, range(4, 13)))
              for rule in (157, 199, 29, 71)}
    assert len(series) == 1
    assert {tuple(r["D"] for r in AS.code_growth(rule, range(4, 13)))
            for rule in (28, 70)} != series


@pytest.mark.parametrize("rule", (73, 109, 28, 157))
def test_the_whole_space_outgrows_its_largest_enclosure(rule):
    rows = AS.code_growth(rule, range(6, 15))
    a = np.log2(rows[-1]["D"] / rows[-4]["D"]) / 3
    b = np.log2(rows[-1]["d_max"] / rows[-4]["d_max"]) / 3
    assert a > b + 0.05, (rule, a, b)


# --- R24 sec.6.1: the parent identity is general ----------------------------

def test_the_four_parents_are_the_single_V_rules():
    words = {p: "".join(rules.wolfram_to_tuple(p))
             for p in set(AS.COHERENT_PARENT.values())}
    assert words == {201: "VIII", 108: "IIIV", 156: "IIVI", 198: "IVII"}
    for p, w in words.items():
        assert w.count("V") == 1
        assert rules.is_unitary(rules.wolfram_to_tuple(p))


@pytest.mark.parametrize("rule", (28, 70, 157, 199, 29, 71, 73, 109))
@pytest.mark.parametrize("N", (7, 9))
def test_every_enclosure_is_a_krylov_sector_of_the_parent(rule, N):
    z = AS.parent_on_every_enclosure(rule, N)
    assert z["difference"] == 0.0, z
    assert z["escaped"] == 0
    assert z["kraus_labels"] == 1
    assert z["unitarity"] < 1e-14
    assert z["n_enclosures"] > 0


@pytest.mark.parametrize("rule", (28, 70, 157, 199, 29, 71))
@pytest.mark.parametrize("N", (7, 9))
def test_the_six_carry_a_bare_product_of_hadamards(rule, N):
    """Asymmetric projector => subcube enclosures => a product unitary, so R23's
    period 2 and zero entanglement follow with no further measurement."""
    z = AS.parent_on_every_enclosure(rule, N)
    assert z["product_hadamard"] < 1e-14, z


@pytest.mark.parametrize("rule", (73, 109))
def test_the_symmetric_pair_is_not_a_product(rule):
    z = AS.parent_on_every_enclosure(rule, 9)
    assert z["product_hadamard"] > 1e-6 or z["product_hadamard"] == np.inf
