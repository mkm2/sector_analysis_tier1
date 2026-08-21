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
