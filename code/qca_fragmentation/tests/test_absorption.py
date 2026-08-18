"""Tests for the R22 absorption diagnostic."""
import numpy as np
import pytest

from qca_fragmentation.scaling import absorption as AB
from qca_fragmentation.graph import scc
from qca_fragmentation.core import rules


@pytest.mark.parametrize("rule", (203, 36, 73, 150))
@pytest.mark.parametrize("N", (6, 8))
def test_the_transition_matrix_is_a_markov_matrix(rule, N):
    P = AB.transition_matrix(rule, N)
    assert np.allclose(P.sum(axis=0), 1.0)
    assert (P >= 0).all()


@pytest.mark.parametrize("rule", (108, 150, 105, 156))
@pytest.mark.parametrize("N", (6, 8))
def test_unitary_rules_give_a_doubly_stochastic_matrix(rule, N):
    """R22 Prop. 1: this is what kills the transient part."""
    assert rules.is_unitary(rules.wolfram_to_tuple(rule))
    P = AB.transition_matrix(rule, N)
    assert np.allclose(P.sum(axis=0), 1.0)
    assert np.allclose(P.sum(axis=1), 1.0)


@pytest.mark.parametrize("rule", (203, 36, 44, 233))
@pytest.mark.parametrize("N", (6, 8))
def test_dissipative_rules_are_not_doubly_stochastic(rule, N):
    P = AB.transition_matrix(rule, N)
    assert not np.allclose(P.sum(axis=1), 1.0)


@pytest.mark.parametrize("rule", (203, 36, 73, 109))
@pytest.mark.parametrize("N", (6, 8))
def test_absorption_probabilities_sum_to_one(rule, N):
    phi, term = AB.absorption(rule, N)
    assert np.allclose(phi.sum(axis=1), 1.0)
    assert len(term) > 0


@pytest.mark.parametrize("rule", (203, 36, 73, 109, 217, 219))
@pytest.mark.parametrize("N", (6, 8, 10))
def test_the_support_of_phi_matches_the_engines_terminal_count(rule, N):
    """The weighted chain must find exactly the terminal classes scc.py finds."""
    phi, term = AB.absorption(rule, N)
    g = scc.analyze(rule, N, "obc0", rules.wolfram_to_tuple(rule),
                    detect_ergodic=False)
    assert len(term) == g.n_recurrent


@pytest.mark.parametrize("rule,N", ((73, 10), (109, 10)))
def test_idempotent_exactly_when_every_sector_has_one_attractor(rule, N):
    """R22 Prop. 2, the forward direction: n_rec == n_wcc => certainty."""
    d = AB.summary(rule, N)
    assert d["n_wcc"] == d["n_terminal"]
    assert d["idempotent"]
    assert d["certain_fraction"] == pytest.approx(1.0)
    assert d["eff_attractors_mean"] == pytest.approx(1.0)


@pytest.mark.parametrize("rule,N", ((203, 10), (203, 12), (36, 10)))
def test_not_idempotent_when_one_sector_holds_many_attractors(rule, N):
    """R22 Prop. 2, the converse: the noise, not the initial state, decides."""
    d = AB.summary(rule, N)
    assert d["n_terminal"] > d["n_wcc"]
    assert not d["idempotent"]
    assert d["certain_fraction"] < 1.0
    assert d["eff_attractors_mean"] > 1.0


def test_the_certainty_of_w203_falls_with_N():
    """The gap widens: fewer states have a determined fate as N grows."""
    a = AB.summary(203, 10)
    b = AB.summary(203, 12)
    assert b["certain_fraction"] < a["certain_fraction"]
    assert b["eff_attractors_max"] > a["eff_attractors_max"]


def test_the_numbers_R22_quotes_are_the_numbers_the_module_returns():
    quoted = {
        (73, 10):  (41, 41, 1.000),
        (109, 10): (54, 54, 1.000),
        (36, 10):  (10, 60, 0.725),
        (203, 10): (1,  28, 0.461),
        (203, 12): (1,  60, 0.391),
    }
    for (rule, N), (n_wcc, n_term, certain) in quoted.items():
        d = AB.summary(rule, N)
        assert d["n_wcc"] == n_wcc, (rule, N)
        assert d["n_terminal"] == n_term, (rule, N)
        assert d["certain_fraction"] == pytest.approx(certain, abs=5e-4), (rule, N)


def test_strong_symmetry_helper_agrees_with_the_summary():
    assert AB.strong_symmetry_holds(73, 10)
    assert not AB.strong_symmetry_holds(203, 10)


# --- R22 Prop. 1 / Prop. 2: when can the two counts separate at all? --------

@pytest.mark.parametrize("rule", (0, 4, 12, 76, 200, 240, 255))
@pytest.mark.parametrize("N", (6, 8, 10))
def test_deterministic_rules_never_show_the_gap(rule, N):
    """No V in the word => out-degree 1 => one attractor per weak component."""
    from qca_fragmentation import results_io
    assert "V" not in rules.wolfram_to_tuple(rule)
    g = scc.analyze(rule, N, "obc0", rules.wolfram_to_tuple(rule),
                    detect_ergodic=False)
    w = results_io.load_wcc_results(rule, "obc0").get(N)
    if w is not None:
        assert g.n_recurrent == w["n_wcc"], (rule, N)


def test_every_rule_showing_the_gap_is_branching_and_non_unital():
    """R22: the gap needs both a V and a reset.  The eight are minimal in V."""
    gap = (36, 44, 100, 104, 203, 217, 219, 233)
    for r in gap:
        t = rules.wolfram_to_tuple(r)
        assert sum(s == "V" for s in t) == 1, r
        assert any(s in ("D", "E") for s in t), r


# --- R22 Thm. 1: the idempotents cannot outnumber the weak components -------

@pytest.mark.parametrize("rule,N,expected", ((73, 8, 19), (36, 8, 9), (203, 8, 0)))
def test_the_idempotent_count_is_what_R22_quotes(rule, N, expected):
    assert AB.n_idempotent(rule, N) == expected


@pytest.mark.parametrize("rule", (36, 203, 219, 73))
@pytest.mark.parametrize("N", (6, 8))
def test_idempotents_never_exceed_the_weak_component_count(rule, N):
    from qca_fragmentation import results_io
    w = results_io.load_wcc_results(rule, "obc0").get(N)
    if w is not None:
        assert AB.n_idempotent(rule, N) <= w["n_wcc"], (rule, N)


# --- R22 section 3.5: how many bits survive ---------------------------------

@pytest.mark.parametrize("rule,N", ((73, 10), (109, 10)))
def test_a_strong_symmetry_makes_the_memory_noiseless(rule, N):
    d = AB.retained_information(rule, N)
    assert d["H_T_given_X"] == pytest.approx(0.0, abs=1e-12)
    assert d["noiseless"]
    assert d["I_XT"] == pytest.approx(d["H_T"])


@pytest.mark.parametrize("rule,N", ((203, 10), (219, 10), (36, 10)))
def test_without_it_the_branching_destroys_memory(rule, N):
    d = AB.retained_information(rule, N)
    assert d["H_T_given_X"] > 0.0
    assert not d["noiseless"]
    assert d["I_XT"] < d["H_T"]


@pytest.mark.parametrize("rule", (73, 109, 203, 219))
def test_the_retained_information_is_extensive_in_both_regimes(rule):
    """The point of R22 sec 3.5: case B remembers just as many bits, less surely."""
    vals = [AB.retained_information(rule, N)["I_XT"] for N in (8, 10, 12)]
    assert vals[0] < vals[1] < vals[2]
    d1, d2 = vals[1] - vals[0], vals[2] - vals[1]
    assert d1 == pytest.approx(d2, rel=0.05), (rule, vals)
    assert 0.7 < d1 < 1.0, (rule, d1)


def test_the_noise_term_itself_grows_with_N():
    prev = 0.0
    for N in (8, 10, 12):
        h = AB.retained_information(203, N)["H_T_given_X"]
        assert h > prev
        prev = h
