"""Is the channel attractor basis-spanned?  (Tier 1b structural test)

Companion to test_peripheral.py::test_graph_faithfulness_two_regimes, which
tests the WEAKER leakage criterion.  The cases here pin the difference: rule 27
leaks nothing yet is still undescribable by any state-space graph.
"""
import math

import numpy as np
import pytest
import scipy.sparse as sp

from qca_fragmentation.core import rules
from qca_fragmentation.quantum import attractor_structure as ast
from qca_fragmentation.quantum import peripheral as pp


@pytest.mark.parametrize("rule", [0, 22, 27, 28, 200])
@pytest.mark.parametrize("bc", ["pbc", "obc0"])
def test_site_kraus_is_a_channel(rule, bc):
    """sum_mu A_mu^dag A_mu = 1 for every site -- the sparse Kraus pair is CPTP."""
    N, t = 4, rules.wolfram_to_tuple(rule)
    for i in range(N):
        A0, A1 = ast.site_kraus_sparse(i, N, t, bc)
        S = (A0.getH() @ A0).toarray()
        if A1 is not None:
            S = S + (A1.getH() @ A1).toarray()
        assert np.allclose(S, np.eye(1 << N), atol=1e-12), (rule, bc, i)


def test_cycle_agrees_with_dense_superoperator():
    """The sparse per-site cycle reproduces the dense 4^N superoperator."""
    N, bc = 3, "pbc"
    for rule in (22, 28, 200):
        t = rules.wolfram_to_tuple(rule)
        dim = 1 << N
        rng = np.random.default_rng(0)
        A = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        rho = A @ A.conj().T
        rho /= np.trace(rho).real

        got = ast.apply_cycle(rho, ast.kraus_list(N, t, bc))
        want = (pp.full_channel_superoperator(N, t, bc)
                @ rho.reshape(-1)).reshape(dim, dim)
        assert np.allclose(got, want, atol=1e-9), rule


def test_rule27_is_coherent_but_does_not_leak():
    """The two failure modes are genuinely different.

    Rule 27 (VEVD) passes the leakage test -- nothing sits on graph-transient
    states -- yet its attractor is a single pure superposition, not the nine
    basis states the graph reports.  Only the structural test sees this.
    """
    t = rules.wolfram_to_tuple(27)
    assert "".join(t) == "VEVD"

    st = ast.attractor_structure(4, t, "pbc")
    assert st["converged"]
    assert st["dim_att"] == 1              # ONE stationary state...
    assert st["basis_in_att"] == 0         # ...and it is not a basis state
    assert st["coh_dirs"] == 1
    assert not ast.basis_spanned(4, t, "pbc")

    d = pp.graph_faithfulness(4, t, "pbc")
    assert d["n_recurrent_states"] == 9    # the graph says nine
    assert abs(d["leak_flow"]) < 1e-9      # yet the weak test sees nothing


def test_classical_rule_is_basis_spanned():
    # Rule 200 (DIII) has no Hadamard: Kraus operators are permutations composed
    # with resets, so the attractor is spanned by basis states by construction.
    st = ast.attractor_structure(4, rules.wolfram_to_tuple(200), "pbc")
    assert st["converged"] and st["coh_dirs"] == 0
    assert st["dim_att"] == st["basis_in_att"] == 10


def test_hadamard_free_rules_are_never_coherent():
    """The census' structural claim, on a sample: no V => basis-spanned."""
    checked = 0
    for r in range(256):
        t = rules.wolfram_to_tuple(r)
        if rules.is_unitary(t) or "V" in t or r % 5:
            continue
        st = ast.attractor_structure(4, t, "pbc")
        assert st["converged"] and st["coh_dirs"] == 0, (r, "".join(t), st)
        checked += 1
    assert checked >= 8


def test_convergence_is_gap_driven_not_burnin_driven():
    """Rule 19 at N=6 reports a bogus 35-dim attractor at a short burn-in.

    With burn-in 600 the kept eigenvalues are still-decaying transients and there
    is no spectral gap; the doubling loop must run far past that to the true
    5-dimensional answer.
    """
    t = rules.wolfram_to_tuple(19)
    short = ast.attractor_structure(6, t, "pbc", burn0=600, avg=60, max_burn=600)
    assert not short["converged"]
    assert short["dim_att"] > 5            # transients miscounted as attractor

    full = ast.attractor_structure(6, t, "pbc")
    assert full["converged"] and full["gap"] > ast.GAP_MIN
    assert full["dim_att"] == 5
    assert full["burn"] > 600
