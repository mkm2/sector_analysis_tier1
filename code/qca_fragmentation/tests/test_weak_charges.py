"""Tier 1d diagonal strong/weak charge detector."""
import pytest

from qca_fragmentation.core.rules import wolfram_to_tuple
from qca_fragmentation.quantum.weak_charges import analyze_charges


def _c(rule, N=6, bc="pbc", r=3):
    return analyze_charges(rule, N, bc, wolfram_to_tuple(rule), r=r)


def test_strong_charges_of_frozen_rules():
    # rule 200 (D,I,I,I) freezes local structure -> local-density strong charges;
    # they must be constant on WCCs and hence invisible as weak.
    r200 = _c(200, r=2)
    assert r200.n_strong == 2 and r200.n_weak == 0
    r232 = _c(232, r=2)
    assert r232.n_strong == 1 and r232.n_weak == 0


def test_no_vacuous_weak_charges_when_coherence_decays():
    # rule 0 collapses everything to |0...0>; off-diagonal coherences die in one
    # step, so there is no protected coherence to grade.
    r0 = _c(0, r=2)
    assert r0.n_weak == 0 and r0.n_strong == 0


def test_wall_hadamard_core_weak_charge_grades_coherence():
    # The protected coherence of the wall-Hadamard core (22, 23, 146, 151, 178)
    # is graded by a weak charge invisible to the monitored dynamics.
    for rule in (22, 146, 151):
        res = _c(rule, r=3)
        assert res.n_strong == 0
        assert res.n_weak >= 1
        assert res.weak_grades_coherence is True
        assert len(res.d_values_on_coherence) >= 2


def test_unitary_rules_have_no_diagonal_charges():
    # unitary rules carry a single Kraus label; the detector short-circuits.
    res = _c(150, r=2)
    assert res.n_strong == 0 and res.n_weak == 0
