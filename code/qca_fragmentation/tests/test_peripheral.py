"""Tier 1b restricted-superoperator / attractor-classification tests."""
import numpy as np
import pytest

from qca_fragmentation.core import rules
from qca_fragmentation.graph import scc
from qca_fragmentation.quantum import peripheral as pp


def _R(rule, N, bc):
    return scc.recurrent_classes(rule, N, bc, rules.wolfram_to_tuple(rule))


@pytest.mark.parametrize("bc", ["pbc", "obc0"])
@pytest.mark.parametrize("N", [3, 4])
def test_restricted_superop_matches_full_block(bc, N):
    for rule in (22, 28, 0, 50, 178):
        t = rules.wolfram_to_tuple(rule)
        S = pp.full_channel_superoperator(N, t, bc)
        dim = 1 << N
        for R in _R(rule, N, bc):
            Rs = sorted(R); k = len(Rs); idx = {x: i for i, x in enumerate(Rs)}
            PhiR = pp.restricted_superoperator(Rs, N, t, bc)
            block = np.zeros((k * k, k * k))
            for y in Rs:
                for yp in Rs:
                    rr = idx[y] * k + idx[yp]
                    for x in Rs:
                        for xp in Rs:
                            block[rr, idx[x] * k + idx[xp]] = S[y * dim + yp, x * dim + xp]
            assert np.allclose(PhiR, block, atol=1e-9), (rule, N, bc, Rs)


def test_point_attractors_are_mixing():
    # rule 22 (I,V,V,D): 3 classical pointer states (vacuum + 2 Neel)
    t = rules.wolfram_to_tuple(22)
    R = _R(22, 8, "pbc")
    assert len(R) == 3
    for cls in R:
        at = pp.classify_attractor(cls, 8, t, "pbc")
        assert at.size == 1 and at.kind == "mixing" and at.d_k == 1


def test_rule28_has_coherent_attractors():
    # rule 28 (I,I,V,D): multi-state recurrent classes carry protected coherence
    t = rules.wolfram_to_tuple(28)
    kinds = {pp.classify_attractor(R, 8, t, "pbc").kind for R in _R(28, 8, "pbc")}
    assert "coherent" in kinds


def test_cesaro_rank_and_coherence_gap():
    # Pure decay (rule 0): fixed space is 1-dim, no coherence gap.
    assert pp.cesaro_rank(4, rules.wolfram_to_tuple(0), "pbc") == 1
    # Weakly dissipative rule 22: full fixed-point space >> classical pointer
    # count -> large protected-coherence gap (graph under-resolves; context 5).
    t22 = rules.wolfram_to_tuple(22)
    cr = pp.cesaro_rank(4, t22, "pbc")
    classical = sum(pp.classify_attractor(R, 4, t22, "pbc").d_k ** 2
                    for R in _R(22, 4, "pbc"))
    assert cr > classical  # protected coherence present


def test_geometric_vs_algebraic_multiplicity():
    # geometric multiplicity of a defective 2x2 Jordan block at lambda=1 is 1
    J = np.array([[1.0, 1.0], [0.0, 1.0]])
    assert pp.geometric_multiplicity(J, 1.0) == 1
