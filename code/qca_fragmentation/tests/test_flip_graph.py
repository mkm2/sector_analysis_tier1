"""
The flip reduction (graph/flip_graph.py) against the streamed engine.

Theorem: for a unitary rule the sector partition equals the connected components
of the single-flip graph x ~ x XOR 2^i over the sites where the Hadamard fires.
These tests are the validation cited by R8 sec.2.
"""

import pytest

from qca_fragmentation.core import rules
from qca_fragmentation.graph import scc, flip_graph as fg

UNITARY = [r for r in range(256) if rules.is_unitary(rules.wolfram_to_tuple(r))]


def test_sixteen_unitary_rules():
    assert UNITARY == [51, 54, 57, 60, 99, 102, 105, 108,
                       147, 150, 153, 156, 195, 198, 201, 204]


@pytest.mark.parametrize("rule", UNITARY)
@pytest.mark.parametrize("bc", ["obc0", "pbc"])
def test_flip_reduction_matches_engine(rule, bc):
    """The reduction reproduces the engine's sector sizes exactly, both the
    pure-python union-find and the vectorised label propagation."""
    t = rules.wolfram_to_tuple(rule)
    for N in range(3, 10):
        eng = scc.analyze(rule, N, bc, t, detect_ergodic=False).sizes_recurrent
        assert fg.flip_components(N, t, bc) == eng, (rule, bc, N, "union-find")
        assert fg.flip_components_np(N, t, bc) == eng, (rule, bc, N, "numpy")


@pytest.mark.parametrize("bc", ["obc0", "pbc"])
def test_flip_reduction_larger_N_sample(bc):
    """A couple of larger lengths for the two rules R8 cares about."""
    for rule in (150, 105):
        t = rules.wolfram_to_tuple(rule)
        for N in (10, 11, 12):
            eng = scc.analyze(rule, N, bc, t,
                              detect_ergodic=False).sizes_recurrent
            assert fg.flip_components_np(N, t, bc) == eng, (rule, bc, N)


def test_chunking_and_inplace_paths_agree():
    """The chunked in-place pointer jumping must not depend on the chunk size."""
    t = rules.wolfram_to_tuple(150)
    for N in (7, 8, 9):
        ref = fg.flip_components_np(N, t, "obc0", chunk=1 << 20)
        for ch in (1, 3, 1 << 4, 1 << 6):
            assert fg.flip_components_np(N, t, "obc0", chunk=ch) == ref, (N, ch)


def test_v_mask_matches_symbol_table():
    """v_mask is the set of sites whose symbol is V under the current controls."""
    from qca_fragmentation.core.cycle import trigger_symbol
    for rule in UNITARY:
        t = rules.wolfram_to_tuple(rule)
        for bc in ("obc0", "pbc"):
            for N in (4, 5):
                for x in range(1 << N):
                    m = fg.v_mask(x, N, t, bc)
                    for i in range(N):
                        fired = bool(m >> i & 1)
                        assert fired == (trigger_symbol(x, i, N, t, bc) == "V")


def test_rejects_non_unitary_rule():
    t = rules.wolfram_to_tuple(22)          # has a reset symbol
    with pytest.raises(ValueError):
        fg.flip_components(4, t, "obc0")
    with pytest.raises(ValueError):
        fg.flip_components_np(4, t, "obc0")
