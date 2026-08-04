"""R12: the spy plots of the C156 transition matrix."""

import numpy as np
import pytest

from qca_fragmentation.graph import wcc
from qca_fragmentation.viz import spy


def _fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


@pytest.mark.parametrize("N", [6, 8, 10])
def test_the_blocks_are_the_tier1e_sectors(N):
    """The picture must partition the basis exactly as the Tier-1e pass does, or
    it is illustrating something other than the reports."""
    blocks = spy.wcc_blocks(156, N, "obc0")
    r = wcc.weak_components(156, N, "obc0")
    assert sorted((len(b) for b in blocks), reverse=True) == r.sizes_wcc
    flat = sorted(v for b in blocks for v in b)
    assert flat == list(range(1 << N))          # a partition, no repeats


@pytest.mark.parametrize("N", [6, 7, 8, 9, 10, 11, 12])
def test_no_nonzero_crosses_a_block(N):
    """R12's central claim, and the one the sorted panel makes visually."""
    st = spy.check_block_diagonal(156, N, "obc0")
    assert st["off_block_nonzeros"] == 0
    assert st["nnz"] > 0


@pytest.mark.parametrize("N", [6, 7, 8, 9, 10, 11, 12])
def test_the_block_count_is_fibonacci(N):
    """n_wcc = F_{N+2} exactly at obc0, which is where R2's phi comes from."""
    st = spy.check_block_diagonal(156, N, "obc0")
    assert st["n_blocks"] == _fib(N + 2)


def test_the_permutation_is_a_permutation():
    N = 9
    blocks = spy.wcc_blocks(156, N, "obc0")
    perm, starts = spy.sorted_permutation(blocks)
    assert sorted(perm.tolist()) == list(range(1 << N))
    assert len(starts) == len(blocks)
    assert starts[0] == 0
    # blocks laid out largest first, and the starts match the sizes
    sizes = [len(b) for b in blocks]
    assert sizes == sorted(sizes, reverse=True)
    assert starts == list(np.cumsum([0] + sizes[:-1]))


def test_permuting_really_does_concentrate_the_mass():
    """The sorted matrix is block-diagonal, so every nonzero has |row-col| less
    than the largest block; in the computational basis it does not."""
    N = 9
    rows, cols = spy.transition_pattern(156, N, "obc0")
    blocks = spy.wcc_blocks(156, N, "obc0")
    perm, _ = spy.sorted_permutation(blocks)
    pos = np.empty(1 << N, dtype=int)
    pos[perm] = np.arange(1 << N)
    d_max = max(len(b) for b in blocks)
    assert np.abs(pos[rows] - pos[cols]).max() < d_max
    assert np.abs(rows - cols).max() > d_max     # unsorted spreads much wider


def test_the_pattern_is_the_support_of_the_successor_map():
    N = 8
    rows, cols = spy.transition_pattern(156, N, "obc0")
    succ = wcc.make_succ(156, N, "obc0")
    seen = set(zip(cols.tolist(), rows.tolist()))
    want = {(x, y) for x in range(1 << N) for y in succ(x)}
    assert seen == want


def test_unitary_rule_has_no_frozen_column():
    """156 is unitary, so every column has at least one nonzero -- no state maps
    to nothing.  A dropped edge would show up here before it reached a figure."""
    N = 9
    _, cols = spy.transition_pattern(156, N, "obc0")
    assert len(set(cols.tolist())) == 1 << N


def test_figures_and_table_write(tmp_path):
    out = str(tmp_path / "spy.pdf")
    st = spy.fig_spy(156, 7, "obc0", out=out)
    assert st["off_block_nonzeros"] == 0
    import os
    assert os.path.exists(out) and os.path.getsize(out) > 0
    spy.fig_spy_zoom(156, 7, "obc0", span=40,
                     out=str(tmp_path / "zoom.pdf"))
    t = spy.block_size_table(156, Ns=(6, 7), bc="obc0",
                             out=str(tmp_path / "tab.tex"))
    text = open(t).read()
    assert text.startswith("\\begin{tabular}") and "\\end{tabular}" in text


def test_the_nonzero_count_obeys_an_exact_silver_ratio_recurrence():
    """R12.  nnz(N) = 2a_{N-1} + 2a_{N-3} + a_{N-4}, characteristic polynomial
    (x^2-2x-1)(x^2+1): base 1+sqrt(2) with a period-4 modulation from the +-i.
    A third exact constant for the same rule, alongside phi for the block count
    and 4^(1/5) for D_max."""
    from qca_fragmentation.scaling.fits import find_integer_recurrence
    ys = [spy.check_block_diagonal(156, N, "obc0")["nnz"]
          for N in range(6, 15)]
    assert ys[:5] == [204, 493, 1189, 2870, 6930]
    rec = find_integer_recurrence(list(ys))
    assert rec["ok"]
    assert rec["coeffs"] == [2, 0, 2, 1]
    assert rec["base"] == pytest.approx(1 + 2 ** 0.5, abs=1e-9)
    # and the recurrence really does reproduce the series
    for i in range(4, len(ys)):
        assert ys[i] == 2 * ys[i - 1] + 2 * ys[i - 3] + ys[i - 4]


def test_the_three_constants_are_different_questions_about_one_matrix():
    """phi counts blocks, 1+sqrt2 counts nonzeros; they must not be conflated."""
    from qca_fragmentation.scaling.fits import find_integer_recurrence
    Ns = list(range(6, 15))
    blocks = [spy.check_block_diagonal(156, N, "obc0")["n_blocks"] for N in Ns]
    nnz = [spy.check_block_diagonal(156, N, "obc0")["nnz"] for N in Ns]
    rb = find_integer_recurrence(list(blocks))
    rn = find_integer_recurrence(list(nnz))
    assert rb["ok"] and rn["ok"]
    assert rb["base"] == pytest.approx((1 + 5 ** 0.5) / 2, abs=1e-9)
    assert rn["base"] == pytest.approx(1 + 2 ** 0.5, abs=1e-9)
    assert abs(rb["base"] - rn["base"]) > 0.7


# --- Frobenius normal form (R12 sec.4) ---------------------------------------

@pytest.mark.parametrize("rule", spy.FROBENIUS_RULES)
@pytest.mark.parametrize("N", [7, 9])
def test_frobenius_form_is_upper_triangular(rule, N):
    """The defining property: no nonzero below the block diagonal."""
    st = spy.check_frobenius(rule, N, "obc0")
    assert st["nnz_below_diagonal"] == 0
    assert st["nnz_in_blocks"] + st["nnz_strictly_upper"] == st["nnz"]


@pytest.mark.parametrize("rule", spy.FROBENIUS_RULES)
def test_scc_blocks_partition_and_refine_the_wcc_blocks(rule):
    """SCCs refine weak components -- each strong component lies inside one weak
    one, which is what makes the triangular form a refinement of the diagonal
    one rather than a different decomposition."""
    N = 9
    sb, _ = spy.scc_blocks(rule, N, "obc0")
    wb = spy.wcc_blocks(rule, N, "obc0")
    assert sorted(v for b in sb for v in b) == list(range(1 << N))
    owner = spy.block_of(wb, 1 << N)
    for b in sb:
        assert len({owner[v] for v in b}) == 1, (rule, b[:4])


@pytest.mark.parametrize("rule", [156, 150])
def test_a_unitary_rule_has_no_drainage(rule):
    """R9's A2, drawn: |U|^2 doubly stochastic => no transient states => weak and
    strong components coincide and the Frobenius form IS the diagonal form."""
    from qca_fragmentation.core import rules as R
    assert R.is_unitary(R.wolfram_to_tuple(rule))
    for N in (7, 8, 9):
        st = spy.check_frobenius(rule, N, "obc0")
        assert st["n_scc"] == st["n_wcc"] == st["n_terminal"]
        assert st["recurrent"] == st["dim"]
        assert st["nnz_strictly_upper"] == 0


@pytest.mark.parametrize("rule", [157, 109, 158])
def test_a_dissipative_rule_has_strictly_more_sccs_than_wccs(rule):
    for N in (8, 9, 10):
        st = spy.check_frobenius(rule, N, "obc0")
        assert st["n_scc"] > st["n_wcc"]
        assert st["nnz_strictly_upper"] > 0
        assert st["recurrent"] < st["dim"]


def test_150_and_158_have_the_identical_sector_partition():
    """R12 sec.4.2, the sharpest case in the report: same sectors -- the same
    SETS of states, not merely the same sizes -- yet one is unitary with every
    state recurrent and the other retains 31 of 1024.  No sector observable can
    separate them; the Frobenius form does at a glance."""
    for N in (8, 10, 12):
        a = {frozenset(b) for b in spy.wcc_blocks(150, N, "obc0")}
        b = {frozenset(x) for x in spy.wcc_blocks(158, N, "obc0")}
        assert a == b, N
    s150 = spy.check_frobenius(150, 10, "obc0")
    s158 = spy.check_frobenius(158, 10, "obc0")
    assert s150["n_wcc"] == s158["n_wcc"] == 6
    assert (s150["n_scc"], s158["n_scc"]) == (6, 936)
    assert (s150["recurrent"], s158["recurrent"]) == (1024, 31)


def test_terminal_scc_count_equals_wcc_count_for_these_five():
    """None of the five is among R9 sec.7.3's twelve multi-attractor rules, so
    sectors and attractors are in bijection here."""
    for rule in spy.FROBENIUS_RULES:
        st = spy.check_frobenius(rule, 10, "obc0")
        assert st["n_terminal"] == st["n_wcc"], rule


def test_frobenius_figure_writes(tmp_path):
    out = str(tmp_path / "frob.pdf")
    st = spy.fig_frobenius(158, 8, "obc0", out=out)
    import os
    assert os.path.exists(out) and os.path.getsize(out) > 0
    assert st["nnz_below_diagonal"] == 0
