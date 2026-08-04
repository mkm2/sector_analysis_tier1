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


# --- the nesting of the two normal forms (R12 sec.4, corrected 2026-08-04) ----

ALL_SPY_RULES = list(spy.FROBENIUS_RULES) + [r for r, _ in
                                             spy.MULTI_ATTRACTOR_RULES]


@pytest.mark.parametrize("rule", ALL_SPY_RULES)
def test_no_nonzero_ever_crosses_a_sector(rule):
    """The fact that makes the nesting possible, and the reason an apparent
    inter-sector transition in a Frobenius picture is always a layout artefact:
    an edge is a witness of weak connectivity, so it cannot join two sectors."""
    N = 9
    rows, cols = spy.transition_pattern(rule, N, "obc0")
    wb = spy.wcc_blocks(rule, N, "obc0")
    own = spy.block_of(wb, 1 << N)
    assert int(np.count_nonzero(own[rows] != own[cols])) == 0


@pytest.mark.parametrize("rule", ALL_SPY_RULES)
def test_the_frobenius_order_nests_inside_the_sector_blocks(rule):
    """Default ordering: each sector's states are CONTIGUOUS, so the third panel
    carries the second panel's blocks and merely refines them.  Without this the
    intra-sector drainage is drawn far from the diagonal and reads as flow
    between sectors -- which the test above shows is impossible."""
    N = 9
    n = 1 << N
    sb, _ = spy.scc_blocks(rule, N, "obc0")
    perm, _ = spy.sorted_permutation(sb)
    pos = np.empty(n, dtype=int)
    pos[perm] = np.arange(n)
    for b in spy.wcc_blocks(rule, N, "obc0"):
        idx = pos[np.asarray(b)]
        assert idx.max() - idx.min() + 1 == len(b), (rule, len(b))


@pytest.mark.parametrize("rule", ALL_SPY_RULES)
def test_both_orderings_are_triangular(rule):
    """Nesting is a choice among valid reverse-topological orders, not a
    weakening of one: the global sweep and the nested sweep are both exactly
    triangular."""
    N = 9
    rows, cols = spy.transition_pattern(rule, N, "obc0")
    for nest in (False, True):
        sb, _ = spy.scc_blocks(rule, N, "obc0", nest_in_wcc=nest)
        own = spy.block_of(sb, 1 << N)
        assert int(np.count_nonzero(own[rows] > own[cols])) == 0, (rule, nest)


def test_nesting_does_not_change_any_count():
    """Only the layout moves; every quantity in the tables is order-independent."""
    for rule in ALL_SPY_RULES:
        a, ta = spy.scc_blocks(rule, 9, "obc0", nest_in_wcc=False)
        b, tb = spy.scc_blocks(rule, 9, "obc0", nest_in_wcc=True)
        assert {frozenset(x) for x in a} == {frozenset(x) for x in b}
        assert sum(ta) == sum(tb)


def test_spy_module_exposes_every_figure_main_draws():
    """The __main__ guard once sat mid-module here too, so `python -m ...spy`
    NameError'd on FROBENIUS_RULES after writing only the first three figures."""
    import re
    src = open(spy.__file__).read()
    guard = src.index('if __name__ == "__main__"')
    assert guard > max(m.start() for m in re.finditer(r"^def ", src, re.M))
    for name in ("fig_spy", "fig_spy_zoom", "fig_frobenius",
                 "fig_recurrent_corner", "frobenius_table",
                 "multi_attractor_table"):
        assert hasattr(spy, name), name


@pytest.mark.parametrize("rule,cls", spy.MULTI_ATTRACTOR_RULES)
def test_the_multi_attractor_examples_have_a_sector_with_several_sinks(rule, cls):
    """One example per sector-count class, and each must actually exhibit the
    property it is there to illustrate."""
    from collections import Counter
    from qca_fragmentation.scaling import sectors as S
    st = spy.check_frobenius(rule, 10, "obc0")
    assert st["n_terminal"] > st["n_wcc"]
    sb, term = spy.scc_blocks(rule, 10, "obc0")
    wb = spy.wcc_blocks(rule, 10, "obc0")
    own = spy.block_of(wb, 1 << 10)
    per = Counter(own[b[0]] for b, t in zip(sb, term) if t)
    assert max(per.values()) > 1
    d = S.load("obc0")
    got = {p["rule"]: p for p in d["points"]}[rule]["n_wcc"]["cls"]
    assert got == cls, (rule, got, cls)


def test_29_and_157_have_the_identical_attractor_set():
    """The dual of the 150/158 pair: same attractors, different sectors.  157 has
    one per sector, 29 merges them, so an attractor-only observable cannot tell
    them apart just as a sector-only one cannot separate 150 from 158."""
    for N in (9, 10, 11):
        a, ta = spy.scc_blocks(29, N, "obc0")
        b, tb = spy.scc_blocks(157, N, "obc0")
        A = {frozenset(x) for x, t in zip(a, ta) if t}
        B = {frozenset(x) for x, t in zip(b, tb) if t}
        assert A == B, N
        n29 = len(spy.wcc_blocks(29, N, "obc0"))
        n157 = len(spy.wcc_blocks(157, N, "obc0"))
        assert n157 == len(B)          # 157: one attractor per sector
        assert n29 < n157              # 29: merged


@pytest.mark.parametrize("rule", ALL_SPY_RULES)
def test_each_sector_holds_its_sinks_at_the_front_of_its_own_range(rule):
    """What the per-sector gold shading depends on.  The nested sweep runs
    sinks-first WITHIN each sector, so a sector's terminal SCCs occupy the start
    of its range -- not the start of the matrix.  A single gold corner at the
    top-left would be wrong under this ordering, and was the bug that lost the
    marking entirely when the sector outlines were added."""
    N = 9
    n = 1 << N
    sb, terminal = spy.scc_blocks(rule, N, "obc0")
    perm, _ = spy.sorted_permutation(sb)
    pos = np.empty(n, dtype=int)
    pos[perm] = np.arange(n)
    wb = spy.wcc_blocks(rule, N, "obc0")
    wown = spy.block_of(wb, n)
    for i, sector in enumerate(wb):
        lo = int(pos[np.asarray(sector)].min())
        rec = sorted(pos[v] for b, t in zip(sb, terminal) if t
                     for v in b if wown[v] == i)
        if not rec:
            continue
        # contiguous, and starting exactly at the sector's first index
        assert rec == list(range(lo, lo + len(rec))), (rule, i)


def test_a_unitary_rule_is_irreducible_on_every_sector():
    """Why 150's and 156's blocks are not triangular: each sector is a SINGLE
    SCC, and the Frobenius form of an irreducible matrix is the matrix itself.
    Triangularity is visible only where drainage exists to make it."""
    from qca_fragmentation.core import rules as R
    for rule in (156, 150):
        assert R.is_unitary(R.wolfram_to_tuple(rule))
        for N in (8, 9, 10):
            sb, terminal = spy.scc_blocks(rule, N, "obc0")
            wb = spy.wcc_blocks(rule, N, "obc0")
            assert len(sb) == len(wb)
            assert all(terminal)
            assert {frozenset(b) for b in sb} == {frozenset(b) for b in wb}


# --- what a recurrent block actually looks like (R12 sec.5) -------------------

def test_a_recurrent_block_is_diagonal_only_when_every_attractor_is_a_fixed_point():
    """R12 sec.5.  'Recurrent' constrains the SET -- closed and strongly
    connected -- not the individual states.  Rule 100 is the one rule here whose
    recurrent part is diagonal, and only because every SCC is a singleton."""
    c = spy.block_type_census(100, 10, "obc0")
    assert c["diagonal_recurrent"] is True
    assert (c["fixed"], c["cycle"], c["branching"]) == (60, 0, 0)
    assert c["max_block"] == 1
    for rule in (156, 157, 109, 150, 158, 203, 29):
        assert spy.block_type_census(rule, 10, "obc0")["diagonal_recurrent"] \
            is False, rule


def test_no_hadamard_attractor_is_a_pure_cycle():
    """Every multi-state attractor of an H-gate rule branches: a Hadamard puts
    the state into a superposition of successors inside the attractor.  Pure
    permutation cycles are the X-gate case (R10), not this one."""
    for rule in ALL_SPY_RULES:
        assert spy.block_type_census(rule, 10, "obc0")["cycle"] == 0, rule


def test_rule_203_two_state_attractors_are_full_blocks_not_2_cycles():
    """The smallest concrete case: 2x2 FULL, not the anti-diagonal swap."""
    from qca_fragmentation.graph import wcc as W
    N = 10
    succ = W.make_succ(203, N, "obc0")
    sb, term = spy.scc_blocks(203, N, "obc0")
    pairs = [b for b, t in zip(sb, term) if t and len(b) == 2]
    assert len(pairs) == 6
    for b in pairs:
        S = set(b)
        for x in b:                      # both columns carry both successors
            assert sorted(y for y in succ(x) if y in S) == sorted(b)


def test_column_weight_is_the_branching_number():
    """A column's nonzero count is how many successors the state has; for a
    unitary rule that is set by how many Hadamards fire at once."""
    N = 9
    rows, cols = spy.transition_pattern(156, N, "obc0")
    from collections import Counter
    w = Counter(cols.tolist())
    assert min(w.values()) >= 1
    assert max(w.values()) > 4            # genuine multi-way branching
    # and it agrees with the successor map itself
    from qca_fragmentation.graph import wcc as W
    succ = W.make_succ(156, N, "obc0")
    for x in (0, 1, 37, 255):
        assert w[x] == len(succ(x))


# --- R12 sec.4.1: when the normal form is triangular and when it is not -------

def _condensation_edges(rule, N, bc="obc0"):
    """Number of nonzeros joining two DIFFERENT strong components."""
    rows, cols = spy.transition_pattern(rule, N, bc)
    sb, _ = spy.scc_blocks(rule, N, bc)
    own = spy.block_of(sb, 1 << N)
    return int(np.count_nonzero(own[rows] != own[cols]))


@pytest.mark.parametrize("rule", ALL_SPY_RULES)
def test_strictly_upper_is_nonzero_exactly_when_a_state_is_transient(rule):
    """R12 sec.4.1.  k >= 2 is NOT the criterion -- components with no edge
    between them give a block-DIAGONAL form.  The strictly upper part is nonzero
    precisely when the condensation has an edge, i.e. when some state can leave
    its component and never return."""
    N = 9
    st = spy.check_frobenius(rule, N, "obc0")
    has_transient = st["recurrent"] < st["dim"]
    assert (st["nnz_strictly_upper"] > 0) == has_transient, rule
    assert (_condensation_edges(rule, N) > 0) == has_transient, rule


@pytest.mark.parametrize("rule", [156, 150])
def test_a_unitary_rule_has_no_condensation_edge_at_all(rule):
    """So its Frobenius form is block-DIAGONAL, and each block is irreducible:
    there is nothing a permutation could triangularise."""
    for N in (8, 9, 10):
        assert _condensation_edges(rule, N) == 0
        st = spy.check_frobenius(rule, N, "obc0")
        assert st["nnz_strictly_upper"] == 0
        assert st["n_scc"] == st["n_terminal"]        # every component a sink


@pytest.mark.parametrize("rule", [157, 109, 158, 203, 100, 29])
def test_a_dissipative_rule_is_genuinely_triangular(rule):
    for N in (8, 9, 10):
        assert _condensation_edges(rule, N) > 0
        assert spy.check_frobenius(rule, N, "obc0")["nnz_strictly_upper"] > 0


def test_trivial_blocks_dominate_the_dissipative_pictures():
    """A singleton component with no self-loop is a 1x1 ZERO block; these are
    why the dissipative pictures carry so little on the diagonal."""
    from qca_fragmentation.graph import wcc as W
    N = 10
    for rule, want_trivial, want_total in ((157, 669, 758), (100, 880, 1024)):
        succ = W.make_succ(rule, N, "obc0")
        sb, _ = spy.scc_blocks(rule, N, "obc0")
        trivial = sum(1 for b in sb if len(b) == 1 and b[0] not in succ(b[0]))
        assert (trivial, len(sb)) == (want_trivial, want_total), rule


def test_a_self_loop_does_not_make_a_state_recurrent():
    """R12 sec.4.1's warning.  Rule 100 has 144 states with x -> x but only 60
    terminal components: the other 84 can also step away and never return."""
    from qca_fragmentation.graph import wcc as W
    N = 10
    succ = W.make_succ(100, N, "obc0")
    loops = sum(1 for x in range(1 << N) if x in succ(x))
    assert loops == 144
    st = spy.check_frobenius(100, N, "obc0")
    assert st["n_terminal"] == 60 and st["recurrent"] == 60
    assert loops > st["n_terminal"]
