"""R19: the closed forms annotated on the paper figures, and the size stats.

The figures carry tags like $F_{N+2}$ and $2^{N-1}$ next to the curves. A tag is
a claim, and these are the checks that keep it honest -- if a sweep is extended
and a law breaks, the notebook refuses to draw and this suite says which one.
"""

import pytest

from qca_fragmentation.scaling import paper_figures as PF

BC = "obc0"


@pytest.mark.parametrize("rule,key", sorted(PF.LAWS, key=lambda t: (t[1], t[0])))
def test_every_annotated_law_holds_at_every_computed_N(rule, key):
    """Not a fit and not a tail check -- every N in the sweep."""
    c = PF.check_law(rule, key, BC)
    assert c["ok"], (rule, key, c["mismatch"])
    assert c["n_points"] >= 10, c          # a law verified on 3 points is not one


def test_the_plastic_recurrence_is_exact_not_fitted():
    """W108's sector count satisfies an integer recurrence whose dominant root
    is rho^2 = 1.754878. R9 fitted that base; R18 derived it from the wall
    grammar; this is the third, purely arithmetic, route to the same number."""
    assert PF.plastic_recurrence(BC)


def test_w201_is_w108_shifted_by_one_site():
    """The two are spin-flip partners and their sector counts are the same
    sequence offset by one, which is why both carry the same rho^{2N} tag."""
    assert PF.shifted_partner(BC)


def test_w156_dmax_converges_to_the_room_packing_base():
    """4^{1/5} is derived (R2 sec.3 saddle point), not fitted, and the
    five-step ratio has to agree with it -- but only asymptotically, since at
    finite N the largest sector mixes rooms of length 2 and 3. Six digits by
    N = 21 is the evidence; an exact recurrence would be the wrong assertion."""
    rb = PF.room_base_check(BC)
    assert rb["ok"]
    assert rb["abs_error"] < 1e-6, rb


def test_w150_and_w105_differ_exactly_at_N_equiv_1_mod_4():
    """The figure draws them as one track but two curves, and this is why.

    At N = 1 (mod 4) the largest sector of 105 sits at ODD domain-wall number
    (N=9: C(10,5)=252 against 150's C(10,4)=210) and 105 has one sector fewer.
    At every other N the two agree term for term. Both series break at the same
    residue -- one extra level of the wall number is exactly one fewer sector --
    so this is one phenomenon, not two.
    """
    N150, n150, d150 = PF.series(150, BC)
    N105, n105, d105 = PF.series(105, BC)
    common = sorted(set(N150) & set(N105))
    n1, n2 = dict(zip(N150, n150)), dict(zip(N105, n105))
    d1, d2 = dict(zip(N150, d150)), dict(zip(N105, d105))
    assert any(N % 4 == 1 for N in common)
    assert all((n1[N] == n2[N]) == (N % 4 != 1) for N in common)
    assert all((d1[N] == d2[N]) == (N % 4 != 1) for N in common)
    assert all(n2[N] == n1[N] - 1 for N in common if N % 4 == 1)


def test_w60_and_w102_really_are_term_for_term_identical():
    """The other track IS collapsible, and the figure labels it `W60 = W102`
    with an equals sign on the strength of this."""
    _, n60, d60 = PF.series(60, BC)
    _, n102, d102 = PF.series(102, BC)
    assert n60 == n102 and d60 == d102


@pytest.mark.parametrize("rule", PF.HIST_RULES)
@pytest.mark.parametrize("N", PF.HIST_NS)
def test_the_size_histogram_is_complete_where_the_size_list_is_truncated(rule, N):
    """`sizes_recurrent` stops at 2048 entries for these N, so Figure 3 reads
    `size_hist` instead. That is only safe because the histogram carries every
    class and its mass sums to 2^N, which size_stats asserts."""
    s = PF.size_stats(rule, N, BC)
    assert s["n_sectors"] > 2048 or N < 14
    assert 0 < s["d_max_fraction"] <= 1.0
    assert s["mean"] >= 1.0


@pytest.mark.parametrize("rule", PF.HIST_RULES)
def test_the_typical_sector_is_far_smaller_than_the_largest(rule):
    """The point of Figure 3's bottom row. Quoting D_max alone overstates where
    a random state actually lands, and the gap widens with N."""
    ratios = []
    for N in PF.HIST_NS:
        s = PF.size_stats(rule, N, BC)
        assert s["median"] < s["d_max"]
        ratios.append(s["d_max"] / s["median"])
    assert ratios[-1] > ratios[0]


def test_w108_is_dominated_by_tiny_sectors_and_w156_is_not():
    """The visible contrast between the two columns of Figure 3: W108's size
    distribution is monotone decreasing with a huge singleton population, W156's
    is peaked away from 1. If this flips, the figure's caption is wrong."""
    s108 = PF.size_stats(108, 21, BC)
    s156 = PF.size_stats(156, 21, BC)
    assert s108["n_singletons"] / s108["n_sectors"] > 0.15
    assert s156["n_singletons"] / s156["n_sectors"] < 0.01


def test_the_report_tables_render_the_checked_values(tmp_path, monkeypatch):
    """The tables are generated from `verify`/`size_stats`, so R19 cannot quote
    a number this suite has not checked."""
    monkeypatch.setattr(PF, "TEX_DIR", str(tmp_path))
    paths = PF.write_tables(BC)
    assert len(paths) == 2
    laws = open(paths[0]).read()
    sizes = open(paths[1]).read()
    assert "NO" not in laws.split("\\midrule")[1]      # every law holds
    assert "$F_{N+2}$" in laws and "$2^{N-1}$" in laws
    for rule in PF.HIST_RULES:
        assert str(rule) in sizes
