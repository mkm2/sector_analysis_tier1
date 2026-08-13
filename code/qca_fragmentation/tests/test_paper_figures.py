"""R19: the closed forms annotated on the paper figures, and the size stats.

The figures carry tags like $F_{N+2}$ and $2^{N-1}$ next to the curves. A tag is
a claim, and these are the checks that keep it honest -- if a sweep is extended
and a law breaks, the notebook refuses to draw and this suite says which one.

Since the census was widened to all sixteen unitary rules, the same file also
guards the two things that widening depends on: that the ergodic units are
loaded at all (the growth-fit loader drops them), and that the two sweeps which
between them cover those units agree wherever they overlap.
"""

import pytest

from qca_fragmentation.core import rules
from qca_fragmentation.graph.wcc import make_succ
from qca_fragmentation.scaling import paper_figures as PF
from qca_fragmentation.scaling.summary import load_series

BC = "obc0"


@pytest.mark.parametrize("rule,key", sorted(PF.LAWS, key=lambda t: (t[1], t[0])))
def test_every_annotated_law_holds_at_every_computed_N(rule, key):
    """Not a fit and not a tail check -- every N in the sweep."""
    c = PF.check_law(rule, key, BC)
    assert c["ok"], (rule, key, c["mismatch"])
    assert c["n_points"] >= 10, c          # a law verified on 3 points is not one


def test_the_verify_bundle_is_green():
    """One call the notebook makes before drawing anything."""
    assert PF.verify(BC)["ok"]


# --------------------------------------------------------- the census is whole

def test_every_unitary_rule_is_classified_and_present():
    assert len(PF.UNITARY) == 16
    assert set(PF.CLASS_OF) == set(PF.UNITARY)
    rows = PF.census(BC)
    assert {r["rule"] for r in rows} == set(PF.UNITARY)
    assert all(r["n_points"] >= 10 for r in rows)


#: the four (rule, series) slots that deliberately have no closed form
_NO_CLOSED_FORM = {(156, "d_max"), (198, "d_max"),
                   (108, "n_recurrent"), (201, "n_recurrent")}


def test_every_rule_and_series_has_a_law_except_the_four_that_cannot():
    """W156/W198's largest sector only approaches 4^{N/5} and W108/W201's
    sector counts satisfy a recurrence, not a closed form. Everything else --
    all 44 remaining slots -- is exact, and a silent gap would mean a curve
    drawn without a certified tag."""
    missing = {(r, k) for r in PF.UNITARY for k in PF.SERIES_KEYS
               if (r, k) not in PF.LAWS}
    assert missing == _NO_CLOSED_FORM


def test_the_growth_loader_hides_exactly_the_ergodic_rules():
    """Why this module has its own loader.

    `summary.load_series` drops ergodic units, which is right for fitting a
    growth law and leaves six of the sixteen rules with an EMPTY series. If
    this test ever fails the notebook's merge has stopped being necessary --
    or, worse, has started hiding something.
    """
    hidden = [r for r in PF.UNITARY if not load_series(r, BC)["N"]]
    assert sorted(hidden) == sorted(PF.ERGODIC_SIX + (54,))
    for r in hidden:
        assert len(PF.unit_series(r, BC)["N"]) >= 10


def test_the_two_sweeps_agree_wherever_both_ran():
    """For a unitary rule the weak and strong components coincide, so Tier-1a
    (Tarjan, deeper) and Tier-1e (union-find, wider) must give the same numbers
    on every shared unit. The merge in `unit_series` is only legitimate because
    of this."""
    assert PF.wcc_scc_agreement(BC)


def test_no_rule_is_left_behind_at_a_shorter_frontier():
    """The ergodic rules used to stop at N = 16 while W150 reached 22, purely
    because the sweep stopped enlarging N once a rule was classified ergodic.
    `extend_unitary` closed that gap with the flip reduction. The figure still
    annotates a short track's N_max in the legend -- ranges may legitimately
    differ again -- but the census is no longer ragged for no reason."""
    nmax = {r: PF.unit_series(r, BC)["N"][-1] for r in PF.UNITARY}
    assert min(nmax.values()) >= 22, nmax
    assert max(nmax.values()) - min(nmax.values()) <= 1, nmax


def test_no_series_has_a_hole_in_it():
    """Consecutive N with no gaps, so a line drawn between two markers never
    spans an N nobody computed."""
    for r in PF.UNITARY:
        Ns = PF.unit_series(r, BC)["N"]
        assert Ns == list(range(Ns[0], Ns[-1] + 1)), r


# ------------------------------------------------------------ the four classes

def test_the_ergodic_six_are_one_curve_and_the_identity_is_the_other_extreme():
    assert PF.ergodic_six_coincide(BC)
    for r in PF.ERGODIC_SIX:
        s = PF.unit_series(r, BC)
        assert set(s["n_recurrent"]) == {1}
        assert all(d == 2 ** N for N, d in zip(s["N"], s["d_max"]))
    s = PF.unit_series(204, BC)
    assert set(s["d_max"]) == {1}
    assert all(n == 2 ** N for N, n in zip(s["N"], s["n_recurrent"]))


def test_only_the_trivial_rules_sit_on_the_hyperbola():
    """n_wcc * D_max >= 2^N with equality iff every sector is the same size.
    Exactly two behaviours manage that, and they are the two that do not
    fragment; the figure's frame is that statement."""
    for r in PF.UNITARY:
        h = PF.saturation(r, BC)
        assert h["n"] * h["d_max"] >= 1 << h["N"]
        assert h["saturates"] == (r in PF.ERGODIC_SIX or r == 204), r
    assert PF.saturation(201, BC)["ratio"] > 1000


def test_w54_is_ergodic_plus_exactly_one_frozen_state_and_it_is_the_vacuum():
    """r00 = I freezes |0...0>, and the census says W54 has exactly one frozen
    state, so that one is it. This is the whole difference between W54 and the
    other six -- and between W54 and its spin-flip partner W147."""
    N = 8
    f = make_succ(54, N, BC)
    frozen = [x for x in range(1 << N) if set(f(x)) == {x}]
    assert frozen == [0]
    assert PF.check_law(54, "n_frozen", BC)["ok"]
    assert PF.unit_series(147, BC)["n_frozen"][-1] == 0


# ------------------------------------------------------------- the symmetries

def test_reflection_is_an_obc0_symmetry_so_sixteen_rules_give_twelve_curves():
    refl = PF.reflection_identities(BC)
    assert set(refl) == {(57, 99), (60, 102), (153, 195), (156, 198)}
    assert all(refl.values())
    classes = {frozenset({r, rules.reflect_wolfram(r)}) for r in PF.UNITARY}
    assert len(classes) == 12


def test_the_spin_flip_survives_only_where_it_is_also_a_reflection():
    """X^{tensor N} preserves the transition graph under pbc, but obc0 pins the
    padding to |0> and the flip sends it to |1>. Of the six flip pairs the only
    two whose series match are the two that are reflection pairs as well, so
    nothing here is the flip's own doing."""
    assert PF.flip_survives_only_via_reflection(BC)
    rows = {row["pair"]: row for row in PF.spinflip_break(BC)}
    assert set(rows) == {(54, 147), (57, 99), (60, 195), (102, 153),
                         (108, 201), (156, 198)}
    for pair, expect in (((57, 99), True), ((156, 198), True),
                         ((54, 147), False), ((60, 195), False),
                         ((102, 153), False), ((108, 201), False)):
        assert rows[pair]["identical"] is expect
        assert rows[pair]["is_reflection"] is expect


def test_the_vacuum_alone_decides_whether_w60_and_w102_fragment():
    """The sharpest consequence: W195 and W153 are the spin-flip images of two
    polynomially fragmented rules and are completely ergodic. Same gate set,
    same lattice, different vacuum."""
    for frag, erg in ((60, 195), (102, 153)):
        assert PF.spinflip_partner(frag) == erg
        assert PF.CLASS_OF[frag] == "polynomial"
        assert PF.CLASS_OF[erg] == "ergodic"
        assert PF.unit_series(erg, BC)["n_recurrent"][-1] == 1
        assert PF.unit_series(frag, BC)["n_recurrent"][-1] > 10


# ------------------------------------------------------------- the older laws

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


def test_w105_loses_its_frozen_states_at_the_same_residue():
    """A third face of the same parity obstruction: N = 1 (mod 4) is exactly
    where W105 has NO frozen state, while W150 never drops below one."""
    s105, s150 = PF.unit_series(105, BC), PF.unit_series(150, BC)
    zero = {N for N, f in zip(s105["N"], s105["n_frozen"]) if f == 0}
    assert zero == {N for N in s105["N"] if N % 4 == 1}
    assert min(f for f in s150["n_frozen"]) >= 1


def test_w60_and_w102_really_are_term_for_term_identical():
    """The other track IS collapsible, and the figure labels it `W60 = W102`
    with an equals sign on the strength of this."""
    _, n60, d60 = PF.series(60, BC)
    _, n102, d102 = PF.series(102, BC)
    assert n60 == n102 and d60 == d102


def test_the_frozen_count_of_w108_factorises_over_the_sublattices():
    """A state is frozen iff nothing moves on either sublattice, and the two
    decouple, so the count is a PRODUCT of two Fibonacci numbers of nearly
    equal index -- the same even/odd split that carries R18's two independent
    wall charges. Checked against the raw series, not against itself."""
    s = PF.unit_series(108, BC)
    for N, f in zip(s["N"], s["n_frozen"]):
        assert f == PF.fib(N // 2 + 2) * PF.fib((N + 1) // 2 + 2), N
    # and the exponential rules are the only ones with exponentially many
    frozen_at_max = {r: PF.unit_series(r, BC)["n_frozen"][-1] for r in PF.UNITARY}
    assert frozen_at_max[108] > 10000 and frozen_at_max[156] < 20


# ---------------------------------------------------------------- the histogram

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
    """The tables are generated from `verify`/`census`/`size_stats`, so R19
    cannot quote a number this suite has not checked."""
    monkeypatch.setattr(PF, "TEX_DIR", str(tmp_path))
    paths = PF.write_tables(BC)
    assert len(paths) == 3
    laws, cens, sizes = (open(p).read() for p in paths)
    assert "FAILS" not in laws and "NO" not in laws.split("\\midrule")[1]
    assert "$F_{N+2}$" in laws and "$2^{N-1}$" in laws
    for rule in PF.UNITARY:                      # every rule appears somewhere
        assert str(rule) in cens
    assert cens.count("\\\\") >= 16
    for rule in PF.HIST_RULES:
        assert str(rule) in sizes
