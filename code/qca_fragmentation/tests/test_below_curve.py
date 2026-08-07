"""R16: the seven rules that a finite-N fit puts under the ab >= 2 hyperbola.

The claim these tests defend is narrow and worth stating plainly: the seven do
not violate the partition bound, their fitted product is an internally
inconsistent descriptor, and the correction that fixes it is derived rather than
tuned.  Each of those is checkable, so each gets a test.
"""

import math

import pytest

from qca_fragmentation import results_io
from qca_fragmentation.scaling import below_curve as B
from qca_fragmentation.scaling import sector_figure, sectors

BC = "obc0"
SEVEN = (6, 14, 20, 74, 84, 88, 229)


# --- the correction itself ---------------------------------------------------

def test_the_override_covers_exactly_the_rules_that_were_below():
    """R16_OVERRIDES is what below_curve lifts to reconstruct the pre-correction
    fit.  If it drifted away from the set of rules actually below the curve the
    module would be explaining a different seven than the one it corrected."""
    with B.without_r16_override():
        raw = sorted(h["rule"] for h in B.raw_below(BC))
    assert raw == list(SEVEN)
    assert sorted(r for r, bc in sectors.R16_OVERRIDES) == list(SEVEN)
    assert {bc for _, bc in sectors.R16_OVERRIDES} == {BC}


def test_no_rule_is_below_the_curve_once_the_override_is_in_place():
    assert B.corrected_below(BC) == []


def test_lifting_the_override_is_reversible():
    """The context manager mutates a module-level dict; a leak would silently
    change every later fit in the same process."""
    before = dict(sectors.ANALYTIC)
    with B.without_r16_override():
        assert not (sectors.R16_OVERRIDES & set(sectors.ANALYTIC))
    assert sectors.ANALYTIC == before


def test_the_override_pins_a_equals_one_and_b_equals_two():
    for r in SEVEN:
        e = sectors.ANALYTIC[(r, BC)]
        assert e["n_wcc"][0] == 1.0
        assert e["d_max_wcc"][0] == 2.0
        # b = 2 is derived, the power that goes with it is not, and the entry
        # says so by leaving alpha None for series_descriptor to refit.
        assert e["d_max_wcc"][1] is None


def test_analytic_source_strings_parse_as_mathtext():
    """These strings are rendered by matplotlib in the figures as well as by
    LaTeX in the tables, so a report-only macro in one of them breaks a figure
    at draw time and nowhere earlier."""
    from matplotlib import mathtext
    parser = mathtext.MathTextParser("path")
    for r in SEVEN:
        for key in ("n_wcc", "d_max_wcc"):
            src = sectors.ANALYTIC[(r, BC)][key][2]
            for chunk in src.split("$")[1::2]:
                parser.parse("$" + chunk + "$")


# --- the evidence ------------------------------------------------------------

def test_the_sector_counts_are_exact_staircases():
    """a = 1 is not fitted here, it is read off a closed form that holds at every
    N on record.  Everything else in R16 rests on it."""
    for r in SEVEN:
        st = B.staircase_check(r, BC)
        assert st["closed_form_exact"], (r, st["n_wcc"])
        assert len(st["N"]) >= 15


@pytest.mark.parametrize("rules,form", [
    ((6, 14, 20, 84), lambda N: (N + 1) // 2 + 1),
    ((74, 88, 229), lambda N: (N + 4) // 3),
])
def test_the_two_closed_forms_are_the_ones_claimed(rules, form):
    for r in rules:
        recs = results_io.load_wcc_results(r, BC)
        for N in sorted(recs):
            assert recs[N]["n_wcc"] == form(N), (r, N)


def test_the_finite_N_partition_bound_holds_at_every_N():
    """n_wcc * D_max >= 2^N is the sum rule, not an asymptotic statement.  If it
    ever failed, the sectors would not be a partition and the fit would be the
    least of the problems."""
    for r in SEVEN:
        assert B.squeeze(r, BC)["min_finite_theorem"] >= 1.0


def test_the_squeeze_forces_base_two():
    for r in SEVEN:
        assert B.squeeze(r, BC)["asymptotic_base"] == 2.0


def test_the_certified_control_looks_like_the_seven_over_this_window():
    """W134's D_max IS the central binomial, so its base is 2 by derivation.  If
    a finite-N fit puts it in the same band as the seven, no better estimator
    on this window separates them -- which is the whole argument for deriving b
    instead of fitting it."""
    c = B.load(BC) or B.build(BC)
    ctrl = c["control_rule"]["cumulative"][-1][1]
    band = [p["cumulative"][-1][1] for p in c["per_rule"]]
    assert 1.85 < ctrl < 2.0
    assert min(band) - 0.05 < ctrl < max(band) + 0.05


def test_a_genuine_base_is_flat_where_a_power_law_prefactor_climbs():
    """The estimator's signature, established on synthetic series where the
    answer is known: a real base sits still as the window grows, c*2^N*N^-1/2
    climbs.  This is a property of the estimator, not of any rule."""
    c = B.load(BC) or B.build(BC)
    geo = [b for _, b in c["controls"]["cumulative_geometric"]]
    assert max(geo) - min(geo) < 1e-6
    binom = [b for _, b in c["controls"]["cumulative_binomial"]]
    assert binom[-1] > binom[0] + 0.02


def test_only_the_ceil_group_climbs_and_the_report_must_not_claim_more():
    """Honest scope for the climbing argument.  W6/14/20/84 climb monotonically
    like the binomial control; W74/88/229 carry a period-3 modulation from their
    staircase that a base fitted across residues absorbs, and their traces
    wander instead.  If that ever changes, the R16 text needs changing too."""
    c = B.load(BC) or B.build(BC)
    by_rule = {p["rule"]: p for p in c["per_rule"]}
    for r in (6, 14, 20, 84):
        cu = [b for _, b in by_rule[r]["cumulative"]]
        assert cu[-1] > cu[0] + 0.03, r
    wander = [r for r in (74, 88, 229)
              if by_rule[r]["cumulative"][-1][1] <= by_rule[r]["cumulative"][0][1]]
    assert wander, "R16 says the floor-group does not climb; it now does"


def test_the_tail_secant_beats_every_fitted_base():
    """Model-free: a pure exponential c*b^N has EVERY same-residue secant equal
    to b.  All seven grow faster than their own fitted base over the tail, so
    the fit is not merely imprecise, it is the wrong model."""
    c = B.load(BC) or B.build(BC)
    for p in c["per_rule"]:
        tail = p["secant"]["best_tail"]["tail"]["base"]
        assert tail > p["fitted_b"], p["rule"]
        assert tail <= 2.0 + 0.02                       # D_max <= 2^N caps it


def test_one_rule_refutes_a_sub_two_base_from_the_data_alone():
    """W229's odd branch stops decaying, which no base below 2 can do.  The
    theorem covers all seven; this covers one of them without the theorem."""
    c = B.load(BC) or B.build(BC)
    refuters = [p["rule"] for p in c["per_rule"] if p["secant"]["refutes_sub_two"]]
    assert 229 in refuters


def test_the_secant_does_not_fire_on_a_genuine_sub_two_rule():
    """W28's base really is about 1.70.  A diagnostic that also fired here would
    be measuring noise."""
    s = B.secant(28, BC)
    assert not s["refutes_sub_two"]
    assert s["best_tail"]["tail"]["base"] < 1.8


# --- the rendering artefact --------------------------------------------------

def test_the_family_offset_matches_the_figure_it_explains():
    """R16 blames a specific number of markers on a specific nudge.  If F1's
    nudge changed and this copy did not, the explanation would be of a figure
    that no longer exists."""
    assert B.FAM_DX == sector_figure._FAM_DX


def test_the_offset_draws_on_curve_rules_below_the_curve():
    a = 1.0 + B.FAM_DX["classical"]
    assert 2.0 / a > 2.0                       # the curve rises as a falls
    art = B.visual_below(BC, pre=False)
    assert art["n"] >= 35
    assert art["families"] == ["classical"]
    # every rule counted is exactly on the curve, not near it
    for p in art["detail"]:
        assert math.isclose(p["ab"], 2.0, rel_tol=1e-9)
        assert p["ab_drawn"] < 2.0


def test_the_correction_moves_the_seven_onto_that_same_cell():
    """Five of the seven are V-free, so correcting them enlarges the artefact
    cell rather than clearing it -- worth pinning, because the count in the
    figure annotation comes from here."""
    after = B.visual_below(BC, pre=False)["n"]
    before = B.visual_below(BC, pre=True)["n"]
    assert after > before


# --- the map that consumes all of it -----------------------------------------

def test_the_sector_map_reports_no_violation_and_no_degenerate_verdict():
    d = sectors.load(BC) or sectors.build(BC)
    assert d["verdicts"].get("degenerate_subexponential", 0) == 0
    assert not d["violations"]
    assert not d["raw_below"]
    for p in d["points"]:
        a, b = p["n_wcc"]["base"], p["d_max_wcc"]["base"]
        if a is not None and b is not None:
            assert a * b >= 2.0 - 1e-9, p["rule"]


def test_the_anchors_did_not_move():
    """The override touches seven rules.  The rules whose bases are known exactly
    must be bit-for-bit where they were."""
    d = sectors.load(BC) or sectors.build(BC)
    by_rule = {p["rule"]: p for p in d["points"]}
    for r in (204, 51, 150):
        p = by_rule[r]
        assert math.isclose(p["n_wcc"]["base"] * p["d_max_wcc"]["base"],
                            2.0, rel_tol=1e-9), r
