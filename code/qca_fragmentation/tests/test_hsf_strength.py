"""
Strong vs weak fragmentation -- the checks that keep R21 honest.

The expensive parts (N = 20) live in the cached analytics JSON; these tests
recompute the small-N facts from scratch, because the claims that matter are
structural and visible by N = 12.
"""

import numpy as np
import pytest

from qca_fragmentation.scaling import hsf_strength as HS
from qca_fragmentation.scaling import wall_charges as WC

BC = "obc0"


# --- the detector itself ------------------------------------------------------

@pytest.mark.parametrize("rule", HS.RULES)
def test_the_detector_finds_r18s_two_charges_and_no_more(rule):
    """R18 certifies exactly two independent range-2 charges per rule, by
    Tier-2's null-space detector.  This module finds them by exact elimination
    on count-vector differences, with no notion of a wall anywhere."""
    for N in (10, 11, 12):
        d = HS.conserved_charges(rule, N, BC, 2)
        assert d["n_independent"] == 2, (rule, N, d["n_independent"])
        assert d["value_rank_with_constant"] == 3, (rule, N)


@pytest.mark.parametrize("rule", HS.RULES)
def test_those_charges_are_the_sublattice_resolved_wall_counts(rule):
    """R18 names them (|W_even|, |W_odd|) for the rule's minimal wall word.
    Two independent routes, one partition."""
    for N in (10, 11, 12):
        m = HS.matches_r18(rule, N, BC)
        assert m["same_partition"], (rule, N, m)
        assert m["n_mine"] == m["n_r18"] == m["n_joint"], (rule, N, m)


def test_there_is_no_single_site_charge_for_any_of_them():
    """R18: what fails is range, not basis -- the gate flips a site outright,
    so magnetisation is not conserved."""
    for rule in HS.RULES:
        assert HS.conserved_charges(rule, 12, BC, 1)["n_independent"] == 0, rule


def test_the_two_site_unit_cell_is_what_makes_the_pair_visible():
    """With p = 1 (uniform coefficients) only one charge survives; the second
    lives on a sublattice, which is exactly R18's parity split."""
    for rule in HS.RULES:
        uniform = HS.conserved_charges(rule, 12, BC, 2, p=1)["n_independent"]
        staggered = HS.conserved_charges(rule, 12, BC, 2, p=2)["n_independent"]
        assert uniform == 1, (rule, uniform)
        assert staggered == 2, (rule, staggered)


@pytest.mark.parametrize("rule", (108, 201, 156, 198, 60, 150, 105))
def test_a_basis_found_at_even_N_is_not_conserved_at_odd_N(rule):
    """The bug the parity split fixes, pinned so it cannot come back.

    On an open chain the two sublattices swap roles with N mod 2, so a charge
    detected at N = 12 need not be conserved at N = 13.  It happens to survive
    at range 2, which is why the earlier pass of this module went unnoticed;
    at range 4 every rule in play has at least one basis vector that breaks.
    """
    r = 4
    qs = HS.conserved_charges(rule, 12, BC, r)["nonconstant"]
    kl = HS.krylov_labels(rule, 13, BC)
    ks = range(int(kl.max()) + 1)

    def breaks(q):
        Q = HS.charge_values(q, 13, BC, r)
        return any(len(set(Q[kl == k].tolist())) > 1 for k in ks)

    assert any(breaks(q) for q in qs), (rule, "expected an odd-N break")
    # and the parity-resolved basis does not break
    for q in HS.conserved_charges(rule, 13, BC, r)["nonconstant"]:
        assert not breaks(q), (rule, q)


# --- the ratios ---------------------------------------------------------------

@pytest.mark.parametrize("rule", HS.RULES)
def test_every_krylov_sector_lies_inside_one_symmetry_sector(rule):
    for N in (10, 11, 12, 13):
        qs = HS.conserved_charges(rule, N, BC, 2)["nonconstant"]
        assert HS.strength(rule, N, BC, qs, 2)["krylov_inside_sym"], (rule, N)


@pytest.mark.parametrize("rule", HS.RULES)
def test_the_symmetry_sectors_stay_polynomial_in_N(rule):
    """The whole reason the symmetry cannot rescue the ratio.  At range 2 the
    joint level sets of the two charges number O(N^2); an exponential count
    would break the argument, so it is asserted rather than assumed."""
    ns = []
    for N in (8, 10, 12, 14):
        qs = HS.conserved_charges(rule, N, BC, 2)["nonconstant"]
        ns.append(HS.strength(rule, N, BC, qs, 2)["n_sym"])
    # doubling N must not square the count; a quadratic law gives ~4x
    assert ns[-1] < 6 * ns[0], (rule, ns)
    assert ns == sorted(ns), (rule, ns)


@pytest.mark.parametrize("rule", HS.RULES)
def test_the_dominant_symmetry_sector_holds_a_constant_times_2N_over_N(rule):
    """dim(s*) = Theta(2^N / N), which is what turns the O(N^2) level sets into
    a merely polynomial rescue."""
    vals = []
    for N in (10, 12, 14):
        qs = HS.conserved_charges(rule, N, BC, 2)["nonconstant"]
        vals.append(HS.strength(rule, N, BC, qs, 2)
                    ["dim_sym_max_times_N_over_2N"])
    assert all(1.0 < v < 4.0 for v in vals), (rule, vals)


@pytest.mark.parametrize("rule", HS.RULES)
def test_the_ratio_falls_on_the_even_sublattice(rule):
    """Structure only.  At N <= 14 the O(N^2) prefactor still eats much of the
    (base/2)^N decay -- W108 falls by 1.7x over N = 10..14 where the bare
    exponential would give 2.3x -- so the sharp rate is asserted against the
    N = 20 run in test_the_measured_decay_brackets_the_predicted_one, and what
    is checked here is that the ratio decreases at every step."""
    vals = []
    for N in (8, 10, 12, 14):
        qs = HS.conserved_charges(rule, N, BC, 2)["nonconstant"]
        vals.append(HS.strength(rule, N, BC, qs, 2)["ratio_dominant"])
    assert vals == sorted(vals, reverse=True), (rule, vals)
    assert vals[-1] < vals[0] / 1.5, (rule, vals)


# --- the controls -------------------------------------------------------------

def test_w60_and_w102_keep_exactly_half_the_space_in_one_sector():
    """The weak signature, and a rule of this project rather than a foreign
    model: D_max = 2^(N-1), no charge below range 3, ratio 1/2 forever."""
    for rule in (60, 102):
        for N in (10, 12, 14):
            qs = HS.conserved_charges(rule, N, BC, 2)["nonconstant"]
            s = HS.strength(rule, N, BC, qs, 2)
            assert qs == [], (rule, N)
            assert s["n_sym"] == 1
            assert s["d_max_global"] == 1 << (N - 1)
            assert s["ratio_dominant"] == 0.5
            assert s["n_krylov"] == N + 1


def test_w150_and_w105_are_not_fragmented_at_all():
    """Their Krylov sectors ARE the level sets of a single range-2 charge, so
    the ratio is exactly 1 and no amount of range changes it."""
    for rule in (150, 105):
        for N in (10, 11, 12, 13):
            d = HS.conserved_charges(rule, N, BC, 2)
            assert d["n_independent"] == 1, (rule, N)
            s = HS.strength(rule, N, BC, d["nonconstant"], 2)
            assert s["n_sym"] == s["n_krylov"], (rule, N, s)
            assert s["ratio_dominant"] == 1.0
            assert s["explored_fraction"] == 1.0


@pytest.mark.parametrize("rule", (150, 105))
@pytest.mark.parametrize("N", range(3, 17))
def test_the_closed_form_gives_the_whole_sector_distribution(rule, N):
    """Not just D_max: every sector size, keyed by its charge value.

    W150: size(D) = C(N+1, D) for D even.
    W105: size(S) = C(N+1, S + ceil(N/2)) for S even.
    """
    assert HS.distributions_agree(rule, N), (rule, N)


@pytest.mark.parametrize("rule", (150, 105))
def test_the_closed_form_reproduces_the_r19_and_r20_laws(rule):
    """Sector count, D_max and the frozen count all fall out of the same row of
    Pascal's triangle, so R19's closed forms and R20's OEIS entries must agree
    with it."""
    from math import comb
    for N in range(3, 23):
        d = HS.sector_distribution(rule, N)
        assert sum(d.values()) == 1 << N, (rule, N)
        assert len(d) == HS.n_sectors_closed(rule, N), (rule, N)
        assert HS.d_max_closed(rule, N) == max(d.values())
        if rule == 150:
            assert HS.n_frozen_closed(150, N) == 1 + (N % 2), N
        else:
            assert HS.n_frozen_closed(105, N) == {0: 1, 1: 0, 2: 1, 3: 2}[N % 4], N
    # W105 always reaches the central binomial; W150 misses it exactly at
    # N = 1 (mod 4).  When N+1 is odd the peak value sits at two indices of
    # opposite parity, so there both classes reach it.
    for N in range(3, 23):
        central = comb(N + 1, (N + 1) // 2)
        assert HS.d_max_closed(105, N) == central, N
        assert (HS.d_max_closed(150, N) == central) == (N % 4 != 1), N


def test_the_two_distributions_differ_exactly_at_N_congruent_1_mod_4():
    """Same parity class or not: the reflection w -> N+1-w swaps classes only
    when N+1 is odd, so the size multisets part company precisely at
    N = 1 (mod 4) -- the same residue at which W105 has no frozen states."""
    for N in range(3, 25):
        assert HS.same_multiset(N) == (N % 4 != 1), N
        if N % 4 == 1:
            assert HS.n_frozen_closed(105, N) == 0, N


def test_the_divergences_R20_observed_are_the_parity_class():
    """R20 reports W150's D_max leaving the central binomials at N = 9 (210 vs
    252) and N = 13 (3003 vs 3432).  Both are the excluded odd centre."""
    assert HS.d_max_closed(150, 9) == 210 and HS.d_max_closed(105, 9) == 252
    assert HS.d_max_closed(150, 13) == 3003 and HS.d_max_closed(105, 13) == 3432


def test_the_single_charge_is_D_for_150_and_S_for_105():
    """Total domain-wall number against staggered domain-wall number -- and in
    each case the OTHER one is not conserved."""
    for rule, conserved, broken in ((150, WC.charge_total, WC.charge_staggered),
                                    (105, WC.charge_staggered, WC.charge_total)):
        for N in (10, 11, 12):
            kl = HS.krylov_labels(rule, N, BC)
            good = np.array([conserved(x, N, BC) for x in range(1 << N)])
            bad = np.array([broken(x, N, BC) for x in range(1 << N)])
            ks = range(int(kl.max()) + 1)
            assert all(len(set(good[kl == k].tolist())) == 1 for k in ks), \
                (rule, N, "expected conserved")
            assert any(len(set(bad[kl == k].tolist())) > 1 for k in ks), \
                (rule, N, "expected broken")


# --- the verdict --------------------------------------------------------------

def test_the_verdict_separates_the_four_rules_from_the_controls():
    d = HS.load(BC)
    if d is None:
        pytest.skip("analytics/hsf_strength_obc0.json missing; run --rebuild")
    HS.annotate(d)
    got = {int(e["rule"]): e["verdict"] for e in d["rules"]}
    for rule in HS.RULES:
        assert got[rule] == "strong", (rule, got[rule])
    for rule in (60, 102):
        assert got[rule] == "weak", (rule, got[rule])
    for rule in (150, 105):
        assert got[rule] == "unfragmented", (rule, got[rule])


def test_the_verdict_is_strong_at_every_range_that_was_tested():
    """A "strong" that only survives at the range one happens to pick would be
    worthless.  All four rules must come out strong at r = 1, 2, 3 and 4."""
    d = HS.load(BC)
    if d is None:
        pytest.skip("analytics/hsf_strength_obc0.json missing")
    HS.annotate(d)
    for e in d["rules"]:
        if int(e["rule"]) not in HS.RULES:
            continue
        for r in HS.RANGES:
            assert e["verdict_by_range"][str(r)] == "strong", (e["rule"], r)


def test_the_measured_decay_brackets_the_predicted_one():
    """The N^m prefactor makes the measured rate a little SLOWER than base/2;
    it must not be faster, and it must not be near 1."""
    d = HS.load(BC)
    if d is None:
        pytest.skip("analytics/hsf_strength_obc0.json missing")
    for e in d["rules"]:
        rule = int(e["rule"])
        if rule not in HS.RULES:
            continue
        predicted = HS.DMAX_BASE[rule] / 2.0
        got = e["by_range"]["2"]["decay_dominant"]
        assert predicted <= got < 0.9, (rule, predicted, got)


def test_on_the_ring_the_detector_reproduces_r18s_certified_counts():
    """R18 quotes Tier-2's FULL certified charge set at pbc as 13, 18, 25 level
    sets for W108 at N = 8, 10, 12 and 12, 17, 23 for W156 -- its own check that
    nothing conserved had been left out.  A third detector, same numbers."""
    want = {108: [13, 18, 25], 156: [12, 17, 23]}
    for rule, expect in want.items():
        got = []
        for N in (8, 10, 12):
            qs = HS.conserved_charges(rule, N, "pbc", 2)["nonconstant"]
            got.append(HS.strength(rule, N, "pbc", qs, 2)["n_sym"])
        assert got == expect, (rule, got, expect)


def test_the_ring_merges_108_and_201_but_obc0_does_not():
    """The spin flip is a pbc symmetry, not an obc0 one (R19)."""
    for N in (10, 12):
        a = HS.strength(108, N, "pbc",
                        HS.conserved_charges(108, N, "pbc", 2)["nonconstant"], 2)
        b = HS.strength(201, N, "pbc",
                        HS.conserved_charges(201, N, "pbc", 2)["nonconstant"], 2)
        assert (a["n_krylov"], a["n_sym"]) == (b["n_krylov"], b["n_sym"]), N
    a = HS.strength(108, 12, BC,
                    HS.conserved_charges(108, 12, BC, 2)["nonconstant"], 2)
    b = HS.strength(201, 12, BC,
                    HS.conserved_charges(201, 12, BC, 2)["nonconstant"], 2)
    assert a["n_krylov"] != b["n_krylov"]


def test_the_verdict_is_strong_at_pbc_too():
    d = HS.load("pbc")
    if d is None:
        pytest.skip("analytics/hsf_strength_pbc.json missing")
    HS.annotate(d)
    for e in d["rules"]:
        if int(e["rule"]) not in HS.RULES:
            continue
        for r in HS.RANGES:
            assert e["verdict_by_range"][str(r)] == "strong", (e["rule"], r)


def test_the_theorem_bound_says_strong_exactly_for_the_four_rules():
    for rule in HS.RULES:
        assert HS.theorem_bound(rule, HS.DMAX_BASE[rule], 2)["strong"]
    for rule in HS.CONTROLS:
        assert not HS.theorem_bound(rule, HS.DMAX_BASE[rule], 2)["strong"]
