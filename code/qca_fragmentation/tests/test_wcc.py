"""
Tier 1e: weakly connected components as the sector-level observable.

Ground-truth regression targets from the task spec sec.6, plus the hard
assertions A1-A4 of sec.2.  Everything here runs at small N so the suite stays
fast; the reference series that need large N are checked against their closed
forms, which are themselves verified against the computed values where both are
affordable.
"""

from math import comb

import pytest

from qca_fragmentation import results_io
from qca_fragmentation.core import rules
from qca_fragmentation.graph import scc, wcc
from qca_fragmentation.scaling.fits import find_integer_recurrence

PHI = (1 + 5 ** 0.5) / 2
PLASTIC = 1.324717957244746


# --- A1: the sum rule, which is what makes WCC a sector axis at all ----------

@pytest.mark.parametrize("rule", [0, 22, 28, 51, 90, 105, 108, 150, 156, 201,
                                  204, 232])
@pytest.mark.parametrize("bc", ["obc0", "pbc"])
def test_sum_rule(rule, bc):
    r = wcc.weak_components(rule, 8, bc)
    assert sum(r.sizes_wcc) == 2 ** 8
    assert r.n_wcc == len(r.sizes_wcc)
    assert r.d_max_wcc == max(r.sizes_wcc)
    assert r.n_frozen == sum(1 for s in r.sizes_wcc if s == 1)


def test_components_are_never_aborted_by_default():
    """weak_components must finish the pass, or A1 and the hyperbola anchors
    would both be unavailable for the ergodic rules."""
    r = wcc.weak_components(51, 8, "obc0")
    assert not r.aborted
    assert r.ergodic                      # classification, not truncation
    assert r.n_wcc == 1 and r.d_max_wcc == 256


# --- the anchors of the exclusion curve (task sec.4 V2) ----------------------

@pytest.mark.parametrize("N", [6, 8, 10])
def test_rule_204_is_all_singletons(N):
    """204 = IIII: nothing ever moves.  a = 2, b = 1."""
    r = wcc.weak_components(204, N, "obc0")
    assert r.n_wcc == 2 ** N
    assert r.d_max_wcc == 1
    assert r.n_frozen == 2 ** N


@pytest.mark.parametrize("N", [6, 8, 10])
def test_rule_51_is_one_sector(N):
    """51 = VVVV: a Hadamard everywhere, one sector.  a = 1, b = 2."""
    r = wcc.weak_components(51, N, "obc0")
    assert r.n_wcc == 1
    assert r.d_max_wcc == 2 ** N


# --- rule 150 obc0: the exact reference series -------------------------------

_R150_NSEC = [5, 6, 6, 7, 7, 8, 8, 9, 9, 10]          # N = 8..17
_R150_DMAX = [126, 210, 462, 924, 1716, 3003, 6435, 12870, 24310, 43758]


def _r150_closed_form(N):
    sizes = [comb(N + 1, w) for w in range(0, N + 2, 2)]
    return len(sizes), max(sizes)


def test_rule_150_reference_series_closed_form():
    """The closed form of R8 sec.4 reproduces the task's reference series."""
    for i, N in enumerate(range(8, 18)):
        n, d = _r150_closed_form(N)
        assert n == _R150_NSEC[i], (N, n)
        assert d == _R150_DMAX[i], (N, d)


@pytest.mark.parametrize("N", range(8, 14))
def test_rule_150_computed_matches_reference(N):
    """And the streamed union-find reproduces it where it is affordable."""
    r = wcc.weak_components(150, N, "obc0")
    assert r.n_wcc == _R150_NSEC[N - 8]
    assert r.d_max_wcc == _R150_DMAX[N - 8]
    assert r.sizes_wcc == sorted(
        (comb(N + 1, w) for w in range(0, N + 2, 2)), reverse=True)


# --- rule 156 pbc: Lucas ------------------------------------------------------

def _lucas(n):
    a, b = 2, 1
    for _ in range(n):
        a, b = b, a + b
    return a


@pytest.mark.parametrize("N", range(6, 13))
def test_rule_156_pbc_is_lucas_plus_one(N):
    r = wcc.weak_components(156, N, "pbc")
    assert r.n_wcc == _lucas(N) + 1


def test_rule_156_pbc_dmax_base_is_4_to_the_fifth():
    """b -> 4^(1/5) = 1.31951 for the largest sector."""
    ys = [wcc.weak_components(156, N, "pbc").d_max_wcc for N in range(6, 15)]
    ratios = [ys[i + 1] / ys[i] for i in range(len(ys) - 1)]
    assert 4 ** 0.2 - 0.12 < sum(ratios[-4:]) / 4 < 4 ** 0.2 + 0.12, ys


# --- rule 22 pbc: the whole point of the tier --------------------------------

@pytest.mark.parametrize("N", [6, 8, 10, 12])
def test_rule_22_pbc_sectors_partition_while_attractors_do_not(N):
    """
    3 monitored attractors, all of size 1, covering 3 of 2^N states; the WCCs
    cover everything.  transient_fraction -> 1.
    """
    r = wcc.weak_components(22, N, "pbc")
    assert sum(r.sizes_wcc) == 2 ** N
    assert r.n_wcc == 2                       # few sectors, one of them huge
    rec = results_io.load_results(22, "pbc").get(N)
    if rec and not rec.get("ergodic_flag"):
        assert rec["n_recurrent"] == 3
        assert results_io.sizes_from_record(rec, "sizes_recurrent") == [1, 1, 1]
        tf = 1 - 3 / 2 ** N
        assert tf > 0.95
        assert tf > 1 - 10.0 / 2 ** N         # -> 1


def test_rule_22_transient_fraction_increases_with_N():
    tf = []
    for N in (6, 8, 10, 12):
        rec = results_io.load_results(22, "pbc").get(N)
        if not rec or rec.get("ergodic_flag"):
            continue
        sizes = results_io.sizes_from_record(rec, "sizes_recurrent")
        tf.append(1 - sum(sizes) / 2 ** N)
    assert len(tf) >= 3
    assert all(tf[i] < tf[i + 1] for i in range(len(tf) - 1)), tf


# --- rule 28: the wall-transparency prediction, and its failure --------------

def test_rule_28_sector_count_follows_the_plastic_number_not_phi():
    """
    Task sec.6 predicts a_wcc >= phi for rule 28 = (IIVD), inheriting the 01
    wall grammar of rule 156.  It does NOT hold: the obc0 sector count obeys the
    EXACT recurrence a_n = a_{n-1} + a_{n-2} - a_{n-4}, whose characteristic
    polynomial factors as (x-1)(x^3-x-1), giving the plastic number
    rho = 1.32472 -- well below phi = 1.61803.  This test pins the violation so
    it cannot be quietly lost; see R9 for the discussion.
    """
    ys = [wcc.weak_components(28, N, "obc0").n_wcc for N in range(6, 15)]
    assert ys == [11, 15, 20, 27, 36, 48, 64, 85, 113]
    rec = find_integer_recurrence(ys)
    assert rec["ok"]
    assert rec["coeffs"] == [1, 1, 0, -1]
    assert rec["base"] == pytest.approx(PLASTIC, abs=1e-9)
    assert rec["base"] < PHI - 0.25          # the violation, quantified


# --- A2/A3/A4 against the archived Tier-1a records ---------------------------

@pytest.mark.parametrize("rule", [60, 102, 105, 108, 150, 156, 198, 201, 204])
def test_A2_unitary_wcc_equals_the_sector_partition(rule):
    """
    Doubly stochastic => weak, strong and forward-closure partitions agree.

    The parametrisation lists Wolfram numbers that are unitary IN THIS ENCODING
    (rules.UNITARY_RULES), i.e. whose symbol tuple uses only I and V.  Note the
    Wolfram number is not the classical ECA rule: W90 is the tuple DEED and is
    dissipative, so A2 must not be applied to it.
    """
    assert rule in rules.UNITARY_RULES
    recs = results_io.load_results(rule, "obc0")
    checked = 0
    for N in (8, 9, 10):
        rec = recs.get(N)
        if not rec or rec.get("ergodic_flag"):
            continue
        r = wcc.weak_components(rule, N, "obc0")
        out = wcc.check_against_scc(r, rec, unitary=True)
        assert out["a2"] is True
        checked += 1
    assert checked >= 1


def test_A2_does_not_apply_to_a_dissipative_rule_with_a_unitary_looking_number():
    """W90 = DEED: n_wcc and n_recurrent may agree while the SIZES differ wildly,
    because most of the space is transient.  This is the P1 problem the tier
    exists to fix, not a violation."""
    assert 90 not in rules.UNITARY_RULES
    r = wcc.weak_components(90, 8, "obc0")
    rec = results_io.load_results(90, "obc0").get(8)
    assert r.sizes_wcc == [112, 112, 16, 16]
    assert results_io.sizes_from_record(rec, "sizes_recurrent") == [7, 7, 1, 1]
    assert sum(r.sizes_wcc) == 256                     # sectors partition
    assert sum([7, 7, 1, 1]) == 16                     # attractors do not
    wcc.check_against_scc(r, rec)                      # A3/A4 still hold


@pytest.mark.parametrize("rule", [22, 28, 76, 108, 156, 201, 232])
def test_A3_A4_against_archive(rule):
    recs = results_io.load_results(rule, "obc0")
    for N in (8, 10):
        rec = recs.get(N)
        if not rec or rec.get("ergodic_flag"):
            continue
        r = wcc.weak_components(rule, N, "obc0")
        out = wcc.check_against_scc(r, rec)
        assert out["a3"] is True             # n_wcc <= n_scc
        assert out["a4"] is True             # every WCC holds >=1 terminal SCC


def test_A4_is_not_symmetric():
    """n_recurrent can exceed n_wcc by a lot -- that is the whole point, and
    it is why the task forbids asserting the reverse inequality."""
    rec = results_io.load_results(22, "pbc").get(10)
    if rec and not rec.get("ergodic_flag"):
        r = wcc.weak_components(22, 10, "pbc")
        assert rec["n_recurrent"] > r.n_wcc


# --- the shared code path (task sec.2) ---------------------------------------

def test_scc_sectors_union_find_delegates_to_wcc():
    N, rule = 9, 150
    t = rules.wolfram_to_tuple(rule)
    sizes, erg, mx = scc.sectors_union_find(N, scc._make_succ(rule, N, "obc0", t),
                                            detect_ergodic=False)
    r = wcc.weak_components(rule, N, "obc0")
    assert sizes == r.sizes_wcc
    assert mx == r.d_max_wcc


def test_edges_are_symmetrised():
    """union(x, y) for every y in succ(x): a rule whose successor relation is
    strictly one-way must still land in ONE component."""
    seen = {0: [1], 1: [2], 2: [2], 3: [3]}       # 0->1->2 chain, 3 isolated

    def succ(x):
        return seen[x]

    sizes, erg, mx = wcc.union_find_components(2, succ, detect_ergodic=False)
    assert sorted(sizes, reverse=True) == [3, 1]


# --- rule 150's family (R9 sec.6) --------------------------------------------

_R150_FAMILY = [134, 142, 148, 158, 212, 214]      # dissipative, same sectors
_R150_EXCEPTIONS = [132, 222]                      # IDDI and IEEI differ


@pytest.mark.parametrize("rule", _R150_FAMILY)
@pytest.mark.parametrize("N", [8, 10, 12])
def test_dissipative_rules_share_rule_150_sectors(rule, N):
    """The domain-wall charge is not a property of unitarity: replacing either
    Hadamard of IVVI by a reset leaves the sector multiset untouched."""
    ref = wcc.weak_components(150, N, "obc0").sizes_wcc
    got = wcc.weak_components(rule, N, "obc0").sizes_wcc
    assert got == ref
    assert ref == sorted((comb(N + 1, w) for w in range(0, N + 2, 2)),
                         reverse=True)


@pytest.mark.parametrize("rule", _R150_EXCEPTIONS)
def test_same_reset_type_pair_breaks_the_wall_charge(rule):
    """IDDI and IEEI -- both active symbols the same reset type -- do NOT."""
    N = 12
    ref = wcc.weak_components(150, N, "obc0").sizes_wcc
    got = wcc.weak_components(rule, N, "obc0").sizes_wcc
    assert got != ref
    assert sum(got) == 2 ** N            # still a partition, of course


def test_i_pattern_does_not_determine_the_sector_partition():
    """
    The tempting generalisation -- V, D and E all generate the same undirected
    flip edge, so only the I-pattern should matter -- is false, because the
    odd-layer symbol is read off the post-even-layer state.  IDDI vs IVVI is a
    counterexample within one I-pattern group.
    """
    N = 10
    pat = lambda r: tuple(s == "I" for s in rules.wolfram_to_tuple(r))
    assert pat(132) == pat(150)
    assert (wcc.weak_components(132, N, "obc0").sizes_wcc
            != wcc.weak_components(150, N, "obc0").sizes_wcc)


# --- the finite-N form of the exclusion curve --------------------------------

@pytest.mark.parametrize("rule", [0, 22, 28, 36, 51, 74, 90, 132, 150, 156,
                                  165, 173, 204, 222])
def test_finite_hyperbola_holds(rule):
    """n_wcc(N) * D_max(N) >= 2^N, exactly, with no fitting -- the actual
    content of the exclusion curve (R9 eq. finite)."""
    for N in (7, 9, 11):
        r = wcc.weak_components(rule, N, "obc0")
        assert r.n_wcc * r.d_max_wcc >= (1 << N), (rule, N)


def test_finite_hyperbola_is_tight_for_the_identity_rule():
    """Rule 204 saturates it: 2^N sectors of size 1."""
    for N in (7, 9, 11):
        r = wcc.weak_components(204, N, "obc0")
        assert r.n_wcc * r.d_max_wcc == (1 << N)


# --- R2 sec.3: the rate must not come from the M2 fit ------------------------

_PHI_FAMILY = [140, 156, 196, 198, 206, 220]
_R2_DMAX = [6, 9, 12, 16, 20, 27, 36, 48, 64, 81, 108]      # N = 6..16


@pytest.mark.parametrize("rule", _PHI_FAMILY)
def test_phi_family_shares_the_r2_dmax_series(rule):
    """All six carry the series R2 sec.3 derives the base of, so the
    room-packing derivation covers all of them."""
    from qca_fragmentation.scaling import sectors
    s = sectors.load_series(rule, "obc0", 16)
    if len(s["N"]) < len(_R2_DMAX):
        pytest.skip("sweep has not reached N=16 for this rule")
    assert s["d_max_wcc"] == _R2_DMAX


def test_m2_rate_is_biased_and_is_not_used():
    """
    R2 sec.3's "purely numerical trap": M2's alpha*ln N absorbs growth, so its
    kappa understates the rate.  The descriptor must not report it.
    """
    from qca_fragmentation.scaling import sectors
    from qca_fragmentation.scaling.fits import fit_series
    s = sectors.load_series(156, "obc0", 16)
    if len(s["N"]) < len(_R2_DMAX):
        pytest.skip("sweep has not reached N=16")
    m2 = fit_series(s["N"], s["d_max_wcc"])["base"]
    assert m2 < 1.28                      # the biased estimate
    d = sectors.series_descriptor(156, "obc0", "d_max_wcc",
                                  s["N"], s["d_max_wcc"])
    assert d["base"] == pytest.approx(4 ** 0.2, abs=1e-9)
    assert "R2" in d["source"]


def test_rule_156_sits_at_phi_times_4_to_the_fifth():
    """V3: the anchor that must lie ABOVE the curve, at phi * 4^(1/5)."""
    from qca_fragmentation.scaling import sectors
    s = sectors.load_series(156, "obc0", 16)
    if len(s["N"]) < len(_R2_DMAX):
        pytest.skip("sweep has not reached N=16")
    p = sectors.rule_point(156, "obc0", 16)
    assert p["n_wcc"]["base"] == pytest.approx(PHI, abs=1e-9)
    assert p["n_wcc"]["exact"]
    assert p["d_max_wcc"]["base"] == pytest.approx(4 ** 0.2, abs=1e-9)
    assert p["product_ab"] == pytest.approx(PHI * 4 ** 0.2, abs=1e-9)
    assert sectors.hyperbola_check(p)["verdict"] == "above"


def test_binomial_rules_get_base_two_not_the_undershoot():
    """The mirror bias: the two-parameter fit has no alpha to absorb the
    N^{-1/2} prefactor and reports ~1.92 for W134, whose D_max is exactly rule
    150's central binomials.  The power-law-prefactor test must recover 2."""
    from qca_fragmentation.scaling import sectors
    for rule in (134, 142, 148, 158, 212, 214):
        s = sectors.load_series(rule, "obc0", 16)
        if len(s["N"]) < 8:
            pytest.skip("sweep too short")
        d = sectors.series_descriptor(rule, "obc0", "d_max_wcc",
                                      s["N"], s["d_max_wcc"])
        assert d["base"] == pytest.approx(2.0, abs=1e-9), (rule, d)
        assert d["alpha"] < 0.0           # a decaying prefactor


# --- record schema ------------------------------------------------------------

def test_wcc_record_roundtrip_and_derived_fields():
    r = wcc.weak_components(150, 10, "obc0")
    scc_rec = results_io.load_results(150, "obc0").get(10)
    rec = results_io.record_from_wcc_result(r, scc_rec)
    assert set(results_io.WCC_FIELDS) >= set(rec)
    assert rec["n_wcc"] == 6 and rec["d_max_wcc"] == 462
    assert rec["d_max_ratio"] == pytest.approx(462 / 1024)
    assert results_io.sizes_from_wcc_record(rec) == r.sizes_wcc
    if scc_rec and not scc_rec.get("ergodic_flag"):
        assert rec["transient_fraction"] == pytest.approx(0.0)
        assert rec["att_per_sector"] == pytest.approx(1.0)


def test_size_histogram_reconstructs_a_truncated_multiset():
    """rule 204 at N=12 has 4096 singleton sectors, twice the 2048 cap."""
    r = wcc.weak_components(204, 12, "obc0")
    rec = results_io.record_from_wcc_result(r, None)
    assert rec["wcc_truncated"] is True
    assert len(rec["sizes_wcc"]) == 2048
    full = results_io.sizes_from_wcc_record(rec)
    assert len(full) == 4096 and sum(full) == 4096


# --- coherent correspondent of a dissipative rule (R9 sec.6) -----------------

def test_coherent_part_switches_resets_off():
    assert rules.coherent_part(("I", "V", "D", "E")) == ("I", "V", "I", "I")
    assert rules.coherent_parent(134) == 198          # IVDI -> IVII
    assert rules.coherent_parent(148) == 156          # IDVI -> IIVI
    for r in range(256):
        t = rules.coherent_part(rules.wolfram_to_tuple(r))
        assert set(t) <= {"I", "V"}
        assert rules.is_unitary(t)


def test_clusters_partition_the_v_plus_reset_family():
    """A parent with v Hadamards has exactly 3^(4-v)-1 children, and the 14
    non-empty clusters cover all 160 V+reset rules exactly once."""
    mixed = [r for r in range(256)
             if not rules.is_unitary(rules.wolfram_to_tuple(r))
             and "V" in rules.wolfram_to_tuple(r)]
    assert len(mixed) == 160
    seen, nonempty = [], 0
    for p in rules.UNITARY_RULES:
        kids = rules.dissipative_children(p)
        v = sum(1 for s in rules.wolfram_to_tuple(p) if s == "V")
        assert len(kids) == (3 ** (4 - v) - 1 if v >= 1 else 0), (p, v)
        if kids:
            nonempty += 1
        seen.extend(kids)
    assert nonempty == 14
    assert sorted(seen) == sorted(mixed)              # a partition, exactly
    assert rules.dissipative_children(51) == []       # VVVV has no I to spoil


def test_dissipation_destroys_sector_structure_without_exception():
    """
    R9 sec.6: of the six parents with non-trivial sector structure, not one of
    their 120 children keeps the parent's position in the sector plane.
    """
    from qca_fragmentation.scaling import sectors
    from qca_fragmentation.scaling.sector_figure import _plane_points
    d = sectors.load("obc0")
    if d is None:
        pytest.skip("sector map not built")
    xy = _plane_points(d, "sector")
    n_kids = n_keep = 0
    for p in rules.UNITARY_RULES:
        if p not in xy:
            continue
        pa, pb = xy[p]
        if abs(pa - 1) < 1e-6 and abs(pb - 2) < 1e-6:
            continue                                  # nothing to destroy
        for c in rules.dissipative_children(p):
            if c not in xy:
                continue
            n_kids += 1
            if abs(xy[c][0] - pa) < 1e-6 and abs(xy[c][1] - pb) < 1e-6:
                n_keep += 1
    assert n_kids >= 100
    assert n_keep == 0, f"{n_keep} of {n_kids} children kept their parent's point"


# --- fragmentation surviving dissipation (R9 sec.6.2) ------------------------

_OPEN_FRAG = {73, 109, 28, 70, 157, 199, 71, 29}
PLASTIC_ = 1.324717957244746
SUPERGOLDEN = 1.4655712318767682


def test_open_system_fragmented_set():
    """
    The V+reset rules whose sector count still grows exponentially.  A sector is
    an enclosure, so this is fragmentation of the OPEN system, exact for the
    unmonitored channel too -- not a unitary circuit with noise bolted on.
    """
    from qca_fragmentation.scaling import sectors
    d = sectors.load("obc0")
    if d is None:
        pytest.skip("sector map not built")
    frag = sectors.open_system_fragmented(d)
    assert {r["rule"] for r in frag} == _OPEN_FRAG
    for r in frag:
        t = rules.wolfram_to_tuple(r["rule"])
        assert "V" in t                                  # still quantum
        assert rules.channel_kraus_symbols(t)            # genuinely dissipative
        assert r["a"] > 1.02
        assert r["product"] > 2.0 - 1e-9                 # still above the curve
        # inherited: the coherent correspondent is itself fragmented
        assert r["parent"] in (156, 198, 108, 201)


def test_open_fragmented_bases_are_exact_algebraic_numbers():
    """Six of the eight have integer recurrences: the plastic number for
    28/70/157/199 and the supergolden ratio for 73/109."""
    from qca_fragmentation.scaling import sectors
    d = sectors.load("obc0")
    if d is None:
        pytest.skip("sector map not built")
    by = {r["rule"]: r for r in sectors.open_system_fragmented(d)}
    for rule in (28, 70, 157, 199):
        assert by[rule]["a"] == pytest.approx(PLASTIC_, abs=1e-9), rule
        assert by[rule]["a_exact"], rule
    for rule in (73, 109):
        assert by[rule]["a"] == pytest.approx(SUPERGOLDEN, abs=1e-9), rule
        assert by[rule]["a_exact"], rule
    # The b values.  70/157/199 all have D_max = 2*3^(N/2-1) on even N, so
    # b = sqrt(3) exactly -- 70 used to be reported at 2.0, which the corrected
    # volume-fraction guard (R9 sec.6.3, 2026-07-31) removed along with seven
    # other spurious base-2 descriptors.
    assert by[70]["b"] == pytest.approx(3 ** 0.5, abs=1e-9)
    assert by[157]["b"] == pytest.approx(3 ** 0.5, abs=1e-9)
    assert by[199]["b"] == pytest.approx(3 ** 0.5, abs=1e-9)
    # 28 is the one case where the two parities do not share a base: its even
    # branch settles to a ratio of exactly 3 per two sites (sqrt 3) and its odd
    # branch to 1 + sqrt 3, so the single fitted number lies between them.
    assert 3 ** 0.5 > by[28]["b"] > (1 + 3 ** 0.5) ** 0.5


def test_supergolden_and_plastic_are_the_roots_they_should_be():
    assert SUPERGOLDEN ** 3 == pytest.approx(SUPERGOLDEN ** 2 + 1, abs=1e-12)
    assert PLASTIC_ ** 3 == pytest.approx(PLASTIC_ + 1, abs=1e-12)


# --- growth class, not base: what survives for 150 and 105 (R9 sec.6.3) ------

def test_base_plane_cannot_see_polynomial_sector_structure():
    """Rules 51 and 150 both sit at (a,b) = (1,2), but 51 has ONE sector and
    150 has floor((N+1)/2)+1 of them.  The class distinguishes them."""
    from qca_fragmentation.scaling import sectors
    d = sectors.load("obc0")
    if d is None:
        pytest.skip("sector map not built")
    for r in (51, 150):
        p = next(x for x in d["points"] if x["rule"] == r)
        assert p["n_wcc"]["base"] == pytest.approx(1.0)
        assert p["d_max_wcc"]["base"] == pytest.approx(2.0)
    assert sectors.sector_growth_class(51, "obc0", d) == "constant"
    assert sectors.sector_growth_class(150, "obc0", d) == "polynomial"
    s = sectors.load_series(150, "obc0", 16)
    assert s["n_wcc"] == [(N + 1) // 2 + 1 for N in s["N"]]


def test_nothing_survives_for_the_polynomial_parents():
    """R9 sec.6.3: every child of a polynomial-sector parent drops to a
    CONSTANT sector count -- 0 of 32, so polynomial structure is more fragile
    than exponential structure, where 22 of 104 keep some growth."""
    from qca_fragmentation.scaling import sectors
    d = sectors.load("obc0")
    if d is None:
        pytest.skip("sector map not built")
    r = sectors.survival_by_parent_class(d, "obc0")
    poly = r["by_parent_class"]["polynomial"]
    assert set(poly["parents"]) == {60, 102, 105, 150}
    assert poly["children"] == 32
    assert poly["exponential"] == 0 and poly["polynomial"] == 0
    assert poly["constant"] == 32
    exp = r["by_parent_class"]["exponential"]
    assert exp["exponential"] == 8 and exp["polynomial"] == 14


def test_the_wall_charge_is_a_sink_reachable_from_the_exponential_rules():
    """Six children of 156/198 land on rule 150's EXACT sector series, so the
    wall charge is reachable by dissipation from above even though it has no
    dissipative descendants of its own."""
    from qca_fragmentation.scaling import sectors
    d = sectors.load("obc0")
    if d is None:
        pytest.skip("sector map not built")
    ref = sectors.load_series(150, "obc0", 16)["n_wcc"]
    hit = {c for c in (20, 148, 158, 6, 134, 214)
           if sectors.load_series(c, "obc0", 16)["n_wcc"] == ref}
    assert hit == {20, 148, 158, 6, 134, 214}
    for c in hit:
        assert rules.coherent_parent(c) in (156, 198)


def test_rule_105_is_not_rule_150_at_N_1_mod_4():
    """105 = VIIV fires when the neighbours AGREE; at N = 1 (mod 4) its sectors
    carry ODD wall number, so the multiset differs from 150's."""
    for N in (9, 13):
        s150 = results_io.sizes_from_wcc_record(
            results_io.load_wcc_results(150, "obc0")[N])
        s105 = results_io.sizes_from_wcc_record(
            results_io.load_wcc_results(105, "obc0")[N])
        even = sorted((comb(N + 1, w) for w in range(0, N + 2, 2)), reverse=True)
        odd = sorted((comb(N + 1, w) for w in range(1, N + 2, 2)), reverse=True)
        assert s150 == even, N
        assert s105 == odd, N
        assert s150 != s105
    for N in (8, 12):                     # and they agree elsewhere
        assert (results_io.sizes_from_wcc_record(
                    results_io.load_wcc_results(150, "obc0")[N])
                == results_io.sizes_from_wcc_record(
                    results_io.load_wcc_results(105, "obc0")[N]))


# --- the pinned-frontier family (R9 sec.6.4) ---------------------------------

def _components(rule, N, bc="obc0"):
    """{root: [members]} of the WCC partition."""
    from array import array
    succ = wcc.make_succ(rule, N, bc)
    parent = array("q", range(1 << N))
    size = array("q", [1]) * (1 << N)

    def find(a):
        r = a
        while parent[r] != r:
            r = parent[r]
        while parent[a] != r:
            parent[a], a = r, parent[a]
        return r

    for x in range(1 << N):
        rx = find(x)
        for y in succ(x):
            ry = find(y)
            if rx != ry:
                if size[rx] < size[ry]:
                    rx, ry = ry, rx
                parent[ry] = rx
                size[rx] += size[ry]
    out = {}
    for x in range(1 << N):
        out.setdefault(find(x), []).append(x)
    return out


@pytest.mark.parametrize("rule,which", [(110, "high"), (230, "high"),
                                        (124, "low"), (188, "low"),
                                        (44, "low")])
def test_frontier_charge_is_the_extremal_excitation(rule, which):
    """Every sector is a level set of the index of the outermost 1."""
    N = 9
    key = ((lambda x: x.bit_length() - 1 if x else -1) if which == "high"
           else (lambda x: (x & -x).bit_length() - 1 if x else -1))
    for mem in _components(rule, N).values():
        assert len({key(v) for v in mem}) == 1, (rule, mem[:6])


@pytest.mark.parametrize("rule", [110, 124, 188, 230])
def test_frontier_rules_have_N_plus_1_sectors_and_dyadic_sizes(rule):
    """n_wcc = N+1, sizes 2^{N-1},...,2,1,1, D_max = 2^{N-1} exactly."""
    for N in (8, 10, 12):
        r = wcc.weak_components(rule, N, "obc0")
        assert r.n_wcc == N + 1, (rule, N)
        assert r.d_max_wcc == 2 ** (N - 1), (rule, N)
        assert r.sizes_wcc == [2 ** k for k in range(N - 1, -1, -1)] + [1]
        assert sum(r.sizes_wcc) == 2 ** N


def test_frontier_and_wall_charge_are_different_mechanisms():
    """Both sit at (a,b) = (1,2) but differ in every structural respect."""
    N = 12
    front = wcc.weak_components(110, N, "obc0")
    wall = wcc.weak_components(150, N, "obc0")
    assert front.n_wcc == N + 1 == 13
    assert wall.n_wcc == (N + 1) // 2 + 1 == 7
    assert front.d_max_wcc == 2 ** (N - 1)                     # half the space
    assert wall.d_max_wcc == comb(N + 1, 6)                    # binomial
    assert front.d_max_wcc / 2 ** N == pytest.approx(0.5)
    assert wall.d_max_wcc / 2 ** N < 0.45                      # shrinking
    assert front.sizes_wcc != wall.sizes_wcc


# --- boundary robustness -----------------------------------------------------
# NOT part of R9, which is strictly obc0, and NOT the pbc sweep.  These three
# tests pin a handful of pbc values at small N because they answer a question
# R9 sec.6.4 raises and cannot settle: the frontier charge is defined by
# reference to an end of the chain, so it should vanish on a ring, while the
# wall charge should not.  Kept here so the fact is not lost before the pbc
# report is written.

@pytest.mark.parametrize("rule", [44, 60, 100, 102, 110, 124, 188, 230])
def test_frontier_family_is_an_obc0_artefact(rule):
    """
    The pinned-frontier charge is the position of the extremal excitation, which
    is not definable on a ring -- so the whole N+1 dyadic tower must collapse at
    pbc, and it does: 2 sectors at every N.  Includes the two UNITARY members,
    60 and 102, which carry the same charge without any reset.
    """
    for N in (7, 9, 11):
        assert wcc.weak_components(rule, N, "obc0").n_wcc == N + 1 or rule in (44, 100)
        assert wcc.weak_components(rule, N, "pbc").n_wcc == 2, (rule, N)


@pytest.mark.parametrize("rule", [28, 70, 157, 199, 73, 109, 29, 71])
def test_open_system_fragmentation_survives_the_ring(rule):
    """The eight exponentially fragmented rules are BULK: their sector counts
    still grow on the ring, unlike the frontier family."""
    ys = [wcc.weak_components(rule, N, "pbc").n_wcc for N in (7, 9, 11, 13)]
    assert len(set(ys)) > 1
    assert ys[-1] > 2 * ys[0], (rule, ys)          # genuinely growing


def test_wall_charge_survives_the_ring_with_r8_closed_form():
    """Rule 150's pbc sector count is R8's formula: N/2+3 (even), (N+3)/2 (odd)."""
    for N in range(6, 14):
        want = N // 2 + 3 if N % 2 == 0 else (N + 3) // 2
        assert wcc.weak_components(150, N, "pbc").n_wcc == want, N


# --- which constant describes what (R9 sec.6.2a) ------------------------------

def test_phi_plays_different_roles_in_the_two_parent_pairs():
    """
    Why "phi-fragmented" is the wrong label for the four structured parents:
    for 156/198 phi is the base of the sector COUNT (Fibonacci-many sectors),
    for 108/201 it is the base of the largest sector SIZE, while their count
    grows as the root of x^3 = 2x^2 - x + 1.
    """
    from qca_fragmentation.scaling import sectors
    for rule in (156, 198):
        s = sectors.load_series(rule, "obc0", 16)
        rn = find_integer_recurrence(s["n_wcc"])
        assert rn["ok"] and rn["coeffs"] == [1, 1]           # Fibonacci count
        assert rn["base"] == pytest.approx(PHI, abs=1e-9)
        # and D_max is NOT a recurrence: its base comes from theory (R2 sec.3)
        assert not find_integer_recurrence(s["d_max_wcc"]).get("ok")
    for rule in (108, 201):
        s = sectors.load_series(rule, "obc0", 16)
        rn = find_integer_recurrence(s["n_wcc"])
        assert rn["ok"] and rn["coeffs"] == [2, -1, 1]
        assert rn["base"] == pytest.approx(1.7548776662, abs=1e-6)
        rd = find_integer_recurrence(s["d_max_wcc"])
        assert rd["ok"] and rd["coeffs"] == [1, 1]           # Fibonacci SIZE
        assert rd["base"] == pytest.approx(PHI, abs=1e-9)


def test_the_degree_raising_pattern_holds_only_for_the_fibonacci_count_pair():
    """
    156/198 count by x^2 = x+1 and their rho children by x^3 = x+1: degree up.
    201/108 already count by a cubic, and their psi children stay cubic
    (x^3 = x^2+1): degree unchanged.  So no single statement covers all eight.
    """
    assert rules.coherent_parent(28) == 156
    assert rules.coherent_parent(70) == 198
    assert rules.coherent_parent(73) == 201
    assert rules.coherent_parent(109) == 108
    # rho and psi are both cubic; phi is quadratic
    assert PLASTIC_ ** 3 == pytest.approx(PLASTIC_ + 1, abs=1e-12)
    assert SUPERGOLDEN ** 3 == pytest.approx(SUPERGOLDEN ** 2 + 1, abs=1e-12)
    assert PHI ** 2 == pytest.approx(PHI + 1, abs=1e-12)


# --- the corner view and the recurrent/transient map (R9 sec.6) ---------------

def test_attractor_point_agrees_with_attractor_deficit():
    """The corner view must not use a different convention from the map it
    accompanies: same window, same descriptors, same bases."""
    from qca_fragmentation.scaling import sectors
    for rule in (22, 28, 150, 156, 201, 204):
        p = sectors.attractor_point(rule, "obc0")
        d = sectors.attractor_deficit(rule, "obc0", sectors.UNIFORM_N_CAP)
        if p is None or d is None:
            continue
        if p["n_recurrent"]["base"] is None or p["d_max"]["base"] is None:
            continue
        assert p["n_recurrent"]["base"] == pytest.approx(d["a_att"])
        assert p["d_max"]["base"] == pytest.approx(d["b_att"])


def test_unitary_rules_have_no_transient_at_all():
    """|Rec| = 2^N exactly for a unitary rule, so the recurrent-mass base is 2
    and the recurrent fraction is 1.  This anchors the top of F9."""
    from qca_fragmentation.scaling import sectors
    from qca_fragmentation.core import rules as R
    seen = 0
    for rule in R.UNITARY_RULES:
        m = sectors.recurrent_mass(rule, "obc0")
        if m is None or m["mass"]["base"] is None:
            continue
        seen += 1
        assert m["mass"]["base"] == pytest.approx(2.0, abs=1e-9), rule
        assert m["recurrent_fraction"] == pytest.approx(1.0, abs=1e-12), rule
    assert seen >= 5


def test_no_dissipative_rule_keeps_a_finite_recurrent_fraction():
    """R9 sec.6.2: one reset is enough to make the recurrent set an
    exponentially vanishing share of the basis."""
    from qca_fragmentation.scaling import sectors
    from qca_fragmentation.core import rules as R
    bad = []
    for rule in range(256):
        if R.is_unitary(R.wolfram_to_tuple(rule)):
            continue
        m = sectors.recurrent_mass(rule, "obc0")
        if m is None or m["mass"]["base"] is None:
            continue
        if m["mass"]["base"] > 2.0 - 0.02:
            bad.append(rule)
    assert bad == [], bad


def test_alpha_and_base_describe_the_same_model():
    """R9 sec.6.5.  The descriptor's (base, alpha) is a PAIR: y ~ c N^alpha b^N.
    Every branch that overrides the base after M1/M2 has been fitted must
    re-derive alpha against the base it is actually reporting, or the two halves
    describe different models.  Rule 28's D_max is the type case: the series
    4,4,8,8,8,16,16,16,32,32,32 is exactly c*2^{N/3} with no prefactor, and the
    stale alpha reported it as 2^{1/3} with alpha = 2.34."""
    from qca_fragmentation.scaling import sectors
    bad = []
    for rule in range(256):
        p = sectors.attractor_point(rule, "obc0")
        if p is None:
            continue
        for key in ("n_recurrent", "d_max"):
            d = p[key]
            # M0/M1 fit at base 1 and are coherent as they stand; the analytic,
            # saturated and volume-fraction branches supply their own pair.
            if d["base"] is None or d["alpha_source"] in (
                    None, "analytic", "saturated", "M0", "M1 fit",
                    "volume-fraction discriminator"):
                continue
            s = sectors.load_series(rule, "obc0", sectors.UNIFORM_N_CAP)
            # alpha must be the one a fit at THIS base would give
            from qca_fragmentation.scaling.dissipative import load_series as ls
            t = ls(rule, "obc0")
            Ns = [N for N in t["N"] if N <= sectors.UNIFORM_N_CAP]
            ys = (t["n_recurrent"] if key == "n_recurrent"
                  else t["d_max"])[:len(Ns)]
            if len(Ns) < 3 or min(ys) <= 0:
                continue
            want, _ = sectors._alpha_at_base(Ns, ys, d["base"])
            if abs(want - d["alpha"]) > 1e-6:
                bad.append((rule, key, d["alpha"], want, d["alpha_source"]))
    assert bad == [], bad[:8]


def test_rule_28_dmax_is_a_clean_staircase_with_no_prefactor():
    """The concrete number that the incoherent pair got wrong."""
    from qca_fragmentation.scaling import sectors
    d = sectors.attractor_point(28, "obc0")["d_max"]
    assert d["base"] == pytest.approx(2 ** (1 / 3), rel=1e-6)
    assert abs(d["alpha"]) < 0.1, d["alpha"]      # was 2.339


def test_sector_figure_exposes_every_figure_it_draws():
    """The __main__ guard once sat mid-module, so `python -m sector_figure`
    raised NameError on fig_corner and silently left F7/F8/F9 stale."""
    from qca_fragmentation.scaling import sector_figure as sf
    for name in ("fig_corner", "fig_recurrent_transient", "fig_sector_map",
                 "fig_dissipation_clusters"):
        assert hasattr(sf, name), name
    import re
    src = open(sf.__file__).read()
    guard = src.index('if __name__ == "__main__"')
    # every figure function, not just the ones that existed when this was
    # written -- appending a new fig_* to the end of the module re-creates the
    # bug otherwise, which is exactly what happened once already.
    last = max(m.start() for m in re.finditer(r"^def fig_", src, re.M))
    assert guard > last, \
        "the __main__ guard must come after every fig_* definition"


def _mass_rows():
    from qca_fragmentation.scaling import sectors
    rows = [sectors.recurrent_mass(r, "obc0") for r in range(256)]
    return [r for r in rows if r and r["mass"]["base"] is not None
            and r["count"]["base"] is not None]


def test_mass_base_medians_are_an_artefact_of_the_atoms():
    """R9 sec.6.2 warning.  The fitted bases are algebraic and pile up on a few
    values, so the family medians land on named constants by rank accident and
    sit one handful of rules away from flipping.  Pin both facts, so that nobody
    reads a gap of 0.47 into 1 vs psi again."""
    rows = _mass_rows()
    for family, n_at_one, total in (("mixed", 68, 155), ("classical", 40, 78)):
        b = sorted(r["mass"]["base"] for r in rows if r["family"] == family)
        assert len(b) == total, (family, len(b))
        assert sum(1 for x in b if x < 1.0001) == n_at_one, family
        # the atom at 1 is within a handful of rules of carrying the median
        assert abs(n_at_one - total / 2) <= 10, family
    # ... and psi, the "median" V+reset base, is a six-rule block
    b = [r["mass"]["base"] for r in rows if r["family"] == "mixed"]
    assert sum(1 for x in b if abs(x - 1.465571) < 1e-4) == 6


def test_the_extreme_contractors_are_the_memoryless_rules():
    """R9 sec.6.2: the V-free family's low recurrent mass is carried by the
    all-reset rules, which keep no memory and collapse to a single state."""
    from qca_fragmentation.core import rules as R
    rows = [r for r in _mass_rows() if r["family"] == "classical"]
    reset = [r for r in rows
             if all(s in "DE" for s in R.wolfram_to_tuple(r["rule"]))]
    rest = [r for r in rows if r not in reset]
    assert len(reset) == 14 and len(rest) == 64
    # all-reset: bounded recurrent set, and the median is a lone fixed point
    assert sum(1 for r in reset if r["mass"]["base"] < 1.0001) == 12
    med = sorted(r["recurrent_fraction"] for r in reset)[len(reset) // 2]
    assert med == pytest.approx(2.0 ** -16, rel=1e-6)
    # the rest of the family is nowhere near that floor
    med_rest = sorted(r["recurrent_fraction"] for r in rest)[len(rest) // 2]
    assert med_rest > 50 * med


def test_highlight_rings_land_on_the_markers_they_ring():
    """R9 F1.  _scatter_cells nudges each family sideways by _FAM_DX so the three
    families do not hide under one another; _highlight did not, so every gold
    ring sat 0.019 to the left of its own dot -- the ring at W29's quoted
    (1.2147, 1.8059) was empty and W29's marker was inside W71's ring.  Reported
    from the report's own coordinates, 2026-08-04."""
    from qca_fragmentation.scaling import sectors, sector_figure as sf
    d = sectors.load("obc0") or sectors.build("obc0")
    fam = {p["rule"]: p["family"] for p in d["points"]}
    xy = {p["rule"]: (p["n_wcc"]["base"], p["d_max_wcc"]["base"])
          for p in d["points"] if p["n_wcc"]["base"] is not None}

    class _Rec:
        def __init__(self):
            self.pts = []

        def scatter(self, x, y, **kw):
            self.pts.append((round(x, 6), round(y, 6)))

        def annotate(self, *a, **kw):
            pass

    ax = _Rec()
    frag = sf._highlight(ax, d, xy, fam=fam)
    assert len(frag) == 8
    # every ring must coincide with the nudged marker position
    want = {(round(xy[r["rule"]][0] + sf._FAM_DX[fam[r["rule"]]], 6),
             round(xy[r["rule"]][1], 6)) for r in frag}
    assert set(ax.pts) == want
    # and the nudge is real, so the ring is NOT at the raw coordinate
    raw = {(round(xy[r["rule"]][0], 6), round(xy[r["rule"]][1], 6))
           for r in frag}
    assert set(ax.pts).isdisjoint(raw), "all eight are mixed, so all shift"
