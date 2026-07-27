"""
Level statistics of C150 (R8 sec.7): the pipeline, the generated reference
ensembles, and the structural facts the interpretation rests on.
"""

import numpy as np
import pytest

from qca_fragmentation.quantum import rule150_levels as lv
from qca_fragmentation.quantum import rule150_spectra as rs


# --- pipeline ----------------------------------------------------------------

def test_distinct_upper_phases_collapses_and_halves():
    """Conjugate pairs are folded to the upper half circle, +-1 dropped, exact
    degeneracies collapsed."""
    th = np.array([0.3, 0.3, 0.3, 1.0, 2.0])
    ev = np.concatenate([np.exp(1j * th), np.exp(-1j * th),
                         [1.0, 1.0, -1.0]])          # +1 twice, -1 once
    ph, diag = lv.distinct_upper_phases(ev)
    assert np.allclose(ph, [0.3, 1.0, 2.0])
    # collapsed: 2 extra at -0.3, 2 extra at +0.3, 1 extra at phase 0
    assert diag["n_collapsed"] == 5
    assert diag["n_upper"] == 3


def test_r_tilde_is_in_range_and_scale_free():
    rng = np.random.default_rng(3)
    ph = np.sort(rng.uniform(0, np.pi, 4000))
    rt, n = lv.r_tilde(ph)
    assert n == 3998                       # 4000 phases -> 3999 gaps -> 3998 ratios
    assert 0.0 < rt <= 1.0
    # r~ is a ratio statistic: rescaling every phase cannot change it
    rt2, _ = lv.r_tilde(0.5 * ph)
    assert abs(rt - rt2) < 1e-12


def test_r_tilde_needs_at_least_three_spacings():
    assert np.isnan(lv.r_tilde(np.array([0.1, 0.2]))[0])


def test_unfolded_spacings_have_unit_mean():
    rng = np.random.default_rng(5)
    ph = np.sort(rng.uniform(0, np.pi, 800))
    s = lv.unfolded_spacings(ph)
    assert s.size == 799
    assert abs(s.mean() - 1.0) < 1e-12


# --- reference ensembles -----------------------------------------------------

def test_reference_ensembles_are_ordered_as_expected():
    """Poisson < superposed Haar-O < COE < Haar-O <= CUE, all through the same
    pipeline.  This is what makes the C150 numbers interpretable."""
    refs = lv.reference_table(D=300, samples=4, seed=11)
    m = {k: refs[k]["r_tilde_mean"] for k in refs}
    assert m["poisson"] < m["orth_x2"] < m["coe"] < m["orth"]
    assert m["orth"] <= m["cue"] + 0.03
    assert 0.36 < m["poisson"] < 0.41           # 2 ln 2 - 1 = 0.3863
    assert 0.50 < m["coe"] < 0.57
    assert 0.57 < m["cue"] < 0.65


def test_superposition_pushes_toward_poisson():
    refs = lv.reference_table(D=300, samples=4, seed=13)
    assert refs["orth_x3"]["r_tilde_mean"] < refs["orth_x2"]["r_tilde_mean"]
    assert refs["orth_x2"]["r_tilde_mean"] < refs["orth"]["r_tilde_mean"]


# --- structural facts the interpretation depends on --------------------------

@pytest.mark.parametrize("N", [5, 6, 7, 8])
def test_half_layers_are_symmetric_involutions(N):
    """U is a product of two involutions, which is why the spectrum comes in
    +-theta pairs and why Fix(U) decomposes as it does."""
    d = rs.involution_defect(N)
    for k in ("LA_involution", "LB_involution", "LA_symmetric", "LB_symmetric",
              "LA_inverts_U", "matches_engine"):
        assert d[k] < 1e-12, (N, k, d[k])


@pytest.mark.parametrize("N", [5, 6, 7, 8, 9])
def test_fix_decomposition_and_empty_minus_sector(N):
    d = rs.fix_pm_decomposition(N)
    assert d["adds_up"]
    assert d["dim_minus_minus"] == 0
    assert d["dim_fix_U"] == d["dim_plus_plus"] == 2 ** ((N + 1) // 2)


@pytest.mark.parametrize("N,w", [(9, 4), (11, 4), (11, 6)])
def test_particle_hole_partner_shells_share_a_spectrum(N, w):
    """For odd N the shells w and N+1-w have the same spectrum, so counting both
    would double count.  (Not a symmetry of U -- see R8 open items.)"""
    assert lv.particle_hole_partner_agrees(N, w) < 1e-12


def test_particle_hole_has_no_partner_for_even_N():
    """For even N, N+1 is odd, so w and N+1-w have opposite parities and the
    partner is not a sector at all."""
    from math import comb
    N, w = 12, 4
    assert (N + 1 - w) % 2 == 1
    assert comb(N + 1, w) == 715


# --- a real shell ------------------------------------------------------------

def test_shell_levels_reports_blocks_and_repulsion():
    """N=11, w=6 is at half filling and, once reflection is resolved, shows
    repulsion well above Poisson."""
    recs = lv.shell_levels(11, 6)
    assert [r["block"] for r in recs] == ["P=+1", "P=-1"]
    for r in recs:
        assert r["filling"] == pytest.approx(0.5)
        assert r["shell_dim"] == 924
        assert r["r_tilde"] > 0.48            # far above Poisson 0.386
        assert r["n_ratios"] > 50


def test_even_N_shell_has_a_single_block():
    recs = lv.shell_levels(10, 6)
    assert [r["block"] for r in recs] == ["all"]
    assert recs[0]["shell_dim"] == 462


def test_control_pipeline_runs_on_another_rule():
    """The methods control: same pipeline, a different unitary rule, small N."""
    recs = lv.control_rule_levels(57, 10)
    assert len(recs) == 1
    r = recs[0]
    assert r["rule"] == 57 and r["block_dim"] > 100
    assert 0.2 < r["r_tilde"] < 0.8


def test_ks_distances_cover_every_reference():
    refs = lv.reference_table(D=200, samples=3, seed=17)
    rng = np.random.default_rng(19)
    s = lv.unfolded_spacings(np.sort(rng.uniform(0, np.pi, 400)))
    ks = lv.ks_distances(s.tolist(), refs)
    assert set(ks) == set(lv.REFERENCES)
    # a Poisson sample should sit closer to Poisson than to CUE
    assert ks["poisson"] < ks["cue"]


# --- the distinct-eigenvalue law ---------------------------------------------

@pytest.mark.parametrize("N", range(4, 12))
def test_distinct_eigenvalue_closed_form(N):
    assert rs.distinct_spectrum_count(N) == rs.distinct_spectrum_closed_form(N)


def test_distinct_count_is_tolerance_independent():
    """There is no threshold to choose: at N=10 the gaps split into a population
    below 1e-12 and one above 1e-4 with nothing in between, so the count is the
    same over eight orders of magnitude of tolerance."""
    ph = np.sort(np.angle(np.linalg.eigvals(rs._full_unitary(10))))
    g = np.diff(ph)
    assert ((g >= 1e-12) & (g < 1e-4)).sum() == 0
    counts = {1 + int((g > t).sum()) for t in (1e-13, 1e-11, 1e-9, 1e-7, 1e-5)}
    assert counts == {rs.distinct_spectrum_closed_form(10)}


def test_distinct_fraction_decays_like_sqrt3_over_2():
    """(sqrt3/2)^N = 0.866^N: the spectrum is exponentially degenerate."""
    from math import sqrt
    for N in (8, 10, 12):
        frac = rs.distinct_spectrum_closed_form(N) / float(1 << N)
        assert abs(frac - (sqrt(3) / 2) ** N) < 0.02 * frac + 1e-3, N
    assert (rs.distinct_spectrum_closed_form(12) / 4096
            < rs.distinct_spectrum_closed_form(8) / 256)


@pytest.mark.parametrize("N", [8, 10])
def test_one_universal_spectrum_per_N(N):
    """Every shell's distinct spectrum is a subset of a single universal set of
    size 3^m, and the largest shell realises all of it.  This is why the level
    set is not a random point process of any kind (R8 sec.7.5)."""
    r = rs.universal_spectrum_check(N)
    assert r["universal_size"] == r["closed_form"]
    assert r["all_shells_are_subsets"]
    assert r["largest_shell_covers_all"]
    # the largest shell is the one realising D_max
    from qca_fragmentation import c150
    assert r["largest_shell_w"] == c150.d_max_argmax_obc0(N)
    # small shells realise only a fraction
    assert min(s["coverage"] for s in r["shells"]) < 0.5


def test_no_near_degeneracies_among_distinct_levels():
    """The sub-Poisson r~ of the biggest shells is not unresolved near-degeneracy:
    no gap among the distinct levels is below a hundredth of the median."""
    U, _ = rs.sector_unitary(12, 6)
    ph = np.sort(np.angle(np.linalg.eigvals(U)))
    keep = []
    for p in ph:
        if not keep or p - keep[-1] >= 1e-9:
            keep.append(p)
    g = np.diff(np.array(keep))
    assert g.min() / np.median(g) > 0.01
