"""
Tests for the dissipative growth-law fitter, especially the period splitting.

The interesting failure this guards against is a REAL one: before the period
generalisation, `fit_with_period` only knew about parity, so a series that
oscillates with period 3 or 4 in N was stamped "irregular" -- 32 of the 240 pbc
D_max series, of which 22 turned out to be perfectly regular once split at the
right period.  Rule 28 (IIVD) is the canonical case.
"""

import numpy as np
import pytest

from qca_fragmentation.scaling.dissipative import fit_with_period
from qca_fragmentation.scaling.fits import (find_integer_recurrence,
                                            find_recurrence_by_period)

NS = list(range(6, 18))          # the sizes the deep sweep actually reaches

# W28 IIVD, pbc: three interleaved doubling sequences, base 2^(1/3).
W28_DMAX = [4, 2, 4, 8, 4, 8, 16, 8, 16, 32, 16, 32]
# W11 VEDD, pbc: period 4, two arithmetic classes and two frozen at 2.
W11_DMAX = [9, 11, 2, 2, 15, 17, 2, 2, 21, 23, 2, 2]


def test_period_three_series_is_not_irregular():
    f = fit_with_period(NS, W28_DMAX)
    assert f["period"] == 3
    assert not f["irregular"]
    # base 2 per step of 3 => 2^(1/3) per site
    assert f["kappa_eff"] == pytest.approx(np.log(2) / 3, rel=1e-6)


def test_period_four_is_refused_when_the_sizes_cannot_settle_it():
    """W11 VEDD *looks* period-4 (9,15,21 / 11,17,23 / 2,2,2 / 2,2,2) but at
    N<=17 each class has 3 points against a 2-parameter line, and no exact
    recurrence certifies it either.  Claiming period 4 here would be exactly
    the overfitting the BIC margin exists to prevent, so it must NOT be split.
    """
    f = fit_with_period(NS, W11_DMAX)
    assert f["period"] == 1


def test_period_four_is_accepted_once_there_are_enough_sizes():
    """The same shape with 20 sizes has 5 points per class and is claimed."""
    ns = list(range(6, 26))
    ys = [(2 if n % 4 in (2, 3) else 9 + 6 * (n // 4) + (n % 4)) for n in ns]
    f = fit_with_period(ns, ys)
    assert f["period"] == 4


def test_an_exact_recurrence_can_license_a_period_bic_cannot_afford():
    """The recurrence route is stricter than any residual test (0/4000 on
    surrogates), so it is allowed to certify a period with small classes."""
    f = fit_with_period(NS, W28_DMAX)
    assert f["period"] == 3
    assert f["period_via"] in ("bic", "recurrence")


def test_a_clean_exponential_is_not_split():
    """Guard the other direction: splitting must not fire on a smooth series."""
    ys = [int(round(1.5 ** n)) for n in NS]
    f = fit_with_period(NS, ys)
    assert f["period"] == 1
    assert f["growth_eff"] == "exponential"
    # rounding to integers biases the slope slightly; 1% is plenty to pin 1.5
    assert f["kappa_eff"] == pytest.approx(np.log(1.5), rel=1e-2)


def test_a_period_two_series_prefers_two_over_four():
    """Every period-2 series is also period-4 consistent (4 refines 2), so the
    search must return the SMALLEST period that works, not the best-fitting."""
    ys = [(3 ** (n // 2) if n % 2 == 0 else 5 * 3 ** (n // 2)) for n in NS]
    f = fit_with_period(NS, ys)
    assert f["period"] == 2


def test_period_needs_enough_points_per_class():
    """With 6 sizes, period 4 leaves at most 2 points per class -- which fits
    any two points exactly -- so it must be refused."""
    ns = NS[:6]
    f = fit_with_period(ns, W11_DMAX[:6])
    assert f["period"] in (1, 2, 3)


def test_genuinely_ragged_series_stays_irregular():
    """W90 DEED: D_max is governed by the 2-adic structure of N, not by any
    growth law, and must not be rescued by a period split."""
    f = fit_with_period(NS, [1, 15, 1, 7, 3, 63, 2, 127, 7, 21, 1, 73])
    assert f["irregular"]
    assert f["kappa_eff"] is None


def test_recurrence_by_period_requires_all_classes_to_agree():
    """Two classes with different bases means there is no single growth base."""
    seq = [2 ** (n // 2) if n % 2 == 0 else 3 ** (n // 2) for n in NS]
    assert not find_recurrence_by_period(NS, seq, period=2)["ok"]


def test_recurrence_by_period_reports_the_per_site_base():
    """A recurrence over a step-3 subsequence has its base taken to the 1/3
    power, or a period-3 rule would report a base three times too large."""
    r = find_recurrence_by_period(NS, W28_DMAX, period=3)
    assert r["ok"] and r["period"] == 3
    assert r["base"] == pytest.approx(2 ** (1 / 3), rel=1e-9)


def test_null_false_split_rate_stays_low():
    """Calibration guard on _BIC_MARGIN / _MIN_PER_CLASS.

    Surrogates are smooth exponentials with 25% multiplicative noise -- no
    period structure whatsoever -- so every split is a false positive.  The
    ratio-of-residuals rule this replaced scored 21% here.  Kept as a test
    because the thresholds are the kind of thing that gets "tuned" later.
    """
    from qca_fragmentation.scaling.overfit_audit import null_period_rate
    r = null_period_rate(noise=0.25, trials=400, seed=7)
    assert r["false_split_rate"] < 0.08, r


def test_null_false_recurrence_rate_is_zero():
    """The exact-recurrence search must not fire on structureless data at all;
    this is what licenses it to certify periods BIC cannot afford."""
    from qca_fragmentation.scaling.overfit_audit import null_recurrence_rate
    r = null_recurrence_rate(trials=1500, seed=3)
    assert r["false_positives"] == 0, r


def test_whole_series_recurrence_already_covers_periodicity():
    """Period-p oscillation IS a recurrence of order >= p, so the exact-base
    search never needed the split -- the split only fixes the growth CLASS.
    This is why generalising the period changed no exact-base count."""
    r = find_integer_recurrence(W28_DMAX)
    assert r["ok"] and r["base"] == pytest.approx(2 ** (1 / 3), rel=1e-9)
