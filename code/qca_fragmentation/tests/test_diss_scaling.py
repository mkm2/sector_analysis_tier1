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


def test_period_four_series_is_not_irregular():
    f = fit_with_period(NS, W11_DMAX)
    assert f["period"] == 4
    assert not f["irregular"]


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


def test_whole_series_recurrence_already_covers_periodicity():
    """Period-p oscillation IS a recurrence of order >= p, so the exact-base
    search never needed the split -- the split only fixes the growth CLASS.
    This is why generalising the period changed no exact-base count."""
    r = find_integer_recurrence(W28_DMAX)
    assert r["ok"] and r["base"] == pytest.approx(2 ** (1 / 3), rel=1e-9)
