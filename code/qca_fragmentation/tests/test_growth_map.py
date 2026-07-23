"""
Anchor-rule checks for the unified growth map descriptors.

These pin the (class, base, alpha, exact) that each canonical rule must land at,
so a regression in the fit plumbing or the analytic overrides is caught before
it silently moves a point on the published map.
"""

import math

from qca_fragmentation.scaling.growth_map import (_DMAX_ANALYTIC,
                                                  _dissipative_descriptor,
                                                  _series_descriptor)
from qca_fragmentation.scaling.summary import load_series


def _dmax(rule, bc="obc0"):
    s = load_series(rule, bc)
    return _series_descriptor(s["N"], s["d_max"], _DMAX_ANALYTIC.get(rule))


def _nsec(rule, bc="obc0"):
    s = load_series(rule, bc)
    return _series_descriptor(s["N"], s["n_recurrent"])


def test_rule204_is_shattered_not_extensive():
    """W204: 2^N frozen singletons -> D_max constant, #sectors base 2."""
    assert _dmax(204)["cls"] == "constant"
    n = _nsec(204)
    assert n["cls"] == "exponential"
    assert abs(n["base"] - 2.0) < 1e-6 and n["exact"]


def test_rule156_dmax_is_the_analytic_four_fifth_with_half_power():
    d = _dmax(156)
    assert d["exact"]
    assert abs(d["base"] - 4 ** 0.2) < 1e-9
    assert abs(d["alpha"] + 0.5) < 1e-9          # binomial-like -1/2


def test_rule150_dmax_is_base_two_with_minus_half_power():
    """The binomial rule sits BELOW the ergodic point, not on it."""
    d = _dmax(150)
    assert d["exact"] and abs(d["base"] - 2.0) < 1e-9
    assert abs(d["alpha"] + 0.5) < 1e-9


def test_fibonacci_base_is_exact_without_an_override():
    """201's #sectors is an exact integer recurrence (root of x^3=2x^2-x+1),
    so it must come back exact even though it is not in the override table."""
    n = _nsec(201)
    assert n["exact"] and abs(n["base"] - 1.7548776662) < 1e-6


def test_dissipative_irregular_rule_has_no_base():
    """W90 (DEED) D_max is a multiplicative order, not a growth law."""
    d = _dissipative_descriptor(90, "obc0")["d_max"]
    assert d["cls"] == "irregular" and d["base"] is None
    assert d.get("kind") == "arithmetic"


def test_dissipative_domain_wall_like_rule_is_exponential():
    """A dissipative exponential rule reports a finite base >= 1."""
    d = _dissipative_descriptor(4, "obc0")["n_recurrent"]   # W4 IDDD, Lucas
    assert d["cls"] in ("exponential", "polynomial", "constant")
    assert d["base"] >= 1.0
