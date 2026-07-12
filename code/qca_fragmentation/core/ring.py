"""
Exact arithmetic in the ring Z[1/sqrt2].

Every amplitude in a QCA branch is represented as

    value = (a + b*sqrt2) / 2^m ,      a, b  arbitrary-precision Python ints,
                                        m     a single int SHARED by the whole
                                              branch dictionary.

Rationale (context Tier 1 sec.2): a conditional-Hadamard layer multiplies the
*fired* components by 1/sqrt2 and leaves the non-fired components untouched.  A
per-entry sqrt2-exponent would force mixed-parity additions (integer + integer*
sqrt2) that are not integer-closed.  With the (a, b, shared m) form every layer
update is closed over the integers:

  conditional-H at a site:  m -> m + 1
      non-fired entry:  (a, b) -> (2a, 2b)                    [value unchanged]
      fired  entry:     (a, b) -> +-(2b, a)                   [value * 1/sqrt2]

  because (a + b*sqrt2)/sqrt2 = (2b + a*sqrt2)/2.

Exact zero:  a == 0 AND b == 0.  No tolerance, ever (sqrt2 irrational =>
a + b*sqrt2 = 0 with integer a,b forces a = b = 0).

A "branch" here is a plain dict  {bitmask: (a, b)}  plus a single int m.  This
module holds only the ring-level primitives and validation helpers; the gate /
layer logic lives in core/cycle.py.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Dict, Tuple

# An amplitude numerator is a 2-tuple of Python ints (a, b) meaning a + b*sqrt2.
Num = Tuple[int, int]
BranchAmps = Dict[int, Num]


def add(p: Num, q: Num) -> Num:
    """Componentwise addition of two numerators over a shared denominator."""
    return (p[0] + q[0], p[1] + q[1])


def neg(p: Num) -> Num:
    return (-p[0], -p[1])


def is_zero(p: Num) -> bool:
    """Exact zero test: a == 0 and b == 0."""
    return p[0] == 0 and p[1] == 0


def double(p: Num) -> Num:
    """(a, b) -> (2a, 2b): value unchanged when m is simultaneously incremented."""
    return (2 * p[0], 2 * p[1])


def div_sqrt2(p: Num) -> Num:
    """(a + b*sqrt2)/sqrt2 = (2b + a*sqrt2)/2  ->  numerator (2b, a), m -> m+1."""
    return (2 * p[1], p[0])


def prune(amps: BranchAmps) -> BranchAmps:
    """Drop entries whose numerator is exactly (0, 0)."""
    return {k: v for k, v in amps.items() if not (v[0] == 0 and v[1] == 0)}


def renorm(amps: BranchAmps, m: int) -> Tuple[BranchAmps, int]:
    """
    If every numerator has both a and b even, halve all of them and decrement m.
    Keeps the integers from growing without bound.  Repeats greedily.
    """
    while amps and m > 0 and all((a & 1) == 0 and (b & 1) == 0 for (a, b) in amps.values()):
        amps = {k: (a >> 1, b >> 1) for k, (a, b) in amps.items()}
        m -= 1
    return amps, m


def prob_int(a: int, b: int) -> Tuple[int, int]:
    """
    |a + b*sqrt2|^2 = (a^2 + 2 b^2) + (2 a b) * sqrt2, exact over the shared 4^m.
    Returns the (rational-part, sqrt2-part) integer pair of the numerator.
    """
    return (a * a + 2 * b * b, 2 * a * b)


def branch_norm(amps: BranchAmps, m: int) -> Tuple[Fraction, Fraction]:
    """
    Exact squared norm of a branch as (rational_part, sqrt2_part) over 4^m,
    returned as two Fractions.  For a normalized *unitary* branch this must be
    exactly (1, 0).  The sqrt2 part must vanish for any physical (real) total.
    """
    r = 0
    s = 0
    for (a, b) in amps.values():
        pr, ps = prob_int(a, b)
        r += pr
        s += ps
    denom = 1 << (2 * m)  # 4^m
    return Fraction(r, denom), Fraction(s, denom)


def prob_of(a: int, b: int, m: int) -> Fraction:
    """
    Exact real probability |(a + b*sqrt2)/2^m|^2 as a Fraction *when it is
    rational* (i.e. a*b == 0).  Raises if the sqrt2 part does not vanish; used
    only in validation paths where rationality is guaranteed.
    """
    pr, ps = prob_int(a, b)
    if ps != 0:
        raise ValueError("probability has non-vanishing sqrt2 part; use prob_int")
    return Fraction(pr, 1 << (2 * m))
