"""Unit tests for the exact Z[1/sqrt2] arithmetic (context Tier 1 sec.2)."""
from fractions import Fraction

from qca_fragmentation.core import ring


def test_div_sqrt2_identity():
    # (a + b sqrt2)/sqrt2 = (2b + a sqrt2)/2 ; applied twice = value/2, i.e.
    # numerator doubled after m -> m+2 (since /2 == *4/2^2 in shared denom).
    a, b = 3, 5
    p = ring.div_sqrt2((a, b))            # -> (2b, a) = (10, 3), m+1
    q = ring.div_sqrt2(p)                 # -> (2*3, 10) = (6, 10), m+2
    # value of q should equal (a+b sqrt2)/2 with m increased by 2:
    # (a+b sqrt2)/2 has numerator (2a,2b)=(6,10) over 2^{m+2}. matches.
    assert q == (2 * a, 2 * b)


def test_double_and_renorm_roundtrip():
    amps = {0: (3, 5), 1: (2, 4)}
    m = 2
    d2 = {k: ring.double(v) for k, v in amps.items()}
    d2, m2 = ring.renorm(d2, m + 1)
    assert (d2, m2) == (amps, m)


def test_exact_zero_and_prune():
    assert ring.is_zero((0, 0))
    assert not ring.is_zero((0, 1))   # sqrt2 part alone is nonzero
    assert not ring.is_zero((1, 0))
    d = {0: (0, 0), 1: (1, 0), 2: (0, 0)}
    assert ring.prune(d) == {1: (1, 0)}


def test_branch_norm_normalized_state():
    # |+> = (|0> + |1>)/sqrt2 : both amps 1/sqrt2 -> (a,b)=(0,1), m=1
    amps = {0: (0, 1), 1: (0, 1)}
    r, s = ring.branch_norm(amps, 1)
    assert r == Fraction(1) and s == Fraction(0)


def test_interference_cancellation():
    # (1 + sqrt2) + (-1 - sqrt2) = 0 exactly
    assert ring.add((1, 1), (-1, -1)) == (0, 0)
    assert ring.is_zero(ring.add((1, 1), (-1, -1)))


def test_prob_int_and_prob_of():
    # |1/sqrt2|^2 = 1/2
    assert ring.prob_int(0, 1) == (2, 0)
    assert ring.prob_of(0, 1, 1) == Fraction(1, 2)
    # (1+sqrt2)/2 has irrational square -> prob_of must refuse
    pr, ps = ring.prob_int(1, 1)
    assert (pr, ps) == (3, 2)
    try:
        ring.prob_of(1, 1, 1)
        assert False, "expected ValueError"
    except ValueError:
        pass
