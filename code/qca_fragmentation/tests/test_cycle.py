"""succ(x) and branch-semantics tests (context Tier 1 sec.3)."""
from fractions import Fraction

import pytest

from qca_fragmentation.core import cycle, rules

UNITARY = rules.UNITARY_RULES


@pytest.mark.parametrize("bc", ["obc0", "pbc"])
@pytest.mark.parametrize("N", [3, 4, 5, 6])
def test_unitary_single_branch_and_norm(bc, N):
    for r in UNITARY:
        t = rules.wolfram_to_tuple(r)
        for x in range(1 << N):
            brs = cycle.one_cycle_branches(x, N, t, bc)
            assert len(brs) == 1                      # single Kraus branch
            assert cycle.branch_norms(x, N, t, bc) == Fraction(1)


@pytest.mark.parametrize("bc", ["obc0", "pbc"])
@pytest.mark.parametrize("rule", [22, 0, 28, 29, 22 ^ 0])
def test_dissipative_norm_sums_to_one(bc, rule):
    t = rules.wolfram_to_tuple(rule)
    for N in (3, 4, 5):
        for x in range(1 << N):
            assert cycle.branch_norms(x, N, t, bc) == Fraction(1)


def test_identity_rule_204_is_identity():
    # rule 204 (I,I,I,I): succ(x) = {x} for all x
    t = rules.wolfram_to_tuple(204)
    for bc in ("obc0", "pbc"):
        for N in (4, 6):
            for x in range(1 << N):
                assert cycle.succ(x, N, t, bc) == [x]


def test_all_hadamard_51_full_support():
    # rule 51 (V,V,V,V): H^{tensor N}, succ(x) = all 2^N states
    t = rules.wolfram_to_tuple(51)
    for bc in ("obc0", "pbc"):
        for N in (3, 4, 5):
            full = list(range(1 << N))
            for x in range(1 << N):
                assert cycle.succ(x, N, t, bc) == full


def test_reflection_isomorphism_succ_obc0_odd_N():
    # Under obc0 with ODD N the even/odd sublattices are each reflection-
    # invariant AND there is no periodic seam, so reflecting the rule +
    # bit-reversing the states permutes succ sets EXACTLY.  (For even N
    # reflection swaps the two brickwork layers, and for pbc odd N the wrap-
    # around seam breaks per-state equality; only sector SIZES survive there,
    # and only for unitary rules -- see test_scc.)
    import random
    bc = "obc0"
    rng = random.Random(1)
    N = 5
    def revbits(x):
        return int(f"{x:0{N}b}"[::-1], 2)
    for _ in range(40):
        r = rng.randrange(256)
        t = rules.wolfram_to_tuple(r)
        tr = rules.reflect_tuple(t)
        for x in range(1 << N):
            s1 = {revbits(y) for y in cycle.succ(x, N, t, bc)}
            s2 = set(cycle.succ(revbits(x), N, tr, bc))
            assert s1 == s2
