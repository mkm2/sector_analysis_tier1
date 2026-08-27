"""Regression tests for R30: the sector-size distribution of the four
one-Hadamard rules.

The point of the module is that ONE 2x2 matrix decides everything, so the tests
pin the chain of reductions rather than the numbers alone:

    flip graph (2^N)  ==  renewal product over the label  ==  the O(N) DP
    and, for the moments, ==  the s-fold tensor transfer matrix.

Every claim that could be a coincidence of one rule at one size is checked for
all four rules at both boundary conditions.
"""

import math
from collections import Counter

import pytest

from qca_fragmentation.scaling import sector_sizes as S

BCS = ("obc0", "pbc")


# --- the rank-one factorisation ----------------------------------------------

def test_m1_is_a_matrix_unit_and_m0_is_its_complement_in_J():
    for rule in S.RULES:
        m0, m1 = S.wall_matrices(rule)
        assert sum(sum(r) for r in m1) == 1, rule       # rank one
        for i in (0, 1):
            for j in (0, 1):
                assert m0[i][j] + m1[i][j] == 1, rule   # M_0 + M_1 = J


@pytest.mark.parametrize("rule", S.RULES)
def test_jordan_type_matches_the_family(rule):
    """Diagonal word -> Fibonacci matrix; off-diagonal word -> unipotent."""
    (m00, m01), (m10, m11) = S.wall_matrices(rule)[0]
    tr, det = m00 + m11, m00 * m11 - m01 * m10
    disc = tr * tr - 4 * det
    if S.FAMILY[rule] == "hyperbolic":
        assert (tr, det) == (1, -1), rule               # char poly t^2 - t - 1
        assert disc == 5
    else:
        assert (tr, det) == (2, 1), rule                # (t - 1)^2
        assert disc == 0


def test_the_gap_kernel_is_fibonacci_or_linear():
    for g in range(0, 12):
        assert S.gap_kernel(156, g) == g
        assert S.gap_kernel(198, g) == g
        assert S.gap_kernel(201, g) == S.fib(g - 1)
        assert S.gap_kernel(108, g) == S.fib(g - 1)


# --- the three routes to the same multiset -----------------------------------

@pytest.mark.parametrize("bc", BCS)
@pytest.mark.parametrize("rule", S.RULES)
@pytest.mark.parametrize("N", [4, 6, 8, 9])
def test_renewal_product_equals_dp_equals_flip_graph(bc, rule, N):
    enum = Counter()
    for _walls, size in S.enumerate_sectors(rule, N, bc):
        enum[size] += 1
    if bc == "pbc" and rule in (156, 198):       # the ring defect
        enum[2] -= 1
        if not enum[2]:
            del enum[2]
        enum[1] += 2
    assert dict(enum) == S.hist(rule, N, bc) == S.hist_bruteforce(rule, N, bc)


@pytest.mark.parametrize("bc", BCS)
@pytest.mark.parametrize("rule", S.RULES)
@pytest.mark.parametrize("N", [10, 12, 14])
def test_dp_equals_flip_graph_at_larger_N(bc, rule, N):
    assert S.hist(rule, N, bc) == S.hist_bruteforce(rule, N, bc)


@pytest.mark.parametrize("bc", BCS)
@pytest.mark.parametrize("rule", S.RULES)
@pytest.mark.parametrize("N", [6, 11, 15])
def test_the_sum_rule(bc, rule, N):
    h = S.hist(rule, N, bc)
    assert sum(k * v for k, v in h.items()) == 1 << N


# --- the tensor transfer matrix (Tier-2 R-T20's route) -----------------------

@pytest.mark.parametrize("bc", BCS)
@pytest.mark.parametrize("rule", S.RULES)
def test_tensor_moments_equal_histogram_moments(bc, rule):
    for N in (6, 9, 13):
        h = S.hist(rule, N, bc)
        for s in (1, 2, 3, 4):
            assert (S.moment_Z(rule, N, bc, s)
                    == sum(v * k ** s for k, v in h.items())), (rule, N, bc, s)


@pytest.mark.parametrize("bc", BCS)
@pytest.mark.parametrize("rule", S.RULES)
def test_s_equals_one_is_the_sum_rule_for_free(bc, rule):
    """M_0 + M_1 = J, so T_1 = J and Z_N(1) = 2^N is a theorem."""
    for N in (5, 10, 40):
        assert S.moment_Z(rule, N, bc, 1) == 1 << N


def test_moment_Z_rejects_s_zero():
    with pytest.raises(ValueError):
        S.moment_Z(156, 8, "obc0", 0)


# --- closed forms -------------------------------------------------------------

@pytest.mark.parametrize("bc", BCS)
@pytest.mark.parametrize("rule", S.RULES)
def test_closed_forms_against_the_histogram(bc, rule):
    for N in range(3, 19):
        h = S.hist(rule, N, bc)
        assert max(h) == S.d_max_closed(rule, N, bc), ("d_max", rule, N, bc)
        assert h.get(1, 0) == S.n_frozen_closed(rule, N, bc), \
            ("n_frozen", rule, N, bc)
        assert sum(h.values()) == S.n_sectors_closed(rule, N, bc), \
            ("n_sectors", rule, N, bc)


def test_d_max_is_the_wall_free_sector_in_the_hyperbolic_family():
    """F(N+2) for W201 at obc0 but F(N) for W108: the 0-pad is a free collar for
    the word `11` and a block letter for `00`.  Same rule, two Fibonacci indices
    apart -- the correction Tier-2 R-T20 caught in the preliminary brief."""
    for N in range(3, 20):
        assert S.d_max_closed(201, N, "obc0") == S.fib(N + 2)
        assert S.d_max_closed(108, N, "obc0") == S.fib(N)
        assert S.d_max_closed(201, N, "pbc") == S.lucas(N)
        assert S.d_max_closed(108, N, "pbc") == S.lucas(N)


def test_the_parabolic_d_max_is_the_best_product_of_gap_lengths():
    """4^{N/5}: a gap of length g costs g+1 letters and is worth g, and
    4^{1/5} = 1.319508 beats 3^{1/4} = 1.316074."""
    assert [S.d_max_closed(156, N, "pbc") for N in range(3, 17)] == \
        [2, 3, 4, 5, 6, 9, 12, 16, 20, 27, 36, 48, 64, 81]
    # obc0 is the same series one site later: the chain has N+1 label letters
    assert [S.d_max_closed(156, N, "obc0") for N in range(3, 17)] == \
        [3, 4, 5, 6, 9, 12, 16, 20, 27, 36, 48, 64, 81, 108]
    for N in range(12, 40):                       # G(B+5) = 4 G(B)
        assert (S.d_max_closed(156, N + 5, "pbc")
                == 4 * S.d_max_closed(156, N, "pbc"))


def test_frozen_sectors_are_linear_for_the_ascent_rules():
    for N in range(3, 25):
        for rule in (156, 198):
            assert S.hist(rule, N, "obc0").get(1) == (N - 1) // 2 + 2
            assert S.hist(rule, N, "pbc").get(1) == (4 if N % 2 == 0 else 2)


def test_frozen_sectors_are_exponential_for_the_symmetric_rules():
    """K(g) = 1 for g in {0,2,3}, and x + x^3 + x^4 = 1 at x = 1/phi."""
    phi = (1 + math.sqrt(5)) / 2
    for rule in (108, 201):
        f = [S.hist(rule, N, "pbc")[1] for N in (18, 22, 26)]
        assert abs((f[2] / f[0]) ** (1 / 8) - phi) < 5e-3, (rule, f)


def test_sector_counts_reproduce_R18():
    """R9/R18's bases, derived here rather than fitted."""
    assert [S.n_sectors_closed(156, N, "obc0") for N in range(3, 12)] == \
        [S.fib(N + 2) for N in range(3, 12)]
    assert [S.n_sectors_closed(156, N, "pbc") for N in range(3, 12)] == \
        [S.lucas(N) + 1 for N in range(3, 12)]
    # a(N) = 2a(N-1) - a(N-2) + a(N-3) for the symmetric pair, both bc
    for rule in (108, 201):
        for bc in BCS:
            a = [S.n_sectors_closed(rule, N, bc) for N in range(3, 20)]
            for i in range(3, len(a)):
                assert a[i] == 2 * a[i - 1] - a[i - 2] + a[i - 3], (rule, bc, i)


# --- the free energy tau(s) ---------------------------------------------------

@pytest.mark.parametrize("rule", S.RULES)
def test_lambda_one_is_exactly_two(rule):
    assert abs(S.lambda_s(rule, 1.0) - 2.0) < 1e-12


def test_lambda_zero_is_the_sector_count_base():
    phi = (1 + math.sqrt(5)) / 2
    assert abs(S.lambda_s(156, 0.0) - phi) < 1e-10
    assert abs(S.lambda_s(198, 0.0) - phi) < 1e-10
    for rule in (108, 201):                      # rho^2, R18's plastic number
        assert abs(S.lambda_s(rule, 0.0) - 1.754877666247) < 1e-10


def test_lambda_s_matches_the_tensor_growth_rate():
    """The transfer-matrix ratio at large N must reproduce the root of
    B_s(x) = 1, and must be the SAME at obc0 and pbc."""
    for rule in (156, 108):
        for s in (2, 3):
            for bc in BCS:
                a = S.moment_Z(rule, 300, bc, s)
                b = S.moment_Z(rule, 301, bc, s)
                assert abs(b / a - S.lambda_s(rule, float(s))) < 1e-9, \
                    (rule, s, bc)


def test_minimal_polynomials_from_tier2_solve_the_scalar_equation():
    """Tier-2 R-T20's characteristic polynomials, checked against B_s(1/lam)=1."""
    import numpy as np
    minpoly = {                                  # coefficients, highest first
        (156, 2): [1, -3, 2, -2], (108, 2): [1, -2, -2],
        (156, 3): [1, -4, 5, -8], (108, 3): [1, -4, -3, 8],
    }
    for (rule, s), coef in minpoly.items():
        roots = np.roots(coef)
        lam = max(r.real for r in roots if abs(r.imag) < 1e-9)
        assert abs(float(S.bulk_series(rule, float(s), 1 / lam, dps=40)) - 1) \
            < 1e-12, (rule, s, lam)


def test_alpha_max_is_the_d_max_exponent():
    for rule, N in ((201, 200), (156, 200)):
        exact = math.log(S.d_max_closed(rule, N, "pbc")) / N
        assert abs(exact - S.alpha_max(rule)) < 5e-3, rule


def test_strong_fragmentation():
    """tau'(1) < ln 2 strictly: a random state's sector is an exponentially
    small fraction of the Hilbert space (R21)."""
    for rule in S.RULES:
        a1, _ = S.tau_derivatives(rule, 1.0)
        assert 0 < a1 < math.log(2) - 0.4, rule


# --- the log-normal law and its boundary offsets ------------------------------

@pytest.mark.parametrize("bc", BCS)
@pytest.mark.parametrize("rule", S.RULES)
def test_cumulant_dp_matches_the_exact_histogram(bc, rule):
    for N in (6, 11, 16):
        h = S.hist(rule, N, bc)
        n = sum(h.values())
        m1 = sum(v * math.log(k) for k, v in h.items()) / n
        m2 = sum(v * math.log(k) ** 2 for k, v in h.items()) / n
        c = S.cumulants(rule, N, bc)
        assert c["n_sectors"] == n
        assert abs(c["mean"] - m1) < 1e-9
        assert abs(c["var"] - (m2 - m1 * m1)) < 1e-9


def test_the_ring_has_no_finite_size_offset():
    """At pbc both cumulants sit on their asymptotes at finite N; the residual
    is the truncation of alpha_typ and sigma^2, so it grows like N, not like a
    constant."""
    for rule in S.RULES:
        for N in (60, 120):
            o = S.offsets(rule, "pbc", N)
            assert abs(o["c1"]) < 3e-6 and abs(o["c2"]) < 3e-6, (rule, N, o)


def test_the_chain_offset_is_rule_dependent_and_can_be_positive():
    """Tier-2 R-T20 rev.2's correction: this is NOT one universal pad cost.
    W201's offset is POSITIVE -- the 0-pad is a free collar for the word `11`."""
    got = {rule: S.offsets(rule, "obc0", 120)["c1"] for rule in S.RULES}
    assert abs(got[156] - (-0.246953)) < 1e-5
    assert abs(got[198] - (-0.246953)) < 1e-5
    assert abs(got[108] - (-0.100987)) < 1e-5
    assert abs(got[201] - (+0.245187)) < 1e-5
    assert got[201] > 0 > got[108] > got[156]


# --- the label really is the sector ------------------------------------------

@pytest.mark.parametrize("bc", BCS)
@pytest.mark.parametrize("rule", S.RULES)
def test_the_wall_set_is_a_complete_invariant(bc, rule):
    """Constant on sectors and injective across them -- except the one ring
    defect of the ascent rules, where 0^N and 1^N share the empty label."""
    expected = 1 if (bc == "pbc" and rule in (156, 198)) else 0
    for N in (7, 10, 13):
        v = S.sector_is_wall_set(rule, N, bc)
        assert v["constant"] is True, (rule, N, bc, v)
        assert v["collisions"] == expected, (rule, N, bc, v)


# --- n_N(sigma): the multiplicity function -----------------------------------

@pytest.mark.parametrize("bc", BCS)
@pytest.mark.parametrize("rule", S.RULES)
def test_multiplicity_matches_the_histogram(bc, rule):
    """The divisor-restricted renewal DP against the full histogram, size by
    size -- including sizes that never occur, which must come back 0."""
    for N in (8, 13, 17):
        h = S.hist(rule, N, bc)
        for sigma in sorted(set(h) | set(range(1, 26))):
            assert S.multiplicity(rule, N, bc, sigma) == h.get(sigma, 0), \
                (rule, N, bc, sigma)


def _is_fibonacci_product(v):
    fibs = [S.fib(j) for j in range(3, 40) if S.fib(j) <= v]
    if v == 1:
        return True
    return any(v % f == 0 and _is_fibonacci_product(v // f) for f in fibs)


def test_every_integer_up_to_N_is_a_parabolic_sector_size():
    for N in (18, 22, 24):
        sizes = set(S.hist(156, N, "obc0"))
        assert all(k in sizes for k in range(1, N + 1)), N


def test_the_chain_sizes_are_exactly_the_fibonacci_semigroup():
    for rule in (108, 201):
        for N in range(3, 26):
            for k in S.hist(rule, N, "obc0"):
                assert _is_fibonacci_product(k), (rule, N, k)


def test_the_ring_adds_exactly_one_lucas_size_per_N():
    """THE EXCEPTION, and the reason the earlier version of this test was
    vacuous: the wall-free label is a TRACE at pbc, not a corner of M_0^N, so
    its size is the Lucas number L(N), which is generally not a Fibonacci
    product.  7 = L(4), 11 = L(5), 29 = L(7) really are sector sizes.  The old
    test only passed because its N list (16, 20, 24, 30) never hit the N at
    which those Lucas numbers appear.  Caught by Tier-2 R-T20 rev.3."""
    for rule in (108, 201):
        for N in (4, 5, 7, 9, 11, 13):
            L = S.lucas(N)
            assert S.multiplicity(rule, N, "pbc", L) >= 1, (rule, N, L)
            assert S.hist_bruteforce(rule, N, "pbc").get(L, 0) >= 1, (rule, N)
    assert S.multiplicity(201, 4, "pbc", 7) == 1
    assert S.multiplicity(108, 5, "pbc", 11) == 1
    assert S.multiplicity(201, 7, "pbc", 29) == 1
    for rule in (108, 201):
        for N in range(3, 24):
            luc = {S.lucas(m) for m in range(1, 40)}
            for k in S.hist(rule, N, "pbc"):
                assert _is_fibonacci_product(k) or k in luc, (rule, N, k)


def test_the_sizes_excluded_at_both_boundaries():
    """Neither a Fibonacci product nor a Lucas number."""
    never = [14, 17, 19, 22, 23, 28, 31, 33, 35, 37, 38]
    luc = {S.lucas(m) for m in range(1, 40)}
    for k in never:
        assert not _is_fibonacci_product(k) and k not in luc, k
        assert S.multiplicity_degree(108, k) == -1, k
    for rule in (108, 201):
        for bc in BCS:
            for N in range(3, 32):
                for k in never:
                    assert S.multiplicity(rule, N, bc, k) == 0, (rule, N, bc, k)


def test_the_parabolic_multiplicity_is_polynomial_of_degree_omega_plus_one():
    """n_N(sigma) ~ P(sigma) (N/2)^{Omega+1}/(Omega+1)!, so the sector census
    reads off the prime factorisation of sigma."""
    for sigma in (1, 2, 4, 6, 12, 16):
        r = [S.multiplicity(156, N, "obc0", sigma)
             / S.multiplicity_asymptotic(156, N, "obc0", sigma)
             for N in (400, 1600)]
        assert abs(r[1] - 1) < abs(r[0] - 1) + 1e-12, (sigma, r)
        assert abs(r[1] - 1) < 0.02, (sigma, r)
    # n(6) is asymptotically twice n(4): same Omega, twice the orderings
    a = S.multiplicity(156, 1600, "obc0", 6) / S.multiplicity(156, 1600, "obc0", 4)
    assert abs(a - 2) < 0.05, a


def test_a_size_occurs_at_both_parities_iff_four_divides_it():
    """Tier-2 R-T20 rev.3's sharpening of the parity rule: merging factors a, b
    shifts sopfr + Omega by (a-1)(b-1) - 2, odd iff both are even, so the
    opposite parity is reachable only when sigma has a factorisation with two
    even factors."""
    for sigma in range(2, 33):
        parities = {N % 2 for N in range(3, 48)
                    if S.multiplicity(156, N, "pbc", sigma)}
        assert (len(parities) == 2) == (sigma % 4 == 0), (sigma, parities)


def test_the_ring_multiplicity_has_a_parity_selection_rule():
    """At pbc the cost is exact, so sigma appears at leading order only when
    N = sopfr(sigma) + Omega(sigma) (mod 2); the other parity drops a degree."""
    assert S.multiplicity(156, 400, "pbc", 2) == 0
    assert S.multiplicity(156, 401, "pbc", 2) == 401
    assert S.multiplicity(156, 400, "pbc", 6) == 0
    # sigma = 4 exists at both parities but one degree apart: N^2 vs N^1
    assert S.multiplicity(156, 400, "pbc", 4) == 39600
    assert S.multiplicity(156, 401, "pbc", 4) == 401
    assert (S.multiplicity(156, 800, "pbc", 4)
            / S.multiplicity(156, 801, "pbc", 4)) > 2 * (
        S.multiplicity(156, 400, "pbc", 4)
        / S.multiplicity(156, 401, "pbc", 4)) - 1e-9
    for sigma in (1, 4, 12):
        N = 800 + ((S.sopfr(sigma) + S.omega(sigma) - 800) % 2)
        r = (S.multiplicity(156, N, "pbc", sigma)
             / S.multiplicity_asymptotic(156, N, "pbc", sigma))
        assert abs(r - 1) < 0.02, (sigma, N, r)


def test_the_hyperbolic_multiplicity_grows_like_phi_to_the_N():
    """A(sigma) N^m phi^N: the 1-gap is threefold degenerate (g in {0,2,3}) and
    x + x^3 + x^4 = 1 at x = 1/phi, so padding is exponentially many ways."""
    phi = (1 + math.sqrt(5)) / 2
    for sigma in (1, 2, 4, 8):
        m = S.multiplicity_degree(108, sigma)
        vals = [S.multiplicity(108, N, "obc0", sigma) / (N ** m * phi ** N)
                for N in (200, 400)]
        assert 0.9 < vals[1] / vals[0] < 1.1, (sigma, m, vals)
    assert S.multiplicity_degree(108, 1) == 0
    assert S.multiplicity_degree(108, 2) == 1
    assert S.multiplicity_degree(108, 8) == 3        # 8 = 2*2*2, all F(3)


def test_multiplicity_asymptotic_refuses_the_hyperbolic_family():
    with pytest.raises(NotImplementedError):
        S.multiplicity_asymptotic(108, 100, "obc0", 4)
