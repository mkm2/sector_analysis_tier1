"""Growth-class fit tests (context Tier 1 sec.7)."""
import math

from qca_fragmentation.scaling.fits import (find_integer_recurrence,
                                            find_recurrence_by_parity,
                                            fit_pure_exponential, name_base,
                                            fit_series)


NS = list(range(6, 19))


def test_constant():
    f = fit_series(NS, [5] * len(NS))
    assert f["ok"] and f["growth_class"] == "constant"


def test_linear_is_polynomial():
    # rule 60 style: n_sectors = N + 1  (must NOT be labelled exponential)
    f = fit_series(NS, [N + 1 for N in NS])
    assert f["growth_class"] == "polynomial"


def test_power_law_is_polynomial():
    f = fit_series(NS, [N * N for N in NS])
    assert f["growth_class"] == "polynomial"


def test_exponential_base_two():
    # rule 60 D_max = 2^{N-1}
    f = fit_series(NS, [1 << (N - 1) for N in NS])
    assert f["growth_class"] == "exponential"
    assert abs(f["base"] - 2.0) < 1e-6


def test_fibonacci_is_golden_exponential():
    # rule 201 D_max ~ Fibonacci -> base phi
    fib = {}
    a, b = 1, 1
    seq = []
    for N in NS:
        # Fibonacci indexed so growth base is phi
        val = round((((1 + 5 ** 0.5) / 2) ** N) / (5 ** 0.5))
        seq.append(val)
    f = fit_series(NS, seq)
    assert f["growth_class"] == "exponential"
    assert abs(f["base"] - (1 + 5 ** 0.5) / 2) < 1e-2


# --- exact recurrences and the bounded rate model -----------------------------

def test_lucas_recurrence_is_exact_and_gives_phi():
    # rule 4 (IDDD) attractor count, pbc, N = 6..13
    seq = [18, 29, 47, 76, 123, 199, 322, 521]
    r = find_integer_recurrence(seq)
    assert r["ok"] and r["order"] == 2 and r["coeffs"] == [1, 1]
    assert abs(r["base"] - (1 + 5 ** 0.5) / 2) < 1e-12
    assert r["name"] == "golden $\\varphi$"


def test_recurrence_must_verify_on_every_term():
    # Lucas for six terms, then broken: no recurrence may be returned.
    assert not find_integer_recurrence([18, 29, 47, 76, 123, 199, 322, 999])["ok"]


def test_recurrence_rejects_non_integer_solution():
    assert not find_integer_recurrence([1, 2, 3, 5, 8, 13, 21, 35])["ok"]


def test_pure_exponential_stays_inside_the_physical_bound():
    # rule 90 D_max (pbc, N = 6..13): ragged, and M2 fits it with base ~11 by
    # trading kappa against alpha ln N.  The 2-parameter model cannot do that.
    ragged = [1, 15, 1, 7, 3, 63, 2, 127]
    assert fit_series(list(range(6, 14)), ragged)["base"] > 2.0
    e = fit_pure_exponential(list(range(6, 14)), ragged)
    assert e["ok"] and e["bounded"] and e["base"] <= 2.0


def test_pure_exponential_recovers_a_clean_rate():
    e = fit_pure_exponential(NS, [1 << (N - 1) for N in NS])
    assert abs(e["base"] - 2.0) < 1e-9 and e["rms"] < 1e-9 and e["bounded"]


def test_parity_recurrence_reports_the_per_site_base():
    # A series growing as 3^(N/2) on both parities: the per-parity recurrence
    # has root 3, but the base per SITE is sqrt(3), not 3.
    Ns = list(range(6, 20))
    seq = [3 ** (n // 2) if n % 2 == 0 else 2 * 3 ** (n // 2) for n in Ns]
    r = find_recurrence_by_parity(Ns, seq)
    assert r["ok"] and r["step"] == 2
    assert abs(r["base"] - 3 ** 0.5) < 1e-9
    assert r["name"] == "$\\sqrt{3}$"


def test_parity_recurrence_rejects_disagreeing_parities():
    Ns = list(range(6, 20))
    seq = [3 ** (n // 2) if n % 2 == 0 else 2 ** (n // 2) for n in Ns]
    assert not find_recurrence_by_parity(Ns, seq)["ok"]


def test_step_one_is_unchanged():
    seq = [18, 29, 47, 76, 123, 199, 322, 521]
    assert find_integer_recurrence(seq)["step"] == 1
    assert (find_integer_recurrence(seq)["base"]
            == find_integer_recurrence(seq, step=1)["base"])


def test_repeated_unit_root_is_snapped_to_one():
    # A quadratic series obeys a(n)=3a(n-1)-3a(n-2)+a(n-3), i.e. (x-1)^3, and
    # np.roots splits that triple root into a cluster ~1e-5 wide.  An unsnapped
    # base would label polynomial growth as exponential.
    seq = [n * n for n in range(6, 20)]
    r = find_integer_recurrence(seq)
    assert r["ok"] and r["coeffs"] == [3, -3, 1]
    assert r["base"] == 1.0


def test_parity_doubled_bases_are_named():
    # a(n) = 4a(n-2) - 2a(n-4): x^4 = 4x^2 - 2, so the base is sqrt(2+sqrt2).
    assert name_base(2 ** 0.5 * 0 + (2 + 2 ** 0.5) ** 0.5) is not None
    assert "cos" in name_base((2 + 2 ** 0.5) ** 0.5)
