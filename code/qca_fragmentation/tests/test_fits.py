"""Growth-class fit tests (context Tier 1 sec.7)."""
import math

from qca_fragmentation.scaling.fits import fit_series


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
