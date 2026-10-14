"""R32: Floquet spectral diagnostics -- reference values, symmetry claims, and the
cross-check between the exact Python engine and the Julia spectral data."""

import os

import numpy as np
import pytest
from scipy.integrate import quad

from qca_fragmentation.spectral import floquet as F
from qca_fragmentation.spectral import stats as S


def test_hsf_wolfram_bijection():
    assert [F.wolfram_to_hsf(w) for w in F.UNITARY_WOLFRAM] == list(range(16))
    assert F.hsf_to_wolfram(4) == 156 and F.hsf_to_wolfram(6) == 150
    assert F.hsf_to_wolfram(8) == 108 and F.hsf_to_wolfram(15) == 51


@pytest.mark.parametrize("beta,mean", [(0, 2 * np.log(2) - 1), (1, 0.5359), (2, 0.6027)])
def test_surmise_normalised_and_mean(beta, mean):
    assert quad(lambda r: S.surmise(r, beta), 0, 1)[0] == pytest.approx(1, abs=1e-9)
    assert quad(lambda r: r * S.surmise(r, beta), 0, 1)[0] == pytest.approx(mean, abs=2e-4)


def test_page_entropy():
    assert S.page_entropy(6, 6) == pytest.approx(np.log(64) - 0.5, abs=1e-3)


@pytest.mark.parametrize("w", F.UNITARY_WOLFRAM)
def test_U_real_orthogonal_and_two_reflections(w):
    """U is real orthogonal for both gates (the basis of the half-spectrum rule)."""
    for gate in ("H", "X"):
        U = F.floquet_matrix(w, 6, gate)
        assert np.allclose(U.T @ U, np.eye(64))
        ev = np.linalg.eigvals(U)                     # spectrum closed under conjugation
        assert np.allclose(np.sort_complex(ev), np.sort_complex(ev.conj()), atol=1e-9)


def test_unique_levels_folds_and_dedups():
    th = np.array([0.3, -0.3, 0.3, 1.0, -1.0, np.pi])
    assert np.allclose(S.unique_levels(th), [0.3, 1.0, np.pi])


def test_random_reflection_product_is_beta1():
    """Control for the reference value: a product of two Haar-random reflections."""
    rng = np.random.default_rng(3)
    rs = []
    for _ in range(4):
        def refl():
            Q, _ = np.linalg.qr(rng.normal(size=(300, 300)))
            return Q @ np.diag(np.r_[np.ones(150), -np.ones(150)]) @ Q.T
        th = -np.angle(np.linalg.eigvals(refl() @ refl()))
        rs.append(S.ratios(S.unique_levels(th)))
    r = np.concatenate(rs)
    assert abs(r.mean() - S.R_GOE) < 0.03 and abs(r.mean() - S.R_GUE) > 0.04


@pytest.mark.parametrize("hsf", range(16))
def test_engine_matches_julia_spectrum(hsf):
    """Independent constructions of U: exact Python engine vs HSF/Julia build."""
    w = F.hsf_to_wolfram(hsf)
    for N in (6, 7, 8):
        p = F.spec_path("H", N, w)
        if not os.path.exists(p):
            pytest.skip("Julia spectral data not generated")
        d = F.load(p)
        assert len(d["theta"]) == 2 ** N
        assert d["max_residual"] < 1e-8 and d["max_offblock"] < 1e-8
        ev = np.linalg.eigvals(F.floquet_matrix(w, N, "H"))
        assert F.circle_distance(-np.angle(ev), d["theta"]) < 1e-9


def test_circle_distance_is_wrap_safe():
    a = np.array([-1e-16, 1.0, 2.0])
    b = np.array([0.0, 1.0, 2.0])
    assert F.circle_distance(a, b) < 1e-12
    assert F.circle_distance(a, b + 0.1) == pytest.approx(0.1)
