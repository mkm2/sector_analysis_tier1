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


def test_block_ratios_fold_vs_full_circle():
    """Symmetric spectra are folded to [0, pi]; others use the full circle."""
    x = np.array([0.3, -0.3, 1.0, -1.0, 2.0, -2.0, 2.9, -2.9])
    assert np.allclose(S.block_ratios(x), S.ratios(S.unique_levels(x)))
    y = np.array([0.1, 0.5, 1.4, 3.0, -2.0])                  # not symmetric
    u = np.sort(np.mod(y, 2 * np.pi))
    sp = np.diff(np.r_[u, u[0] + 2 * np.pi])
    expect = np.minimum(sp, np.roll(sp, -1)) / np.maximum(sp, np.roll(sp, -1))
    assert np.allclose(S.block_ratios(y), expect)


def test_cue_through_full_circle_branch():
    """A Haar-random complex unitary (no conjugation symmetry) gives the CUE value."""
    rng = np.random.default_rng(11)
    rs = []
    for _ in range(4):
        Z = (rng.normal(size=(300, 300)) + 1j * rng.normal(size=(300, 300))) / np.sqrt(2)
        Q, R = np.linalg.qr(Z)
        Q = Q * (np.diag(R) / np.abs(np.diag(R)))
        rs.append(S.block_ratios(-np.angle(np.linalg.eigvals(Q))))
    r = np.concatenate(rs)
    assert abs(r.mean() - S.R_GUE) < 0.02


def test_conj_symmetric_level_at_zero():
    # a level at 0 that rounds to 2 pi - eps must not break the symmetry test
    x = np.array([-1e-16, 0.7, -0.7, 2.0, -2.0, np.pi])
    assert S.conj_symmetric(x)
    assert not S.conj_symmetric(np.array([1e-16, 0.7, -0.7, 2.0, -1.9]))
    # the symmetric branch folds: ratios from the half spectrum {0, 0.7, 2, pi}
    assert np.allclose(S.block_ratios(x), S.ratios(S.unique_levels(np.array([0, .7, 2, np.pi]))))


@pytest.mark.skipif(not os.path.exists(F.spec_path("H", 8, 57, "pbc")), reason="no pbc data")
def test_pbc_momentum_zero_block_is_real():
    # ring, T2 momentum j = 0 is a real subspace: its block must be classified as
    # conjugation-symmetric and equal the exact engine's spectrum (U = e^{-i theta})
    d = F.load(F.spec_path("H", 8, 57, "pbc"))
    m = d["stat_label"] == 0
    assert S.conj_symmetric(d["theta"][m]) and S.conj_symmetric(d["phi"][m])
    N, n = 8, 256
    U = F.floquet_matrix(57, N, "H", "pbc")
    T2 = np.zeros((n, n))
    for x in range(n):
        T2[((x << 2) | (x >> (N - 2))) & (n - 1), x] = 1
    P = sum(np.linalg.matrix_power(T2, k) for k in range(N // 2)) / (N // 2)
    w, V = np.linalg.eigh(P)
    B = V[:, w > 0.5]
    e = np.angle(np.linalg.eigvals(B.T @ U @ B))
    th = -d["theta"][m]
    assert th.size == e.size
    dist = np.abs(np.exp(1j * th)[:, None] - np.exp(1j * e)[None, :]).min(axis=1)
    assert dist.max() < 1e-8


def test_shape_test_recognises_poisson_and_cue():
    from qca_fragmentation.spectral import analysis as A
    rng = np.random.default_rng(1)
    poi = S.block_ratios(rng.uniform(0, 2 * np.pi, 20000))
    t = A.shape_test(poi)
    assert t["chi2_poisson"] < min(t["chi2_goe"], t["chi2_gue"])
    # CUE: eigenphases of a Haar unitary (QR of a complex Ginibre matrix)
    z = (rng.normal(size=(400, 400)) + 1j * rng.normal(size=(400, 400))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    q = q * (np.diag(r) / np.abs(np.diag(r)))
    cue = np.concatenate([S.block_ratios(np.angle(np.linalg.eigvals(q)))])
    t = A.shape_test(cue)
    assert t["chi2_gue"] < min(t["chi2_goe"], t["chi2_poisson"])


@pytest.mark.skipif(not os.path.exists(F.spec_path("H", 14, 153, "pbc")), reason="no pbc N=14 data")
def test_ring_classes_at_N14():
    # R32 ring result: 57 is CUE in the complex momentum sectors; the one-sided rule 153
    # (no mirror symmetry) is nevertheless beta = 1 in shape, not a Poisson-CUE crossover
    from qca_fragmentation.spectral import analysis as A
    t57 = A.shape_test(A.complex_sector_ratios(F.load(F.spec_path("H", 14, 57, "pbc"))))
    assert abs(t57["r_mean"] - S.R_GUE) < 0.01 and t57["chi2_gue"] < t57["chi2_goe"]
    t153 = A.shape_test(A.complex_sector_ratios(F.load(F.spec_path("H", 14, 153, "pbc"))))
    assert abs(t153["r_mean"] - S.R_GOE) < 0.01
    assert t153["chi2_goe"] < 0.2 * min(t153["chi2_gue"], t153["chi2_mix"])
