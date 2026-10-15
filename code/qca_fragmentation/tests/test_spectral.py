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


@pytest.mark.parametrize("w,bc", [(57, "obc0"), (57, "pbc"), (150, "obc0"), (60, "pbc")])
def test_xgate_cycles_give_the_spectrum(w, bc):
    # independent of the Julia pipeline: the eigenphases of the dense X-gate U are the
    # union over cycles of the L-th roots of unity
    from qca_fragmentation.spectral import xgate as XG
    N = 8
    L = XG.cycle_lengths(w, N, bc)
    assert L.sum() == 2 ** N
    ev = np.sort(np.mod(np.angle(np.linalg.eigvals(F.floquet_matrix(w, N, "X", bc))), 2 * np.pi))
    ref = np.sort(np.concatenate([2 * np.pi * np.arange(l) / l for l in L]))
    # multiset match on sorted lists, after mapping 2 pi -> 0
    ev = np.sort(np.where(2 * np.pi - ev < 1e-9, 0.0, ev))
    assert np.allclose(ev, ref, atol=1e-9)


def test_class_N_is_common_within_a_class():
    from qca_fragmentation.spectral import analysis as A
    summ = {f"{w}_{N}": {} for w, Ns in {108: [18, 19, 20], 156: [19, 20], 198: [20], 201: [19],
                                         57: [13, 14]}.items() for N in Ns}
    summ["198_19"] = {}
    assert A.class_N(summ, 108) == 19 and A.class_N(summ, 201) == 19
    assert A.class_N(summ, 57) == 14


# --- R32 answers (2026-10-15): symmetry, scars, segments, X charges, block-first ---------

def _kr(N, fac):
    M = np.ones((1, 1))
    for i in range(N):
        M = np.kron(fac(i), M)             # site 0 = LSB = last factor
    return M


def _perm(N, f):
    n = 1 << N; x = np.arange(n); y = np.zeros(n, dtype=np.int64)
    for i in range(N):
        y |= ((x >> i) & 1) << f(i)
    P = np.zeros((n, n)); P[y, x] = 1
    return P


@pytest.mark.parametrize("w", [153, 195])
def test_one_sided_dual_reflection_gives_time_reversal(w):
    # S = B_e X^N R_bond B_e^T (B_e: Hadamard eigenbasis on even sites, R_bond: i -> 1 - i):
    # real, [S, U] = 0, S T2 S^-1 = T2^-1; A = K L_A S preserves momentum, inverts U and
    # Omega_T, A^2 = +1  =>  COE in every ring momentum sector although W is not mirror symmetric
    from qca_fragmentation.spectral import blockfirst as B
    N = 8
    V = np.column_stack([B.H_PLUS, B.H_MINUS])
    Be = _kr(N, lambda i: V if i % 2 == 0 else np.eye(2))
    S = Be @ _kr(N, lambda i: np.array([[0, 1.], [1, 0]])) @ _perm(N, lambda i: (1 - i) % N) @ Be.T
    LA = B.layer_sparse(w, N, 0, "pbc").toarray(); U = B.layer_sparse(w, N, 1, "pbc").toarray() @ LA
    T1, T2 = _perm(N, lambda i: (i + 1) % N), _perm(N, lambda i: (i + 2) % N)
    Om = T1 @ LA; LS = LA @ S
    assert np.allclose(S @ U, U @ S) and np.allclose(S @ T2 @ S.T, T2.T)
    assert np.allclose(LS @ U @ LS.T, U.T) and np.allclose(LS @ Om @ LS.T, Om.T)
    assert np.allclose(LS @ LS, np.eye(1 << N))


def test_h_plus_product_state_is_eigenstate_of_every_gate():
    from qca_fragmentation.spectral import blockfirst as B
    N = 6
    v = _kr(N, lambda i: B.H_PLUS.reshape(2, 1)).ravel()
    for w in (54, 57, 60, 105, 108, 147, 150, 153, 156, 201):
        for bc in ("obc0", "pbc"):
            for i in range(N):
                assert np.allclose(B.gate_sparse(w, N, i, bc) @ v, v)


def test_rule54_singlet_closed_form():
    # the entangled singlet of 54 = |h+>^N projected off the frozen vacuum; its entropy
    import json
    from qca_fragmentation.spectral import analysis as A
    o = json.load(open(A.summary_path("H")))
    c = np.cos(np.pi / 8)
    for N in range(6, 15):
        NA, NB = N // 2, N - N // 2
        oA, oB, x = c ** NA, c ** NB, c ** N
        bas = lambda o_: np.array([[1.0, o_], [0.0, np.sqrt(1 - o_ * o_)]])
        C = np.outer(bas(oA)[:, 0], bas(oB)[:, 0]) - x * np.outer(bas(oA)[:, 1], bas(oB)[:, 1])
        p = np.linalg.svd(C / np.linalg.norm(C), compute_uv=False) ** 2
        S = -(p[p > 1e-16] * np.log(p[p > 1e-16])).sum()
        data = [b["S_vN"][0] for b in o[f"54_{N}"]["special_blocks"] if b["dim"] == 1]
        assert len(data) == 1 and abs(data[0] - S) < 1e-6


def test_rule156_segment_dispersion():
    # one mobile kink per segment of L free sites: cos(theta_m) = -sin^2(pi m/(L+1)), plus theta = 0
    from qca_fragmentation.spectral import blockfirst as B
    for L in range(1, 7):
        N = L + 3
        U = (B.layer_sparse(156, N, 1, "obc0") @ B.layer_sparse(156, N, 0, "obc0")).toarray()
        idx = [sum(1 << i for i in [0, L + 2] + list(range(1, 1 + c))) for c in range(L + 1)]
        th = np.angle(np.linalg.eigvals(U[np.ix_(idx, idx)]))
        pred = [0.0]
        for m in range(1, (L + 1) // 2 + 1):
            t = np.arccos(-np.sin(np.pi * m / (L + 1)) ** 2)
            pred += [t] if 2 * m == L + 1 else [t, -t]
        assert np.allclose(np.sort(np.exp(1j * th)), np.sort(np.exp(1j * np.array(pred))), atol=1e-9) or \
            np.allclose(np.sort(np.mod(th, 2 * np.pi)), np.sort(np.mod(pred, 2 * np.pi)), atol=1e-9)


def test_x57_domain_walls_conserved_on_ring_only():
    from qca_fragmentation.permutation import xca
    def walls(x, N, ring):
        b = [(x >> i) & 1 for i in range(N)]
        pairs = range(N) if ring else range(N - 1)
        return sum(b[i] != b[(i + 1) % N] for i in pairs)
    N = 10
    f = np.asarray(xca.step_table(57, N, "pbc"))
    assert all(walls(int(f[x]), N, True) == walls(x, N, True) for x in range(1 << N))
    f = np.asarray(xca.step_table(57, N, "obc0"))
    assert any(walls(int(f[x]), N, False) != walls(x, N, False) for x in range(1 << N))


def test_blockfirst_matches_pipeline():
    # ring: per-momentum phi spectra; open chain: the largest tower block of 153 and 60
    from qca_fragmentation.spectral import blockfirst as B
    for w in (57, 153):
        path = F.spec_path("H", 10, w, "pbc")
        if not os.path.exists(path):
            continue
        d = F.load(path)
        R = B.ring_phases(w, 10)
        mine = np.concatenate([np.concatenate(v["phi"]) for v in R.values()])
        assert mine.size == 2 ** 10
        ref = np.exp(1j * d["phi"][np.isfinite(d["phi"])])       # 1-state sectors carry no phi
        dist = np.abs(np.exp(1j * mine)[:, None] - ref[None, :]).min(axis=1)
        assert np.sum(dist > 1e-9) <= np.sum(~np.isfinite(d["phi"]))
    for w, N in ((153, 10), (60, 10)):
        Ub = B.fixed_site_block(w, N, B.TOWER_BLOCKS[w](N))
        assert np.allclose(Ub.T @ Ub, np.eye(Ub.shape[0]))
        d = F.load(F.spec_path("H", N, w))
        big = max(S.groups(S.block_keys(d)), key=lambda km: km[1].size)[1]
        assert np.allclose(S.unique_levels(np.angle(np.linalg.eigvals(Ub))), S.unique_levels(d["theta"][big]), atol=1e-9)


def test_x102_is_the_deterministic_floquet_quantum_east_model():
    # DFQEM (Bertini, De Fazio, Garrahan, Klobas, PRL 132, 120402): CNOT brickwork on a ring, a qubit flips iff its
    # RIGHT neighbour is 1.  Equal to our X-gate rule 102 up to the one-site shift.
    from qca_fragmentation.spectral import blockfirst as B
    N = 8; n = 1 << N; x = np.arange(n)
    def layer(targets):
        y = x.copy()
        for t in targets:
            y = y ^ (((x >> ((t + 1) % N)) & 1) << t)
        P = np.zeros((n, n)); P[y, x] = 1
        return P
    U_d = layer(range(0, N, 2)) @ layer(range(1, N, 2))
    U = (B.layer_sparse(102, N, 1, "pbc", gate="X") @ B.layer_sparse(102, N, 0, "pbc", gate="X")).toarray()
    T1 = _perm(N, lambda i: (i + 1) % N)
    assert np.allclose(U, U_d) or np.allclose(T1 @ U @ T1.T, U_d) or np.allclose(T1.T @ U @ T1, U_d)


def test_thermal_tools():
    from qca_fragmentation.spectral import thermal as T
    # RMT form factors: CUE ramp tau, COE 2 tau - tau ln(1 + 2 tau); both -> 1 at large tau
    assert np.isclose(T.sff_rmt(0.3, 2), 0.3) and np.isclose(T.sff_rmt(0.3, 1), 0.6 - 0.3 * np.log(1.6))
    assert abs(T.sff_rmt(20.0, 1) - 1) < 1e-2
    # Poisson levels: SFF ~ 1 at all t
    rng = np.random.default_rng(0)
    t, K = T.sff([rng.uniform(-np.pi, np.pi, 2000) for _ in range(5)], 50)
    assert abs(K[5:].mean() - 1) < 0.2
    # the scar |h+>^N never moves; a computational basis state of a chaotic rule spreads
    q = T.quench(57, 8, "obc0", "hplus", 20)
    assert np.allclose(q["R"], 1) and np.allclose(q["S"], 0, atol=1e-10)
    q = T.quench(57, 8, "pbc", "neel", 50)
    assert q["S"][-10:].mean() > 1.5
