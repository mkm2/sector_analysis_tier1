"""
C150: the full computational-basis entanglement census and its random-state
references (R8 sec.10).
"""

from math import comb, log

import numpy as np
import pytest

from qca_fragmentation import c150
from qca_fragmentation.quantum import rule150_basis_census as bc
from qca_fragmentation.quantum import rule150_entanglement as re
from qca_fragmentation.quantum import rule150_spectra as rs


# --- the batched machinery agrees with the scalar reference ------------------

@pytest.mark.parametrize("N,w", [(7, 4), (9, 4), (9, 6)])
def test_batch_entropies_matches_scalar(N, w):
    states = rs.sector_states(N, w)
    rows, cols, nb, na = rs.schmidt_index(N, states)
    rng = np.random.default_rng(0)
    V = rng.standard_normal((len(states), 6))
    V /= np.linalg.norm(V, axis=0)
    ref = np.array([rs.entropy_from_amplitudes(V[:, j], rows, cols, nb, na)
                    for j in range(6)])
    assert np.abs(bc.batch_entropies(V, rows, cols, nb, na) - ref).max() < 1e-12


def test_shell_traces_match_basis_trace():
    """Column j of U_w^t IS the trajectory of basis state j."""
    N, w = 9, 4
    tr, states = bc.shell_basis_traces(N, w, t_max=10)
    for k in (0, 5, 31):
        w2, ref = re.basis_trace(N, states[k], t_max=10)
        assert w2 == w
        assert np.abs(tr[k] - ref).max() < 1e-11


def test_traces_start_at_zero_and_respect_the_ceiling():
    N, w = 9, 4
    tr, _ = bc.shell_basis_traces(N, w, t_max=40)
    assert np.abs(tr[:, 0]).max() < 1e-12          # product states
    assert tr.max() <= re.kinematic_ceiling(N, w) + 1e-9


# --- Page's formula ----------------------------------------------------------

def test_page_value_two_qubits():
    """The textbook case: one qubit out of two, Page average = 1/3."""
    assert bc.page_value(2, 2) == pytest.approx(1.0 / 3.0, abs=1e-12)


def test_page_value_below_the_volume_law():
    for c in (3, 4, 5):
        assert bc.page_value(2 ** c, 2 ** c) < c * log(2.0)


# --- the census bookkeeping ---------------------------------------------------

def test_census_covers_the_whole_basis_and_splits_the_variance():
    N = 7
    d = bc.census(N, t_max=60, burn=20, n_haar=200, verbose=False)
    assert d["n_states"] == 2 ** N
    assert sum(s["size"] for s in d["shells"]) == 2 ** N
    assert [s["size"] for s in d["shells"]] == \
        [comb(N + 1, w) for w in range(0, N + 2, 2)]
    g = d["global"]
    # one-way ANOVA identity, exactly
    assert g["within_var"] + g["between_var"] == pytest.approx(g["var"],
                                                              abs=1e-12)
    assert 0.0 <= g["eta_squared"] <= 1.0


def test_frozen_singletons_have_zero_entropy():
    N = 7
    d = bc.census(N, t_max=40, burn=10, n_haar=100, verbose=False)
    for s in d["shells"]:
        if s["size"] == 1:
            assert s["mean"] == 0.0 and s["std"] == 0.0


def test_partner_shells_have_identical_statistics():
    """For odd N the shells w and N+1-w are spectrally identical (R8 open
    item 10); the entanglement census sees that too."""
    N = 9
    d = bc.census(N, t_max=80, burn=20, n_haar=100, verbose=False)
    by_w = {s["w"]: s for s in d["shells"]}
    for w in range(0, (N + 1) // 2 + 1, 2):
        p = N + 1 - w
        if p in by_w and w != p:
            assert by_w[w]["mean"] == pytest.approx(by_w[p]["mean"], abs=1e-9)
            assert by_w[w]["std"] == pytest.approx(by_w[p]["std"], abs=1e-9)


# --- the random-state references ---------------------------------------------

def test_phase_frame_reconstructs_powers_of_U():
    """U^t v = sum_theta [cos(theta t) a + sin(theta t) b] + P_+ v +/- P_- v."""
    N, w = 9, 4
    U, states = rs.sector_unitary(N, w, check=False)
    V, plus, minus, pos = bc._real_phase_frame(U)
    D = len(states)
    assert np.abs(V.conj().T @ V - np.eye(D)).max() < 1e-8
    j = 3
    c = np.conj(V[j, :])
    thv = np.array([t for t, _ in pos])
    A = np.column_stack([2 * (V[:, g] @ c[g]).real for _, g in pos])
    B = np.column_stack([-2 * (V[:, g] @ c[g]).imag for _, g in pos])
    pp = sum((V[:, g] @ c[g]).real for g in plus) if plus else np.zeros(D)
    pm = sum((V[:, g] @ c[g]).real for g in minus) if minus else np.zeros(D)
    for t in (1, 4, 9):
        rec = A @ np.cos(thv * t) + B @ np.sin(thv * t) + pp + (-1) ** t * pm
        assert np.abs(rec - np.linalg.matrix_power(U, t)[:, j]).max() < 1e-9


def test_dephasing_overshoots_the_true_time_average():
    """
    The point of R8 sec.10: even a random-state model that keeps the exact
    initial state AND the exact eigenspaces overshoots, because C150's phases
    are not rationally independent -- they are integer combinations of only
    m = ceil(N/2) fundamentals, so the orbit fills an m-torus, not the full one.
    """
    N, w = 9, 4
    dp = bc.dephasing_reference(N, w, n_states=6, n_phase=200, seed=0)
    U, states = rs.sector_unitary(N, w, check=False)
    rows, cols, nb, na = rs.schmidt_index(N, states)
    D = len(states)
    M = np.eye(D)
    acc = np.zeros(D)
    T = 800
    for _ in range(T + 1):
        acc += bc.batch_entropies(M, rows, cols, nb, na)
        M = U @ M
    acc /= T + 1
    true = acc[[states.index(s) for s in dp["states"]]].mean()
    assert dp["mean"] > true + 0.5, (dp["mean"], true)
    # and there really are far more distinct phases than fundamentals
    assert dp["n_distinct_pos_phases"] > (N + 1) // 2


def test_haar_shell_mean_exceeds_the_measured_saturation():
    N, w = 9, 4
    d = bc.census(N, t_max=200, burn=50, n_haar=400, verbose=False)
    s = [x for x in d["shells"] if x["w"] == w][0]
    assert s["haar_shell"]["mean"] > s["mean"]
    assert s["mean"] <= s["ceiling"] + 1e-9


# --- the substructure inside a shell ------------------------------------------

@pytest.mark.parametrize("N", [3, 7, 9, 11, 13, 15, 19])
def test_mirror_complement_count(N):
    """R = P o (xor A) has fixed points only when N = 3 mod 4, and then
    exactly 4^{(N+1)/4} of them."""
    fx = bc.mirror_complement_fixed(N)
    assert len(fx) == bc.mirror_complement_count(N)
    assert (len(fx) > 0) == (N % 4 == 3)


@pytest.mark.parametrize("N", [3, 7, 11, 15])
def test_mirror_complement_forces_the_middle_shell(N):
    """In bond variables the condition is b_{N-j} = 1 xor b_j, so each mirror
    pair of bonds holds exactly one wall and w = (N+1)/2 follows."""
    for x in bc.mirror_complement_fixed(N):
        b = c150.wall_string(x, N, "obc0")
        for j in range(N + 1):
            assert ((b >> j) & 1) ^ ((b >> (N - j)) & 1) == 1
        assert c150.wall_number(x, N, "obc0") == (N + 1) // 2


def test_mirror_complement_states_top_their_shell_with_a_gap():
    """At N=7 the 16 R-fixed states sit above the whole rest of the w=4 shell,
    separated by an empty gap -- the isolated spike of R8 sec.10."""
    d = bc.spike_separation(7, t_max=600, burn=100)
    assert d["n_fixed"] == 16 and d["w"] == 4
    assert d["separated"]
    assert d["gap"] > 0.02


def test_no_spike_when_N_is_1_mod_4():
    assert bc.spike_separation(9)["n_fixed"] == 0


def test_wall_complement_is_an_exact_entropy_symmetry():
    """x -> x^A maps shell w to N+1-w and preserves the saturation entropy
    exactly, which is why the partner shells have identical censuses."""
    N = 7
    cs = bc.census(N, t_max=300, burn=100, n_haar=50, verbose=False)
    assert bc.complement_symmetry_check(N, cs)["max_abs_deviation"] < 1e-12


def test_haar_full_is_close_to_page():
    N = 9
    hf = bc.haar_full_stats(N, n_samples=400, seed=0)
    # real (beta=1) sits just below the complex Page value, both below ln 2^c
    assert hf["mean_real"] < hf["page_complex"] + 0.02
    assert hf["page_complex"] < hf["volume_law"]
