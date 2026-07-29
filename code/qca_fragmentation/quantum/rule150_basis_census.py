"""
C150 obc0, N=11: entanglement saturation for EVERY computational basis state,
clustered by sector, against what a random (Haar) state would have predicted.

The whole computational basis is 2^N states, and every one of them is a product
state, so S(0) = 0 for all of them.  Each lies in exactly one wall shell w, and
the shell sizes are the even-weight entries of Pascal's row N+1 (R8 sec.4):

    N = 11:  w = 0  2   4    6    8    4   12
             |S_w| 1  66  495  924  495  66   1      (sum 2048 = 2^11)

Because the dynamics is block diagonal, the trajectories of ALL |S_w| basis
states of one shell are the columns of U_w^t.  So one matrix power per time step
gives the whole shell at once -- no per-state loop -- and the entropies come
from a batched eigendecomposition of the compressed Schmidt Gram matrices.

Two questions this answers.

1.  How much of the saturation value does the sector fix?  Reported as the
    one-way variance decomposition (eta^2) over the 2048 states: between-cluster
    variance / total variance.  R8 sec.9 showed the sector fixes the *type*
    (area vs volume) and the *ceiling*; here we ask how tightly it fixes the
    number.

2.  Could a Haar random state have told us the answer more cheaply?  Three
    reference ensembles, in increasing order of how much structure they know:

      full     Haar on the whole 2^N space           (knows nothing)
      shell    Haar on S_w                           (knows the sector)
      measured the actual time-averaged basis states (knows the dynamics)

    S is not a linear functional of rho, so E[S] is not S(E[rho]) -- but that
    alone is not the obstruction, because Page's formula computes E[S] for the
    Haar ensemble exactly.  The obstruction is that Haar predicts the WRONG
    SPREAD: concentration of measure makes Var[S] ~ 1/D^2 for a random state,
    while the basis states of one shell are strongly non-typical and spread out
    over an O(0.1) range.  See haar_shell_stats() vs census().

    python -m qca_fragmentation.quantum.rule150_basis_census --run
"""

from __future__ import annotations

import argparse
import json
import os
import time
from math import comb, log
from typing import Dict, List, Optional

import numpy as np

from .. import c150
from ..results_io import REPO_ROOT
from . import rule150_entanglement as re
from . import rule150_spectra as rs

CENSUS_PATH = os.path.join(REPO_ROOT, "analytics", "c150_basis_census.json")

#: default system size -- big enough for a 924-state shell, small enough that
#: the full basis is exactly enumerable.
N_DEFAULT = 11
#: C150 is NOT chaotic: there is no fast relaxation to a plateau, only a
#: quasi-periodic S(t) whose *time average* exists.  The spectral factorisation
#: (R-T13) writes every phase as sum_i eps_i phi_i with m = ceil(N/2) = 6
#: fundamentals, so S(t) wanders on a 6-torus and the running mean converges
#: like a 6-frequency quasi-periodic average -- slowly.  Empirically the
#: cumulative mean of the w=6 shell is stable to +-0.002 only after t ~ 800,
#: and a naive t in [200,400] window is 2% low.  Hence the long window.
T_MAX = 4000
BURN = 200
N_HAAR = 4000

#: batched entropy chunk (states per call); bounds the (chunk, n_b, n_a) tensor
CHUNK = 512


# --- batched half-chain entropy ---------------------------------------------

def batch_entropies(M: np.ndarray, rows, cols, n_b: int, n_a: int,
                    chunk: int = CHUNK) -> np.ndarray:
    """
    Half-chain entropies of many sector-supported states at once.

    `M` is (D, n_vec): column j holds the amplitudes of state j on the shell
    basis.  Same compressed Schmidt layout as
    rule150_spectra.entropy_from_amplitudes, but the (n_b, n_a) matrices are
    stacked and diagonalised in one batched call, which is what makes a
    full-basis census affordable.
    """
    n_vec = M.shape[1]
    out = np.empty(n_vec)
    small = min(n_b, n_a)
    for s in range(0, n_vec, chunk):
        e = min(s + chunk, n_vec)
        T = np.zeros((e - s, n_b, n_a), dtype=np.float64)
        T[:, rows, cols] = M[:, s:e].T
        G = T @ T.transpose(0, 2, 1) if n_b == small else T.transpose(0, 2, 1) @ T
        ev = np.linalg.eigvalsh(G)
        np.clip(ev, 0.0, None, out=ev)
        ev /= ev.sum(axis=1, keepdims=True)
        lg = np.where(ev > 1e-24, np.log(np.where(ev > 1e-24, ev, 1.0)), 0.0)
        out[s:e] = -(ev * lg).sum(axis=1)
    return out


def shell_basis_traces(N: int, w: int, *, t_max: int = T_MAX,
                       cut: Optional[int] = None, U=None, states=None):
    """
    S(t) for every basis state of shell w, as an (|S_w|, t_max+1) array.

    Column j of U_w^t is the time-t state that started from basis state j, so
    the whole shell evolves with one matrix multiply per step.
    """
    if U is None:
        U, states = rs.sector_unitary(N, w, check=False)
    rows, cols, n_b, n_a = rs.schmidt_index(N, states, cut)
    D = len(states)
    M = np.eye(D)
    out = np.empty((D, t_max + 1))
    for t in range(t_max + 1):
        out[:, t] = batch_entropies(M, rows, cols, n_b, n_a)
        M = U @ M
    return out, states


# --- reference ensembles ------------------------------------------------------

def page_value(d_a: int, d_b: int) -> float:
    """Page's exact average entropy of a Haar random COMPLEX pure state on a
    d_a x d_b bipartition, d_a <= d_b:  sum_{k=d_b+1}^{d_a d_b} 1/k
    - (d_a-1)/(2 d_b)."""
    if d_a > d_b:
        d_a, d_b = d_b, d_a
    k = np.arange(d_b + 1, d_a * d_b + 1, dtype=np.float64)
    return float(np.sum(1.0 / k) - (d_a - 1) / (2.0 * d_b))


def haar_shell_stats(N: int, w: int, *, n_samples: int = N_HAAR,
                     seed: int = 0, cut: Optional[int] = None,
                     states=None, complex_: bool = False) -> dict:
    """
    Mean and variance of S over states drawn Haar-uniformly from the shell S_w.

    Real by default: U is real orthogonal, so the states the dynamics actually
    produces from real basis states are real, and the matching random-state
    ensemble is the uniform measure on the real unit sphere of S_w (beta = 1),
    not the complex one.
    """
    if states is None:
        states = rs.sector_states(N, w)
    D = len(states)
    if D < 2:
        return {"n_samples": 0, "mean": 0.0, "var": 0.0, "std": 0.0, "sem": 0.0}
    rows, cols, n_b, n_a = rs.schmidt_index(N, states, cut)
    rng = np.random.default_rng(seed)
    vals = []
    done = 0
    while done < n_samples:
        m = min(CHUNK, n_samples - done)
        V = rng.standard_normal((D, m))
        if complex_:
            V = V + 1j * rng.standard_normal((D, m))
        V /= np.linalg.norm(V, axis=0, keepdims=True)
        if complex_:                      # batch_entropies is the real path
            vals.append(np.array([rs.entropy_from_amplitudes(
                V[:, j], rows, cols, n_b, n_a) for j in range(m)]))
        else:
            vals.append(batch_entropies(V, rows, cols, n_b, n_a))
        done += m
    v = np.concatenate(vals)
    return {"n_samples": int(v.size), "mean": float(v.mean()),
            "var": float(v.var(ddof=1)), "std": float(v.std(ddof=1)),
            "sem": float(v.std(ddof=1) / np.sqrt(v.size))}


def haar_full_stats(N: int, *, n_samples: int = N_HAAR, seed: int = 0,
                    cut: Optional[int] = None) -> dict:
    """The naive reference: Haar on the whole 2^N space, sector structure
    ignored."""
    c = N // 2 if cut is None else cut
    d_a, d_b = 2 ** c, 2 ** (N - c)
    rng = np.random.default_rng(seed)
    vals = []
    done = 0
    while done < n_samples:
        m = min(CHUNK, n_samples - done)
        V = rng.standard_normal((m, d_b, d_a))
        V /= np.linalg.norm(V.reshape(m, -1), axis=1)[:, None, None]
        G = V.transpose(0, 2, 1) @ V if d_a <= d_b else V @ V.transpose(0, 2, 1)
        ev = np.clip(np.linalg.eigvalsh(G), 0.0, None)
        ev /= ev.sum(axis=1, keepdims=True)
        lg = np.where(ev > 1e-24, np.log(np.where(ev > 1e-24, ev, 1.0)), 0.0)
        vals.append(-(ev * lg).sum(axis=1))
        done += m
    v = np.concatenate(vals)
    return {"n_samples": int(v.size), "mean_real": float(v.mean()),
            "var_real": float(v.var(ddof=1)),
            "std_real": float(v.std(ddof=1)),
            "page_complex": page_value(d_a, d_b),
            "volume_law": min(c, N - c) * log(2.0)}


# --- the sharpest random-state model: random phases on the real spectrum ------

def _real_phase_frame(U: np.ndarray, tol: float = 1e-9):
    """
    Decompose a real orthogonal U into its real invariant frame.

    U^t v = sum_{theta in (0,pi)} [cos(theta t) a_theta + sin(theta t) b_theta]
            + P_+ v + (-1)^t P_- v,

    with a_theta = 2 Re(P_theta v), b_theta = -2 Im(P_theta v), where P_theta
    projects on the eigenspace of e^{i theta}.  Returns the eigenvector matrix
    together with the index groups, so that the frame of any initial vector is
    one pass over the groups.
    """
    ev, V = np.linalg.eig(U)
    th = np.angle(ev)
    order = np.argsort(th)
    th, V = th[order], V[:, order]
    groups, start = [], 0
    for k in range(1, len(th) + 1):
        if k == len(th) or th[k] - th[start] > tol:
            groups.append((float(th[start:k].mean()),
                           np.arange(start, k)))
            start = k
    # np.linalg.eig does NOT return an orthonormal basis inside a degenerate
    # eigenspace, and C150's spectrum is massively degenerate (3^m distinct
    # phases on 2^N states).  P = sum_k v_k v_k^dag is only a projector for an
    # orthonormal basis, so re-orthonormalise each cluster.
    for _, g in groups:
        if g.size > 1:
            V[:, g] = np.linalg.qr(V[:, g])[0]
    err = np.abs(V.conj().T @ V - np.eye(V.shape[0])).max()
    assert err < 1e-8, f"eigenbasis not unitary after cleanup: {err:.2e}"
    plus = [g for t, g in groups if abs(t) <= tol]
    minus = [g for t, g in groups if abs(abs(t) - np.pi) <= tol]
    pos = [(t, g) for t, g in groups if tol < t < np.pi - tol]
    return V, plus, minus, pos


def dephasing_reference(N: int, w: int, *, n_states: int = 24,
                        n_phase: int = 400, seed: int = 0,
                        cut: Optional[int] = None, verify: bool = True) -> dict:
    """
    The best random-state model that still respects the dynamics: keep the exact
    initial state and the exact eigenspace decomposition, and replace the
    coherent phases theta_lambda * t by INDEPENDENT uniform random phases.

    This is the "maximal dephasing" / diagonal-ensemble estimate: it is what one
    would predict if the eigenphases were rationally independent, so that Weyl
    equidistribution filled the whole (#distinct phase) torus.  For C150 they
    are NOT independent -- the spectral factorisation writes every phase as
    theta = sum_i eps_i phi_i over only m = ceil(N/2) fundamentals -- so the
    trajectory is confined to an m-torus inside a much bigger one.  The gap
    between this reference and the true time average is exactly the price of
    that resonance structure.
    """
    U, states = rs.sector_unitary(N, w, check=False)
    rows, cols, nb, na = rs.schmidt_index(N, states, cut)
    D = len(states)
    V, plus, minus, pos = _real_phase_frame(U)
    rng = np.random.default_rng(seed)
    idx = rng.choice(D, size=min(n_states, D), replace=False)

    n_th = len(pos)
    means, tavg = [], []
    for j in idx:
        c = np.conj(V[j, :])                       # <v_k | e_j >
        A = np.empty((D, n_th)); B = np.empty((D, n_th))
        for q, (_, g) in enumerate(pos):
            z = V[:, g] @ c[g]
            A[:, q] = 2.0 * z.real
            B[:, q] = -2.0 * z.imag
        p_plus = sum((V[:, g] @ c[g]).real for g in plus) if plus \
            else np.zeros(D)
        p_minus = sum((V[:, g] @ c[g]).real for g in minus) if minus \
            else np.zeros(D)
        if verify:                                  # the frame reproduces U^t
            for t in (1, 5, 13):
                thv = np.array([t_ for t_, _ in pos])
                rec = (A @ np.cos(thv * t) + B @ np.sin(thv * t)
                       + p_plus + (-1) ** t * p_minus)
                ref = np.linalg.matrix_power(U, t)[:, j]
                assert np.abs(rec - ref).max() < 1e-8, np.abs(rec - ref).max()
            verify = False
        al = rng.uniform(0, 2 * np.pi, size=(n_th, n_phase))
        sg = rng.choice([-1.0, 1.0], size=n_phase)
        W = (A @ np.cos(al) + B @ np.sin(al)
             + p_plus[:, None] + sg[None, :] * p_minus[:, None])
        W /= np.linalg.norm(W, axis=0, keepdims=True)
        means.append(float(batch_entropies(W, rows, cols, nb, na).mean()))
    return {"w": w, "n_states": int(idx.size), "n_phase": n_phase,
            "n_distinct_pos_phases": n_th,
            "states": [int(states[j]) for j in idx],
            "per_state_mean": means,
            "mean": float(np.mean(means))}


# --- the census ---------------------------------------------------------------

def census(N: int = N_DEFAULT, *, t_max: int = T_MAX, burn: int = BURN,
           cut: Optional[int] = None, n_haar: int = N_HAAR,
           seed: int = 0, verbose: bool = True) -> dict:
    """
    Evolve every computational basis state, cluster by sector, and compare the
    saturation statistics with the Haar references.
    """
    c = N // 2 if cut is None else cut
    t0 = time.time()
    shells: List[dict] = []
    all_sat, all_w = [], []
    for w in range(0, N + 2, 2):
        D = comb(N + 1, w)
        if verbose:
            print(f"  w={w:2d}  D={D:5d}", flush=True)
        if D == 1:                                    # frozen singleton
            states = rs.sector_states(N, w)
            sat = np.zeros(1)
            tstd = np.zeros(1)
            haar = {"n_samples": 0, "mean": 0.0, "var": 0.0, "std": 0.0,
                    "sem": 0.0}
        else:
            tr, states = shell_basis_traces(N, w, t_max=t_max, cut=c)
            sat = tr[:, burn:].mean(axis=1)
            tstd = tr[:, burn:].std(axis=1)
            haar = haar_shell_stats(N, w, n_samples=n_haar, seed=seed + w,
                                    cut=c, states=states)
        all_sat.append(sat)
        all_w.append(np.full(sat.size, w))
        ceil_ = re.kinematic_ceiling(N, w, c)

        def spin(x: int) -> str:                      # site 0 leftmost
            return format(x, f"0{N}b")[::-1]

        shells.append({
            "w": w, "size": int(D), "ceiling": ceil_,
            "argmin_state": spin(states[int(np.argmin(sat))]),
            "argmax_state": spin(states[int(np.argmax(sat))]),
            "mean": float(sat.mean()),
            "var": float(sat.var(ddof=1)) if D > 1 else 0.0,
            "std": float(sat.std(ddof=1)) if D > 1 else 0.0,
            "min": float(sat.min()), "max": float(sat.max()),
            "time_std_mean": float(tstd.mean()),
            "frac_of_ceiling": float(sat.mean() / ceil_) if ceil_ > 0 else 0.0,
            "haar_shell": haar,
            "haar_minus_measured": haar["mean"] - float(sat.mean()),
        })
    sat = np.concatenate(all_sat)
    wv = np.concatenate(all_w)
    assert sat.size == 2 ** N, (sat.size, 2 ** N)

    # one-way variance decomposition: how much does knowing w buy you?
    grand = float(sat.mean())
    within = float(np.sum([(s.size) * s.var() for s in all_sat]) / sat.size)
    between = float(np.sum([s.size * (s.mean() - grand) ** 2
                            for s in all_sat]) / sat.size)
    total = float(sat.var())

    haar_full = haar_full_stats(N, n_samples=n_haar, seed=seed, cut=c)
    # sector-weighted Haar: the best a random-state argument can do while still
    # respecting the block structure
    haar_weighted = float(np.sum([sh["size"] * sh["haar_shell"]["mean"]
                                  for sh in shells]) / sat.size)

    out = {
        "N": N, "cut": c, "t_max": t_max, "burn": burn,
        "n_states": int(sat.size), "n_haar": n_haar, "seed": seed,
        "shells": shells,
        "global": {
            "mean": grand, "var": total, "std": float(sat.std()),
            "min": float(sat.min()), "max": float(sat.max()),
            "median": float(np.median(sat)),
            "within_var": within, "between_var": between,
            "eta_squared": between / total if total > 0 else 0.0,
        },
        "haar_full": haar_full,
        "haar_sector_weighted": haar_weighted,
        "runtime": time.time() - t0,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    # keep the raw per-state numbers: cheap at 2^11 and needed for the histogram
    out["raw"] = {"w": wv.astype(int).tolist(),
                  "S_sat": np.round(sat, 9).tolist()}
    return out


def split_half_check(N: int = N_DEFAULT, w: int = 6, *, t_max: int = T_MAX,
                     burn: int = BURN, cut: Optional[int] = None) -> dict:
    """
    Is the intra-cluster spread real, or just residual time-averaging noise?

    Average each basis state over two DISJOINT halves of the window.  If the
    spread were noise the two halves would be uncorrelated; if it is a genuine
    state-to-state property they agree.  The noise level is estimated from the
    half-to-half difference: sigma_noise = std(A - B)/2 for the full-window
    average.
    """
    tr, _ = shell_basis_traces(N, w, t_max=t_max, cut=cut)
    mid = (burn + t_max) // 2
    a, b = tr[:, burn:mid].mean(axis=1), tr[:, mid:].mean(axis=1)
    full = tr[:, burn:].mean(axis=1)
    noise = float(np.std(a - b, ddof=1) / 2.0)
    spread = float(full.std(ddof=1))
    return {"w": w, "t_max": t_max, "burn": burn,
            "spread": spread, "noise_estimate": noise,
            "half_correlation": float(np.corrcoef(a, b)[0, 1]),
            "signal_to_noise": spread / noise if noise > 0 else float("inf"),
            "true_spread": float(np.sqrt(max(spread ** 2 - noise ** 2, 0.0)))}


def one_design_check(N: int = N_DEFAULT, ts=(0, 1, 7, 33),
                     cut: Optional[int] = None) -> dict:
    """
    Why a first-moment (Haar) argument can say nothing about S.

    The computational basis is an exact 1-design: E_x[|x><x|] = I/2^N.  Unitary
    evolution preserves that, so E_x[rho_A(t)] = I_A/d_A at EVERY t.  Hence

      * every LINEAR observable has the same basis average as the Haar average,
        exactly, and that average does not move under the dynamics at all;
      * S is concave, so E_x[S] <= S(E_x[rho_A]) = ln d_A -- the volume law --
        which is true but empty: it holds with equality nowhere and is saturated
        by the t=0 product states where every S is exactly 0.

    Returns the deviation of the averaged reduced state from I_A/d_A (machine
    precision) alongside the two sides of the concavity bound.
    """
    c = N // 2 if cut is None else cut
    d_a, d_b = 2 ** c, 2 ** (N - c)
    D = 2 ** N
    full = np.zeros((D, D))
    for w in range(0, N + 2, 2):
        U, states = rs.sector_unitary(N, w, check=False)
        ix = np.array(states)
        full[np.ix_(ix, ix)] = U
    assert np.abs(full.T @ full - np.eye(D)).max() < 1e-9
    out = []
    M = np.eye(D)
    for t in range(max(ts) + 1):
        if t in ts:
            psi = M.reshape(d_b, d_a, D)              # (x_B, x_A, which state)
            rho = np.einsum("bai,bci->ac", psi, psi) / D
            ent = _full_entropies(M, d_a, d_b)
            out.append({
                "t": t,
                "dev_from_maximally_mixed": float(
                    np.abs(rho - np.eye(d_a) / d_a).max()),
                "S_of_mean_rho": float(log(d_a)),
                "mean_S": float(ent.mean()),
            })
        M = full @ M
    return {"cut": c, "d_a": d_a, "checks": out}


def _full_entropies(M: np.ndarray, d_a: int, d_b: int) -> np.ndarray:
    """Half-chain entropies of every column of M, in the uncompressed layout."""
    D = M.shape[1]
    out = np.empty(D)
    for s in range(0, D, CHUNK):
        e = min(s + CHUNK, D)
        T = M[:, s:e].T.reshape(e - s, d_b, d_a)
        G = (T.transpose(0, 2, 1) @ T if d_a <= d_b
             else T @ T.transpose(0, 2, 1))
        ev = np.clip(np.linalg.eigvalsh(G), 0.0, None)
        ev /= ev.sum(axis=1, keepdims=True)
        lg = np.where(ev > 1e-24, np.log(np.where(ev > 1e-24, ev, 1.0)), 0.0)
        out[s:e] = -(ev * lg).sum(axis=1)
    return out


def convergence_check(N: int = N_DEFAULT, w: int = 6, *, t_max: int = T_MAX,
                      cut: Optional[int] = None) -> dict:
    """
    Is the averaging window long enough?  Reports the cumulative time average of
    the shell-averaged S at growing horizons (should stop moving) and the
    non-overlapping block means (should scatter about it without trend).
    """
    tr, _ = shell_basis_traces(N, w, t_max=t_max, cut=cut)
    sm = tr.mean(axis=0)
    cum = np.cumsum(sm) / np.arange(1, sm.size + 1)
    horizons = [h for h in (100, 200, 400, 800, 1600, 3200, t_max)
                if h <= t_max]
    nb = 8
    edges = np.linspace(0, t_max + 1, nb + 1).astype(int)
    blocks = [float(sm[a:b].mean()) for a, b in zip(edges[:-1], edges[1:])]
    return {"w": w, "t_max": t_max,
            "cumulative": [{"t": int(h), "mean": float(cum[h])}
                           for h in horizons],
            "block_means": blocks,
            "block_scatter": float(np.std(blocks, ddof=1)),
            "revival": {"min_shell_mean_after_t0": float(sm[1:].min()),
                        "argmin_t": int(1 + np.argmin(sm[1:]))}}


def run(*, force: bool = False, N: int = N_DEFAULT) -> dict:
    if os.path.exists(CENSUS_PATH) and not force:
        with open(CENSUS_PATH) as f:
            return json.load(f)
    print(f"C150 obc0 N={N}: full computational-basis census")
    out = census(N)
    out["convergence"] = convergence_check(N, 6)
    print("  dephasing reference:", flush=True)
    out["dephasing"] = [dephasing_reference(N, w) for w in (2, 4, 6)]
    print("  1-design check:", flush=True)
    out["one_design"] = one_design_check(N)
    out["cut_alt"] = None
    if N % 2 == 1:                       # odd N: the other half-chain cut
        alt = census(N, cut=N - N // 2, n_haar=1000, verbose=False)
        out["cut_alt"] = {"cut": alt["cut"], "global": alt["global"],
                          "haar_sector_weighted": alt["haar_sector_weighted"],
                          "shells": [{k: s[k] for k in
                                      ("w", "mean", "std", "ceiling")}
                                     for s in alt["shells"]]}
    os.makedirs(os.path.dirname(CENSUS_PATH), exist_ok=True)
    with open(CENSUS_PATH, "w") as f:
        json.dump(out, f)
    return out


def summarise(d: dict) -> str:
    g, N = d["global"], d["N"]
    L = [f"C150 obc0 N={N}, cut={d['cut']}, all {d['n_states']} basis states, "
         f"S = mean over t in [{d['burn']},{d['t_max']}]", "",
         f"{'w':>3} {'|S_w|':>6} {'mean':>7} {'std':>7} {'min':>7} {'max':>7} "
         f"{'ceil':>7} {'frac':>6} {'Haar_w':>7} {'Haar-S':>7}"]
    for s in d["shells"]:
        L.append(f"{s['w']:>3} {s['size']:>6} {s['mean']:>7.4f} "
                 f"{s['std']:>7.4f} {s['min']:>7.4f} {s['max']:>7.4f} "
                 f"{s['ceiling']:>7.4f} {s['frac_of_ceiling']:>6.3f} "
                 f"{s['haar_shell']['mean']:>7.4f} "
                 f"{s['haar_minus_measured']:>7.4f}")
    hf = d["haar_full"]
    if d.get("dephasing"):
        L += ["", "random-phase (dephasing) reference vs measured:"]
        by_w = {s["w"]: s["mean"] for s in d["shells"]}
        for dp in d["dephasing"]:
            L.append(f"  w={dp['w']:>2}  {dp['n_distinct_pos_phases']:>4} "
                     f"distinct phases -> dephased {dp['mean']:.4f}  "
                     f"vs measured {by_w[dp['w']]:.4f}")
    L += ["",
          f"global mean {g['mean']:.4f}  std {g['std']:.4f}  "
          f"median {g['median']:.4f}  range [{g['min']:.4f}, {g['max']:.4f}]",
          f"variance split: within {g['within_var']:.4f}  "
          f"between {g['between_var']:.4f}  eta^2 = {g['eta_squared']:.3f}",
          "",
          f"Haar, sector-weighted   {d['haar_sector_weighted']:.4f}",
          f"Haar, full space (real) {hf['mean_real']:.4f} "
          f"+- {hf['std_real']:.4f}",
          f"Page, full space (cplx) {hf['page_complex']:.4f}",
          f"volume law ln2*{min(d['cut'], N - d['cut'])}      "
          f"{hf['volume_law']:.4f}"]
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="C150: entanglement saturation over the whole basis")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("-N", type=int, default=N_DEFAULT)
    args = ap.parse_args(argv)
    if args.run:
        d = run(force=args.force, N=args.N)
        print()
        print(summarise(d))
        print(f"\nsaved {CENSUS_PATH} [{d['runtime']:.0f}s]")


if __name__ == "__main__":
    main()
