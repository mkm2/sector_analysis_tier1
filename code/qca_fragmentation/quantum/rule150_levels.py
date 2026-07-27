"""
Level statistics of C150 (rule 150), obc0, wall shell by wall shell.

Methodology, and why it is not simply "compare to GUE".

1.  The one-cycle operator is REAL orthogonal -- every amplitude lies in
    Z[1/sqrt2] -- so complex conjugation K is an antiunitary symmetry, K U K = U,
    and the spectrum is closed under conjugation: eigenvalues come in pairs
    e^{+i theta}, e^{-i theta} plus real eigenvalues +-1.  That mirror pairing is
    a kinematic constraint, not a spectral correlation, so all statistics are
    computed on the INDEPENDENT half of the spectrum, theta in (0, pi), with the
    real eigenvalues +-1 removed.  Feeding the full circle to a spacing
    statistic would measure the mirror, not the dynamics.

2.  U_w has EXACT degeneracies (R8 sec.6): dim Fix(U_w) = C(ceil(N/2), w/2) at
    theta = 0, and further exact multiplicities 2, 3, 4, 6, 10, 15 that
    reflection does not explain.  Exact degeneracy means a commutant, and a
    spacing statistic run over a degenerate spectrum measures the degeneracies.
    Each exact degeneracy is therefore collapsed to a single level.

3.  For ODD N the site reflection P commutes with U exactly, so the shell splits
    into P = +-1 blocks and the raw spectrum is a SUPERPOSITION of two
    independent ones.  Superposition depresses every repulsion measure toward
    Poisson, so odd-N shells are analysed inside each parity block.  For even N
    reflection instead conjugates U to U^{-1} and gives no blocks.

4.  Reference ensembles are GENERATED and pushed through the identical pipeline
    rather than taken from remembered surmise constants:

      Poisson       uniform phases on (0, pi)                -- no repulsion
      COE           U = W^T W, W Haar CUE (symmetric unitary) -- beta = 1
      Haar O(D)     real orthogonal, the ensemble that shares C150's
                    conjugate-pairing constraint                -- the matched null
      CUE           Haar unitary                              -- beta = 2
      m x Haar O(D) m independent Haar-O spectra superposed    -- what unresolved
                    symmetry blocks look like

    Comparing to a reference that has been through the same half-circle
    restriction and degeneracy collapse is the only apples-to-apples version of
    the question.

Primary statistic: <r~> = <min(r, 1/r)> with r_i = s_i / s_{i-1}, which needs no
unfolding and so cannot be biased by a bad unfolding of a Floquet spectrum.
Secondary: the unfolded spacing distribution P(s) and its two-sample
Kolmogorov-Smirnov distance to each reference.

    python -m qca_fragmentation.quantum.rule150_levels --run
    python -m qca_fragmentation.quantum.rule150_levels --run --control 54
"""

from __future__ import annotations

import argparse
import json
import os
import time
from math import ceil, comb, pi
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import rule150_spectra as rs
from ..results_io import REPO_ROOT

LEVELS_PATH = os.path.join(REPO_ROOT, "analytics", "c150_levels.json")

# Degeneracy tolerance.  R8 sec.6: over the whole shell dataset the exact
# degeneracies close to 8.8e-15 while the smallest genuinely distinct gap is
# 4.8e-6, so 1e-9 separates them with three orders of margin either side.
DEG_TOL = 1e-9

REFERENCES = ("poisson", "coe", "orth", "cue", "orth_x2", "orth_x3")


# --- pipeline ----------------------------------------------------------------

def distinct_upper_phases(ev: np.ndarray, tol: float = DEG_TOL):
    """
    Eigenvalues -> independent, non-degenerate phases in (0, pi).

    Returns (phases ascending, diagnostics dict).
    """
    ph = np.sort(np.angle(ev))
    keep: List[float] = []
    n_exact = 0
    for p in ph:
        if keep and p - keep[-1] < tol:
            n_exact += 1
            continue
        keep.append(float(p))
    arr = np.array(keep)
    upper = np.sort(arr[(arr > tol) & (arr < pi - tol)])
    diag = {
        "dim": int(ev.shape[0]),
        "n_distinct": len(keep),
        "n_collapsed": n_exact,
        "degenerate_fraction": n_exact / max(1, ev.shape[0]),
        "n_upper": int(upper.shape[0]),
    }
    return upper, diag


def r_tilde(phases: np.ndarray) -> Tuple[float, int]:
    """<min(r, 1/r)> over consecutive spacing ratios.  Unfolding-free."""
    s = np.diff(phases)
    s = s[s > 0]
    if s.shape[0] < 3:
        return float("nan"), 0
    r = s[1:] / s[:-1]
    rt = np.minimum(r, 1.0 / r)
    return float(rt.mean()), int(rt.shape[0])


def unfolded_spacings(phases: np.ndarray, window: Optional[int] = None):
    """
    Spacings divided by the local mean spacing (centred moving average), then
    renormalised to unit mean.  Local unfolding avoids assuming a uniform
    spectral density on the half circle.
    """
    s = np.diff(phases)
    s = s[s > 0]
    n = s.shape[0]
    if n < 10:
        return np.array([])
    if window is None:
        window = max(11, (n // 20) | 1)
    window = min(window, n if n % 2 else n - 1)
    pad = window // 2
    padded = np.concatenate([s[:pad][::-1], s, s[-pad:][::-1]])
    kern = np.ones(window) / window
    local = np.convolve(padded, kern, mode="valid")[:n]
    out = s / local
    return out / out.mean()


# --- reference ensembles -----------------------------------------------------

def _haar_cue(D: int, rng) -> np.ndarray:
    A = (rng.standard_normal((D, D)) + 1j * rng.standard_normal((D, D))) / np.sqrt(2)
    Q, R = np.linalg.qr(A)
    d = np.diag(R)
    return Q * (d / np.abs(d))


def _haar_orth(D: int, rng) -> np.ndarray:
    A = rng.standard_normal((D, D))
    Q, R = np.linalg.qr(A)
    return Q * np.sign(np.diag(R))


def reference_phases(kind: str, D: int, rng) -> np.ndarray:
    """One sample of a reference ensemble, through the identical pipeline."""
    if kind == "poisson":
        return np.sort(rng.uniform(0.0, pi, D // 2))
    if kind == "cue":
        return distinct_upper_phases(np.linalg.eigvals(_haar_cue(D, rng)))[0]
    if kind == "coe":
        W = _haar_cue(D, rng)
        return distinct_upper_phases(np.linalg.eigvals(W.T @ W))[0]
    if kind == "orth":
        return distinct_upper_phases(np.linalg.eigvals(_haar_orth(D, rng)))[0]
    if kind.startswith("orth_x"):
        m = int(kind.split("x")[1])
        parts = [distinct_upper_phases(np.linalg.eigvals(_haar_orth(D // m, rng)))[0]
                 for _ in range(m)]
        return np.sort(np.concatenate(parts))
    raise ValueError(f"unknown reference {kind!r}")


def reference_table(D: int = 600, samples: int = 12, seed: int = 20260727) -> dict:
    """<r~> and a pooled unfolded-spacing sample for every reference."""
    rng = np.random.default_rng(seed)
    out: Dict[str, dict] = {}
    for kind in REFERENCES:
        rts, pooled = [], []
        for _ in range(samples):
            ph = reference_phases(kind, D, rng)
            rt, n = r_tilde(ph)
            if np.isfinite(rt):
                rts.append(rt)
            pooled.append(unfolded_spacings(ph))
        out[kind] = {
            "r_tilde_mean": float(np.mean(rts)),
            "r_tilde_std": float(np.std(rts)),
            "n_samples": samples,
            "D": D,
            "spacings": np.concatenate(pooled).tolist(),
        }
    return out


# --- C150 shells -------------------------------------------------------------

def _parity_blocks(N: int, w: int):
    """
    For odd N, the P = +-1 blocks of the shell (P commutes with U exactly);
    for even N, the single unresolved block.  Yields (label, U_block).
    """
    U, states = rs.sector_unitary(N, w)
    if N % 2 == 0:
        yield "all", U
        return
    P = rs.reflection_matrix(N, states)
    assert np.abs(P @ U - U @ P).max() < 1e-10, "P must commute for odd N"
    D = U.shape[0]
    for sgn, lab in ((+1, "P=+1"), (-1, "P=-1")):
        Q = (np.eye(D) + sgn * P) / 2
        u, s, _ = np.linalg.svd(Q)
        B = u[:, s > 0.5]
        yield lab, B.T @ U @ B


def particle_hole_partner_agrees(N: int, w: int) -> float:
    """
    For ODD N the shells w and N+1-w have the SAME spectrum (both weights are
    even because N+1 is even).  Returns the largest phase discrepancy, which is
    at machine precision -- so those shells must not be counted twice.
    """
    U1, _ = rs.sector_unitary(N, w)
    U2, _ = rs.sector_unitary(N, N + 1 - w)
    if U1.shape != U2.shape:
        return float("nan")
    a = np.sort(np.angle(np.linalg.eigvals(U1)))
    b = np.sort(np.angle(np.linalg.eigvals(U2)))
    return float(np.abs(a - b).max())


def shell_levels(N: int, w: int) -> List[dict]:
    """Level statistics of one wall shell, one record per symmetry block."""
    out = []
    for lab, Ub in _parity_blocks(N, w):
        ph, diag = distinct_upper_phases(np.linalg.eigvals(Ub))
        rt, n = r_tilde(ph)
        rec = {
            "N": N, "w": w, "block": lab,
            # filling of the BOND lattice: N+1 bonds carry w hard-core walls.
            # Fixed w with growing N is a few-body problem, so the chaos
            # question only has content near half filling.
            "filling": w / (N + 1.0),
            "shell_dim": comb(N + 1, w),
            "block_dim": int(Ub.shape[0]),
            "r_tilde": rt, "n_ratios": n,
            "spacings": unfolded_spacings(ph).tolist(),
        }
        rec.update(diag)
        out.append(rec)
    return out


def distinct_count_table(n_max: int = 13) -> List[dict]:
    """
    How many DISTINCT eigenvalues the full-space operator has, against the closed
    form of rs.distinct_spectrum_closed_form.  This is the decisive measurement of
    R8 sec.7: an exponentially degenerate spectrum with a closed-form count is
    not something any random-matrix ensemble produces.
    """
    out = []
    for N in range(4, n_max + 1):
        got = rs.distinct_spectrum_count(N)
        pred = rs.distinct_spectrum_closed_form(N)
        out.append({"N": N, "dim": 1 << N, "n_distinct": got,
                    "closed_form": pred, "matches": got == pred,
                    "distinct_fraction": got / float(1 << N)})
        print(f"  distinct N={N} 2^N={1<<N:6d} #distinct={got:6d} "
              f"closed_form={pred:6d} {'OK' if got == pred else 'MISMATCH'}",
              flush=True)
    return out


def ks_distances(spacings: List[float], refs: dict) -> Dict[str, float]:
    """Two-sample Kolmogorov-Smirnov distance to each reference's pooled P(s)."""
    from scipy.stats import ks_2samp
    s = np.asarray(spacings)
    if s.size < 30:
        return {}
    return {k: float(ks_2samp(s, np.asarray(v["spacings"])).statistic)
            for k, v in refs.items()}


def control_rule_levels(rule: int, N: int, bc: str = "obc0") -> List[dict]:
    """
    The same pipeline applied to the largest sector of another unitary rule --
    a methods control: it shows the pipeline can report both repulsion and its
    absence, and it is run at small N only (C150 is the only rule taken to large
    N in this project).
    """
    from ..core import rules
    from ..core.cycle import one_cycle_branches
    from ..graph import flip_graph

    t = rules.wolfram_to_tuple(rule)
    _, labels = flip_graph.flip_components_np(N, t, bc, return_labels=True)
    lab = np.asarray(labels)
    vals, counts = np.unique(lab, return_counts=True)
    biggest = vals[np.argmax(counts)]
    states = np.nonzero(lab == biggest)[0].tolist()
    idx = {x: k for k, x in enumerate(states)}
    D = len(states)
    U = np.zeros((D, D))
    for k, x in enumerate(states):
        (amps, m), = one_cycle_branches(x, N, t, bc)
        sc = 2.0 ** (-m)
        for y, (a, b) in amps.items():
            U[idx[y], k] = (a + b * np.sqrt(2.0)) * sc
    assert np.abs(U.T @ U - np.eye(D)).max() < 1e-9, "control U not orthogonal"
    ph, diag = distinct_upper_phases(np.linalg.eigvals(U))
    rt, n = r_tilde(ph)
    rec = {"rule": rule, "N": N, "bc": bc, "block": "largest sector",
           "block_dim": D, "r_tilde": rt, "n_ratios": n,
           "spacings": unfolded_spacings(ph).tolist()}
    rec.update(diag)
    return [rec]


# --- driver ------------------------------------------------------------------

#: Extra shells near half filling, past the general cap.  These are the only
#: ones that can speak to many-body chaos: a shell with fixed w and growing N is
#: w hard-core walls on N+1 bonds, i.e. a few-body problem, and must drift to
#: Poisson however chaotic the dense model is.
BIG_HALF_FILLING = ((13, 6), (14, 6), (15, 8), (16, 8))


def run(*, dim_cap: int = 3200, dim_floor: int = 200,
        big_shells: Tuple[Tuple[int, int], ...] = BIG_HALF_FILLING,
        controls: Tuple[int, ...] = (54, 57, 156),
        ref_D: int = 600, ref_samples: int = 12,
        force: bool = False) -> dict:
    """
    Level statistics for every shell we can diagonalise that is big enough to
    give meaningful statistics, plus references and controls.  Cached.
    """
    if os.path.exists(LEVELS_PATH) and not force:
        with open(LEVELS_PATH) as f:
            return json.load(f)

    t0 = time.time()
    refs = reference_table(D=ref_D, samples=ref_samples)
    print(f"references done [{time.time()-t0:.0f}s]", flush=True)

    shells: List[dict] = []
    phole: List[dict] = []
    for N in range(8, 20):
        for w in range(2, N + 2, 2):
            D = comb(N + 1, w)
            if D < dim_floor or D > dim_cap:
                continue
            if D <= N + 1:            # extremal shell: a one-particle problem
                continue
            if N % 2 == 1 and w > (N + 1) // 2:
                # odd N: this shell's spectrum is that of N+1-w, already done.
                # Verify the identity once instead of double counting it.
                d = particle_hole_partner_agrees(N, w)
                phole.append({"N": N, "w": w, "partner": N + 1 - w,
                              "max_phase_diff": d})
                print(f"  N={N} w={w} == w={N+1-w} (particle-hole), "
                      f"max|dphase|={d:.1e} -- not counted twice", flush=True)
                continue
            for rec in shell_levels(N, w):
                rec["ks"] = ks_distances(rec["spacings"], refs)
                shells.append(rec)
                print(f"  N={N} w={w} nu={rec['filling']:.2f} {rec['block']:6s} "
                      f"dim={rec['block_dim']:5d} <r~>={rec['r_tilde']:.4f} "
                      f"(n={rec['n_ratios']}) deg_frac="
                      f"{rec['degenerate_fraction']:.3f}", flush=True)

    done = {(r["N"], r["w"]) for r in shells}
    for (N, w) in big_shells:
        if (N, w) in done:
            continue
        print(f"  big shell N={N} w={w} dim={comb(N+1, w)} ...", flush=True)
        for rec in shell_levels(N, w):
            rec["ks"] = ks_distances(rec["spacings"], refs)
            rec["big"] = True
            shells.append(rec)
            print(f"  N={N} w={w} nu={rec['filling']:.2f} {rec['block']:6s} "
                  f"dim={rec['block_dim']:5d} <r~>={rec['r_tilde']:.4f} "
                  f"(n={rec['n_ratios']}) deg_frac="
                  f"{rec['degenerate_fraction']:.3f}", flush=True)

    ctrl: List[dict] = []
    for rule in controls:
        for N in (10, 12):
            try:
                for rec in control_rule_levels(rule, N):
                    if rec["n_ratios"] >= 40:
                        ctrl.append(rec)
                        print(f"  control W{rule} N={N} dim={rec['block_dim']} "
                              f"<r~>={rec['r_tilde']:.4f} (n={rec['n_ratios']})",
                              flush=True)
            except AssertionError as e:
                print(f"  control W{rule} N={N} skipped: {e}", flush=True)
    for r in ctrl:
        r["ks"] = ks_distances(r["spacings"], refs)

    out = {"references": refs, "shells": shells, "controls": ctrl,
           "particle_hole": phole, "distinct": distinct_count_table(),
           "deg_tol": DEG_TOL, "dim_cap": dim_cap, "dim_floor": dim_floor,
           "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "runtime": time.time() - t0}
    os.makedirs(os.path.dirname(LEVELS_PATH), exist_ok=True)
    with open(LEVELS_PATH, "w") as f:
        json.dump(out, f)
    return out


def summarise(data: dict) -> dict:
    """
    Group <r~> by FILLING.  A shell with fixed w and growing N is a few-body
    problem (w hard-core walls on N+1 bonds), so only the shells near half
    filling can say anything about many-body chaos; the dilute ones must and do
    drift to Poisson.
    """
    sh = data["shells"]

    def agg(rows):
        v = [r["r_tilde"] for r in rows if np.isfinite(r["r_tilde"])]
        if not v:
            return {"n_blocks": 0}
        return {"n_blocks": len(v), "r_tilde_mean": float(np.mean(v)),
                "r_tilde_min": float(np.min(v)), "r_tilde_max": float(np.max(v)),
                "deg_frac_mean": float(np.mean([r["degenerate_fraction"]
                                                for r in rows]))}

    dense = [r for r in sh if r["filling"] >= 0.35]
    dilute = [r for r in sh if r["filling"] < 0.35]
    return {
        "near_half_filling (nu >= 0.35)": agg(dense),
        "dilute (nu < 0.35)": agg(dilute),
        "dense_even_N": agg([r for r in dense if r["N"] % 2 == 0]),
        "dense_odd_N_parity_resolved": agg([r for r in dense if r["N"] % 2 == 1]),
        "controls": {f"W{r['rule']}_N{r['N']}": round(r["r_tilde"], 4)
                     for r in data.get("controls", [])},
        "references": {k: (round(v["r_tilde_mean"], 4), round(v["r_tilde_std"], 4))
                       for k, v in data["references"].items()},
        "particle_hole_max_phase_diff":
            max([r["max_phase_diff"] for r in data.get("particle_hole", [])],
                default=None),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="C150 level statistics")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dim-cap", type=int, default=3200)
    ap.add_argument("--ref-samples", type=int, default=12)
    args = ap.parse_args(argv)
    if args.run:
        data = run(dim_cap=args.dim_cap, ref_samples=args.ref_samples,
                   force=args.force)
        print(json.dumps(summarise(data), indent=1))


if __name__ == "__main__":
    main()
