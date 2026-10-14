"""
Level-spacing ratios and entanglement summaries of the Floquet spectra (R32).

Symmetry resolution.  A "block" is (Krylov sector, gate-algebra block, spatial
label), as stored by HSF/generate_spectral_data.jl.  Inside a block:

  * Exact degeneracies are removed (each degenerate eigenspace counts once).
  * U is real, so the block spectrum is symmetric under theta -> -theta (complex
    conjugation commutes with U and with every block).  U = L_B L_A is a product
    of two reflections, so by Jordan's lemma it is a direct sum of 2x2 rotations
    by +-theta_j (plus +-1 eigenvalues).  The independent data are the theta_j in
    [0, pi]; ratios are computed on that half spectrum (the mirror half would add
    duplicated, correlated ratios and spurious small gaps near 0 and pi).
  * Reference values: Poisson <r> = 2 ln 2 - 1 = 0.3863; GOE/COE (beta = 1)
    0.5307; GUE/CUE (beta = 2) 0.5996 (Atas et al. 2013).  For generic reflections
    the principal angles follow a real (beta = 1) Jacobi ensemble, so a chaotic
    block is expected at the beta = 1 value.

r_n = min(s_n, s_{n+1}) / max(s_n, s_{n+1}) over consecutive spacings of the
sorted, de-duplicated half spectrum (no wrap-around: [0, pi] is an interval).
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

R_POISSON = 2 * np.log(2) - 1          # 0.38629
R_GOE = 0.5307                         # Atas et al. 2013, large-N numerics
R_GUE = 0.5996
TOL = 1e-8                              # level coincidence


def surmise(r, beta):
    """Atas et al. surmise P(r) on r in [0, 1] (tilde-r, folded), normalised."""
    r = np.asarray(r, dtype=float)
    if beta == 0:
        return 2.0 / (1 + r) ** 2
    Z = {1: 8 / 27, 2: 4 * np.pi / (81 * np.sqrt(3)), 4: 4 * np.pi / (729 * np.sqrt(3))}[beta]
    return 2 * (r + r * r) ** beta / Z / (1 + r + r * r) ** (1 + 1.5 * beta)


def block_keys(d) -> np.ndarray:
    """Integer key per eigenvector identifying its (sector, block, label)."""
    # labels: obc0 v1 in {-1, 0, 1}; v2 / pbc 10 j + (r + 1) <= 10 (N/2) + 2 < 999
    return (d["sector_id"].astype(np.int64) * 1_000_003 + d["block_id"].astype(np.int64)) * 1000 \
        + (d["sym_label"].astype(np.int64) + 1)


def unique_levels(theta: np.ndarray, tol: float = TOL) -> np.ndarray:
    """Sorted, de-duplicated levels of one block on the half circle [0, pi]."""
    h = np.sort(np.abs(theta))                      # fold theta -> |theta|
    if h.size == 0:
        return h
    keep = np.concatenate(([True], np.diff(h) > tol))
    return h[keep]


def ratios(levels: np.ndarray) -> np.ndarray:
    s = np.diff(levels)
    if s.size < 2:
        return np.empty(0)
    a, b = s[:-1], s[1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.minimum(a, b) / np.maximum(a, b)
    return r[np.isfinite(r)]


def block_table(d, nmin: int = 20) -> List[Dict]:
    """One row per block: size, distinct half-levels, <r>, degeneracy share."""
    keys = block_keys(d)
    rows = []
    for k in np.unique(keys):
        m = keys == k
        lev = unique_levels(d["theta"][m])
        r = ratios(lev)
        rows.append({
            "key": int(k), "sector": int(d["sector_id"][m][0]),
            "block": int(d["block_id"][m][0]), "label": int(d["sym_label"][m][0]),
            "dim": int(m.sum()), "n_half_levels": int(lev.size),
            "r_mean": float(r.mean()) if r.size else np.nan, "n_r": int(r.size),
            "deg_frac": float(np.mean(d["degeneracy"][m] > 1)),
            "used": bool(lev.size >= nmin),
        })
    return rows


def pooled_r(d, nmin: int = 20):
    """All ratios from blocks with >= nmin distinct half-levels, and summary."""
    keys = block_keys(d)
    rs, nstates = [], 0
    for k in np.unique(keys):
        m = keys == k
        lev = unique_levels(d["theta"][m])
        if lev.size >= nmin:
            rs.append(ratios(lev)); nstates += int(m.sum())
    r = np.concatenate(rs) if rs else np.empty(0)
    n = r.size
    return r, {"r_mean": float(r.mean()) if n else np.nan,
               "r_err": float(r.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan,
               "n_r": int(n), "n_blocks": len(rs),
               "state_frac": nstates / len(d["theta"])}


def largest_block_r(d):
    """Ratios of the single largest resolved block (no mixing across blocks)."""
    keys = block_keys(d)
    uk, cnt = np.unique(keys, return_counts=True)
    m = keys == uk[np.argmax(cnt)]
    r = ratios(unique_levels(d["theta"][m]))
    return r, int(m.sum())


def page_entropy(NA: int, NB: int) -> float:
    """Page's mean entanglement entropy of a random pure state, m <= n."""
    m, n = 2 ** min(NA, NB), 2 ** max(NA, NB)
    return float(sum(1.0 / k for k in range(n + 1, m * n + 1)) - (m - 1) / (2 * n))


def degeneracy_summary(d) -> Dict:
    deg = d["degeneracy"]
    return {"frac_degenerate": float(np.mean(deg > 1)),
            "max_degeneracy": int(deg.max()),
            "n_distinct_global": int(np.unique(np.round(d["theta"], 9)).size)}
