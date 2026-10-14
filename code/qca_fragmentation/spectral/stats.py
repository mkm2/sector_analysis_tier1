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


def conj_symmetric(x: np.ndarray, tol: float = 1e-7) -> bool:
    """
    True if the multiset {e^{ix}} is closed under complex conjugation, to tol.
    Every -x is matched to its nearest level on the circle.  (Comparing the sorted
    x and -x elementwise is wrong when a level sits at 0 = 2 pi: a rounding of
    +-1e-16 moves it to the other end of the sorted list and shifts every pair.)
    """
    x = np.sort(np.mod(np.asarray(x, float), 2 * np.pi))
    if x.size == 0:
        return True
    y = np.mod(-x, 2 * np.pi)
    i = np.searchsorted(x, y)
    lo, hi = x[(i - 1) % x.size], x[i % x.size]
    circ = lambda a: np.minimum(np.abs(a), 2 * np.pi - np.abs(a))
    return bool(np.all(np.minimum(circ(y - lo), circ(y - hi)) < tol))


def block_ratios(x: np.ndarray, tol: float = TOL) -> np.ndarray:
    """
    Gap ratios of one resolved block from its level variable x (theta, or the
    eigenphase phi of a square-root symmetry), exact degeneracies removed.

    If the block spectrum is symmetric under x -> -x (complex conjugation maps the
    block to itself: every real symmetry sector), the independent data are |x| in
    [0, pi] and ratios are taken on that interval.  Otherwise (e.g. a ring-momentum
    sector j != -j, whose conjugate partner is another sector) the full circle is
    used, with the wrap-around spacing.  The test is numerical, not assumed.
    """
    x = np.mod(np.asarray(x, float), 2 * np.pi)
    if x.size < 3:
        return np.empty(0)
    s = np.sort(x)
    if conj_symmetric(x):
        return ratios(unique_levels(np.where(x > np.pi, x - 2 * np.pi, x), tol))
    keep = np.concatenate(([True], np.diff(s) > tol))
    u = s[keep]
    if u.size > 1 and (u[0] + 2 * np.pi - u[-1]) < tol:
        u = u[:-1]
    if u.size < 3:
        return np.empty(0)
    sp = np.diff(np.concatenate((u, [u[0] + 2 * np.pi])))      # cyclic spacings
    a, b = sp, np.roll(sp, -1)
    return np.minimum(a, b) / np.maximum(a, b)


def block_levels_var(d, m) -> np.ndarray:
    """The level variable of a block: phi where stored and finite, else theta."""
    if "phi" in d:
        ph = d["phi"][m]
        if ph.size and np.all(np.isfinite(ph)):
            return ph
    return d["theta"][m]


def block_keys(d) -> np.ndarray:
    """Integer key per eigenvector identifying its statistics block
    (sector, gate-algebra block, true symmetry label)."""
    lab = d["stat_label"] if "stat_label" in d else d["sym_label"]
    # labels: obc0 v1 in {-1, 0, 1}; v2 stat labels 10 j + (r + 1) <= 10 (N/2) + 2 < 999
    return (d["sector_id"].astype(np.int64) * 1_000_003 + d["block_id"].astype(np.int64)) * 1000 \
        + (lab.astype(np.int64) + 1)


def _block_keys_old(d) -> np.ndarray:
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
        r = block_ratios(block_levels_var(d, m))
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
        r = block_ratios(block_levels_var(d, m))
        if r.size >= nmin - 2:
            rs.append(r); nstates += int(m.sum())
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
    r = block_ratios(block_levels_var(d, m))
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
