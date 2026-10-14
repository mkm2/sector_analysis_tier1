"""
Per-(rule, N) summary of the Floquet spectral data (R32), cached as JSON.

    python -m qca_fragmentation.spectral.analysis [--gate H]   # -> results_spectra/summary_{gate}.json

Three levels of symmetry resolution for the gap ratio, all on the half spectrum
|theta| in [0, pi] with exact degeneracies removed (see stats.py):
  none    the whole spectrum as one list,
  krylov  per Krylov sector,
  full    per (Krylov sector, gate-algebra block, spatial label).
Entanglement:
  eta = s2_haar / s2_block_rand : Renyi-2 entropy of the eigenstate (Haar average
  over its degenerate eigenspace, so basis independent) relative to a random
  state in the same block.  eta ~ 1 thermal within the block; eta << 1 atypically
  weakly entangled (scar-like).  Defined for blocks of dimension >= 2.
"""

from __future__ import annotations

import json
import os
from typing import Dict

import numpy as np

from . import floquet as F
from . import stats as S

NMIN = 20        # minimum distinct half-levels for a block to enter the r statistics
SCAR_ETA = 0.5   # eta below this in a block of dim >= SCAR_DIM counts as scar-like
SCAR_DIM = 20


def _pooled(theta, keys, nmin=NMIN):
    rs, nst = [], 0
    for k in np.unique(keys):
        m = keys == k
        lev = S.unique_levels(theta[m])
        if lev.size >= nmin:
            rs.append(S.ratios(lev)); nst += int(m.sum())
    r = np.concatenate(rs) if rs else np.empty(0)
    n = r.size
    return {"r_mean": float(r.mean()) if n else None,
            "r_err": float(r.std(ddof=1) / np.sqrt(n)) if n > 1 else None,
            "n_r": int(n), "n_blocks": len(rs), "state_frac": nst / len(theta)}


def summarize(d) -> Dict:
    N = d["N"]
    th = d["theta"]
    full = S.block_keys(d)
    krylov = d["sector_id"].astype(np.int64)
    none = np.zeros_like(krylov)
    uk, cnt = np.unique(full, return_counts=True)
    big = uk[np.argmax(cnt)]
    mbig = full == big
    r_big = S.ratios(S.unique_levels(th[mbig]))

    NA, NB = N // 2, N - N // 2
    # eta per state, only in blocks of dim >= 2 with a positive reference
    bdim = dict(zip(uk, cnt))
    dim = np.array([bdim[k] for k in full])
    ref = d.get("s2_block_rand")
    out = {
        "rule": d["rule_wolfram"], "rule_hsf": d["rule_hsf"], "N": N,
        "symmetry": d["symmetry"], "n_sectors": int(len(d["sector_sizes"])),
        "n_blocks": int(len(uk)), "largest_block": int(cnt.max()),
        "r_none": _pooled(th, none), "r_krylov": _pooled(th, krylov),
        "r_full": _pooled(th, full),
        "r_largest": {"r_mean": float(r_big.mean()) if r_big.size else None,
                      "n_r": int(r_big.size), "dim": int(mbig.sum())},
        "deg": S.degeneracy_summary(d),
        "page": S.page_entropy(NA, NB),
        "S_mean_largest_nondeg": float(np.mean(d["entropy"][mbig & (d["degeneracy"] == 1)]))
        if np.any(mbig & (d["degeneracy"] == 1)) else None,
        "checks": {"max_residual": float(d["max_residual"]),
                   "max_offblock": float(d["max_offblock"]),
                   "max_sym_residual": float(d["max_sym_residual"]),
                   "runtime": float(d["runtime"])},
    }
    # gate-algebra blocks of dimension <= 2 inside Krylov sectors of dimension >= 3:
    # simultaneous eigenvectors of every local gate (1-dim irreps of A_0), i.e.
    # exact eigenstates of U for any gate order -- candidate exact scars
    secsize = np.asarray(d["sector_sizes"])[d["sector_id"] - 1]
    special = []
    for k, c in zip(uk, cnt):
        m = full == k
        if c <= 2 and secsize[m][0] >= 3:
            special.append({"dim": int(c), "sector_dim": int(secsize[m][0]),
                            "theta": [round(float(t), 6) for t in th[m]],
                            "S_vN": [round(float(x), 6) for x in d["entropy"][m]]})
    out["special_blocks"] = special
    if ref is not None:
        ok = (dim >= 2) & (ref > 1e-12)
        eta = np.full(th.size, np.nan)
        eta[ok] = d["s2_haar"][ok] / ref[ok]
        sc = ok & (dim >= SCAR_DIM) & (eta < SCAR_ETA)
        out["eta"] = {
            "mean_largest": float(np.nanmean(eta[mbig])) if np.any(ok & mbig) else None,
            "min_largest": float(np.nanmin(eta[mbig])) if np.any(ok & mbig) else None,
            "n_scarlike": int(sc.sum()),
            "n_scarlike_nondeg": int((sc & (d["degeneracy"] == 1)).sum()),
            "s2_rand_largest": float(ref[mbig][0]),
        }
    return out


def summary_path(gate: str = "H", bc: str = "obc0") -> str:
    return os.path.join(F.SPECTRA_DIR, f"summary_{gate}.json" if bc == "obc0"
                        else f"summary_{gate}_{bc}.json")


def build(gate: str = "H", bc: str = "obc0") -> Dict:
    res = {}
    for w, Ns in F.available(gate, bc).items():
        for N in Ns:
            d = F.load(F.spec_path(gate, N, w, bc))
            res[f"{w}_{N}"] = summarize(d)
    path = summary_path(gate, bc)
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", default="H")
    ap.add_argument("--bc", default="obc0")
    a = ap.parse_args()
    res = build(a.gate, a.bc)
    print(f"{len(res)} units -> {summary_path(a.gate, a.bc)}")
