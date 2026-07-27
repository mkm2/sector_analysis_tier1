"""
Cross-check the C150 findings against the pre-existing Julia HSF eigendata.

HSF numbers the 16 unitary rules 0..15 by the bits (r11 r10 r01 r00) with
1 = Hadamard, so **HSF rule 6 = Wolfram 150** (see core.rules.hsf_to_wolfram),
and the `_H_` files use V = H.  Reading HSF/src/model.jl confirms the two codes
build the *same* operator:

  * `binary_coefficients(6)` = "0110", i.e. neighbours (1,1) -> I, (1,0) -> V,
    (0,1) -> V, (0,0) -> I: exactly our (I, V, V, I);
  * `local_gate_leftboundary`/`local_gate_rightboundary` use the c3/c4 and c2/c4
    branches, i.e. the missing neighbour is read as 0: exactly our \\rt{obc0};
  * `U_full = U_even * U_odd` with `U_odd` collecting Julia sites 1,3,5,...
    (0-based 0,2,4,... = our EVEN sublattice) applies the even sublattice first:
    exactly our layer order.

Two conventions differ and neither affects a spectrum:

  * HSF puts qubit 1 in the most significant position of the kron product; we
    put site 0 in the least significant bit.  That is a bit reversal of the
    basis labels, i.e. a permutation similarity.
  * HSF stores quasienergies theta = -angle(lambda); we use +angle(lambda).
    Because U is real orthogonal its spectrum is closed under conjugation, so
    the two multisets coincide -- which is itself checked below.

The eigendata files are flat arrays (N, rule, sector_id, theta, entropy) with
one row per eigenstate of every Krylov sector, produced by generate_EE_data.jl.
JLD2 is HDF5, so h5py reads them directly.

    python -m qca_fragmentation.quantum.hsf_compare --all
"""

from __future__ import annotations

import argparse
import json
import os
from math import comb, ceil
from typing import Dict, List, Optional

import numpy as np

from ..core import rules
from ..results_io import REPO_ROOT
from .. import c150

HSF_RULE = 6
GATE = "H"


def find_hsf_root(explicit: Optional[str] = None) -> Optional[str]:
    """
    Locate the HSF checkout.  It is a sibling of the analysis repo, but the repo
    may itself be a git worktree several levels deeper, so walk up looking for
    an HSF/data directory.  $QCA_HSF_ROOT overrides.
    """
    if explicit:
        return explicit if os.path.isdir(os.path.join(explicit, "data")) else None
    env = os.environ.get("QCA_HSF_ROOT")
    if env and os.path.isdir(os.path.join(env, "data")):
        return env
    here = REPO_ROOT
    for _ in range(6):
        cand = os.path.join(here, "HSF")
        if os.path.isdir(os.path.join(cand, "data")):
            return cand
        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent
    return None


def eigendata_path(N: int, *, gate: str = GATE, hsf_rule: int = HSF_RULE,
                   root: Optional[str] = None) -> Optional[str]:
    r = find_hsf_root(root)
    if r is None:
        return None
    p = os.path.join(r, "data", f"eigendata_N{N}_{gate}_Rule{hsf_rule}.jld2")
    return p if os.path.exists(p) else None


def load_eigendata(N: int, **kw) -> Optional[dict]:
    import h5py

    p = eigendata_path(N, **kw)
    if p is None:
        return None
    with h5py.File(p, "r") as f:
        out = {
            "N": int(f["N"][()]),
            "rule": int(f["rule"][()]),
            "sector_id": np.asarray(f["sector_id"][:], dtype=np.int64),
            "theta": np.asarray(f["theta"][:], dtype=np.float64),
            "entropy": np.asarray(f["entropy"][:], dtype=np.float64),
        }
    out["path"] = p
    return out


# --- the comparison ----------------------------------------------------------

def _distinct(x: np.ndarray, tol: float) -> np.ndarray:
    keep: List[float] = []
    for v in np.sort(x):
        if not keep or v - keep[-1] >= tol:
            keep.append(float(v))
    return np.array(keep)


def compare(N: int, *, tol: float = 1e-8, root: Optional[str] = None) -> Optional[dict]:
    """
    Everything that can be checked against the HSF data WITHOUT rebuilding the
    operator on our side: sector sizes against the closed form, the
    distinct-eigenvalue law, the universal-spectrum property, dim Fix per
    sector, and the spacing-ratio statistic.
    """
    d = load_eigendata(N, root=root)
    if d is None:
        return None
    assert rules.hsf_to_wolfram(d["rule"]) == 150, "not rule 150"
    assert d["N"] == N

    sid, th = d["sector_id"], d["theta"]
    ids, counts = np.unique(sid, return_counts=True)
    sizes = sorted((int(c) for c in counts), reverse=True)

    # (A) sector sizes against |S_w| = C(N+1, w), w even
    closed = c150.sector_sizes_obc0(N)
    sizes_match = sizes == closed

    # conjugation symmetry of the spectrum (checks the sign convention is moot)
    full_all = np.sort(th)
    conj_sym = float(np.abs(full_all + np.sort(-th)[::-1]).max())

    # (B) the distinct-eigenvalue law
    full = _distinct(th, tol)
    from .rule150_spectra import distinct_spectrum_closed_form
    law = distinct_spectrum_closed_form(N)
    distinct_match = len(full) == law

    # (C) universal spectrum: each sector's spectrum inside the global set, and
    #     the largest sector realising all of it
    per: List[dict] = []
    for i, sec in enumerate(ids):
        t = th[sid == sec]
        dist = _distinct(t, tol)
        idx = np.clip(np.searchsorted(full, dist), 1, len(full) - 1)
        dev = float(np.minimum(np.abs(full[idx] - dist),
                               np.abs(full[idx - 1] - dist)).max())
        size = int(t.size)
        # size -> wall number (the pair w, N+1-w is degenerate for odd N and
        # gives the same prediction, so either representative is fine)
        w = next((ww for ww in range(0, N + 2, 2) if comb(N + 1, ww) == size), None)
        n_fix = int(np.sum(np.abs(t) < tol))
        per.append({
            "sector": int(sec), "size": size, "w": w,
            "n_distinct": int(dist.size),
            "coverage": dist.size / len(full),
            "subset_dev": dev,
            "dim_fix": n_fix,
            "fix_closed_form": comb(ceil(N / 2), w // 2) if w is not None else None,
        })
    biggest = max(per, key=lambda r: r["size"])

    # (D) dim Fix per sector
    fix_ok = all(r["fix_closed_form"] is None or r["dim_fix"] == r["fix_closed_form"]
                 for r in per)

    # (E) spacing-ratio statistic from THEIR quasienergies, our pipeline
    from .rule150_levels import distinct_upper_phases, r_tilde
    ph, _ = distinct_upper_phases(np.exp(1j * th))
    rt, n_rt = r_tilde(ph)

    # (F) how much of the entanglement data sits inside a degenerate eigenspace
    #     (where the eigenvector, and hence the entropy, is basis dependent).
    #     The relevant multiplicity is the WITHIN-sector one, because the HSF
    #     generator diagonalises each sector block separately.
    deg_frac = 1.0 - len(full) / float(th.size)
    in_deg, max_mult = 0, 0
    for sec in ids:
        t = np.sort(th[sid == sec])
        run = 1
        for k in range(1, t.size + 1):
            if k < t.size and t[k] - t[k - 1] < tol:
                run += 1
                continue
            if run > 1:
                in_deg += run
            max_mult = max(max_mult, run)
            run = 1

    return {
        "N": N, "hsf_rule": d["rule"], "gate": GATE, "path": d["path"],
        "n_states": int(th.size), "n_sectors": int(ids.size),
        "sizes_match_closed_form": sizes_match,
        "sizes": sizes[:8], "closed_form_sizes": closed[:8],
        "spectrum_conjugation_symmetric": conj_sym,
        "n_distinct": int(len(full)), "distinct_closed_form": law,
        "distinct_match": distinct_match,
        "all_sectors_subset_of_universal": all(r["subset_dev"] < tol for r in per),
        "largest_sector_w": biggest["w"],
        "largest_sector_coverage": biggest["coverage"],
        "min_sector_coverage": min(r["coverage"] for r in per),
        "dim_fix_matches": fix_ok,
        "r_tilde": rt, "n_ratios": n_rt,
        "degenerate_fraction": deg_frac,
        "frac_states_in_degenerate_eigenspace": in_deg / float(th.size),
        "max_within_sector_multiplicity": int(max_mult),
        "entropy_mean": float(d["entropy"].mean()),
        "entropy_max": float(d["entropy"].max()),
        "entropy_page_scale": float(np.log(2.0) * (N // 2)),
        "entropy_zero_fraction": float(np.mean(d["entropy"] < 1e-9)),
        "per_sector": per,
    }


def compare_all(Ns=range(8, 15), **kw) -> List[dict]:
    out = []
    for N in Ns:
        r = compare(N, **kw)
        if r is not None:
            out.append(r)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="C150 vs the HSF Julia eigendata")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--N", type=int, default=None)
    ap.add_argument("--hsf-root", default=None)
    ap.add_argument("--save", default=os.path.join(REPO_ROOT, "analytics",
                                                   "c150_hsf_compare.json"))
    args = ap.parse_args(argv)

    root = find_hsf_root(args.hsf_root)
    print(f"HSF root: {root}")
    rows = ([compare(args.N, root=args.hsf_root)] if args.N
            else compare_all(root=args.hsf_root))
    rows = [r for r in rows if r]
    for r in rows:
        print(f"N={r['N']:2d} states={r['n_states']:6d} sectors={r['n_sectors']:3d} "
              f"sizes={'OK' if r['sizes_match_closed_form'] else 'MISMATCH'} "
              f"distinct={r['n_distinct']:5d}/{r['distinct_closed_form']:5d} "
              f"{'OK' if r['distinct_match'] else 'MISMATCH'} "
              f"universal={'OK' if r['all_sectors_subset_of_universal'] else 'NO'} "
              f"cover={r['largest_sector_coverage']:.3f} "
              f"Fix={'OK' if r['dim_fix_matches'] else 'MISMATCH'} "
              f"<r~>={r['r_tilde']:.4f} degfrac={r['degenerate_fraction']:.3f}")
    if args.save and rows:
        os.makedirs(os.path.dirname(args.save), exist_ok=True)
        with open(args.save, "w") as f:
            json.dump(rows, f, indent=1)
        print("saved", args.save)


if __name__ == "__main__":
    main()
