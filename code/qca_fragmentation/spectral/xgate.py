"""
R32 section 6: the X gate.  U is a permutation of the basis, so its spectrum is fixed by
the cycle structure alone (a cycle of length L contributes the L-th roots of unity, each
once) and the eigenvectors are the Fourier modes of the cycles,

    |k, c> = L^{-1/2} sum_t exp(2 pi i k t / L) |U^t x_c>.

Level statistics measure orbit arithmetic, not chaos.  What is informative:
  * the cycle-length distribution (a classical reversible CA: long cycles = classical
    ergodicity of the orbit, many short cycles = integrable / frozen motion);
  * the entanglement of the Fourier modes, against the Page value, compared with the
    H gate of the same rule.

The cycle structure is computed here independently of the Julia pipeline (from
permutation.xca.step_table), and `check_spectrum` verifies the Julia quasienergies
against it.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np

from . import floquet as F
from . import stats as S
from .analysis import pattern_class


def cycle_lengths(rule: int, N: int, bc: str = "obc0") -> np.ndarray:
    """Lengths of the cycles of the X-gate one-period permutation."""
    from ..permutation import xca
    f = np.asarray(xca.step_table(rule, N, bc))
    seen = np.zeros(f.size, bool)
    out = []
    for x0 in range(f.size):
        if seen[x0]:
            continue
        L, x = 0, x0
        while not seen[x]:
            seen[x] = True; x = f[x]; L += 1
        out.append(L)
    return np.array(sorted(out, reverse=True))


def check_spectrum(d, lengths: np.ndarray) -> float:
    """Max distance between the stored e^{i theta} (either sign convention) and the
    multiset union of L-th roots of unity over the cycles."""
    ref = np.sort(np.concatenate([np.mod(2 * np.pi * np.arange(L) / L, 2 * np.pi) for L in lengths]))
    best = np.inf
    for s in (1, -1):
        x = np.sort(np.mod(s * d["theta"], 2 * np.pi))
        # compare sorted lists on the circle after removing the wrap ambiguity at 0
        x = np.where(2 * np.pi - x < 1e-9, x - 2 * np.pi, x); x = np.sort(x)
        r = np.where(2 * np.pi - ref < 1e-9, ref - 2 * np.pi, ref); r = np.sort(r)
        if x.size == r.size:
            best = min(best, float(np.abs(x - r).max()))
    return best


def summarize_x(d, lengths: np.ndarray) -> Dict:
    N = int(d["N"]); n = 1 << N
    NA, NB = N // 2, N - N // 2
    page = S.page_entropy(NA, NB)
    ent = np.asarray(d["entropy"])
    nondeg = np.asarray(d["degeneracy"]) == 1
    big = lengths[0]
    return {
        "rule": int(d["rule_wolfram"]), "N": N, "bc": str(d["bc"]),
        "n_cycles": int(lengths.size), "longest": int(big), "longest_frac": big / n,
        "frac_in_long": float(lengths[lengths >= 2 ** (N / 2)].sum() / n),  # cycles >= sqrt(dim)
        "mean_len_state": float((lengths ** 2).sum() / n),                    # cycle length seen by a random state
        "n_fixed": int((lengths == 1).sum()),
        "spectrum_check": check_spectrum(d, lengths),
        "page": page,
        "S_mean": float(ent.mean()), "S_mean_nondeg": float(ent[nondeg].mean()) if nondeg.any() else None,
        "frac_degenerate": float(1 - nondeg.mean()),
        "S_max": float(ent.max()),
    }


def summary_path(bc: str = "obc0") -> str:
    return os.path.join(F.SPECTRA_DIR, "summary_X.json" if bc == "obc0" else f"summary_X_{bc}.json")


def build(bc: str = "obc0") -> Dict:
    res = {}
    for w, Ns in F.available("X", bc).items():
        for N in Ns:
            d = F.load(F.spec_path("X", N, w, bc))
            res[f"{w}_{N}"] = summarize_x(d, cycle_lengths(w, N, bc))
    with open(summary_path(bc), "w") as fh:
        json.dump(res, fh, indent=1)
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--bc", default="obc0")
    a = ap.parse_args()
    res = build(a.bc)
    bad = {k: v["spectrum_check"] for k, v in res.items() if not v["spectrum_check"] < 1e-8}
    print(f"{len(res)} units -> {summary_path(a.bc)}; spectrum check failures: {bad or 'none'}")
