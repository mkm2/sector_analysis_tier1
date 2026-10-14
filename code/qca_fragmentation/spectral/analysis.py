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


def _pooled(d, keys, nmin=NMIN):
    """Ratios pooled over blocks (defined by keys) with >= nmin - 2 ratios."""
    rs, nst = [], 0
    for k, m in S.groups(keys):
        if m.size < nmin - 2:          # cannot reach nmin - 2 ratios
            continue
        r = S.block_ratios(S.block_levels_var(d, m))
        if r.size >= nmin - 2:
            rs.append(r); nst += int(m.size)
    r = np.concatenate(rs) if rs else np.empty(0)
    n = r.size
    return {"r_mean": float(r.mean()) if n else None,
            "r_err": float(r.std(ddof=1) / np.sqrt(n)) if n > 1 else None,
            "n_r": int(n), "n_blocks": len(rs), "state_frac": nst / len(keys)}


def _pooled_by_type(d, keys, nmin=NMIN):
    """
    Like _pooled for the fully resolved blocks, but split by block type:
    'real' = conjugation-symmetric (folded; reference beta = 1, GOE 0.5307),
    'complex' = not symmetric (full circle; e.g. ring momentum j != -j, beta = 2, 0.5996).
    """
    out = {"real": [], "complex": []}
    for k, m in S.groups(keys):
        if m.size < nmin - 2:
            continue
        x = np.mod(S.block_levels_var(d, m), 2 * np.pi)
        r = S.block_ratios(x)
        if r.size < nmin - 2:
            continue
        out["real" if S.conj_symmetric(x) else "complex"].append(r)
    res = {}
    for t, rs in out.items():
        r = np.concatenate(rs) if rs else np.empty(0)
        n = r.size
        res[t] = {"r_mean": float(r.mean()) if n else None,
                  "r_err": float(r.std(ddof=1) / np.sqrt(n)) if n > 1 else None,
                  "n_r": int(n), "n_blocks": len(rs)}
    return res


def summarize(d) -> Dict:
    N = d["N"]
    th = d["theta"]
    full = S.block_keys(d)
    krylov = d["sector_id"].astype(np.int64)
    none = np.zeros_like(krylov)
    uk, inv, cnt = np.unique(full, return_inverse=True, return_counts=True)
    big = uk[np.argmax(cnt)]
    mbig = full == big
    r_big = S.block_ratios(S.block_levels_var(d, mbig))

    NA, NB = N // 2, N - N // 2
    # eta per state, only in blocks of dim >= 2 with a positive reference
    dim = cnt[inv.ravel()]
    ref = d.get("s2_block_rand")
    out = {
        "rule": d["rule_wolfram"], "rule_hsf": d["rule_hsf"], "N": N,
        "symmetry": d["symmetry"], "n_sectors": int(len(d["sector_sizes"])),
        "n_blocks": int(len(uk)), "largest_block": int(cnt.max()),
        "r_none": _pooled({"theta": th}, none), "r_krylov": _pooled({"theta": th}, krylov),
        "r_full": _pooled(d, full),
        "r_full_by_type": _pooled_by_type(d, full),
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
    small = np.isin(full, uk[cnt <= 2])
    for k, m in S.groups(np.where(small, full, -1)):
        if k == -1:
            continue
        c = m.size
        if secsize[m][0] >= 3:
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


#: the R32 classes: number k of neighbour patterns (00, 01, 10, 11) that fire the gate
def pattern_class(w: int) -> str:
    t = F.rules.wolfram_to_tuple(w)
    k = sum(s == "V" for s in t)
    if k == 2:
        # one-sided: the firing set depends on one neighbour only
        on = {i for i, s in enumerate(t) if s == "V"}      # index 2m + n
        one_sided = on in ({0, 1}, {2, 3}, {0, 2}, {1, 3})
        return "2 one-sided" if one_sided else "2 XOR"
    return str(k)


def class_N(summary: Dict, w: int) -> int:
    """
    The N at which rule w is compared: the largest N computed for EVERY rule of its class
    (user rule: rules of one class are compared at the same N, not each at its own maximum).
    """
    Ns = lambda v: {int(k.split("_")[1]) for k in summary if k.startswith(f"{v}_")}
    members = {int(k.split("_")[0]) for k in summary}
    common = set.intersection(*[Ns(v) for v in members if pattern_class(v) == pattern_class(w)])
    return max(common) if common else max(Ns(w))


def write_table(summary: Dict, gate: str = "H", bc: str = "obc0", out: str = None) -> str:
    """LaTeX table of the per-rule summary, each class at its common largest N (class_N)."""
    from .. import results_io
    rows = []
    order = ["0", "1", "2 one-sided", "2 XOR", "3", "4"]
    for w in sorted({int(k.split("_")[0]) for k in summary},
                    key=lambda w: (order.index(pattern_class(w)), w)):
        N = class_N(summary, w)
        s = summary[f"{w}_{N}"]
        f = lambda v: "--" if v["r_mean"] is None or v["n_r"] < 30 else f"{v['r_mean']:.3f}"
        e = s.get("eta") or {}
        n1 = sum(1 for b in s["special_blocks"] if b["dim"] == 1)
        rows.append(
            f"{pattern_class(w)} & {w} & {N} & {s['symmetry'].replace('_', ' ')} & "
            f"{s['n_sectors']} & {s['n_blocks']} & {f(s['r_none'])} & {f(s['r_full'])} & "
            f"{s['r_full']['n_r']} & {s['deg']['n_distinct_global'] / 2 ** N:.3g} & "
            + (f"{e['mean_largest']:.2f}" if e.get("mean_largest") is not None else "--")
            + f" & {n1} \\\\")
    body = "\n".join(rows)
    tex = (r"\begin{tabular}{llrlrrrrrrrr}" "\n\\toprule\n"
           r"$k$ & $W$ & $N$ & sym. & sectors & blocks & $\langle r\rangle_\mathrm{none}$ & "
           r"$\langle r\rangle_\mathrm{full}$ & \#$r$ & distinct $\theta/2^N$ & "
           r"$\bar\eta_\mathrm{big}$ & singlets \\" "\n\\midrule\n" + body +
           "\n\\bottomrule\n\\end{tabular}\n")
    out = out or os.path.join(results_io.REPO_ROOT, "reports", "tex",
                              f"tab_r32_summary_{gate}" + ("" if bc == "obc0" else f"_{bc}") + ".tex")
    with open(out, "w") as fh:
        fh.write(tex)
    return out


def write_bc_table(sum_obc: Dict, sum_pbc: Dict, N: int, gate: str = "H", out: str = None) -> str:
    """
    LaTeX table: obc0 vs pbc at the same (even) N, per rule (R32 boundary check).
    On the ring the resolved blocks are T2-momentum sectors j.  j = 0, N/4 are real
    (conjugation maps them to themselves; folded, beta = 1).  j != -j are complex:
    their reference is CUE (beta = 2) unless the rule is mirror symmetric, in which
    case K R maps j to itself (antiunitary) and the reference is COE (beta = 1).
    """
    from .. import results_io
    order = ["0", "1", "2 one-sided", "2 XOR", "3", "4"]
    rules = sorted({int(k.split("_")[0]) for k in sum_pbc},
                   key=lambda w: (order.index(pattern_class(w)), w))
    f = lambda v: "--" if v["r_mean"] is None or v["n_r"] < 30 else f"{v['r_mean']:.3f}"
    n1 = lambda s: sum(1 for b in s["special_blocks"] if b["dim"] == 1)
    rows = []
    for w in rules:
        o, p = sum_obc.get(f"{w}_{N}"), sum_pbc.get(f"{w}_{N}")
        if o is None or p is None:
            continue
        t = p["r_full_by_type"]
        ref = "COE" if is_mirror(w) else "CUE"
        rows.append(f"{pattern_class(w)} & {w} & {o['n_sectors']} & {p['n_sectors']} & "
                    f"{o['n_blocks']} & {p['n_blocks']} & {f(o['r_full'])} & {f(t['real'])} & "
                    f"{f(t['complex'])} & {ref} & {n1(o)} & {n1(p)} \\\\")
    tex = (r"\begin{tabular}{llrrrrrrrlrr}" "\n\\toprule\n"
           r" & & \multicolumn{2}{c}{sectors} & \multicolumn{2}{c}{blocks} & obc & "
           r"\multicolumn{3}{c}{pbc $\langle r\rangle$} & \multicolumn{2}{c}{singlets} \\" "\n"
           r"$k$ & $W$ & obc & pbc & obc & pbc & $\langle r\rangle_\mathrm{full}$ & real $j$ & "
           r"complex $j$ & naive ref. & obc & pbc \\" "\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    out = out or os.path.join(results_io.REPO_ROOT, "reports", "tex", f"tab_r32_bc_{gate}_N{N}.tex")
    with open(out, "w") as fh:
        fh.write(tex)
    return out


def is_mirror(w: int) -> bool:
    """Rule invariant under the spatial reflection (R32: K R is then a symmetry of every ring sector j)."""
    return F.rules.reflect_wolfram(w) == w


#: bins for the shape test of P(r)
_BETA_EDGES = np.linspace(0.0, 1.0, 11)


def shape_test(r: np.ndarray) -> Dict:
    """
    Which ensemble does the SHAPE of P(r) follow (not only its mean)?  chi^2 of the
    10-bin histogram against the Atas surmises (Poisson, beta = 1, beta = 2), and the
    best Poisson + beta=2 mixture: a crossover towards CUE can pass through the COE
    mean, but its small-r tail (P(r) ~ r^beta) is then too heavy.
    Note: the surmise itself deviates from large-N RMT by ~1%, so chi^2 of a few
    tens at n ~ 10^4 is a good fit.
    """
    from scipy.integrate import quad
    e = _BETA_EDGES
    ref = {b: np.array([quad(lambda x: S.surmise(x, b), lo, hi)[0] for lo, hi in zip(e[:-1], e[1:])])
           for b in (0, 1, 2)}
    h = np.histogram(r, e)[0]
    n = r.size
    chi = lambda p: float(((h - n * p) ** 2 / (n * p)).sum())
    mix = min((chi(a * ref[0] + (1 - a) * ref[2]), float(a)) for a in np.linspace(0, 1, 101))
    return {"n_r": int(n), "r_mean": float(r.mean()), "small_r": float((r < 0.15).mean()),
            "chi2_poisson": chi(ref[0]), "chi2_goe": chi(ref[1]), "chi2_gue": chi(ref[2]),
            "chi2_mix": mix[0], "mix_poisson_weight": mix[1]}


def complex_sector_ratios(d) -> np.ndarray:
    """Pooled ratios of the resolved blocks that are NOT conjugation-symmetric (ring j != -j)."""
    keys = S.block_keys(d)
    rs = []
    for k, m in S.groups(keys):
        if m.size < NMIN:
            continue
        x = S.block_levels_var(d, m)
        if not S.conj_symmetric(x):
            rs.append(S.block_ratios(x))
    return np.concatenate(rs) if rs else np.empty(0)


def write_shape_table(N: int, gate: str = "H", bc: str = "pbc", rules=None, out: str = None) -> str:
    """LaTeX table of shape_test on the complex momentum sectors (R32 ring check)."""
    from .. import results_io
    rules = rules or [57, 99, 147, 54, 60, 102, 153, 195]
    rows = []
    for w in rules:
        path = F.spec_path(gate, N, w, bc)
        if not os.path.exists(path):
            continue
        t = shape_test(complex_sector_ratios(F.load(path)))
        rows.append(f"{pattern_class(w)} & {w} & {'yes' if is_mirror(w) else 'no'} & {t['n_r']} & "
                    f"{t['r_mean']:.3f} & {t['small_r']:.3f} & {t['chi2_poisson']:.0f} & "
                    f"{t['chi2_goe']:.0f} & {t['chi2_gue']:.0f} & {t['chi2_mix']:.0f} \\\\")
    tex = (r"\begin{tabular}{llrrrrrrrr}" "\n\\toprule\n"
           r"$k$ & $W$ & mirror & \#$r$ & $\langle r\rangle$ & $P(r<0.15)$ & "
           r"$\chi^2_\mathrm{P}$ & $\chi^2_{\beta=1}$ & $\chi^2_{\beta=2}$ & $\chi^2_\mathrm{P+\beta=2}$ \\"
           "\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    out = out or os.path.join(results_io.REPO_ROOT, "reports", "tex", f"tab_r32_shape_{gate}_{bc}_N{N}.tex")
    with open(out, "w") as fh:
        fh.write(tex)
    return out
