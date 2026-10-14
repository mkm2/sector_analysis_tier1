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


def compare_figure(sum_h: Dict, sum_x: Dict, N: int = 14, out_dir: str = None) -> str:
    """
    H against X for every rule at ONE common N (open chain).
    (a) mean half-chain entanglement of the non-degenerate eigenstates over the Page value;
    (b) X gate: mean cycle length seen by a uniformly random basis state, over 2^N.
    """
    from ..viz import thesis_style as ts
    from .figures import CLASS_ORDER, FIG_DIR
    plt = ts.use(ts._mode)
    fig, (a, b) = ts.subplots(2, 1, aspect=0.75, sharex=True)
    xs, labels, x, groups = [], [], 0, []
    for name, ws in CLASS_ORDER:
        g0 = x
        for w in ws:
            h, xx = sum_h.get(f"{w}_{N}"), sum_x.get(f"{w}_{N}")
            if h is None or xx is None:
                x += 1; continue
            if h.get("S_mean_nondeg") is not None:
                a.plot(x - 0.15, h["S_mean_nondeg"] / h["page"], "o", ms=ts.MS, color=ts.PALETTE[0])
            a.plot(x + 0.15, xx["S_mean"] / xx["page"], "s", ms=ts.MS, color=ts.PALETTE[1])
            b.bar(x, xx["mean_len_state"] / 2 ** N, width=0.6, color=ts.PALETTE[1])
            labels.append(f"{w}"); xs.append(x); x += 1
        groups.append((name, g0, x - 1)); x += 0.6
    a.axhline(1.0, color=ts.GREY, lw=0.5, ls="-", zorder=0)
    a.plot([], [], "o", color=ts.PALETTE[0], ms=ts.MS, label=r"$V=H$ (non-degenerate states)")
    a.plot([], [], "s", color=ts.PALETTE[1], ms=ts.MS, label=r"$V=X$ (cycle Fourier modes)")
    a.legend(loc="upper left", handlelength=1.0, borderaxespad=0.2)
    a.set_ylabel(r"$\bar S_\mathrm{vN}/S_\mathrm{Page}$"); a.set_ylim(-0.05, 1.25)
    b.set_yscale("log"); b.set_ylabel(r"$\langle L\rangle_\mathrm{state}/2^N$ ($X$)")
    b.set_ylim(top=3)
    b.set_xticks(xs, labels, rotation=90); b.set_xlabel(rf"rule $W$, $N={N}$, open chain")
    for name, g0, g1 in groups:
        b.annotate(name, ((g0 + g1) / 2, 1.0), xycoords=("data", "axes fraction"), xytext=(0, -2),
                   textcoords="offset points", ha="center", va="top",
                   fontsize=ts.FS["annot"] - 2, color=ts.GREY)
    for ax, lab in zip((a, b), "ab"):
        ts.panel_label(ax, f"({lab})")
    stem = os.path.join(out_dir or FIG_DIR, f"fig_r32_HX_compare_N{N}")
    ts.save(fig, stem)
    plt.close(fig)
    return stem


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--bc", default="obc0")
    a = ap.parse_args()
    res = build(a.bc)
    bad = {k: v["spectrum_check"] for k, v in res.items() if not v["spectrum_check"] < 1e-8}
    print(f"{len(res)} units -> {summary_path(a.bc)}; spectrum check failures: {bad or 'none'}")
    if a.bc == "obc0":
        from . import analysis as A
        hp = A.summary_path("H")
        if os.path.exists(hp):
            print("wrote", compare_figure(json.load(open(hp)), res))


#: representatives for the cycle-growth figure: (rule, label)
CYCLE_RULES = [(57, r"57/99 ($k=3$)"), (54, r"54/147 ($k=3$, RCA54)"), (150, r"150 (XOR)"),
               (153, r"153 (one-sided)"), (201, r"201 ($k=1$, Floquet-PXP)")]


def cycle_growth(Nmax_obc: int = 18, Nmax_pbc: int = 18) -> Dict:
    """Longest cycle and <L>_state / 2^N against N, straight from the permutation (no Julia)."""
    out = {}
    for w, _ in CYCLE_RULES:
        for bc, Ns in (("obc0", range(6, Nmax_obc + 1)), ("pbc", range(6, Nmax_pbc + 1, 2))):
            rows = []
            for N in Ns:
                L = cycle_lengths(w, N, bc)
                rows.append((N, int(L[0]), float((L.astype(float) ** 2).sum() / 4.0 ** N)))
            out[f"{w}_{bc}"] = rows
    return out


def cycles_figure(growth: Dict, out_dir: str = None) -> str:
    """(a) longest cycle against N; (b) <L>_state / 2^N.  Open chain filled, ring open symbols."""
    from ..viz import thesis_style as ts
    from .figures import FIG_DIR
    plt = ts.use(ts._mode)
    fig, (a, b) = ts.subplots(1, 2, aspect=0.42)
    for i, (w, lab) in enumerate(CYCLE_RULES):
        c = ts.PALETTE[i]
        for bc, mfc, ls in (("obc0", c, "-"), ("pbc", "white", ":")):
            r = np.array(growth[f"{w}_{bc}"])
            kw = dict(color=c, mfc=mfc, ms=ts.MS, lw=1.0, ls=ls, marker="o" if bc == "obc0" else "s")
            a.plot(r[:, 0], r[:, 1], label=lab if bc == "obc0" else None, **kw)
            b.plot(r[:, 0], r[:, 2], **kw)
    N = np.arange(6, 19)
    a.plot(N, 2.0 ** N, color=ts.GREY, lw=0.6, ls="--", zorder=0)
    a.annotate(r"$2^N$", (16.2, 2.0 ** 16.2), color=ts.GREY, fontsize=ts.FS["annot"], ha="right")
    a.set_yscale("log"); b.set_yscale("log")
    a.set_xlabel(r"$N$"); b.set_xlabel(r"$N$")
    a.set_ylabel(r"longest cycle"); b.set_ylabel(r"$\langle L\rangle_\mathrm{state}/2^N$")
    b.plot([], [], "ko-", ms=ts.MS, lw=1.0, label="open chain")
    b.plot([], [], "ks:", mfc="white", ms=ts.MS, lw=1.0, label="ring")
    a.legend(loc="upper left", fontsize=ts.FS["legend"] - 2, handlelength=1.2, borderaxespad=0.2)
    b.legend(loc="lower left", fontsize=ts.FS["legend"] - 2, handlelength=1.6, borderaxespad=0.2)
    for ax, lab in zip((a, b), "ab"):
        ts.panel_label(ax, f"({lab})")
    stem = os.path.join(out_dir or FIG_DIR, "fig_r32_X_cycles")
    ts.save(fig, stem)
    plt.close(fig)
    return stem


def write_x_table(N: int = 14, out: str = None) -> str:
    """LaTeX table: X-gate cycle structure and H vs X eigenstate entanglement, both boundaries, one N."""
    from .. import results_io
    from . import analysis as A
    from .figures import CLASS_ORDER
    sx = {bc: json.load(open(summary_path(bc))) for bc in ("obc0", "pbc")}
    sh = {bc: json.load(open(A.summary_path("H", bc))) for bc in ("obc0", "pbc")}
    rows = []
    for _, ws in CLASS_ORDER:
        for w in ws:
            cells = [pattern_class(w), str(w)]
            for bc in ("obc0", "pbc"):
                x, h = sx[bc][f"{w}_{N}"], sh[bc][f"{w}_{N}"]
                hS = h.get("S_mean_nondeg")
                cells += [str(x["n_cycles"]), str(x["longest"]), f"{x['mean_len_state'] / 2 ** N:.1e}",
                          f"{max(x['S_mean'], 0.0) / x['page']:.2f}", "--" if hS is None else f"{max(hS, 0.0) / h['page']:.2f}"]
            rows.append(" & ".join(cells) + r" \\")
    tex = (r"\begin{tabular}{llrrrrr|rrrrr}" "\n\\toprule\n"
           r" & & \multicolumn{5}{c|}{open chain} & \multicolumn{5}{c}{ring} \\" "\n"
           r"$k$ & $W$ & cycles & longest & $\langle L\rangle/2^N$ & $\bar S_X$ & $\bar S_H$ & "
           r"cycles & longest & $\langle L\rangle/2^N$ & $\bar S_X$ & $\bar S_H$ \\" "\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    out = out or os.path.join(results_io.REPO_ROOT, "reports", "tex", f"tab_r32_X_N{N}.tex")
    with open(out, "w") as fh:
        fh.write(tex)
    return out
