"""R19: the exact laws behind the paper figures, and the tables that quote them.

`notebooks/paper_figures.ipynb` draws three figures -- sector count vs N, D_max
vs N, and the sector-size distribution of W156 and W108.  All the *drawing* code
lives in the notebook, where it is meant to be edited.  What lives here is the
part that must NOT drift: the closed forms annotated on the curves.

Every law in `LAWS` was read off the integer series and is then checked at every
computed N by `verify`.  The notebook calls it before drawing and the test suite
calls it independently, so a figure can never carry a tag the data has stopped
supporting.  Nothing here is fitted -- contrast scaling/sectors.py, where the
`source` field tells you which bases are derived and which are regressions.

The one asymptotic statement, W156's D_max ~ 4^{N/5}, is deliberately NOT in
`LAWS`: at finite N the largest sector is a mixture of rooms of length 2 and 3
(see the derivation in scaling/sectors.py), so no exact recurrence holds over
the computed range.  `room_base_check` reports the five-step ratio instead,
which converges to 4^{1/5} to seven digits by N = 21.
"""

from __future__ import annotations

import os
from math import comb
from typing import Callable, Dict, List, Tuple

from .. import results_io
from ..core import rules
from .summary import load_series

TEX_DIR = os.path.join(results_io.REPO_ROOT, "reports", "tex")

#: the two exponentially fragmented rules whose size distribution Figure 3 draws
HIST_RULES = (156, 108)

#: the N shown in Figure 3, light -> dark
HIST_NS = (13, 17, 21)

_F: List[int] = [0, 1]
while len(_F) < 64:
    _F.append(_F[-1] + _F[-2])


def fib(k: int) -> int:
    """Fibonacci with F_1 = F_2 = 1."""
    return _F[k]


def central(N: int, parity: int) -> int:
    """Largest binomial C(N+1, w) over w of the given parity.

    R8 sec.4: rule 150's sectors are the even-weight levels of the domain-wall
    number on a chain of N+1 bonds, so its largest sector is the largest even
    binomial rather than the central one.  They differ whenever the central
    coefficient sits at odd w.
    """
    return max(comb(N + 1, w) for w in range(N + 2) if w % 2 == parity)


#: (rule, series) -> (closed form, human-readable name, LaTeX)
LAWS: Dict[Tuple[int, str], Tuple[Callable[[int], int], str, str]] = {
    (156, "n_recurrent"): (lambda N: fib(N + 2), "F_{N+2}", r"$F_{N+2}$"),
    (108, "d_max"): (lambda N: fib(N), "F_N", r"$F_{N}$"),
    (201, "d_max"): (lambda N: fib(N + 2), "F_{N+2}", r"$F_{N+2}$"),
    (60, "n_recurrent"): (lambda N: N + 1, "N+1", r"$N+1$"),
    (60, "d_max"): (lambda N: 2 ** (N - 1), "2^(N-1)", r"$2^{N-1}$"),
    (102, "n_recurrent"): (lambda N: N + 1, "N+1", r"$N+1$"),
    (102, "d_max"): (lambda N: 2 ** (N - 1), "2^(N-1)", r"$2^{N-1}$"),
    (150, "n_recurrent"): (lambda N: (N + 1) // 2 + 1,
                           "floor((N+1)/2)+1",
                           r"$\lfloor (N{+}1)/2\rfloor+1$"),
    (150, "d_max"): (lambda N: central(N, 0),
                     "max over even w of C(N+1,w)",
                     r"$\max_{w\ \mathrm{even}}\binom{N+1}{w}$"),
    (105, "n_recurrent"): (lambda N: N // 2 + 1 + (N % 4 == 3),
                           "floor(N/2)+1+[N=3 mod 4]",
                           r"$\lfloor N/2\rfloor+1+[N\equiv3\ (4)]$"),
    (105, "d_max"): (lambda N: central(N, 1 if N % 4 == 1 else 0),
                     "max C(N+1,w), odd w at N=1 mod 4",
                     r"$\max_{w}\binom{N+1}{w}$, odd $w$ at $N\equiv1\ (4)$"),
}

#: series label -> how it prints
_SERIES_TEX = {"n_recurrent": r"$n_{\wcc}$", "d_max": r"$D_{\max}$"}


def series(rule: int, bc: str = "obc0"):
    """(N grid, sector counts, D_max) as plain integer lists."""
    s = load_series(rule, bc)
    return s["N"], s["n_recurrent"], s["d_max"]


def check_law(rule: int, key: str, bc: str = "obc0") -> Dict:
    """Evaluate one closed form at every computed N."""
    f = LAWS[(rule, key)][0]
    Ns, n, d = series(rule, bc)
    y = n if key == "n_recurrent" else d
    bad = [N for N, v in zip(Ns, y) if f(N) != v]
    return {"rule": rule, "key": key, "N_min": Ns[0] if Ns else None,
            "N_max": Ns[-1] if Ns else None, "n_points": len(Ns),
            "mismatch": bad, "ok": not bad and bool(Ns)}


def plastic_recurrence(bc: str = "obc0") -> bool:
    """W108's sector count obeys a_N = 2a_{N-1} - a_{N-2} + a_{N-3}.

    The dominant root is rho^2 = 1.754878, rho the plastic number; R18 derives
    the same recurrence from the wall grammar rather than from the series.
    """
    _, n, _ = series(108, bc)
    return len(n) > 3 and all(n[i] == 2 * n[i - 1] - n[i - 2] + n[i - 3]
                              for i in range(3, len(n)))


def shifted_partner(bc: str = "obc0") -> bool:
    """W201's sector count is W108's, evaluated one site shorter."""
    N108, n108, _ = series(108, bc)
    N201, n201, _ = series(201, bc)
    prev = dict(zip(N108, n108))
    overlap = [(N, v) for N, v in zip(N201, n201) if N - 1 in prev]
    return bool(overlap) and all(v == prev[N - 1] for N, v in overlap)


def room_base_check(bc: str = "obc0") -> Dict:
    """W156's D_max grows like 4^{N/5}; report the five-step ratio.

    D_max(N+5)/D_max(N) is 4 asymptotically, so its fifth root estimates the
    base without fitting anything.  Not an exact law -- see the module
    docstring -- which is why it is reported rather than asserted.
    """
    _, _, d = series(156, bc)
    if len(d) < 6:
        return {"ok": False}
    est = (d[-1] / d[-6]) ** 0.2
    return {"ok": True, "estimate": est, "target": 4 ** 0.2,
            "abs_error": abs(est - 4 ** 0.2)}


def verify(bc: str = "obc0") -> Dict:
    """Every claim the figure annotations make, checked against the data."""
    checks = {(r, k): check_law(r, k, bc) for (r, k) in LAWS}
    return {"laws": checks,
            "plastic_recurrence": plastic_recurrence(bc),
            "shifted_partner": shifted_partner(bc),
            "room_base": room_base_check(bc),
            "ok": all(c["ok"] for c in checks.values())
                  and plastic_recurrence(bc) and shifted_partner(bc)}


def size_stats(rule: int, N: int, bc: str = "obc0") -> Dict:
    """Sector-size summary for one (rule, N), from the complete `size_hist`.

    `sizes_recurrent` in the record is truncated at 2048 entries for the large
    N of these rules; `size_hist` is not.  The two asserts are what make it
    safe to prefer the histogram, and they are the reason Figure 3 can run out
    to N = 21 at all.
    """
    rec = results_io.load_results(rule, bc)[N]
    h = {int(k): int(v) for k, v in rec["size_hist"].items()}
    assert sum(h.values()) == rec["n_recurrent"], (rule, N, "class count")
    assert sum(s * c for s, c in h.items()) == 2 ** N, (rule, N, "mass")

    sizes = sorted(h)
    total = float(2 ** N)
    cum, median = 0.0, sizes[-1]
    for s in sizes:
        cum += s * h[s] / total
        if cum >= 0.5:
            median = s
            break
    d_max = sizes[-1]
    n_sec = sum(h.values())
    return {"rule": rule, "N": N, "n_sectors": n_sec, "d_max": d_max,
            "d_max_fraction": d_max / total, "mean": total / n_sec,
            "median": median, "n_distinct_sizes": len(sizes),
            "n_singletons": h.get(1, 0)}


# ------------------------------------------------------------------ tables

def _tex_laws(bc: str) -> str:
    v = verify(bc)
    rows = []
    for (rule, key) in sorted(LAWS, key=lambda t: (t[1], t[0])):
        c = v["laws"][(rule, key)]
        _, _, tex = LAWS[(rule, key)]
        rows.append(
            f"{rule} & \\rt{{{''.join(rules.wolfram_to_tuple(rule))}}} & "
            f"{_SERIES_TEX[key]} & {tex} & "
            f"${c['N_min']}$--${c['N_max']}$ & "
            f"{'yes' if c['ok'] else 'NO'} \\\\")
    body = "\n".join(rows)
    return (
        "\\begin{table}[htbp]\n\\centering\n\\small\n"
        # The closed-form column wraps: W105's carries a residue qualifier that
        # is wider than the text block on its own.  Plain `p` rather than an
        # array-package variant, so the table needs no preamble beyond booktabs.
        "\\begin{tabular}{rllp{0.34\\textwidth}lc}\n\\toprule\n"
        "rule & tuple & series & closed form & $N$ range & holds \\\\\n"
        "\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n"
        "\\caption{The exact laws annotated on Figures~\\ref{fig:r19sectors} "
        "and~\\ref{fig:r19dmax} (\\rt{" + bc + "}). Each was read off the "
        "integer series and then evaluated at \\emph{every} computed $N$; the "
        "last column is that check, not a goodness of fit. $F_k$ is the "
        "Fibonacci number with $F_1=F_2=1$.}\n"
        "\\label{tab:r19laws}\n\\end{table}\n")


def _tex_sizes(bc: str) -> str:
    rows = []
    for rule in HIST_RULES:
        for N in HIST_NS:
            s = size_stats(rule, N, bc)
            rows.append(
                f"{rule} & {N} & {s['n_sectors']:,} & {s['d_max']:,} & "
                f"{s['d_max_fraction']:.2e} & {s['mean']:.1f} & "
                f"{s['median']:,} & {s['n_singletons']:,} \\\\")
    body = "\n".join(rows).replace(",", "\\,")
    return (
        "\\begin{table}[htbp]\n\\centering\n\\small\n"
        "\\begin{tabular}{rrrrrrrr}\n\\toprule\n"
        "rule & $N$ & sectors & $D_{\\max}$ & $D_{\\max}/2^N$ & mean & "
        "median & singletons \\\\\n\\midrule\n" + body +
        "\n\\bottomrule\n\\end{tabular}\n"
        "\\caption{The numbers behind Figure~\\ref{fig:r19hist} (\\rt{" + bc +
        "}). \\emph{mean} is $2^N/n_{\\wcc}$; \\emph{median} is the size of "
        "the sector a uniformly random computational-basis state lands in, "
        "i.e.\\ where the cumulative curve crosses $1/2$. Both are far below "
        "$D_{\\max}$, which is why the largest sector is a poor description of "
        "the typical one.}\n"
        "\\label{tab:r19sizes}\n\\end{table}\n")


def write_tables(bc: str = "obc0") -> List[str]:
    os.makedirs(TEX_DIR, exist_ok=True)
    out = []
    for name, tex in (("laws", _tex_laws(bc)), ("sizes", _tex_sizes(bc))):
        path = os.path.join(TEX_DIR, f"tab_r19_{name}_{bc}.tex")
        with open(path, "w") as f:
            f.write(tex)
        out.append(path)
    return out


def main(argv=None):
    bc = "obc0"
    v = verify(bc)
    for (rule, key) in sorted(LAWS, key=lambda t: (t[1], t[0])):
        c = v["laws"][(rule, key)]
        print(f"{'ok  ' if c['ok'] else 'FAIL'} W{rule:<4d} {key:<12s} "
              f"= {LAWS[(rule, key)][1]}"
              + (f"   mismatch at N={c['mismatch']}" if c["mismatch"] else ""))
    print(f"{'ok  ' if v['plastic_recurrence'] else 'FAIL'} W108  "
          f"a_N = 2a_(N-1) - a_(N-2) + a_(N-3)   (root rho^2 = 1.754878)")
    print(f"{'ok  ' if v['shifted_partner'] else 'FAIL'} W201  "
          f"n_wcc(N) = W108 n_wcc(N-1)")
    rb = v["room_base"]
    print(f"ok   W156  D_max five-step ratio^(1/5) = {rb['estimate']:.7f} "
          f"vs 4^(1/5) = {rb['target']:.7f}")
    print()
    for p in write_tables(bc):
        print("wrote", os.path.relpath(p, results_io.REPO_ROOT))


if __name__ == "__main__":
    main()
