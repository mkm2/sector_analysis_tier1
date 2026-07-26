"""
Tables, figures and cached data for R8 (rule 150 / C150, obc0).

    python -m qca_fragmentation.scaling.c150_report --quantum      # cache spectra
    python -m qca_fragmentation.scaling.c150_report --tables --figures

The quantum pass is the expensive one (dense eigendecomposition of every wall
shell up to DIM_CAP) so it is checkpointed to analytics/c150_quantum.json and
never recomputed unless --force is given.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from math import comb, ceil, log, pi, sqrt
from typing import Dict, List, Optional

from .. import c150
from ..results_io import REPO_ROOT, load_results, sizes_from_record

ANALYTICS = os.path.join(REPO_ROOT, "analytics")
FIGDIR = os.path.join(REPO_ROOT, "figures")
TEXDIR = os.path.join(REPO_ROOT, "reports", "tex")
QUANTUM_PATH = os.path.join(ANALYTICS, "c150_quantum.json")

DIM_CAP = 3000          # largest shell we diagonalise densely
FREENESS_NS = [5, 6, 7, 8, 9]


# --- cached quantum pass -----------------------------------------------------

def build_quantum(*, force: bool = False, dim_cap: int = DIM_CAP) -> dict:
    from ..quantum import rule150_spectra as rs

    if os.path.exists(QUANTUM_PATH) and not force:
        with open(QUANTUM_PATH) as f:
            return json.load(f)
    os.makedirs(ANALYTICS, exist_ok=True)
    sectors = []
    for N in range(4, 25):
        for w in range(0, N + 2, 2):
            D = comb(N + 1, w)
            if D > dim_cap or D < 2:
                continue
            t0 = time.time()
            rec = rs.spectral_portrait(N, w)
            rec["runtime"] = time.time() - t0
            rec["fix_closed_form"] = comb(ceil(N / 2), w // 2)
            rec["fix_matches"] = rec["mult_plus_one"] == rec["fix_closed_form"]
            sectors.append(rec)
            print(f"  sector N={N} w={w} dim={D} [{rec['runtime']:.1f}s]",
                  flush=True)
    free = rs.freeness_table(FREENESS_NS)
    fix_full = []
    for N in range(3, 13):
        fix_full.append({"N": N, "dim_fix": rs.fix_dim_full(N),
                         "closed_form": 2 ** ceil(N / 2),
                         "n_even_sites": len(range(0, N, 2)),
                         "fix_is_halflayer_intersection":
                             rs.fix_is_halflayer_intersection(N)})
    out = {"sectors": sectors, "freeness": free, "fix_full": fix_full,
           "dim_cap": dim_cap,
           "generated": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with open(QUANTUM_PATH, "w") as f:
        json.dump(out, f, indent=1)
    return out


# --- tables ------------------------------------------------------------------

def _fmt(n: int) -> str:
    """Thousands separator that survives LaTeX math mode."""
    s = f"{n:,}".replace(",", r"\,")
    return s


def tab_frontier() -> str:
    """The two numerical frontiers plus the closed form."""
    eng = load_results(150, "obc0")
    fr = c150.load_frontier()
    Ns = sorted({int(k.split("_")[0]) for k in fr if k.endswith("obc0")}
                | set(eng.keys()))
    rows = []
    for N in Ns:
        if N < 14:
            continue
        e = eng.get(N)
        f = fr.get(f"{N}_obc0")
        cf = c150.sector_sizes_obc0(N)
        def _t(sec: float) -> str:
            if sec < 10:
                return f"{sec:.1f}\\,s"
            if sec < 3600:
                return f"{sec:.0f}\\,s"
            return f"{sec / 3600:.1f}\\,h"

        eng_cell = "---"
        if e:
            ok = sizes_from_record(e) == cf
            eng_cell = (r"\cmark" if ok else r"\xmark") + "~" + _t(e["runtime"])
        fl_cell = "---"
        if f:
            fl_cell = (r"\cmark" if f["closed_form_exact"] else r"\xmark") + \
                "~" + _t(f["runtime"])
        rows.append(
            f"{N} & ${_fmt(1 << N)}$ & ${len(cf)}$ & ${_fmt(cf[0])}$ & "
            f"${c150.d_max_ratio_obc0(N):.4f}$ & {eng_cell} & {fl_cell} \\\\"
        )
    body = "\n".join(rows)
    return (
        "\\begin{tabular}{rrrrrll}\n\\hline\n"
        "$N$ & $2^N$ & $\\#$sectors & $D_{\\max}$ & $D_{\\max}/2^N$ & "
        "engine & flip graph \\\\\n\\hline\n"
        f"{body}\n\\hline\n\\end{{tabular}}\n"
    )


def tab_spectral(data: dict, *, max_rows: int = 22) -> str:
    rows = []
    for r in data["sectors"]:
        if r["dim"] < 10:
            continue
        rows.append(r)
    # keep a readable spread: all w for a few N, then the largest shells
    rows.sort(key=lambda r: (r["N"], r["w"]))
    keep = [r for r in rows if r["N"] in (8, 10, 12, 14, 16, 18, 20)]
    if len(keep) > max_rows:
        keep = keep[:max_rows]
    out = []
    for r in keep:
        refl = {"commutes": r"$[P,U]=0$",
                "conjugates U to U^-1": r"$PUP=U^{-1}$"}.get(r["reflection"], "--")
        # extremal shells (|S_w| <= N+1) are exactly the nondegenerate ones
        mark = r"$^\dagger$" if r["dim"] <= r["N"] + 1 else ""
        out.append(
            f"{r['N']} & {r['w']}{mark} & ${_fmt(r['dim'])}$ & ${_fmt(r['n_distinct'])}$ & "
            f"${r['mult_plus_one']}$ & ${r['fix_closed_form']}$ & "
            f"${max(r['dim_krylov_over_dim']):.3f}$ & "
            f"${min(r['d_eff_over_dim']):.3f}$ & "
            f"${max(r['max_pop_over_uniform']):.2f}$ & {refl} \\\\"
        )
    body = "\n".join(out)
    return (
        "\\begin{tabular}{rrrrrrrrrl}\n\\hline\n"
        "$N$ & $w$ & $|S_w|$ & $\\#\\mathrm{spec}$ & $\\dim\\mathrm{Fix}$ & "
        "$\\binom{\\lceil N/2\\rceil}{w/2}$ & $\\dim\\mathcal{K}/|S_w|$ & "
        "$d_{\\mathrm{eff}}/|S_w|$ & $\\max p\\cdot|S_w|$ & reflection \\\\\n"
        "\\hline\n" f"{body}\n" "\\hline\n\\end{tabular}\n"
    )


def _sci(v: float) -> str:
    """1.1e-16 -> $1.1\\times10^{-16}$."""
    m, e = f"{v:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def tab_freeness(data: dict) -> str:
    rows = []
    for r in data["freeness"]:
        rows.append(
            f"{r['N']} & {_sci(r['bond_vs_engine'])} & ${r['rule150']:.3f}$ & "
            f"${r['no_string']:.3f}$ & ${r['det_phase_only']:.3f}$ & "
            f"{_sci(r['gaussian_ref'])} \\\\"
        )
    body = "\n".join(rows)
    return (
        "\\begin{tabular}{rrrrrr}\n\\hline\n"
        "$N$ & bond vs engine & rule 150 & no string & $\\det$ phase only & "
        "Gaussian ref. \\\\\n\\hline\n" f"{body}\n" "\\hline\n\\end{tabular}\n"
    )


def write_tables(data: dict) -> None:
    os.makedirs(TEXDIR, exist_ok=True)
    for name, txt in (("tab_c150_frontier", tab_frontier()),
                      ("tab_c150_spectral", tab_spectral(data)),
                      ("tab_c150_freeness", tab_freeness(data))):
        path = os.path.join(TEXDIR, f"{name}.tex")
        with open(path, "w") as f:
            f.write(txt)
        print("wrote", path)


# --- figures -----------------------------------------------------------------

def figures(data: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIGDIR, exist_ok=True)
    eng = load_results(150, "obc0")
    fr = c150.load_frontier()
    fr_N = sorted(int(k.split("_")[0]) for k in fr if k.endswith("obc0"))
    eng_N = sorted(eng.keys())

    # --- figure 1: the exact scaling and the two frontiers -------------------
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.4))
    Ns = list(range(4, 41))
    ax[0].semilogy(Ns, [1 << N for N in Ns], "k--", lw=1, label=r"$2^N$")
    ax[0].semilogy(Ns, [c150.d_max_obc0(N) for N in Ns], "-", color="C0",
                   label=r"$D_{\max}=\max_{w\,\mathrm{even}}\binom{N+1}{w}$")
    ax[0].semilogy(Ns, [c150.n_sectors_obc0(N) for N in Ns], "-", color="C1",
                   label=r"$\#\mathrm{sectors}=\lfloor (N{+}1)/2\rfloor+1$")
    if eng_N:
        ax[0].axvline(max(eng_N), color="C3", lw=1, ls=":")
        ax[0].text(max(eng_N) - 0.4, 3e2, "engine frontier", rotation=90,
                   fontsize=7, color="C3", ha="right")
    if fr_N:
        ax[0].axvline(max(fr_N), color="C2", lw=1, ls=":")
        ax[0].text(max(fr_N) - 0.4, 3e2, "flip-graph frontier", rotation=90,
                   fontsize=7, color="C2", ha="right")
    ax[0].set_xlabel("$N$")
    ax[0].set_ylabel("count")
    ax[0].legend(fontsize=7, loc="upper left")
    ax[0].set_title("exact sector structure, rule 150 obc0", fontsize=9)

    r = [c150.d_max_ratio_obc0(N) for N in Ns]
    ax[1].plot(Ns, r, "o-", ms=3, color="C0", label=r"$D_{\max}/2^N$ (exact)")
    ax[1].plot(Ns, [2.0 * sqrt(2.0 / (pi * (N + 1))) for N in Ns], "k--", lw=1,
               label=r"$2\sqrt{2/\pi(N{+}1)}$")
    ax[1].set_xlabel("$N$")
    ax[1].set_ylabel(r"$D_{\max}/2^N$")
    ax[1].legend(fontsize=7)
    ax[1].set_title(r"base is exactly 2; the prefactor is $N^{-1/2}$",
                    fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"fig_c150_frontier.{ext}"), dpi=150)
    plt.close(fig)

    # --- figure 2: what the sector does not fix ------------------------------
    sec = [s for s in data["sectors"] if s["dim"] >= 10]
    inner = [s for s in sec if s["dim"] > s["N"] + 1]
    extr = [s for s in sec if s["dim"] <= s["N"] + 1]
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.4))
    # filled = inner shells, open = extremal (the caption says so); keeping the
    # legend to two entries stops it from covering the data.
    for grp, filled in ((inner, True), (extr, False)):
        if not grp:
            continue
        d = [s["dim"] for s in grp]
        ax[0].semilogx(d, [max(s["dim_krylov_over_dim"]) for s in grp], "o",
                       ms=4, mfc=("C0" if filled else "none"), mec="C0",
                       label=r"$\dim\mathcal{K}(x)/|S_w|$" if filled else None)
        ax[0].semilogx(d, [min(s["d_eff_over_dim"]) for s in grp], "s",
                       ms=4, mfc=("C3" if filled else "none"), mec="C3",
                       label=r"$d_{\mathrm{eff}}(x)/|S_w|$" if filled else None)
    dims = [s["dim"] for s in sec]
    ax[0].axhline(1.0, color="k", ls="--", lw=1)
    ax[0].text(min(dims), 1.03, "monitored: uniform on $S_w$ (exact)",
               fontsize=7, color="k")
    ax[0].set_xlabel(r"sector dimension $|S_w|$")
    ax[0].set_ylabel("fraction of the sector")
    ax[0].set_ylim(0, 1.25)
    ax[0].legend(fontsize=7, loc="lower left", framealpha=0.95)
    ax[0].set_title("the graph gives the container, not the filling",
                    fontsize=9)

    ff = data["fix_full"]
    ax[1].semilogy([r["N"] for r in ff], [r["dim_fix"] for r in ff], "o",
                   ms=4, color="C2", label=r"$\dim\mathrm{Fix}(U)$ (exact)")
    ax[1].semilogy([r["N"] for r in ff], [r["closed_form"] for r in ff], "-",
                   color="C2", lw=1, label=r"$2^{\lceil N/2\rceil}$")
    ax[1].set_xlabel("$N$")
    ax[1].set_ylabel("dimension")
    ax[1].legend(fontsize=7)
    ax[1].set_title("stationary states: dark superpositions inside sectors",
                    fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"fig_c150_quantum.{ext}"), dpi=150)
    plt.close(fig)
    print("wrote figures/fig_c150_frontier.*, figures/fig_c150_quantum.*")


def main(argv=None):
    ap = argparse.ArgumentParser(description="R8 (C150) tables and figures")
    ap.add_argument("--quantum", action="store_true", help="cache the spectra")
    ap.add_argument("--tables", action="store_true")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dim-cap", type=int, default=DIM_CAP)
    args = ap.parse_args(argv)

    data = None
    if args.quantum:
        data = build_quantum(force=args.force, dim_cap=args.dim_cap)
    if args.tables or args.figures:
        data = data or build_quantum()
        if args.tables:
            write_tables(data)
        if args.figures:
            figures(data)


if __name__ == "__main__":
    main()
