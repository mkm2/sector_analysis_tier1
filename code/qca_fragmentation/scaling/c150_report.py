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


REF_LABEL = {
    "poisson": "Poisson",
    "coe": "COE ($\\beta=1$)",
    "orth": "Haar $O(D)$",
    "cue": "CUE ($\\beta=2$)",
    "orth_x2": "$2\\times$Haar $O$",
    "orth_x3": "$3\\times$Haar $O$",
}


def _levels_data() -> Optional[dict]:
    from ..quantum.rule150_levels import LEVELS_PATH
    if not os.path.exists(LEVELS_PATH):
        return None
    with open(LEVELS_PATH) as f:
        return json.load(f)


def tab_distinct(lv: dict) -> str:
    """The distinct-eigenvalue count against its closed form."""
    rows = []
    for r in lv.get("distinct", []):
        m = ceil(r["N"] / 2)
        form = (f"$3^{{{m}}}$" if r["N"] % 2 == 0
                else f"$(3^{{{m}}}{'+1' if m % 2 == 0 else '-1'})/2$")
        mark = r"\cmark" if r["matches"] else r"\xmark"
        rows.append(f"{r['N']} & ${_fmt(r['dim'])}$ & ${_fmt(r['n_distinct'])}$ & "
                    f"{form} & ${_fmt(r['closed_form'])}$ & "
                    f"${r['distinct_fraction']:.4f}$ & {mark} \\\\")
    body = "\n".join(rows)
    return ("\\begin{tabular}{rrrlrrc}\n\\hline\n"
            "$N$ & $2^N$ & $\\#$distinct & closed form & value & fraction & "
            "\\\\\n\\hline\n" f"{body}\n" "\\hline\n\\end{tabular}\n")


def tab_universal(lv: dict) -> str:
    """One universal spectrum per N; per-shell coverage of it."""
    rows = []
    for r in lv.get("universal", []):
        cov = ", ".join(f"$w{{=}}{s['w']}$: ${100*s['coverage']:.1f}\\%$"
                        for s in r["shells"] if s["dim"] >= 8)
        m_sub = r"\cmark" if r["all_shells_are_subsets"] else r"\xmark"
        m_cov = r"\cmark" if r["largest_shell_covers_all"] else r"\xmark"
        rows.append(
            f"{r['N']} & ${_fmt(r['universal_size'])}$ & "
            f"{m_sub} & "
            f"${r['largest_shell_w']}$ & "
            f"{m_cov} & "
            f"${r['min_gap_over_median']:.3f}$ \\\\")
        rows.append(f"\\multicolumn{{6}}{{l}}{{\\quad\\scriptsize coverage: "
                    f"{cov}}} \\\\")
    body = "\n".join(rows)
    return ("\\begin{tabular}{rrcccr}\n\\hline\n"
            "$N$ & $|\\Theta_N|$ & all shells $\\subseteq\\Theta_N$ & "
            "largest $w$ & covers all & $\\min\\text{gap}/\\text{median}$ \\\\\n"
            "\\hline\n" f"{body}\n" "\\hline\n\\end{tabular}\n")


HSF_PATH = os.path.join(ANALYTICS, "c150_hsf_compare.json")


def tab_hsf() -> Optional[str]:
    """C150 against the independent HSF Julia eigendata."""
    if not os.path.exists(HSF_PATH):
        return None
    with open(HSF_PATH) as f:
        rows = json.load(f)
    out = []
    for r in rows:
        ok = (r["sizes_match_closed_form"] and r["distinct_match"]
              and r["dim_fix_matches"] and r["all_sectors_subset_of_universal"]
              and r["largest_sector_coverage"] == 1.0)
        mark = r"\cmark" if ok else r"\xmark"
        out.append(
            f"{r['N']} & ${_fmt(r['n_states'])}$ & ${r['n_sectors']}$ & "
            f"${_fmt(r['n_distinct'])}$ & ${_fmt(r['distinct_closed_form'])}$ & "
            f"${r['min_sector_coverage']*100:.1f}\\%$ & "
            f"${r['frac_states_in_degenerate_eigenspace']:.2f}$ & "
            f"${r['max_within_sector_multiplicity']}$ & "
            f"${r['r_tilde']:.4f}$ & {mark} \\\\")
    body = "\n".join(out)
    return ("\\begin{tabular}{rrrrrrrrrc}\n\\hline\n"
            "$N$ & states & sectors & $\\#$distinct & \\eqref{eq:distinct} & "
            "min cover & deg.\\ states & max mult & $\\langle\\tilde r\\rangle$ & "
            "all checks \\\\\n\\hline\n" f"{body}\n" "\\hline\n\\end{tabular}\n")


def tab_levels(lv: dict) -> str:
    """Reference ensembles, then the C150 blocks, then the controls."""
    rows = ["\\multicolumn{6}{l}{\\itshape reference ensembles, identical "
            "pipeline ($D=%d$, %d samples each)}\\\\"
            % (lv["references"]["orth"]["D"],
               lv["references"]["orth"]["n_samples"])]
    for k in ("poisson", "orth_x3", "orth_x2", "coe", "orth", "cue"):
        r = lv["references"][k]
        rows.append(f"\\quad {REF_LABEL[k]} & --- & --- & --- & "
                    f"${r['r_tilde_mean']:.4f} \\pm {r['r_tilde_std']:.4f}$ & --- \\\\")
    rows.append("\\hline")
    rows.append("\\multicolumn{6}{l}{\\itshape C150 \\rt{obc0} wall shells}\\\\")
    sh = sorted(lv["shells"], key=lambda r: (r["N"], r["w"], r["block"]))
    for r in sh:
        rows.append(
            f"\\quad $N={r['N']}$, $w={r['w']}$ & {r['block'].replace('=','{=}')} & "
            f"${_fmt(r['block_dim'])}$ & ${r['degenerate_fraction']:.2f}$ & "
            f"${r['r_tilde']:.4f}$ & ${r['n_ratios']}$ \\\\")
    if lv.get("controls"):
        rows.append("\\hline")
        rows.append("\\multicolumn{6}{l}{\\itshape control: other unitary rules, "
                    "largest sector, small $N$}\\\\")
        for r in lv["controls"]:
            rows.append(
                f"\\quad \\rt{{W{r['rule']}}}, $N={r['N']}$ & --- & "
                f"${_fmt(r['block_dim'])}$ & ${r['degenerate_fraction']:.2f}$ & "
                f"${r['r_tilde']:.4f}$ & ${r['n_ratios']}$ \\\\")
    body = "\n".join(rows)
    return ("\\begin{tabular}{llrrrr}\n\\hline\n"
            "case & block & dim & deg.\\ frac. & $\\langle\\tilde r\\rangle$ & "
            "$n$ \\\\\n\\hline\n" f"{body}\n" "\\hline\n\\end{tabular}\n")


def _census_data() -> Optional[dict]:
    from ..quantum.rule150_basis_census import CENSUS_PATH
    if not os.path.exists(CENSUS_PATH):
        return None
    with open(CENSUS_PATH) as f:
        return json.load(f)


def tab_census(cs: dict) -> str:
    """Per-sector saturation statistics over the whole computational basis."""
    deph = {d["w"]: d["mean"] for d in cs.get("dephasing", [])}
    rows = []
    for s in cs["shells"]:
        dp = f"${deph[s['w']]:.3f}$" if s["w"] in deph else "---"
        rows.append(
            f"${s['w']}$ & ${s['size']}$ & ${s['mean']:.4f}$ & "
            f"${s['std']:.4f}$ & ${s['min']:.3f}$ & ${s['max']:.3f}$ & "
            f"${s['ceiling']:.3f}$ & ${s['frac_of_ceiling']:.3f}$ & "
            f"${s['haar_shell']['mean']:.3f}$ & {dp} \\\\")
    g = cs["global"]
    body = "\n".join(rows)
    tot = (f"\\hline\nall & ${cs['n_states']}$ & $\\mathbf{{{g['mean']:.4f}}}$ &"
           f" ${g['std']:.4f}$ & ${g['min']:.3f}$ & ${g['max']:.3f}$ & "
           f"--- & --- & ${cs['haar_sector_weighted']:.3f}$ & --- \\\\")
    return ("\\begin{tabular}{rrrrrrrrrr}\n\\hline\n"
            "$w$ & $|S_w|$ & $\\overline{S}$ & $\\sigma_w$ & min & max & "
            "ceiling & frac & Haar$_w$ & deph. \\\\\n\\hline\n"
            f"{body}\n{tot}\n" "\\hline\n\\end{tabular}\n")


def write_tables(data: dict) -> None:
    os.makedirs(TEXDIR, exist_ok=True)
    tables = [("tab_c150_frontier", tab_frontier()),
              ("tab_c150_spectral", tab_spectral(data)),
              ("tab_c150_freeness", tab_freeness(data))]
    cs = _census_data()
    if cs:
        tables.append(("tab_c150_census", tab_census(cs)))
    lv = _levels_data()
    if lv:
        tables.append(("tab_c150_levels", tab_levels(lv)))
        if lv.get("distinct"):
            tables.append(("tab_c150_distinct", tab_distinct(lv)))
        if lv.get("universal"):
            tables.append(("tab_c150_universal", tab_universal(lv)))
    hsf = tab_hsf()
    if hsf:
        tables.append(("tab_c150_hsf", hsf))
    for name, txt in tables:
        path = os.path.join(TEXDIR, f"{name}.tex")
        with open(path, "w") as f:
            f.write(txt)
        print("wrote", path)


# --- figures -----------------------------------------------------------------

def figures(data: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

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

    # --- figure 3: level statistics -----------------------------------------
    lv = _levels_data()
    if not lv:
        print("no analytics/c150_levels.json -- skipping the level figure")
        return
    refs = lv["references"]
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.4))

    # short plain-text names: the LaTeX of REF_LABEL is for the table, not for
    # matplotlib (a bare \times would be rendered literally).
    short = {"poisson": "Poisson", "orth_x3": r"$3\times$Haar $O$",
             "orth_x2": r"$2\times$Haar $O$", "coe": r"COE $\beta{=}1$",
             "orth": r"Haar $O(D)$", "cue": r"CUE $\beta{=}2$"}
    band = (("poisson", "0.55", "-", 0.006), ("orth_x2", "C4", "-.", 0.006),
            ("coe", "C2", "--", 0.006), ("orth", "C1", ":", -0.016),
            ("cue", "C5", "-", 0.006))
    xs = [1, 1e4]
    for k, col, ls, dy in band:
        m, s = refs[k]["r_tilde_mean"], refs[k]["r_tilde_std"]
        ax[0].fill_between(xs, m - s, m + s, color=col, alpha=0.16, lw=0)
        ax[0].plot(xs, [m, m], ls, color=col, lw=1)
        ax[0].text(1.15, m + dy, short[k], fontsize=6, color=col)
    for rows, mk, col, lab in (
            ([r for r in lv["shells"] if r["N"] % 2 == 0], "o", "C0",
             "even $N$ (no parity block)"),
            ([r for r in lv["shells"] if r["N"] % 2 == 1], "^", "C3",
             "odd $N$, parity-resolved")):
        if rows:
            ax[0].semilogx([r["block_dim"] for r in rows],
                           [r["r_tilde"] for r in rows], mk, ms=4.5,
                           color=col, label=lab)
    if lv.get("controls"):
        ax[0].semilogx([r["block_dim"] for r in lv["controls"]],
                       [r["r_tilde"] for r in lv["controls"]], "s", ms=4.5,
                       mfc="none", mec="k", label="control: other rules")
    ax[0].set_xlim(1.1, 1e4)
    ax[0].set_ylim(0.30, 0.66)
    ax[0].set_xlabel("block dimension")
    ax[0].set_ylabel(r"$\langle\tilde r\rangle$")
    ax[0].legend(fontsize=6.5, loc="lower right")
    ax[0].set_title("spacing-ratio statistic against generated references",
                    fontsize=9)

    # pooled P(s): the degeneracy-free / best-resolved C150 blocks
    pool = [s for r in lv["shells"] if r["N"] % 2 == 1 for s in r["spacings"]]
    pool_e = [s for r in lv["shells"] if r["N"] % 2 == 0 for s in r["spacings"]]
    bins = np.linspace(0, 3.5, 36)
    if pool_e:
        ax[1].hist(pool_e, bins=bins, density=True, histtype="step",
                   color="C0", lw=1.6, label="C150, even $N$")
    if pool:
        ax[1].hist(pool, bins=bins, density=True, histtype="step",
                   color="C3", lw=1.6, label="C150, odd $N$ resolved")
    for k, col, ls in (("poisson", "0.55", "-"), ("coe", "C2", "--"),
                       ("orth", "C1", ":"), ("cue", "C5", "-.")):
        h, e = np.histogram(refs[k]["spacings"], bins=bins, density=True)
        ax[1].plot(0.5 * (e[1:] + e[:-1]), h, ls, color=col, lw=1.2,
                   label=REF_LABEL[k])
    ax[1].set_xlabel("unfolded spacing $s$")
    ax[1].set_ylabel("$P(s)$")
    ax[1].legend(fontsize=6.5)
    ax[1].set_title("unfolded spacing distribution", fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"fig_c150_levels.{ext}"), dpi=150)
    plt.close(fig)
    print("wrote figures/fig_c150_levels.*")


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
            figure_entanglement()
            figure_census()




# --- entanglement figure (R8 sec.9) -----------------------------------------

ENT_PATH = os.path.join(ANALYTICS, "c150_entanglement.json")


def figure_entanglement() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if not os.path.exists(ENT_PATH):
        print("no analytics/c150_entanglement.json -- skipping")
        return
    with open(ENT_PATH) as f:
        d = json.load(f)
    from ..quantum.rule150_entanglement import kinematic_ceiling, volume_law_value

    lab = {"single": "single excitation", "neel": "Néel", "pair": "pair"}
    col = {"single": "C0", "neel": "C3", "pair": "C2"}
    fig, ax = plt.subplots(1, 3, figsize=(12.4, 3.5))

    # (a) traces at N=14
    for key, rec in d["traces"].items():
        name, Ns = key.rsplit("_N", 1)
        S = np.array(rec["S"])
        ax[0].plot(np.arange(S.size), S, "-", color=col[name], lw=1.5,
                   label=f"{lab[name]} ($w={rec['w']}$)")
        ax[0].axhline(kinematic_ceiling(int(Ns), rec["w"]), color=col[name],
                      ls="--", lw=0.9)
    ax[0].set_xlabel("cycles $t$")
    ax[0].set_ylabel(r"$S_\psi(t)$")
    ax[0].set_title(r"$N=14$; dashed = kinematic ceiling", fontsize=9)
    ax[0].legend(fontsize=7, loc="lower right")

    # (b) plateau vs N
    for name in ("single", "neel", "pair"):
        rows = [r for r in d["scaling"] if r["state"] == name]
        ax[1].plot([r["N"] for r in rows], [r["s_plateau"] for r in rows],
                   "o-", ms=4, color=col[name], label=lab[name])
        ax[1].plot([r["N"] for r in rows], [r["ceiling"] for r in rows],
                   ":", lw=1, color=col[name])
    Ns = np.arange(8, 40)
    ax[1].plot(Ns, [volume_law_value(int(n)) for n in Ns], "k--", lw=1,
               label=r"$\lfloor N/2\rfloor\ln 2$")
    ax[1].set_xlabel("$N$")
    ax[1].set_ylabel(r"$S_{\rm plateau}$")
    ax[1].set_title("area law vs volume law (dotted = ceiling)", fontsize=9)
    ax[1].legend(fontsize=7, loc="upper left")

    # (c) the dome
    for Ns_, mk in (("12", "s"), ("14", "o")):
        rows = [r for r in d["dome"][Ns_] if not r["frozen"]]
        nu = [r["filling"] for r in rows]
        ax[2].plot(nu, [r["s_plateau"] for r in rows], mk + "-", ms=4,
                   color="C0", label=f"$S_{{\\rm plateau}}$, $N={Ns_}$")
        ax[2].plot(nu, [r["ceiling"] for r in rows], mk + ":", ms=3,
                   color="C1", label=f"ceiling, $N={Ns_}$")
    ax[2].set_xlabel(r"wall filling $\nu = w/(N{+}1)$")
    ax[2].set_ylabel("entropy")
    ax[2].set_title("the entanglement dome", fontsize=9)
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"fig_c150_entanglement.{ext}"), dpi=150)
    plt.close(fig)
    print("wrote figures/fig_c150_entanglement.*")


# --- census figure (R8 sec.10) -----------------------------------------------

def figure_census() -> None:
    """The full-basis census: where the 2^N saturation values actually sit,
    against every random-state prediction."""
    cs = _census_data()
    if cs is None:
        print("no analytics/c150_basis_census.json -- skipping census figure")
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    os.makedirs(FIGDIR, exist_ok=True)
    N, g = cs["N"], cs["global"]
    w = np.array(cs["raw"]["w"])
    S = np.array(cs["raw"]["S_sat"])
    fig, ax = plt.subplots(1, 3, figsize=(13.0, 3.7))

    # (a) histogram, coloured by sector
    ws = sorted(set(w.tolist()))
    bins = np.linspace(0, max(S.max(), cs["haar_full"]["mean_real"]) * 1.05, 90)
    ax[0].hist([S[w == v] for v in ws], bins=bins, stacked=True,
               label=[f"$w={v}$" for v in ws], lw=0)
    for x, c, lab in ((g["mean"], "k", "measured mean"),
                      (cs["haar_sector_weighted"], "C3", "Haar (sector)"),
                      (cs["haar_full"]["mean_real"], "C1", "Haar (full)"),
                      (cs["haar_full"]["volume_law"], "C2", r"$\ln 2^{N/2}$")):
        ax[0].axvline(x, color=c, ls="--", lw=1.2, label=lab)
    ax[0].set_xlabel(r"$\overline{S}$ (time average)")
    ax[0].set_ylabel("# basis states")
    ax[0].set_title(f"all $2^{{{N}}}$ basis states, clustered by sector",
                    fontsize=9)
    ax[0].legend(fontsize=6, ncol=2)

    # (b) per-sector mean +- sd against the references
    sh = cs["shells"]
    xw = [s["w"] for s in sh]
    ax[1].errorbar(xw, [s["mean"] for s in sh], yerr=[s["std"] for s in sh],
                   fmt="o-", ms=5, capsize=3, color="C0",
                   label=r"measured $\pm\sigma_w$")
    ax[1].plot(xw, [s["haar_shell"]["mean"] for s in sh], "s--", ms=4,
               color="C3", label="Haar on $S_w$")
    ax[1].plot(xw, [s["ceiling"] for s in sh], ":", color="C7",
               label="kinematic ceiling")
    deph = cs.get("dephasing", [])
    if deph:
        ax[1].plot([d["w"] for d in deph], [d["mean"] for d in deph], "^-.",
                   ms=5, color="C4", label="random phases")
    ax[1].set_xlabel("wall number $w$")
    ax[1].set_ylabel("entropy")
    ax[1].set_title("every random-state model overshoots", fontsize=9)
    ax[1].legend(fontsize=7)

    # (c) the spread: measured vs Haar, per sector
    big = [s for s in sh if s["size"] > 1]
    xb = np.arange(len(big))
    ax[2].bar(xb - 0.2, [s["std"] for s in big], 0.4, label=r"measured $\sigma_w$")
    ax[2].bar(xb + 0.2, [s["haar_shell"]["std"] for s in big], 0.4,
              label=r"Haar $\sigma_w$")
    ax[2].set_xticks(xb)
    ax[2].set_xticklabels([f"$w{{=}}{s['w']}$\n$D{{=}}{s['size']}$" for s in big],
                          fontsize=7)
    ax[2].set_ylabel(r"s.d. of $\overline{S}$ within the sector")
    ax[2].set_title(r"intra-cluster spread: Haar gets it wrong "
                    r"in both directions", fontsize=9)
    top = max(max(s["std"] for s in big),
              max(s["haar_shell"]["std"] for s in big))
    ax[2].set_ylim(0, top * 1.28)          # headroom so the legend clears
    ax[2].legend(fontsize=7, loc="upper center", ncol=2)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"fig_c150_census.{ext}"), dpi=150)
    plt.close(fig)
    print("wrote figures/fig_c150_census.*")


if __name__ == "__main__":
    main()
