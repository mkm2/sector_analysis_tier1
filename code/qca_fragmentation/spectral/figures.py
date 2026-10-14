"""
R32 figures (thesis standards, viz/thesis_style.py).

    python -m qca_fragmentation.spectral.figures [--gate H] [--rules 150,156] [--paper]

Per rule: fig_r32_rule_{gate}_W{w}.pdf, four panels at the largest computed N:
  (a) quasienergy vs half-chain von Neumann entropy; non-degenerate states
      coloured by block (largest resolved block in palette[0]); states in a
      degenerate eigenspace (basis-dependent entropy) in light grey; Page value.
  (b) eta = S2 / S2_rand(block) vs quasienergy (basis independent, all states).
  (c) P(r) of the fully resolved half spectra vs the Poisson / GOE / GUE surmises.
  (d) <r> vs N at three resolution levels (none / Krylov / full).
"""

from __future__ import annotations

import json
import os

import numpy as np

from .. import results_io
from ..viz import thesis_style as ts
from . import analysis as A
from . import floquet as F
from . import stats as S

FIG_DIR = os.path.join(results_io.REPO_ROOT, "figures", "r32")


def _rule_class(w):
    cls = {204: "frozen", 51: "free", 108: "exp. fragmented", 156: "exp. fragmented",
           198: "exp. fragmented", 201: "exp. fragmented", 60: "polynomial",
           102: "polynomial", 105: "polynomial", 150: "polynomial"}
    return cls.get(w, "one Krylov sector" if w != 54 else "two Krylov sectors")


def rule_figure(w, gate="H", summary=None, out_dir=FIG_DIR, Nplot=None, bc="obc0"):
    plt = ts.use(ts._mode)
    Ns = F.available(gate, bc)[w]
    N = Nplot or max(Ns)
    d = F.load(F.spec_path(gate, N, w, bc))
    th = d["theta"]
    keys = S.block_keys(d)
    uk, inv, cnt = np.unique(keys, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    order = np.argsort(-cnt)
    rank_of = np.empty(uk.size, int); rank_of[order] = np.arange(uk.size)
    rk = rank_of[inv]
    nondeg = d["degeneracy"] == 1

    fig, axs = ts.subplots(2, 2, aspect=0.78)
    (a, b), (c, e) = axs
    # (a) theta vs S_vN
    a.scatter(th[~nondeg], d["entropy"][~nondeg], s=ts.DOT ** 2, color=ts.fill(ts.GREY, 0.6),
              linewidths=0, rasterized=True, zorder=1)
    for lvl, col in ((2, ts.PALETTE[2]), (1, ts.PALETTE[1]), (0, ts.PALETTE[0])):
        m = nondeg & ((rk == lvl) if lvl < 2 else (rk >= 2))
        a.scatter(th[m], d["entropy"][m], s=ts.DOT ** 2, color=ts.dark(col),
                  linewidths=0, rasterized=True, zorder=2 + (2 - lvl))
    # exact eigenstates of every gate: 1-dim gate-algebra blocks inside sectors of dim >= 3
    secsize = np.asarray(d["sector_sizes"])[d["sector_id"] - 1]
    bdim1 = cnt[inv]       # block dimension per state (a dict rebuilt per state was O(blocks x states))
    spec = (bdim1 == 1) & (secsize >= 3)
    if spec.any():
        a.scatter(th[spec], d["entropy"][spec], s=(1.8 * ts.MS) ** 2, marker="D",
                  facecolors="none", edgecolors=ts.INK, linewidths=0.6, zorder=8,
                  label=r"gate singlet")
        a.legend(loc="center right", bbox_to_anchor=(1.0, 0.3), handlelength=1.0,
                 borderaxespad=0.2)
    page = S.page_entropy(N // 2, N - N // 2)
    a.axhline(page, color=ts.INK, lw=0.6, ls="--", zorder=5)
    a.annotate(r"Page", (np.pi, page), xytext=(-2, 2), textcoords="offset points",
               ha="right", va="bottom", fontsize=ts.FS["annot"])
    a.set_xlabel(r"quasienergy $\theta$"); a.set_ylabel(r"$S_\mathrm{vN}$")
    a.set_xlim(-np.pi * 1.03, np.pi * 1.03); a.set_ylim(bottom=-0.25)
    a.set_xticks([-np.pi, 0, np.pi], [r"$-\pi$", r"$0$", r"$\pi$"])

    # (b) eta
    ref = d.get("s2_block_rand")
    dim = cnt[inv]
    if ref is not None:
        ok = (dim >= 2) & (ref > 1e-12)
        eta = np.where(ok, d["s2_haar"] / np.where(ok, ref, 1), np.nan)
        for lvl, col in ((2, ts.PALETTE[2]), (1, ts.PALETTE[1]), (0, ts.PALETTE[0])):
            m = ok & ((rk == lvl) if lvl < 2 else (rk >= 2))
            b.scatter(th[m], eta[m], s=ts.DOT ** 2, color=ts.dark(col), linewidths=0,
                      rasterized=True, zorder=2 + (2 - lvl))
        b.axhline(1, color=ts.INK, lw=0.6, ls="--")
        b.set_ylim(0, max(1.6, np.nanmax(eta) * 1.05) if np.any(ok) else 1.6)
    b.set_xlabel(r"quasienergy $\theta$")
    b.set_ylabel(r"$S_2/S_2^\mathrm{rand}(\mathrm{block})$")
    b.set_xlim(-np.pi, np.pi)
    b.set_xticks([-np.pi, 0, np.pi], [r"$-\pi$", r"$0$", r"$\pi$"])

    # (c) P(r)
    r, info = S.pooled_r(d, A.NMIN)
    x = np.linspace(0, 1, 200)
    if r.size >= 30:
        c.hist(r, bins=np.linspace(0, 1, 21), density=True, color=ts.fill(ts.PALETTE[0]),
               edgecolor=ts.dark(ts.PALETTE[0]), linewidth=0.5)
    c.plot(x, S.surmise(x, 0), color=ts.GREY, ls="--", lw=1.0, label="Poisson")
    c.plot(x, S.surmise(x, 1), color=ts.INK, lw=1.0, label="GOE")
    c.plot(x, S.surmise(x, 2), color=ts.PALETTE[1], ls=":", lw=1.0, label="GUE")
    c.set_xlabel(r"$r$"); c.set_ylabel(r"$P(r)$"); c.set_xlim(0, 1)
    c.set_ylim(0, 2.3)
    c.legend(loc="upper right", handlelength=1.6, borderaxespad=0.2)
    if r.size < 30:
        c.annotate(r"too few levels", (0.5, 1.15), ha="center", fontsize=ts.FS["annot"])

    # (d) <r> vs N, three resolution levels
    sm = summary or {}
    for lvl, mk, col, lab, fc in (("r_none", "o", ts.GREY, "none", "none"),
                                  ("r_krylov", "s", ts.PALETTE[1], "Krylov", "none"),
                                  ("r_full", "o", ts.PALETTE[0], "full", None)):
        xs, ys, es = [], [], []
        for n in Ns:
            s = sm.get(f"{w}_{n}") or A.summarize(F.load(F.spec_path(gate, n, w, bc)))
            v = s[lvl]
            if v["r_mean"] is not None and v["n_r"] >= 30:
                xs.append(n); ys.append(v["r_mean"]); es.append(v["r_err"] or 0)
        if xs:
            e.errorbar(xs, ys, yerr=es, marker=mk, ms=ts.MS, color=col, mfc=fc or col,
                       lw=0.8, capsize=0, label=lab)
    for val, lab, ls in ((S.R_GUE, "GUE", ":"), (S.R_GOE, "GOE", "-"), (S.R_POISSON, "Poisson", "--")):
        e.axhline(val, color=ts.GREY, lw=0.5, ls=ls, zorder=0)
        e.annotate(lab, (1, val), xycoords=("axes fraction", "data"), xytext=(3, 0),
                   textcoords="offset points", ha="left", va="center",
                   fontsize=ts.FS["annot"] - 1, color=ts.GREY, annotation_clip=False)
    e.set_xlabel(r"$N$"); e.set_ylabel(r"$\langle r\rangle$")
    e.set_ylim(0.3, 0.72)
    e.xaxis.get_major_locator().set_params(integer=True)
    e.legend(loc="upper left", ncol=3, handlelength=1.2, columnspacing=0.8,
             borderaxespad=0.2)
    for ax, lab in zip((a, b, c, e), "abcd"):
        ts.panel_label(ax, f"({lab})")
    a.annotate(rf"W{w}, $N={N}$" + ("" if bc == "obc0" else r", periodic"), (0.02, 0.03), xycoords="axes fraction",
               ha="left", va="bottom", fontsize=ts.FS["annot"],
               bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none"))
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"fig_r32_rule_{gate}_W{w}" + ("" if bc == "obc0" else f"_{bc}"))
    ts.save(fig, stem)
    plt.close(fig)
    return stem


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", default="H")
    ap.add_argument("--rules", default="")
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--bc", default="obc0")
    a = ap.parse_args(argv)
    ts.use("paper" if a.paper else "thesis")
    sp = A.summary_path(a.gate, a.bc)
    summary = json.load(open(sp)) if os.path.exists(sp) else None
    rules = [int(x) for x in a.rules.split(",")] if a.rules else list(F.available(a.gate, a.bc))
    for w in rules:
        print("wrote", rule_figure(w, a.gate, summary, bc=a.bc))
    if summary and not a.rules and a.bc == "obc0":
        print("wrote", overview_figure(summary, a.gate))
    po, pp = A.summary_path(a.gate, "obc0"), A.summary_path(a.gate, "pbc")
    if not a.rules and os.path.exists(po) and os.path.exists(pp):
        print("wrote", bc_figure(json.load(open(po)), json.load(open(pp)), a.gate))



#: display order: the R32 classes, by the number k of neighbour patterns that fire
#: the gate (and, for k = 2, whether only one neighbour matters)
CLASS_ORDER = [
    (r"$k=0$", [204]),
    (r"$k=1$", [108, 156, 198, 201]),
    (r"$k=2$ one-sided", [60, 102, 153, 195]),
    (r"$k=2$ XOR", [105, 150]),
    (r"$k=3$", [54, 57, 99, 147]),
    (r"$k=4$", [51]),
]


def overview_figure(summary, gate="H", out_dir=FIG_DIR):
    """(a) resolved <r> at each rule's largest N; (b) degenerate fraction."""
    plt = ts.use(ts._mode)
    fig, (a, b) = ts.subplots(2, 1, aspect=0.75, sharex=True)
    xs, labels, x = [], [], 0
    groups = []
    for name, ws in CLASS_ORDER:
        g0 = x
        for w in ws:
            Ns = sorted(int(k.split("_")[1]) for k in summary if k.startswith(f"{w}_"))
            if not Ns:
                x += 1; continue
            s = summary[f"{w}_{A.class_N(summary, w)}"]     # same N within a class
            for lvl, mk, col, fc, dx in (("r_none", "o", ts.GREY, "none", -0.18),
                                         ("r_full", "o", ts.PALETTE[0], None, 0.18)):
                v = s[lvl]
                if v["r_mean"] is not None and v["n_r"] >= 30:
                    a.errorbar(x + dx, v["r_mean"], yerr=v["r_err"] or 0, marker=mk,
                               ms=ts.MS, color=col, mfc=fc or col, lw=0.8, capsize=0)
            b.bar(x, s["deg"]["n_distinct_global"] / 2 ** s["N"], width=0.6,
                  color=ts.fill(ts.PALETTE[0]), edgecolor=ts.dark(ts.PALETTE[0]),
                  linewidth=0.5)
            labels.append(rf"{w}"); xs.append(x); x += 1
        groups.append((name, g0, x - 1)); x += 0.6
    for val, lab, ls in ((S.R_GUE, "GUE", ":"), (S.R_GOE, "GOE", "-"),
                         (S.R_POISSON, "Poisson", "--")):
        a.axhline(val, color=ts.GREY, lw=0.5, ls=ls, zorder=0)
        a.annotate(lab, (1, val), xycoords=("axes fraction", "data"), xytext=(3, 0),
                   textcoords="offset points", ha="left", va="center",
                   fontsize=ts.FS["annot"] - 1, color=ts.GREY, annotation_clip=False)
    a.set_ylabel(r"$\langle r\rangle$"); a.set_ylim(0.3, 0.68)
    a.plot([], [], "o", color=ts.GREY, mfc="none", ms=ts.MS, label="unresolved")
    a.plot([], [], "o", color=ts.PALETTE[0], ms=ts.MS, label="fully resolved")
    a.legend(loc="upper left", ncol=2, handlelength=1.0, borderaxespad=0.2)
    b.set_yscale("log"); b.set_ylim(1e-5, 60)
    b.set_ylabel(r"distinct $\theta$ / $2^N$")
    b.set_xticks(xs, labels, rotation=90)
    b.set_xlabel(r"rule $W$")
    for name, g0, g1 in groups:
        b.annotate(name, ((g0 + g1) / 2, 1.0), xycoords=("data", "axes fraction"),
                   xytext=(0, -2), textcoords="offset points", ha="center", va="top",
                   fontsize=ts.FS["annot"] - 2, color=ts.GREY)
    for ax, lab in zip((a, b), "ab"):
        ts.panel_label(ax, f"({lab})")
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"fig_r32_overview_{gate}")
    ts.save(fig, stem)
    plt.close(fig)
    return stem


def bc_figure(sum_obc, sum_pbc, gate="H", out_dir=FIG_DIR, N_common=None):
    """
    Do the classes survive the boundary condition?  (a) fully resolved <r> for
    obc0 and pbc at the same (even) N; (b) number of gate-algebra blocks per
    Krylov-sector count, i.e. how much hidden splitting there is, both bc.
    """
    plt = ts.use(ts._mode)
    fig, (a, b) = ts.subplots(2, 1, aspect=0.75, sharex=True)

    def common_N(w):
        # the largest even N present for BOTH boundary conditions and for EVERY rule of
        # w's class (rules of one class are compared at the same N)
        members = [v for _, ws in CLASS_ORDER for v in ws if A.pattern_class(v) == A.pattern_class(w)]
        Ns = set.intersection(*[set(int(k.split("_")[1]) for k in sm if k.startswith(f"{v}_"))
                                for sm in (sum_obc, sum_pbc) for v in members])
        Ns = [n for n in Ns if n % 2 == 0 and (N_common is None or n <= N_common)]
        return max(Ns) if Ns else None

    def dot(ax, x, v, **kw):
        if v["r_mean"] is not None and v["n_r"] >= 30:
            ax.errorbar(x, v["r_mean"], yerr=v["r_err"] or 0, ms=ts.MS, lw=0.8, capsize=0, **kw)

    xs, labels, x, groups = [], [], 0, []
    for name, ws in CLASS_ORDER:
        g0 = x
        for w in ws:
            N = common_N(w)
            if N is None:
                x += 1; continue
            o, p = sum_obc[f"{w}_{N}"], sum_pbc[f"{w}_{N}"]
            dot(a, x - 0.2, o["r_full"], marker="o", color=ts.PALETTE[0])
            dot(a, x + 0.05, p["r_full_by_type"]["real"], marker="s", color=ts.PALETTE[1], mfc="none")
            dot(a, x + 0.25, p["r_full_by_type"]["complex"], marker="s", color=ts.PALETTE[1])
            b.plot(x - 0.18, o["n_blocks"], marker="o", ms=ts.MS, color=ts.PALETTE[0], ls="none")
            b.plot(x + 0.18, p["n_blocks"], marker="s", ms=ts.MS, color=ts.PALETTE[1], ls="none")
            labels.append(f"{w}"); xs.append(x); x += 1
        groups.append((name, g0, x - 1)); x += 0.6
    for val, lab, ls in ((S.R_GUE, "GUE", ":"), (S.R_GOE, "GOE", "-"),
                         (S.R_POISSON, "Poisson", "--")):
        a.axhline(val, color=ts.GREY, lw=0.5, ls=ls, zorder=0)
        a.annotate(lab, (1, val), xycoords=("axes fraction", "data"), xytext=(3, 0),
                   textcoords="offset points", ha="left", va="center",
                   fontsize=ts.FS["annot"] - 1, color=ts.GREY, annotation_clip=False)
    a.plot([], [], "o", color=ts.PALETTE[0], ms=ts.MS, label="open")
    a.plot([], [], "s", color=ts.PALETTE[1], ms=ts.MS, mfc="none", label=r"ring, real $j$")
    a.plot([], [], "s", color=ts.PALETTE[1], ms=ts.MS, label=r"ring, complex $j$")
    a.legend(loc="upper left", ncol=3, handlelength=1.0, borderaxespad=0.2)
    a.set_ylabel(r"$\langle r\rangle$ (resolved)"); a.set_ylim(0.3, 0.68)
    b.set_yscale("log"); b.set_ylabel(r"resolved blocks"); b.set_ylim(top=4e4)
    b.set_xticks(xs, labels, rotation=90); b.set_xlabel(r"rule $W$")
    for name, g0, g1 in groups:
        b.annotate(name, ((g0 + g1) / 2, 1.0), xycoords=("data", "axes fraction"),
                   xytext=(0, -2), textcoords="offset points", ha="center", va="top",
                   fontsize=ts.FS["annot"] - 2, color=ts.GREY)
    for ax, lab in zip((a, b), "ab"):
        ts.panel_label(ax, f"({lab})")
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"fig_r32_bc_compare_{gate}")
    ts.save(fig, stem)
    plt.close(fig)
    return stem


# --- candidate thesis figures (R32 storyline draft; NOT handed off) ------------

CAND_DIR = os.path.join(FIG_DIR, "thesis_candidates")


def cand_chaos_and_scars(w=57, gate="H", out_dir=CAND_DIR):
    """B: one chaotic rule -- entanglement vs quasienergy with exact scars, and P(r)."""
    plt = ts.use(ts._mode)
    N = max(F.available(gate)[w])
    d = F.load(F.spec_path(gate, N, w))
    th, S_ = d["theta"], d["entropy"]
    keys = S.block_keys(d)
    uk, cnt = np.unique(keys, return_counts=True)
    bdim = np.array([dict(zip(uk, cnt))[k] for k in keys])
    secsize = np.asarray(d["sector_sizes"])[d["sector_id"] - 1]
    spec = (bdim == 1) & (secsize >= 3)
    nondeg = (d["degeneracy"] == 1) & ~spec
    fig, (a, c) = ts.subplots(1, 2, aspect=0.42)
    a.scatter(th[nondeg], S_[nondeg], s=ts.DOT ** 2, color=ts.dark(ts.PALETTE[0]),
              linewidths=0, rasterized=True)
    a.scatter(th[spec], S_[spec], s=(1.8 * ts.MS) ** 2, marker="D", facecolors="none",
              edgecolors=ts.PALETTE[1], linewidths=0.8, zorder=5, clip_on=False)
    page = S.page_entropy(N // 2, N - N // 2)
    a.axhline(page, color=ts.INK, lw=0.6, ls="--")
    a.annotate("Page", (0, page), xytext=(0, 2), textcoords="offset points",
               ha="center", va="bottom", fontsize=ts.FS["annot"])
    a.annotate(r"exact scars", (np.pi * 0.55, 0.25), color=ts.PALETTE[1],
               fontsize=ts.FS["annot"], ha="center")
    a.set_xlim(-np.pi * 1.04, np.pi * 1.04); a.set_ylim(-0.3, page * 1.12)
    a.set_xticks([-np.pi, 0, np.pi], [r"$-\pi$", r"$0$", r"$\pi$"])
    a.set_xlabel(r"quasienergy $\theta$"); a.set_ylabel(r"$S_\mathrm{vN}$")
    r, info = S.pooled_r(d, A.NMIN)
    x = np.linspace(0, 1, 200)
    c.hist(r, bins=np.linspace(0, 1, 21), density=True, color=ts.fill(ts.PALETTE[0]),
           edgecolor=ts.dark(ts.PALETTE[0]), linewidth=0.5)
    c.plot(x, S.surmise(x, 0), color=ts.GREY, ls="--", label="Poisson")
    c.plot(x, S.surmise(x, 1), color=ts.INK, label="GOE")
    c.set_xlabel(r"$r$"); c.set_ylabel(r"$P(r)$"); c.set_xlim(0, 1); c.set_ylim(0, 2.2)
    c.legend(loc="upper right", handlelength=1.4)
    c.annotate(rf"$\langle r\rangle={info['r_mean']:.3f}$", (0.97, 0.62), xycoords="axes fraction",
               ha="right", fontsize=ts.FS["annot"])
    for ax, lab in zip((a, c), "ab"):
        ts.panel_label(ax, f"({lab})")
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"cand_B_chaos_scars_W{w}")
    ts.save(fig, stem); plt.close(fig)
    return stem


def cand_resolution(rules=(147, 54), gate="H", summary=None, out_dir=CAND_DIR):
    """C: symmetry resolution decides -- <r> vs N, Krylov-only vs fully resolved."""
    plt = ts.use(ts._mode)
    fig, axs = ts.subplots(1, len(rules), aspect=0.42, sharey=True)
    for ax, w in zip(np.atleast_1d(axs), rules):
        for lvl, mk, col, fc, lab in (("r_krylov", "s", ts.GREY, "none", "Krylov sectors only"),
                                      ("r_full", "o", ts.PALETTE[0], None, "fully resolved")):
            xs, ys, es = [], [], []
            for n in sorted(F.available(gate)[w]):
                v = summary[f"{w}_{n}"][lvl]
                if v["r_mean"] is not None and v["n_r"] >= 30:
                    xs.append(n); ys.append(v["r_mean"]); es.append(v["r_err"] or 0)
            ax.errorbar(xs, ys, yerr=es, marker=mk, ms=ts.MS, color=col, mfc=fc or col,
                        lw=0.8, capsize=0, label=lab)
        for val, lab, ls in ((S.R_GOE, "GOE", "-"), (S.R_POISSON, "Poisson", "--")):
            ax.axhline(val, color=ts.GREY, lw=0.5, ls=ls, zorder=0)
        ax.set_xlabel(r"$N$"); ax.set_ylim(0.32, 0.58)
        ax.annotate(rf"W{w}", (0.03, 0.95), xycoords="axes fraction", ha="left", va="top",
                    fontsize=ts.FS["annot"])
        ax.xaxis.get_major_locator().set_params(integer=True)
    axs = np.atleast_1d(axs)
    axs[0].set_ylabel(r"$\langle r\rangle$")
    axs[-1].annotate("GOE", (1, S.R_GOE), xycoords=("axes fraction", "data"), xytext=(3, 0),
                     textcoords="offset points", va="center", color=ts.GREY,
                     fontsize=ts.FS["annot"] - 1, annotation_clip=False)
    axs[-1].annotate("Poisson", (1, S.R_POISSON), xycoords=("axes fraction", "data"),
                     xytext=(3, 0), textcoords="offset points", va="center", color=ts.GREY,
                     fontsize=ts.FS["annot"] - 1, annotation_clip=False)
    axs[-1].legend(loc="lower right", handlelength=1.2)
    for ax, lab in zip(axs, "ab"):
        ts.panel_label(ax, f"({lab})")
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, "cand_C_resolution")
    ts.save(fig, stem); plt.close(fig)
    return stem


def cand_singlets(summary, gate="H", out_dir=CAND_DIR):
    """D: number of exact gate-singlet eigenstates vs N, by class."""
    plt = ts.use(ts._mode)
    fig, ax = ts.subplots(1, 1, aspect=0.55)
    series = [(r"$k=1$ (108)", 108, ts.PALETTE[2], "o"), (r"$k=1$ (156)", 156, ts.PALETTE[2], "s"),
              (r"$k=2$ one-sided (60)", 60, ts.PALETTE[3], "^"),
              (r"$k=2$ XOR (150)", 150, ts.PALETTE[4], "v"),
              (r"$k=3$ (147)", 147, ts.PALETTE[0], "o"), (r"$k=3$ (57)", 57, ts.PALETTE[0], "s"),
              (r"$k=3$ (54)", 54, ts.PALETTE[0], "D")]
    for lab, w, col, mk in series:
        xs, ys = [], []
        for n in range(6, 20):
            s = summary.get(f"{w}_{n}")
            if s:
                xs.append(n); ys.append(s["n_singlets"])
        ax.plot(xs, ys, marker=mk, ms=ts.MS, color=col, lw=0.8,
                mfc="white" if mk in "sD" else col, label=lab)
    ax.set_yscale("log"); ax.set_xlabel(r"$N$"); ax.set_ylabel(r"exact gate singlets")
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), handlelength=1.6)
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, "cand_D_singlets")
    ts.save(fig, stem); plt.close(fig)
    return stem


if __name__ == "__main__":
    main()
