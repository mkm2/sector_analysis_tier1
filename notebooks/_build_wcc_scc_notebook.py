"""Generate the WCC/SCC stack-figure notebook.  Written via nbformat so the
JSON is guaranteed valid."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(t):  cells.append(nbf.v4.new_markdown_cell(t.strip("\n")))
def code(t): cells.append(nbf.v4.new_code_cell(t.strip("\n")))

md(r"""
# The WCC / SCC plane and the flow between them

This notebook rebuilds the **stack** figure of report R14 from scratch: load the
sweep data, extract the growth bases, assemble the two planes, and draw.

It is written to be taken over and edited. The figure code in §6 is inline and
self-contained — it does not call `qca_fragmentation.viz.journal`, so you can
change it without touching the package.

**What the figure says.** Every rule has a position in the *sector* plane
$(a, b)$ and a position in the *monitored attractor* plane
$(a_\mathrm{att}, b_\mathrm{att})$. Dissipation moves it from the first to the
second. Of the 230 rules that move, 220 move straight down and none moves
horizontally — and §5 shows that is forced rather than observed.

| symbol | meaning |
|---|---|
| $a$ | growth base of the number of sectors, $n_\mathrm{wcc}(N) \sim a^N$ |
| $b$ | growth base of the largest sector, $D_{\max}(N) \sim b^N$ |
| $a_\mathrm{att}$ | growth base of the number of terminal SCCs (attractors) |
| $b_\mathrm{att}$ | growth base of the largest attractor |
""")

code(r"""
import os, sys, json, math
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

# the package lives in ../code relative to this notebook
ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.insert(0, os.path.join(ROOT, "code"))

from qca_fragmentation import results_io
from qca_fragmentation.core import rules as rules_mod
from qca_fragmentation.scaling import sectors
from qca_fragmentation.scaling import fits

BC = "obc0"          # pbc is held pending caveats -- see NEXT_STEPS.md
print("repo root:", results_io.REPO_ROOT)
""")

md(r"""
## 1. Where the numbers come from

Two sweeps feed this figure, both already on disk — nothing here recomputes a
transition graph.

**Sector sweep (WCC).** For each rule and each $N$, the transition graph is
built and its *weak* components are found by union-find. Records live in
`results_wcc/{rule}_{bc}.jsonl` and give `n_wcc` and `d_max_wcc` per $N$.
Weak components are the **sectors**: they partition the basis, and the partition
is exact for both the monitored and the unmonitored experiment.

**Attractor sweep (SCC).** The same graph, Tarjan's algorithm, keeping the
*terminal* strongly connected components. Records live in
`results/{rule}_{bc}.jsonl` and give `n_recurrent` and `d_max`. Terminal SCCs
are the **monitored attractors** — a monitored-only quantity.

`sectors.load_series` and `summary.load_series` are the readers.
""")

code(r"""
RULE = 156                                    # IIVI, unitary, the R2 room-packing rule
s_sec = sectors.load_series(RULE, BC, sectors.UNIFORM_N_CAP)
print("sector series for W%d (%s)" % (RULE, "".join(rules_mod.wolfram_to_tuple(RULE))))
for N, n, d in zip(s_sec["N"], s_sec["n_wcc"], s_sec["d_max_wcc"]):
    print(f"   N={N:<3} n_wcc={n:<6} d_max_wcc={d:<7} n_wcc*d_max={n*d:<9} 2^N={1<<N}")
""")

md(r"""
`UNIFORM_N_CAP = 16` is the largest $N$ reached for *every* rule. All bases are
fitted inside that window so they are comparable across the rule space; a rule
with a longer series from a targeted extension would otherwise be fitted on a
different domain, which is not a like-for-like comparison.

Note the last column: $n_\mathrm{wcc}\cdot D_{\max} \ge 2^N$ always, because the
sectors partition the basis. That is the finite-$N$ form of the exclusion curve
$ab \ge 2$ drawn on the figure.
""")

code(r"""
s_att = None
from qca_fragmentation.scaling.dissipative import load_series as load_att_series
s_att = load_att_series(RULE, BC)
keep = [i for i, N in enumerate(s_att["N"]) if N <= sectors.UNIFORM_N_CAP]
print("attractor series for W%d" % RULE)
for i in keep:
    print(f"   N={s_att['N'][i]:<3} n_recurrent={s_att['n_recurrent'][i]:<6} "
          f"d_max={s_att['d_max'][i]}")
print("\nfor a UNITARY rule these coincide with the sector series -- assertion A2")
""")

md(r"""
## 2. The growth model

Every series here is fitted to

$$ y(N) \;\simeq\; c \cdot N^{\alpha} \cdot b^{N} $$

with **$b$ the growth base** and $\alpha$ a sub-leading power. Both matter:
$b=1$ with $\alpha=1$ is a linear series, $b=1$ with $\alpha=0$ is a constant,
and $b=2$ with $\alpha=-\tfrac12$ is a central binomial — all three occur.

The figure uses only $b$. The next section is about how $b$ is obtained, because
that is where all the care is.
""")

md(r"""
## 3. Base extraction — the decision ladder

`sectors.series_descriptor(rule, bc, key, Ns, ys)` returns a dict with `base`,
`alpha`, `cls`, `exact` and a human-readable `source` recording **which branch
produced the number**. Always read `source`; it is the difference between a
theorem and a regression.

The branches, in the order they are tried:

1. **Analytic override.** A small table of *derived* results
   (`sectors.ANALYTIC`) takes precedence over any fit: rule 204 (`IIII`,
   $2^N$ frozen singletons), rule 51 (`VVVV`, one sector of size $2^N$), rules
   150/105 (central binomials, base 2 with $\alpha=-\tfrac12$, from R8), and the
   room-packing base $4^{1/5}$ for the six domain-wall rules from R2 §3.
   `exact=True`.

2. **Irregularity.** A strongly non-monotone series with a large log-residual
   gets `cls="irregular"` and `base=None` rather than a meaningless number.
   Rules 90 and 165 are the type case: `n_wcc` runs 2, 1, 4, 4, 2, 6, 2, 6, 12,
   1, 20, and forcing a base out of that gives the impossible product $ab=1$.

3. **Volume-fraction discriminator.** Tests whether $D_{\max}=c\,2^N N^{\alpha}$
   — base *exactly* 2 with a power-law prefactor — by comparing a power law and
   an exponential as models of $\ln(D_{\max}/2^N)$. Over a short window $\ln N$
   is nearly affine in $N$, so this test is easy to fool; it carries three
   guards (series eventually non-decreasing, $|\alpha|\le 8$, and the power law
   must win on the **full** series and not merely on a 6-point tail).

4. **Saturation.** A series that stops changing is constant, base 1.

5. **BIC model selection** among M0 (constant), M1 ($cN^{\alpha}$) and
   M2 ($cN^{\alpha}e^{\kappa N}$).

6. **Rate correction — the important one.** If M2 wins, its $\kappa$ is
   **not** used for the base. M2's $\alpha\ln N$ term absorbs part of the growth,
   which biases $\kappa$ downward: for rule 156 it reports 1.2554 against the
   derived $4^{1/5}=1.3195$. The rate instead comes from the two-parameter fit
   $\ln y = c + \kappa N$ (`fits.fit_pure_exponential`). M2 is kept for the
   growth *class* and for $\alpha$, never for the rate. This is the R2 §3
   convention.

7. **Exact integer recurrence.** If the whole series satisfies a linear
   recurrence with integer coefficients, the base is the dominant root of its
   characteristic polynomial and `exact=True`. This is how $\varphi$, $\rho$,
   $\psi$, $\sqrt3$ and the rest enter — they are *derived from an exact
   recurrence*, not read off a regression.

8. **Parity split.** Many series oscillate with $N \bmod 2$. When splitting
   halves the rms log-residual, each branch is fitted separately; an exact
   recurrence on a branch wins, otherwise the branch is *classified* rather than
   merely fitted (rule 44's even-$N$ sector counts are 7, 9, 11, 13, 15, 17 — a
   straight line, and a pure exponential fit to a straight line still returns
   base 1.09).

9. **Class/base reconciliation.** A base recovered from a recurrence or a parity
   split overrides the class BIC assigned to the mixed series.

10. **$\kappa$-band guard.** If the class says exponential but the base is not
    exact and the leave-one-out band of M2's $\kappa$ fails to exclude zero
    rate, the label is unsupported and the series is demoted to polynomial.

11. **$\alpha$ re-derivation.** $\alpha$ is refitted holding the *reported* base
    fixed (`sectors._alpha_at_base`). Before this, $\alpha$ came from M2 while
    the base came from a later branch, so the two halves described different
    models.

12. **Clamp.** Both $n_\mathrm{wcc}$ and $D_{\max}$ lie in $[1, 2^N]$, so both
    bases lie in $[1,2]$. A fit outside that contradicts a theorem; it is
    clamped, and the clamp is recorded in `source`.
""")

code(r"""
def show(rule, key):
    # print the descriptor and, crucially, which branch produced it
    s = sectors.load_series(rule, BC, sectors.UNIFORM_N_CAP)
    d = sectors.series_descriptor(rule, BC, key, s["N"], s[key])
    base = "None" if d["base"] is None else f"{d['base']:.6f}"
    print(f"W{rule:<4} {key:<11} cls={d['cls']:<12} base={base:<10} "
          f"exact={str(d['exact']):<6} named={d['named']}")
    print(f"        source : {d['source']}")
    print(f"        alpha  : {d['alpha']}   (from: {d['alpha_source']})")
    print(f"        series : {s[key]}")
    print()

for rule, key in [(204, "n_wcc"),        # 1. analytic
                  (150, "d_max_wcc"),    # 1. analytic (R8 binomial)
                  (156, "d_max_wcc"),    # 1. analytic (R2 room packing)
                  (90,  "n_wcc"),        # 2. irregular
                  (134, "d_max_wcc"),    # 3. volume fraction
                  (0,   "n_wcc"),        # 4. saturated
                  (156, "n_wcc"),        # 7. exact integer recurrence -> phi
                  (157, "d_max_wcc"),    # 7. exact -> sqrt(3)
                  (29,  "n_wcc")]:       # 8. parity split
    show(rule, key)
""")

md(r"""
### 3.1 Why the rate is not taken from M2

Worked demonstration of step 6 on rule 156, whose true $D_{\max}$ base is the
derived $4^{1/5}=1.319508$.
""")

code(r"""
s = sectors.load_series(156, BC, sectors.UNIFORM_N_CAP)
y = s["d_max_wcc"]
f = fits.fit_series(s["N"], y)
fe = fits.fit_pure_exponential(s["N"], y)
print("series      :", y)
print("M2 kappa    -> base %.6f   (biased: alpha*lnN absorbed part of the growth)"
      % math.exp(f["params"]["M2"][2]))
print("two-param   -> base %.6f" % fe["base"])
print("derived     -> base %.6f   (R2 sec.3 room-packing saddle point, 4^(1/5))"
      % 4 ** 0.2)
""")

md(r"""
### 3.2 An exact recurrence, end to end

Rule 156's sector count is Fibonacci, $n_\mathrm{wcc}(N)=F_{N+2}$. The base is
therefore $\varphi$ *exactly* — not 1.618 to three places.
""")

code(r"""
s = sectors.load_series(156, BC, sectors.UNIFORM_N_CAP)
rec = fits.find_integer_recurrence(list(s["n_wcc"]))
print("series    :", s["n_wcc"])
print("recurrence: a_n =", " + ".join(
    f"{c}*a_(n-{i+1})" for i, c in enumerate(rec["coeffs"]) if c) )
print("base      : %.15f" % rec["base"])
print("phi       : %.15f" % ((1 + 5 ** 0.5) / 2))
print("verified on %d terms" % rec["n_verified"])
""")

md(r"""
## 4. Assembling the two planes

The sector plane comes from the stored map (`analytics/sector_map_{bc}.json`,
built by `sectors.build`); the attractor plane from `attractor_deficits` in the
same file, which is produced by `sectors.attractor_point` — the same descriptor
machinery applied to `n_recurrent` and `d_max`.

For a **unitary** rule the two planes give identical coordinates (assertion A2):
$|U|^2$ is doubly stochastic, so there are no transient states and weak, strong
and forward-closure partitions coincide.
""")

code(r"""
d = sectors.load(BC) or sectors.build(BC)
pts = {p["rule"]: p for p in d["points"]}

rows = []
for x in d["attractor_deficits"]:
    p = pts.get(x["rule"])
    if not p or p["n_wcc"]["base"] is None:
        continue
    rows.append(dict(rule=x["rule"], family=p["family"],
                     a=p["n_wcc"]["base"], b=p["d_max_wcc"]["base"],
                     aa=x["a_att"], bb=x["b_att"]))
print(f"{len(rows)} rules with both descriptors")
print("families:", Counter(r["family"] for r in rows))
for r in rows[:5]:
    print(f"   W{r['rule']:<4} {r['family']:<10} "
          f"({r['a']:.4f}, {r['b']:.4f}) -> ({r['aa']:.4f}, {r['bb']:.4f})")
""")

md(r"""
## 5. Why the flow is vertical

Not an empirical pattern — it is forced:

> If every sector contains exactly **one** attractor then
> $n_\mathrm{scc}=n_\mathrm{wcc}$ identically, hence $a_\mathrm{att}=a$, hence
> the move is vertical.

So a rule can only move sideways if some sector holds **more than one**
attractor. Those are the twelve rules of R9 §7.3; ten of them show it here.
Rules 235 and 249 are multi-attractor but their ratio is a constant 2, which
cancels out of a growth base and leaves both at 1.
""")

code(r"""
moved = [r for r in rows if abs(r["a"] - r["aa"]) > 1e-9 or abs(r["b"] - r["bb"]) > 1e-9]
off   = [r for r in moved if abs(r["a"] - r["aa"]) > 1e-6]      # the exceptions
print(f"rules             : {len(rows)}")
print(f"moved             : {len(moved)}")
print(f"  purely vertical : {sum(1 for r in moved if abs(r['a']-r['aa'])<1e-9)}")
print(f"  b decreases     : {sum(1 for r in moved if r['bb'] < r['b'] - 1e-9)}")
print(f"  off-axis        : {len(off)}  -> {sorted(r['rule'] for r in off)}")
""")

md(r"""
## 6. The figure

Self-contained and editable. Design constraints, in case you re-lay it out:

* **one shared $x$-axis**, drawn once at the bottom — $x$ means the same thing
  in both planes (the growth base of a *count*);
* **two separate $y$-axes**, marked by break slashes on the left spine;
* **the gap is empty** — no spine, no ticks, no panel titles — so an arrow never
  crosses a line or a label;
* the 220 forced-vertical moves are thin grey, the ten exceptions are drawn
  thick and coloured.

Knobs worth turning: `FIGSIZE`, `HSPACE` (gap height), `YLIM`, and
`EXC_COLOUR`.
""")

code(r"""
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch

FAM     = {"unitary": "#1f4e9c", "mixed": "#c62828", "classical": "#7aa87a"}
FAMLAB  = {"unitary": "unitary (V only), 16", "mixed": "V + reset, 160",
           "classical": "V-free, 80"}
ORDER      = ("classical", "mixed", "unitary")     # draw order = z order
EXC_COLOUR = "#d95f0e"
ARROW      = "#8a8f98"
FIGSIZE    = (5.0, 6.4)
HSPACE     = 0.30
XLIM = YLIM = (0.93, 2.11)

def cells(rows, xk, yk):
    c = Counter()
    for r in rows:
        c[(round(r[xk], 3), round(r[yk], 3), r["family"])] += 1
    return c

def scatter(ax, c, base=12.0, per=10.0):
    for fam in ORDER:
        for (x, y, f), n in c.items():
            if f == fam:
                ax.scatter(x, y, s=base + per * np.sqrt(n), facecolor=FAM[fam],
                           edgecolor="white", lw=0.7, alpha=0.9,
                           zorder=4 + ORDER.index(fam))

fig, (axU, axL) = plt.subplots(2, 1, figsize=FIGSIZE, sharex=True,
                               gridspec_kw={"hspace": HSPACE})

for ax in (axU, axL):
    ax.grid(True, color="#ececea", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    a = np.linspace(*XLIM, 400)
    ax.plot(a, 2.0 / a, color="#555555", lw=0.9, ls=":", zorder=2)   # ab = 2

axU.spines["bottom"].set_visible(False)      # keep the gap completely empty
axU.tick_params(axis="x", which="both", length=0, labelbottom=False)
axL.spines["top"].set_visible(False)

scatter(axU, cells(rows, "a", "b"))
scatter(axL, cells(rows, "aa", "bb"))

offset = {id(r) for r in off}
for r in moved:
    is_exc = id(r) in offset
    fig.add_artist(ConnectionPatch(
        xyA=(r["a"], r["b"]),   coordsA=axU.transData,
        xyB=(r["aa"], r["bb"]), coordsB=axL.transData,
        arrowstyle="-|>", mutation_scale=8 if is_exc else 5,
        lw=1.4 if is_exc else 0.5,
        color=EXC_COLOUR if is_exc else ARROW,
        alpha=0.95 if is_exc else 0.30,
        zorder=9 if is_exc else 1, shrinkA=2.5, shrinkB=2.5))

# break slashes on the left spine, marking the split y-axis
x0 = axU.get_position().x0
for yy in (axU.get_position().y0, axL.get_position().y1):
    fig.lines.append(Line2D([x0 - 0.012, x0 + 0.012], [yy - 0.008, yy + 0.008],
                            transform=fig.transFigure, color="#555", lw=0.9,
                            clip_on=False, zorder=20))

axU.set_ylabel(r"$b$   (sectors)", fontsize=9)
axL.set_ylabel(r"$b_{\rm att}$   (attractors)", fontsize=9)
axL.set_xlabel(r"base of the count:  $a$ above,  $a_{\rm att}$ below", fontsize=9)
for ax in (axU, axL):
    ax.tick_params(labelsize=7.5, length=2)

handles = [Line2D([], [], marker="o", ls="", markersize=6, markerfacecolor=FAM[k],
                  markeredgecolor="white", label=FAMLAB[k])
           for k in ("unitary", "mixed", "classical")]
handles.append(Line2D([], [], color=EXC_COLOUR, lw=1.5,
                      label=f"multi-attractor rules ({len(off)})"))
lg = axU.legend(handles=handles, fontsize=6.6, loc="upper right",
                framealpha=0.94, borderpad=0.4, labelspacing=0.35)
lg.get_frame().set_linewidth(0.5)
plt.show()
""")

md(r"""
## 7. Export
""")

code(r"""
OUT = os.path.join(ROOT, "figures", "fig_wcc_scc_stack_notebook.pdf")
fig.savefig(OUT, bbox_inches="tight")
fig.savefig(OUT.replace(".pdf", ".png"), dpi=220, bbox_inches="tight")
print("wrote", OUT)
""")

md(r"""
## Caveats

* **`obc0` only.** The `pbc` sweep is held pending caveats (see
  `NEXT_STEPS.md`), and under `pbc` at odd $N$ the brick-wall layers stop
  commuting, so those fits would need even $N$ only.
* **The attractor plane is monitored-only.** Terminal SCCs describe the measured
  chain; the unmonitored channel can keep weight on states the graph calls
  transient. The sector plane is the one exact for both experiments.
* **Read `source` before quoting a base.** Roughly half are exact recurrences or
  derivations; the rest are fits, and the two should not be mixed in a table
  without saying which is which.
* **Bases are fitted on $N \le 16$** so that all rules share a window.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python",
                   "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
}
import os
out = os.path.join(os.environ["NB_DIR"], "wcc_scc_stack_figure.ipynb")
with open(out, "w") as f:
    nbf.write(nb, f)
print("wrote", out, "with", len(cells), "cells")
