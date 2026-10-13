# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research codebase that classifies the 256 quantum elementary cellular automata (QCA, Wolfram-numbered rules
built from brickwork conditional-Hadamard gates) by the **fragmentation structure** (Krylov sectors / SCCs) of their
one-cycle evolution channel, and predicts that structure analytically at the rule level. Output is scientific:
JSONL sweep records, figures, and LaTeX reports (R1 … R31).

Authoritative specs (read before touching the engine or the analytics):
- `Tier1/context_tier1_graph_numerics.md`: exact graph numerics, the §7 result schema, and the §8 regression table (the correctness gate).
- `Tier2/context_tier2_analytic_machinery.md`: walls, partition function `f(s)`, charges, commutant.
- `PLAN.md`: module API, on-disk formats, checkpoint/reconstruction strategy.
- `NEXT_STEPS.md`: running log of every report task, its findings, corrections, and items **held by the user** (don't pick up held items unprompted). It stops at R22; later reports (R23 to R31) are recorded only in git commit messages and `reports/tex/`.

## Commands

The Python interpreter is `~/pyenvs/main/bin/python` (Python 3.10). Run everything **from `code/`**, because
`code/conftest.py` puts `code/` on `sys.path` and the packages are invoked as modules.

```bash
cd code
~/pyenvs/main/bin/python -m pytest -q                                      # full suite
~/pyenvs/main/bin/python -m pytest qca_fragmentation/tests/test_scc.py -q  # one file
~/pyenvs/main/bin/python -m pytest qca_fragmentation/tests/test_scc.py -k <name> -q

# Tier 1a single rule (idempotent: cached units are skipped unless --force)
~/pyenvs/main/bin/python -m qca_fragmentation.run_rule --rule 150 --N 8-16 --bc obc0
# Checkpointed sweep (resumable; see --help for --which/--rules/--n-min/--n-max)
~/pyenvs/main/bin/python -m qca_fragmentation.sweep --bc obc0 --n-max 20
# Extend the unitary rules via the flip reduction (unitary rules ONLY)
~/pyenvs/main/bin/python -m qca_fragmentation.extend_unitary --n-max 24
# Regenerate summaries/figures/report tables
~/pyenvs/main/bin/python -m qca_fragmentation.scaling.summary
~/pyenvs/main/bin/python -m qca_fragmentation.scaling.figure
~/pyenvs/main/bin/python -m qca_fragmentation.quantum.dissipative_report
# Sector graph renderings (gitignored, ~10 min)
~/pyenvs/main/bin/python -m qca_fragmentation.viz.dot_sectors --N 11 --bc obc0
```

Heavy sweeps can take hours and gigabytes (flip graph memory is 4·2^N bytes). Ask before launching anything large.

Known environment caveat (README): scipy 1.8 / numpy 2.2 were ABI-mismatched at one point; only the numeric
Tier 1b/1c layers need scipy. The exact stages use only Python ints + sympy/mpmath.

## Architecture

Two packages under `code/`:

- **`qca_fragmentation/` (Tier 1, exact numerics)**
  - `core/`: `ring.py` (exact ℤ[1/√2] amplitudes `(a + b√2)/2^m`, no floats), `rules.py` (Wolfram number → channel tuple `(r00,r01,r10,r11)` in `{I,V,D,E}`, reflection/spin-flip), `cycle.py` (brickwork `succ(x)`; unitary = one branch, dissipative D/E = Kraus branches; `pbc` / `obc0` boundaries).
  - `graph/`: `scc.py` (iterative directed Tarjan, condensation, terminal SCCs = recurrent classes, basins, transient depth), `wcc.py`, `flip_graph.py` (flip-reduction, a theorem for unitary rules only), `pair_graph.py` (Tier 1d).
  - `quantum/`: Tier 1b restricted superoperators / peripheral spectrum / attractor classification, rule-150 studies, `hsf_compare.py`.
  - `scaling/`: BIC growth fits and one module per report's figures/tables (`paper_figures.py`, `stripes.py`, `wall_charges.py`, …). Report tables `reports/tex/tab_*.tex` are **generated** by these modules, so a report cannot quote an unchecked number. Change the generator, not the `.tex` table.
  - `permutation/`: the V = X (permutation-circuit) variant of all 256 rules (R10/R11).
  - `viz/`: spy plots, towers, journal/nested figures, DOT sector graphs.
  - Sweep drivers: `run_rule.py`, `sweep.py`, `deep_sweep.py`, `wcc_sweep.py`, `pair_sweep.py`, `basin_recompute.py`, `extend_unitary.py`, `c150.py`.
  - `results_io.py`: append-only I/O, the `_SIZES_CAP` truncation of size lists (with a complete `size_hist`), and the `checkpoints/manifest.jsonl` audit log.
- **`qca_analytics/` (Tier 2, rule-level analytics)**: walls, intervals, partition (`f(s)`, sum rule `f(1)=ln 2`), charges, commutant, counting, sweep. It imports the Tier 1 core.

Data stores (all committed; they are the reconstruction cache):
- `results/{rule}_{bc}.jsonl`: Tier 1a, one record per (rule, N, bc), deepest N.
- `results_tier1e/`: Tier 1e WCC/basin sweep, all 256 rules, obc0. `results_tier1d/` (pair graph), `results_xgate/` (V=X), `results_basins/`, `analytics/` (Tier 2).
- Records are keyed by `ENGINE_VERSION` (in `qca_fragmentation/__init__.py`). Bump it whenever `core/*` semantics change; mismatched entries are ignored, not reused.

Notebooks: `notebooks/paper_figures.ipynb` and `wcc_scc_stack_figure.ipynb` are **generated** by
`notebooks/_build_*_notebook.py` (nbformat). Rebuilding overwrites hand edits, so durable changes go in the builder.
Exact laws drawn in the paper figures live in `scaling/paper_figures.py` (`PF.verify`), not in the notebook.

Reports: `reports/tex/R<n>_*.tex` compile to `reports/pdf/`; each states its checks, findings, and corrections to earlier reports.

## Research conventions (from the project history)

- **Exactness**: no floats or tolerances in the Tier 1a/Tier 2 symbolic stages. The graph is **directed** and never symmetrized for dissipative rules (the Julia `../../HSF` oracle is valid for unitary rules only, read-only, not vendored).
- **No symmetry is a source of data** (user instruction): every rule gets its own computed records. Reflection/spin-flip identities are results to check, never substitutes.
- Default boundary for current work is `obc0` (R9 onward excised the pbc material).
- Cross-check every quantitative claim two ways (independent algorithm + analytic prediction), and add tests that pin it.
- When a result turns out to be wrong, the fix is a visible correction (commit, report revision, `NEXT_STEPS.md` entry), not a silent edit.
- Commit style: `R<n> [rev.k]: <one-line finding in plain words>`, with a body explaining what changed and why.

## Layout and git workflow

- **One checkout, no worktrees** (user-directed; re-unified 2026-10-13). Work directly in this directory on `main`, and push to `origin main` (`git@github.com:mkm2/sector_analysis_tier1.git`). Do not create a git worktree under `.claude/worktrees/`, because past worktrees left `main` and the root checkout stranded on stale commits.
- Untracked files are the user's own manual work (e.g. `notebooks/final/*_copy.ipynb`, `leo.md`, `Notes_on_QCA.pdf`, `Tier1/task_*.md`, `figures/fig_wcc_scc*`). Never delete, overwrite or `git clean` them.
- `analytics/tl_*.py` are standalone scripts that put `../code` on `sys.path` relative to their own location. Keep paths repo-relative, never absolute.

## Figure standards (user rule, 2026-10-14)

Every new figure is drawn at its final print size, so font sizes are true sizes. The binding numbers come from the
thesis figure project (`~/Dokumente/Projects/Figure Generation`, `thesisfig/config.py` and `style.theme()`):

- **Width: thesis text width 422.52348 pt (5.85 in) by default.** Half width for two side-by-side subfigures is 0.49×.
  The maximum height is 508.4 pt. Never stretch a figure to fit, and never use `bbox_inches="tight"`.
- **Paper mode, only on the user's request:** single column, 246 pt wide. Ask for the paper's font sizes the first
  time; don't assume the thesis ones.
- **Text** goes through LaTeX in Latin Modern (`text.usetex`). Sizes: 10 pt for annotations and ticks, 10.95 pt for axis
  and panel labels. Text colour is `#222222` and never the series colour.
- **Colours:** the Okabe-Ito categorical palette, in this order: `#0072B2 #D55E00 #009E73 #E69F00 #CC79A7 #56B4E9`,
  with grey `#7F7F7F` for neutral items. Sector k gets `palette[k]`, and frozen/trivial sectors get grey. Fills are
  1 − 0.24·(1 − c) and dark marks 0.78·c.
- **Strokes:** data lines 1.0 pt, axes 0.6 pt, markers 3.5 pt, dense-scatter dots about 1.05 pt. Raster layers are at
  ≥ 600 dpi; PDF is the deliverable.
- **Hand-off:** figures for the thesis go to `Figure Generation/data/<name>/` as `data.npz` and `meta.yaml`, plus
  `render.py` for custom drawing (see that project's `data/HANDOFF.md`). Never leave files such as `__pycache__` there.
