# Krylov Sector Analysis of Quantum Cellular Automata

Classification of the 256 quantum elementary cellular automata (QCA) by the
**fragmentation structure** of their one-cycle evolution channel, and the
**rule-level analytic prediction** of that structure independent of system size.

This repository is the working area for the *Sector Analysis* project. The
authoritative specification lives in the two context files:

- `Tier1/context_tier1_graph_numerics.md` — exact graph numerics (Tier 1):
  directed transition graph on `2^N` basis states, SCCs / Krylov sectors,
  recurrent classes, per-class quantum structure, scaling fits.
- `Tier2/context_tier2_analytic_machinery.md` — rule-level analytic machinery
  (Tier 2): wall grammars, sector partition functions / free energy,
  conserved-charge detection, commutant algebra, rule-space sweep.

Conceptual background: `QCA_Circuits.pdf` (this folder).

## Status

**Initialized — analysis not yet started.** The directory tree, plan, and
tooling are in place. Simulations will begin on instruction (see `PLAN.md`),
running overnight (see `nighttime.txt`).

## Plan of record

1. **Tier 1a (unitary first)** — exact directed-graph SCC / Krylov-sector
   analysis for the unitary rules, with the ground-truth regression table
   (context Tier 1 §8) as the correctness gate. Cross-check against the
   existing Julia `HSF` code (float sparse-matrix oracle, `../../HSF`).
2. **Tier 2 (unitary)** — walls, interval spectroscopy, partition function
   `f(s)`, predicted `(a, b)`; validate against Tier 1a and the completeness
   certificate `f(1) = ln 2`.
3. **Report** — after each task, a LaTeX report in `reports/`.
4. **Extend to non-unitary (dissipative) rules** — recurrent classes,
   attractors, strong/weak charges, dissipative wall transparency.

See `PLAN.md` for the module API, checkpoint/reconstruction strategy, and the
task breakdown.

## Layout

```
code/
  qca_fragmentation/   Tier 1: exact graph numerics (Python, exact Z[1/sqrt2])
    core/              ring.py, rules.py, cycle.py
    graph/             scc.py (iterative Tarjan, condensation, basins)
    quantum/           peripheral.py (Tier 1b restricted superoperators)
    scaling/           fits.py, figure.py (Tier 1c)
    tests/             pytest regressions (context Tier 1 §8)
  qca_analytics/       Tier 2: walls, intervals, partition, charges, commutant
    tests/             pytest regressions (context Tier 2 validation items)
results/               Tier 1 output: {rule}_{bc}.jsonl  (checkpointed)
analytics/             Tier 2 output: {rule}.json, comparison CSVs
checkpoints/           resumable simulation state (see PLAN.md)
reports/tex, reports/pdf   LaTeX reports per task
figures/               plots (f(s), sigma(kappa), final scatter)
```

`../../HSF` (Julia) is a separate, pre-existing repository used read-only as a
cross-check oracle; it is **not** vendored here.

## Environment

- Python 3.10 (context asks 3.11+; the exact engine uses only pure-Python ints
  + `sympy` + `mpmath`, all present and 3.10-compatible).
- `sympy` 1.9, `mpmath` 1.2.1.
- ⚠️ `scipy` 1.8.0 / `numpy` 2.2.6 are ABI-mismatched; `scipy.sparse.linalg`
  (needed only in Tier 1b) must be fixed before Tier 1b runs. Tier 1a and
  Tier 2 are unaffected (pure Python + sympy/mpmath).
- Julia 1.12.2 (for the HSF oracle).

See `requirements.txt`.
