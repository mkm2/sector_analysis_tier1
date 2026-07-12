# PLAN — implementation, checkpointing, task breakdown

This is the design of record. It fixes the module API, the on-disk formats, and
the checkpoint/reconstruction strategy **before** any heavy simulation, so that
overnight runs are resumable and results reconstructable without re-simulating.

## 0. Principles

- **Exactness in the graph stage.** No floats, no tolerances anywhere in
  Tier 1a/Tier 2 symbolic work. Amplitudes are `(a + b*sqrt2)/2^m` with
  arbitrary-precision Python ints and a per-branch shared `m` (context Tier 1
  §2). Exact zero means `a == 0 and b == 0`.
- **Unitary first, then dissipative.** Get the unitary pipeline fully validated
  against the §8 regression table before touching D/E channels.
- **Cross-check everything.** Every quantitative claim is checked two ways:
  (i) an independent method (e.g. brute-force dense unitary at small N, or the
  Julia `HSF` connected-components oracle), and (ii) the analytic prediction
  (Tier 2) vs the measured value (Tier 1).
- **Checkpoint at the (rule, N, bc) grain.** Each unit is independently
  recomputable and cached; a night that dies mid-sweep loses at most one unit.

## 1. Module API (Tier 1 — `qca_fragmentation`)

- `core/ring.py` — exact `Z[1/sqrt2]` element `(a, b, m)`; branch-dict ops:
  conditional-H layer update, addition, exact zero test, renorm, norm check.
- `core/rules.py` — Wolfram number -> channel tuple `(r00,r01,r10,r11)` in
  `{I,V,D,E}`; reflection representative; V-free spin-flip test. Unit-tested on
  the §1.2 verification cases (150, 156, 22, 204, 51, 201, 0).
- `core/cycle.py` — branch propagation and `succ(x)` (context §3), both bc.
  Invariant tests: unitary -> one branch, probabilities sum to 1 (Fractions);
  dissipative -> branch norms sum to 1; unraveling-independence.
- `graph/scc.py` — iterative Tarjan on the implicit directed graph, condensation
  DAG, terminal SCCs (recurrent classes), basins, transient depth, ergodic
  early-exit (`f_erg` default 0.5). **Directed**, never symmetrized.
- `quantum/peripheral.py` — Tier 1b restricted superoperator per recurrent
  class; peripheral spectrum; attractor classification. (Needs scipy fix.)
- `scaling/fits.py`, `scaling/figure.py` — Tier 1c BIC model selection for
  `ln y = c (+ alpha ln N)(+ kappa N)`; growth-class labels; final scatter.
- `run_rule.py` — CLI `--rule R --N n --bc {pbc,obc0} --tiers 1a[,1b]`.
- `sweep.py` — orchestrates the 136 reflection representatives, checkpointed.

## 2. Module API (Tier 2 — `qca_analytics`)

Imports the Tier 1 exact engine (`core/ring.py`, `core/cycle.py`).

- `walls.py` — A.2 wall detection at both offsets, support vs cancellation
  walls, grammar dedup.
- `intervals.py` — A.3 interval spectroscopy; class-size multisets `{d_j(l)}`;
  growth-law fit.
- `partition.py` — A.4 `g_s(z)`, smallest positive root (mpmath 50 digits),
  `f(s)`, `a = e^{f(0)}`, `b = max_{l,j} d_j(l)^{1/(l+|w|)}`, sum rule
  `f(1) = ln 2`, Legendre `sigma(kappa)`.
- `charges.py` — B.1 edge-constraint null space (comp + bond basis),
  B.2 strong/weak classification, gate-local certificate.
- `commutant.py` — B.3 commutant algebra backstop.
- `counting.py` — B.4 fugacity transfer matrices; charge-resolved sector dims.
- `sweep.py` — Module C: per-rule record + predicted-vs-measured comparison
  table.

## 3. On-disk formats

### Tier 1 results — `results/{rule}_{bc}.jsonl`
One JSON object per `(rule, N, bc)` line (context Tier 1 §7 schema):
`{rule, bc, N, n_scc, n_recurrent, sizes_recurrent, sizes_basins,
  shared_basin_size, transient_depth, ergodic_flag, attractor_types,
  d_max_quantum, runtime, mem, engine_version}`.
All sizes exact ints. Append-only; a `(rule,N,bc)` already present is skipped.

### Tier 2 analytics — `analytics/{rule}.json`
Module C schema: `{rule, tuple, walls, interval_growth_law, f_curve (s grid),
  a_pred, b_pred, sum_rule_ok, charges, commutant_dims, classification, notes}`.

### Comparison — `analytics/comparison.csv`
`rule, a_pred, a_meas, b_pred, b_meas, discrepancy_flag, ...` — the backbone of
the paper's final figure.

### Curves — `figures/*.csv` then `figures/*.png/pdf`.

## 4. Checkpoint / reconstruction strategy

Goal: **reconstruct all conclusions without large re-simulation.**

1. **Content-addressed unit cache.** Each expensive unit keyed by
   `(engine_version, rule, N, bc)`. Result stored as one JSONL line in
   `results/` AND (for the heavy directed-graph objects when worth it) a
   compressed sidecar `checkpoints/{rule}_{bc}_N{N}.npz|json.gz` holding the SCC
   labelling / condensation so Tier 1b can rerun without re-exploring the graph.
2. **`engine_version`** string bumped whenever `core/*` semantics change;
   mismatched cache entries are ignored (not silently reused).
3. **Idempotent sweep.** `sweep.py` reads existing `results/*.jsonl`, computes
   only missing `(rule,N,bc)` units, appends. Safe to kill/restart nightly.
4. **Manifest.** `checkpoints/manifest.jsonl` logs every completed unit with a
   timestamp, wall-time, and a hash of the code — the audit trail for the
   reports and for detecting stale results after an engine bump.
5. **Determinism.** No RNG in Tier 1a / Tier 2 symbolic stages, so a rerun of a
   unit reproduces byte-identical results (used as a self-check on the cache).

## 5. Reporting

After each task, a LaTeX report under `reports/tex/` compiled to
`reports/pdf/`. Each report states: what was computed, the regression/cross-
check results (pass/fail with numbers), the new quantitative findings, and any
open discrepancies flagged for a manual pass. Reports cite the exact
`engine_version` and the `checkpoints/manifest.jsonl` entries they rest on.

## 6. Task breakdown (execution order)

- **T1** Core engine: `ring.py`, `rules.py`, `cycle.py` + their unit tests
  (context §1.2, §2, §3 invariants). Gate: all pass.
- **T2** Tier 1a graph: `scc.py`; reproduce §8 table (204, 51, 150/obc0, 54,
  156/pbc Lucas, 22/pbc, 201). Cross-check vs `HSF`. **Report R1.**
- **T3** Tier 1a unitary sweep (136 reps, feasible N), checkpointed;
  scaling fits `scaling/fits.py`. **Report R2.**
- **T4** Tier 2 unitary: walls, intervals, partition; reproduce 156 targets
  (`a=phi`, `b=4^{1/5}`, `f(1)=ln2`), 201 Fibonacci. Predicted-vs-measured
  table. **Report R3.**
- **T5** Charges/commutant: reproduce rule 150 U(1) bond charge, binomials.
  **Report R4.**
- **T6** Extend to dissipative: cycle-engine D/E branches, recurrent classes,
  strong/weak charges, wall transparency (28/29, 22 multistable). **Report R5.**
- **T7** Full synthesis sweep + final figure. **Report R6.**

## 7. Cross-check oracles

- **Dense brute force** (small N): build the full unitary / channel, extract
  sectors independently, compare to the exact-engine SCCs.
- **Julia `HSF`** (`../../HSF`): `extract_krylov_sectors` via connected
  components on the symmetrized adjacency. Valid as an oracle for *unitary*
  rules (doubly stochastic => all states recurrent, symmetrization harmless);
  NOT a valid oracle for dissipative directedness — used for unitary only.
- **Analytic (Tier 2) vs numeric (Tier 1)**: the primary scientific cross-check.
