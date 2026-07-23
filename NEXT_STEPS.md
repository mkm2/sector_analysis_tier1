# Tier-1 — manual follow-ups after the deep-dataset run

The final regeneration pass ran on the completed sweep (dissipative N≤17,
unitary N≤21; see below). All artefacts and all four reports (R1/R2/R5/R6) were
regenerated and the full pytest suite (142 tests) passes. What is left needs
human judgement:

1. **R5 prose numbers are still the N≤13 snapshot.** The inline counts in
   "Tier 1c: dissipative growth laws" and "Exact algebraic growth bases" still
   read 58/81 exact bases and 43/16 parity splits. The regenerated dissipative
   summary now reports **123 obc0 exact-base series** and period-splits {2:14,
   3:14}; growth classes obc0 #att {const 162, exp 56, poly 22}, Dmax {const
   149, exp 79, poly 10, irregular 2}. The *tables* and *figures* are current;
   only the prose is stale. (Rerun `scaling.dissipative` prints the live counts.)
2. **`tab_diss_scaling` overruns the right margin** (the attractor-class column).
   Narrow/drop a column or use a smaller font / landscape.
3. **Sweep ceiling.** Dissipative W19/W55 cap at N=16 (OOM at N=17 on 32 GB, both
   bc — genuine limit, not a bug). Unitary reached N=21 for 13/16 units; the
   binomial rules 105/150 stopped at N=20 (pbc) — their bases are already exact
   (base 2, α=−½), so nothing is lost. Consider marking these so a future sweep
   doesn't retry them.
4. **Open science question:** the coherent-attractor count is non-monotonic in N
   (84→60→72 at N=4,6,8); looks like a parity effect but is not proved. Also
   push the coherence census beyond N=8 (limited by the 2^N dense ρ).
5. `figures/fig_transient_depth_*` referenced by R5 — verify it exists / is
   regenerated if that section is kept.

The listener `~/qca_tier1_final.sh` and the overnight driver were stopped when
this pass was fired manually; both are no longer running.
