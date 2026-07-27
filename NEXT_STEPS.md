# Tier-1 — follow-up status

Deep dataset: dissipative N<=17, unitary N<=21 (Tier 1a/1c); Tier 1d pair graph
diss N<=9 pbc. **C150 (rule 150) obc0: closed form for all N, numerical
frontier N<=32.** Reports R1/R2/R5/R6/R7/R8 regenerated; full suite 332 tests
pass.

## Done
1. **R5 prose refreshed to the deep dataset.** Exact algebraic bases now
   94 (pbc) / 123 (obc0); growth-class tallies updated (pbc #att
   146/82/8/4, D_max 109/66/45/20; obc0 #att 162/56/22/0). The parity paragraph
   is rewritten as the calibrated period split (p in {2,3,4}, BIC margin >=10 or
   exact recurrence per class, null-calibrated: 3% vs 21%/33% for the naive
   rules, 0/4000 for the recurrence route): 33 pbc parity + 14 period-3 splits
   (14/14 obc0), of which 12 pbc / 10 obc0 period-3 splits are exact-recurrence
   certified (not fitting artifacts). Algebraic-bases table updated with the new
   named constants (2cos(pi/8), sqrt((3+sqrt17)/2), 2^{1/3}, sqrt2, ...).
2. **`tab_diss_scaling` margin fixed.** Smaller font (footnotesize + tighter
   colsep), the redundant constant *names* dropped from the growth cells (base
   number kept; names live in the algebraic-bases table), and wrapped in
   `\resizebox{\textwidth}` as a guaranteed fit. (The remaining overfull boxes
   in R5 are pre-existing unbreakable `\texttt{python -m ...}` command lines and
   the 12-rule wall-core set, unrelated to this table.)
5. **`fig_transient_depth_{pbc,obc0}` verified present** (referenced by R5).
6. **R8 — C150 obc0 (2026-07-26).** Flip reduction theorem for all 16 unitary
   rules; wall/bond bijection -> hard-core hopping -> closed form
   |S_w| = C(N+1,w) (w even), #sectors = floor((N+1)/2)+1, D_max base exactly 2
   with an N^{-1/2} prefactor. Frontier: streamed engine N<=22, flip graph
   N<=32 (4.29e9 states, 21 GB, 815 s). Quantum layer:
   dim Fix(U_w) = C(ceil(N/2), w/2), dim Fix(U) = 2^{ceil(N/2)}, inner sectors
   are never Krylov spaces, monitored uniform vs unmonitored d_eff 6.6-50%,
   and C150 = free Hadamard wall walk + two diagonal sign defects (not free).
   R2 corrected on two points (even-weight vs "lower half" of Pascal's row;
   D_max parity at N = 1 mod 4).

## Held (per user, 2026-07-24)
3. **Mark W19/W55 as capped at N=16** so a future sweep does not retry N=17.
   HELD: user will launch a future sweep on a more powerful machine.
4. **Coherent-attractor count non-monotonicity + census beyond N=8.** HELD.
   (Partial progress in Tier 1d: coherence *support* persists at every N incl.
   odd, ~170/240; exact within-sector count 104/76/58 at N=4/5/6. The dense
   even-N census upturn at N=8 is a distinct quantity out of exact reach.)

7. **Level statistics — DONE (2026-07-27, R8 §7).** C150 is interacting but not
   chaotic. Decisive: #distinct eigenvalues of U on 2^N states = 3^m (N even) /
   (3^m + (-1)^m)/2 (N odd), m = ceil(N/2) = #even sites; verified N=4..13 and
   out of sample at N=14. Distinct fraction decays as 0.866^N. Spacing
   statistics agree (dilute shells Poisson as few-body shells must; half-filled
   drift down with N). beta=2 (GUE/CUE) excluded on symmetry grounds since U is
   real orthogonal. Structural source: U = L_B L_A is a product of two real
   symmetric involutions, which also explains Fix(U) = Fix(L_A) cap Fix(L_B).
   This corrected an over-reaching forward-looking sentence in R8 §5.

## Open from R8 (C150)
8. **Derive dim Fix(U_w) = C(ceil(N/2), w/2)** and the w-resolved version of the
   3^m distinct-eigenvalue law. The full-space law is exact and out-of-sample
   tested but not derived; Fix(U) = Fix(L_A) cap Fix(L_B) is now explained by the
   two-involution structure modulo the (numerically empty) (-1,-1) sector.
9. **Identify the commutant.** The 3^m law forces an exponentially large
   commutant (sum of m_lambda^2). Its generators would prove the law and turn
   the block-resolved spacing statistics from a lower bound into an exact result.
   The exact multiplicities (2,3,4,6,10,15) are not explained by reflection,
   which is abelian.
9b. **The partner-shell intertwiner.** For odd N the shells w and N+1-w have
   identical spectra (1.2e-14), but the bond particle-hole map (x -> x XOR
   alternating mask) does NOT commute with U (checked N=3..9). Unidentified.
10. **C150 on the ring.** Kinematics solved; the quantum layer is obc0 only.
    The 2-to-1 wall map gives the bond frame a Z2 gauge redundancy.
11. **N=33 at the reduction frontier** needs a uint64 label array (68.7 GB), or
    a two-pass compact relabelling to stay inside 32 GB.
