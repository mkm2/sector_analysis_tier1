# Tier-1 — follow-up status

Deep dataset: dissipative N<=17, unitary N<=21 (Tier 1a/1c); Tier 1d pair graph
diss N<=9 pbc. **C150 (rule 150) obc0: closed form for all N, numerical
frontier N<=32.** Reports R1/R2/R5/R6/R7/R8 regenerated; full suite 403 tests pass.
R8 also cross-checked against the independent Julia HSF eigendata (rule 6 = 150).

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

10. **HSF cross-check — DONE (2026-07-27, R8 §8).** HSF rule 6 = Wolfram 150
   (same operator: same symbol table, obc0 boundaries, even-then-odd layers;
   only a bit-reversal of the basis and theta = -angle(lambda) differ). All
   claims reproduce on the independent Julia eigendata for N=8..14: sector sizes
   C(N+1,w), dim Fix = C(ceil(N/2), w/2), #distinct = 3^m INCLUDING the
   out-of-sample N=14 = 2187, the universal-set property, and <r~>.
   Caveat raised for their entropy column: 43%->74% of stored eigenstates lie
   in a within-sector degenerate eigenspace (multiplicity up to 35), where the
   per-eigenstate entanglement entropy is basis dependent (spread up to 0.27
   nats; exactly 0 on simple eigenvalues).

13. **Fig. 3a area-law vs volume-law — RESOLVED (2026-07-27, R8 §9).** The
   three initial states are in three very differently sized sectors, not one:
   single excitation w=2 (|S_w| = C(N+1,2), POLYNOMIAL), Neel w=N (|S_w| = N+1,
   LINEAR -- it is a ONE-HOLE state in wall language, hard-core jammed), pair
   w~N/2 (exponential).  The premise "sectors are binomially large" only holds
   near w=(N+1)/2.  Exact ceiling is NOT ln|S_w| but the bipartite Schmidt rank
   min(|S_w|, sum_{k=max(0,w-1-(N-c))}^{min(w,c)} C(c,k), same for the other
   side) -- the lower limit matters at high filling and is exactly what makes
   Neel constrained.  Verified == the compressed Schmidt dimension for all
   N=6..14, all w.  Measured: single 1.00->1.24 and Neel 0.53->0.58 FLAT to
   N=38 (area law); pair grows, ceiling = volume-law value exactly.  The dome:
   S_plateau peaks at nu=1/2, realised fraction of the ceiling roughly flat
   (0.21-0.52).  Sector IS enough for the type and (for product states) the
   value to +-3%; NOT enough over all states -- dark states of the same shell
   have S(t) constant to 1e-15 at a HIGHER value than the product plateau.

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
10. **The partner-shell intertwiner.** For odd N the shells w and N+1-w have
   identical spectra (1.2e-14), but the bond particle-hole map (x -> x XOR
   alternating mask) does NOT commute with U (checked N=3..9). Unidentified.
11. **C150 on the ring.** Kinematics solved; the quantum layer is obc0 only.
    The 2-to-1 wall map gives the bond frame a Z2 gauge redundancy.
12. **N=33 at the reduction frontier** needs a uint64 label array (68.7 GB), or
    a two-pass compact relabelling to stay inside 32 GB.
