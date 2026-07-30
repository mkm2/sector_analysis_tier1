# Tier-1 — follow-up status

Deep dataset: dissipative N<=17, unitary N<=21 (Tier 1a/1c); Tier 1d pair graph
diss N<=9 pbc. **C150 (rule 150) obc0: closed form for all N, numerical
frontier N<=32.** Reports R1/R2/R5/R6/R7/R8/R9 regenerated; full suite 552 tests pass.
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

14. **Full computational-basis census — DONE (2026-07-29, R8 §10).** N=11 obc0,
   all 2^11 = 2048 basis states to t=4000, clustered by sector.
   **Global average S = 1.6612 nats** (median 1.6710, sd 0.2292, range
   [0, 1.9681]); volume law 5 ln2 = 3.4657, so 48% of it. Per sector
   (w: mean +- sd): 0 -> 0, 2 -> 0.9802 +- 0.0368, 4 -> 1.6057 +- 0.0768,
   6 -> 1.8216 +- 0.0930, and w <-> 12-w identical to every digit (partner-shell
   symmetry now seen in entanglement, not just spectra -> constrains the
   unknown intertwiner, R8 open item 4). w=0 and w=12 are frozen singletons:
   at odd N the Neel state is exactly stationary, S(t) == 0.
   - **eta^2 = 0.870**: the sector explains 87% of the variance of the
     saturation entropy over the whole Hilbert space. The residual 13% is real,
     not noise (split-half correlation 0.9986/0.9988, noise 0.0024/0.0028
     against spreads 0.0768/0.0930 -- 30x above the floor).
   - **No random state predicts it.** Haar on 2^N: 3.2098 (Page 3.2160), off by
     1.93x. Haar within the shell, size-weighted: 2.8248, still off by 70%.
     Random phases on the exact spectrum of the exact initial state (diagonal
     ensemble / maximal dephasing): 1.454/2.640/3.055 for w=2/4/6 vs measured
     0.980/1.606/1.822 -- barely better than plain Haar. Cause: the w=6 shell
     has 182 distinct positive phases but they are integer combinations of only
     m=6 fundamentals, so the orbit fills a 6-torus, not a 182-torus. Control:
     coherent phases alpha=theta*t through the same code reproduce the measured
     time average to 4 decimals (1.3901 vs 1.3901, N=9 w=4).
   - **Why the first-moment route is empty.** The computational basis is an
     exact 1-design, so E_x[rho_A(t)] = I/d_A at EVERY t (verified to 1e-17).
     Every linear observable therefore has basis average = Haar average =
     Tr(O)/2^N exactly, and does not move under the dynamics at all. Concavity
     only gives E[S] <= ln d_A = 3.4657, true even at t=0 where every S is 0,
     while E[S] runs 0 -> 0.208 -> 1.424 -> 1.752 at t=0,1,7,33.
   - **Haar cannot give the intra-cluster spread either**, and errs in both
     directions: measured/Haar sd = 0.0368/0.0888 (w=2, over-predicts 2.4x) but
     0.0930/0.0225 (w=6, under-predicts 4.1x). Haar's spread shrinks with D by
     concentration; the measured one grows.
   - **Substructure inside a shell (asked 2026-07-29): the w=6 histogram is
     bimodal.** Two exact involutions explain it.
     (a) x -> x^A (A = even-site mask) complements EVERY bond, so it maps shell
     w to shell N+1-w and preserves S **exactly** (max deviation 0.0e+00 over
     all 2048 states). That is why partner shells agree to every digit and why
     all 924 values of the self-partner shell come in 462 exact pairs. Note
     this is the bond particle-hole map of R8 open item 4, which does NOT
     commute with U -- not a symmetry of the dynamics, but an exact symmetry of
     the time-averaged entanglement.
     (b) The isolated high spike = the fixed points of **R = P o (xor A)**.
     In bond variables the condition is b_{N-j} = 1 xor b_j (each mirror pair
     of bonds carries exactly one wall), which forces w = (N+1)/2 and therefore
     exists only for **N = 3 mod 4**; count **4^((N+1)/4)** (verified N=3..23:
     4,16,0,64,0,256,1024,4096). At N=11 that is exactly the 64 top states,
     [1.9501,1.9681] vs [1.2762,1.9185] for the other 860 -- gap 0.0316 against
     a 5e-4 spacing inside the spike. Out of sample: N=7 gives 16 states with
     gap 0.045; N=9 and N=13 have none, as predicted.
     (c) The low tail is largely the 20 genuinely reflection-symmetric states
     (Px = x): confined to one reflection sector they populate only 71 of the
     shell's 183 distinct eigenphases (Krylov dim 39% of generic) and sit at
     mean 1.674. 20 = C(6,3) = dim Fix(U_w) is so far an unexplained numerical
     coincidence.
   - **No plateau exists** (C150 is not chaotic): S(t) is quasi-periodic on the
     6-torus forever, <sd_t(S)> ~ 0.35. "Saturation" = infinite-time average;
     the cumulative mean needs t ~ 800 to stabilise and a naive t in [200,400]
     window is 2% low. Cap BLAS threads to 4 -- the default oversubscribes and
     costs a factor 2.

15. **Tier 1e — WCC sectors and basins, obc0 — DONE (2026-07-30, R9).**
   New tier: the weakly connected component as the sector-level observable
   (an ENCLOSURE, so exact for monitored AND unmonitored; partitions the basis;
   = minimal projectors of the DIAGONAL commutant). Vocabulary now fixed:
   sector = WCC, monitored attractor = terminal SCC, channel attractor = block
   of the fixed-point algebra. Never mix them.
   - **Sweep: all 256 rules obc0 at every N=6..16 (uniform), 2892 units.**
     N-major order so a run stopped on a wall-clock budget still leaves uniform
     coverage. 30 diagnostic rules carried to N=17/18; the headline map uses the
     uniform N<=16 window only (comparability).
   - **The theorem, fit-free: n_wcc(N)*D_max(N) >= 2^N holds 256/256 at every
     N, and is TIGHT** (min ratio exactly 1.0000, rule 0 at N=6). This is what
     carries the validation; a*b>=2 is a corollary about bases that degenerates
     when either series is sub-exponential.
   - Base plane: 189 on the curve (ab=2), 60 above, 5 degenerate (polynomial
     sector count), 2 irregular, **0 violations**. Anchors: 204/51/150 on the
     curve to 6 decimals; **156 above at ab=2.0313** (V3 passes) with b=1.2554
     vs the expected 4^(1/5)=1.3195 -- gap open, needs larger N.
   - **Rule 28 FALSIFIES the task's a_wcc >= phi prediction.** Exact recurrence
     a_n = a_{n-1}+a_{n-2}-a_{n-4}, char. (x-1)(x^3-x-1), base = plastic number
     rho = 1.32472 << phi = 1.618. Verified out of sample; pbc agrees (~1.30).
   - **Rule 150's wall charge is not a property of unitarity.** Six DISSIPATIVE
     rules (134 IVDI, 142 IEDI, 148 IDVI, 158 IEVI, 212 IDEI, 214 IVEI) have
     EXACTLY C150's sector multiset C(N+1,even w) at every N=6..16. The only
     two exceptions in the I**I family are 132 IDDI and 222 IEEI -- the
     same-reset-type pairs. Sector-level counterpart of Tier-2 R-T13's DFS
     finding for 134/148, now for six rules and the whole multiset.
     The tempting generalisation (only the I-pattern matters, since V/D/E all
     give the same undirected flip edge) is FALSE: only 3 of 16 I-pattern
     groups collapse ({140,156,220}, {196,198,206}, {204}). Cause: the odd-layer
     symbol reads the post-even-layer state, where V branches and D/E do not.
   - **Basin sum rule was UNANSWERABLE for 76 obc0 units** -- Tier 1a caps
     sizes_basins at 2048 and never stored a basin histogram, so the multiset
     is lost (every failure is a truncated record; no untruncated one fails).
     Per user request, recomputed independently into results_basins/ WITH the
     histogram: **67 units, sum rule holds 67/67, agrees with the surviving
     Tier-1a fields, 1.17 h.**
   - Validation: 2874 units, sum-rule residual identically 0; A2 (unitary
     n_wcc=n_rec=n_scc + identical multisets) 99/99; 11 descriptors clamped to
     the theorem bound [1,2] (each recorded) -- the binomial rules need it.
   - Deviations from the task spec, both flagged in R9: the ergodic early exit
     is NOT used (it would break A1 and lose the V2 anchors), and
     ENGINE_VERSION was NOT bumped (that would invalidate ~6000 Tier-1a records
     and force a full recompute, which the same section forbids) -- the version
     lives on the new tier as TIER1E_VERSION="1e.1".
   - **pbc is deliberately NOT done** (user directive: obc0 first, pbc as its
     own report). Note 156-pbc Lucas(N)+1 and 22-pbc were verified in passing.

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
