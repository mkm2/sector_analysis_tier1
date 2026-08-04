# Tier-1 — follow-up status

Deep dataset: dissipative N<=17, unitary N<=21 (Tier 1a/1c); Tier 1d pair graph
diss N<=9 pbc. **C150 (rule 150) obc0: closed form for all N, numerical
frontier N<=32.** Reports R1/R2/R5/R6/R7/R8/R9/R10/R11 regenerated; full suite 703 tests pass.
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
   - **The V+reset family has NO sector structure.** 186 of 256 rules sit at
     exactly (a,b)=(1,2) -- one sector of size 2^N -- including **144 of the 160
     V+reset rules (90%)**: the reset connects the whole basis into a single
     enclosure. So for those rules the sector axis says nothing and all the
     structure is in the MONITORED map, where they spread over the column
     a_att~1 from b_att=1.0 to 1.89. The V-free baseline is the reverse: it
     populates the sector plane and collapses in the attractor plane. Median
     deficit 2-a_att*b_att: unitary -0.135, V+reset +0.534, V-free +0.909
     (the last means a_att*b_att~1, i.e. O(1) fixed points and an almost
     entirely transient space). Figures now aggregate coincident points with
     marker area + printed counts -- one marker per rule hid all of this.
   - Both map panels use ONE rate convention, so the 9 unitary rules land at
     identical coordinates in both -- assertion A2 made visual. This needed the
     attractor series capped to the same N window as the sector series.
   - **Dissipation destroys sector structure, 120/120 (R9 sec.6).** Every
     V+reset rule has a COHERENT PART: switch the resets off, D,E -> I, leaving
     only I and V, i.e. one of the 16 unitary rules (rules.coherent_part /
     coherent_parent / dissipative_children). The map partitions the 160 V+reset
     rules into exactly 14 clusters: a parent with v Hadamards has 3^(4-v)-1
     children (4*26 + 6*8 + 4*2 = 160; VVVV has none, IIII is nobody's parent).
     Then in the SECTOR plane:
       * 8 parents already sit at (1,2) -- nothing to destroy -- and all 40 of
         their children stay exactly there;
       * 6 parents have genuine sector structure and hold 120 children, and
         **NOT ONE of the 120 keeps its parent's position**; 104 collapse all
         the way to (1,2), the other 16 move somewhere intermediate.
     So adding any reset to a rule with sector structure destroys it, no
     exceptions over the 120 available cases -- and 144 = 104 + 40 reconciles
     exactly with the pile-up count above.
     In the ATTRACTOR plane the moves are large for ALL 14 clusters, including
     the 8 that do not move at all on the sector axis -- which is why both
     planes are kept. Figures fig_dissip_{sector,attractor}_obc0.
   - **FRAGMENTATION IN AN OPEN QUANTUM SYSTEM: 8 rules (R9 sec.6.2).** Of the
     104 children of the four structured parents (156/198/108/201), 94 collapse
     to (1,2), 10 land intermediate, and **8 of those keep an EXPONENTIALLY
     growing sector count**. Because a sector is an enclosure, this is exact for
     the unmonitored channel too -- genuine open-system HSF, not a unitary
     circuit with noise added afterwards. Six of the eight have EXACT algebraic
     bases from integer recurrences:
       * **plastic number rho = 1.32472** (root of x^3 = x+1):
         28 IIVD (b=2), 70 IVID (b=2), 157 EIVI (b=sqrt3), 199 EVII (b=sqrt3)
       * **supergolden psi = 1.46557** (root of x^3 = x^2+1):
         73 VIID (b=1.78187), 109 EIIV (b=1.79363)
       * fitted only, ~1.22: 29 EIVD, 71 EVID (parity-oscillating series, value
         not pinned; both are exponential but likely the same constant)
     Note the parents' bases are phi (x^2=x+1) and psi_201=1.75488, so the reset
     RAISES THE ORDER of the characteristic polynomial rather than perturbing
     the root. They pair by reflection: {28,70} at (rho,2), {157,199} at
     (rho,sqrt3), {29,71}, and 73/109 are self-reflective. 28 and 70 keep b=2
     exactly -- exponentially many sectors AND a volume-law largest sector.
     Highlighted with gold rings in fig_sector_map / fig_dissip_* and pinned by
     3 tests. `sectors.open_system_fragmented()`.
   - Classifier fixes needed to get this right: class and base must agree.
     A polynomial class must report base 1 (six linear-count rules were being
     given a=1.09 by an M2 fit), and a base from a parity split only counts as
     exponential if the parity BRANCH is classified exponential (rule 44's even-N
     counts are 7,9,11,...,17 -- a straight line that a pure exponential fit
     still reports as base 1.09). After the fix: 196/254 rules at (1,2),
     150/160 of the V+reset family, and the deficit medians sharpen to exactly
     0.000 (unitary), +0.534 (V+reset), +1.000 (V-free).
   - **The base plane HIDES a whole tier -- survival must be asked by growth
     CLASS, not base (R9 sec.6.3/6.4).** a=1 conflates "one sector" with "linear
     sector count", so rule 150 (floor((N+1)/2)+1 sectors, 9 at N=16) was
     plotted on top of rule 51 (exactly 1). Resolved by
     `sectors.sector_growth_class` / `survival_by_parent_class`:
       parent class            parents          children | ->exp ->poly ->const
       exponential  108,156,198,201                 104  |   8    14     82
       polynomial   60,102,105,150                   32  |   0     0     32
       constant     54,57,99,147,153,195             24  |   0     0     24
     So for **150 and 105 NOTHING survives** -- all 16 children drop to a
     CONSTANT sector count (150: six to 1 sector, two to 2; 105: all eight to 1).
     Polynomial sector structure is MORE fragile than exponential, not less.
   - **The wall charge is a sink, not a source.** Six of the 14 polynomial
     survivors (20,148,158 from 156; 6,134,214 from 198) have EXACTLY rule 150's
     series 4,5,5,6,6,7,7,8,8,9,9. So dissipation can degrade phi-fragmentation
     DOWN TO the wall charge, while a reset applied TO the wall charge destroys
     it. Hierarchy: exponential(phi,psi_201) -> {rho,psi} (8) -> wall charge (14)
     -> one sector (82), with the wall charge a terminal rung.
   - **A SECOND polynomial mechanism: the pinned frontier (R9 sec.6.4).** The
     other six polynomial survivors -- 110 IEIV, 124 IIEV, 188 IIVE, 230 IVIE,
     44 IIDV, 100 IDIV -- conserve the **position of the extremal excitation**
     (rightmost 1 for 110/230, leftmost for 124/188/44; verified at N=9 that
     every sector is a level set of that index). The outermost excitation is
     pinned by the reset, everything inside is free, giving
     **n_wcc = N+1, sizes 2^(N-1),...,2,1,1 (dyadic tower), D_max = 2^(N-1)
     EXACTLY, so D_max/2^N = 1/2 constant.** Contrast the wall charge: ~N/2
     sectors, binomial sizes, D_max/2^N ~ N^(-1/2) -> 0. Both sit at (a,b)=(1,2)
     and are indistinguishable in the base plane. 44 splits further at odd N
     (N+1 even / N+3 odd); 100 has N+3 with a non-dyadic tail.
   - **105 is NOT 150's reflection partner** (R8 implied it; corrected here). 105
     = VIIV fires when neighbours AGREE, 150 = IVVI when they differ, and at
     **N = 1 (mod 4)** 105's sectors carry ODD wall number: N=9 gives
     [252,120,120,10,10] = C(10,odd w) vs 150's [210,210,45,45,1,1] =
     C(10,even w); same at N=13. Elsewhere they agree exactly.
   - **The frontier family is an obc0 ARTEFACT (R9 sec.6.5; user's point).**
     "Position of the leftmost excitation" is not definable on a ring, so the
     N+1 dyadic tower must vanish at pbc -- and it does: **all eight of
     44,60,100,102,110,124,188,230 collapse to n_wcc = 2 at every N=6..13.**
     This also catches **60 and 102**, which had been filed as polynomial parents
     with alpha~0.82 beside 150/105; their obc0 series is the same N+1 dyadic
     tower, so they belong to the frontier family, and the alpha differed only
     because the fitter saw one series through a different window.
     Rule 150's wall charge SURVIVES pbc with exactly R8's closed form
     (N/2+3 even, (N+3)/2 odd, verified N=6..13) -- bulk, not boundary.
   - **The 8 exponentially fragmented rules ARE bulk** -- the same check gives
     the opposite answer: all eight still fragment on the ring with the SAME
     bases to within finite size. 73/109: pbc 1.4651 vs psi = 1.46557 (3e-4!).
     28/70/157/199: pbc 1.305/1.271-1.310 vs rho = 1.32472. 29/71: 1.158/1.233
     vs ~1.22. So rho and psi are BULK constants and sec.6.2's headline is not a
     boundary effect. Worth having checked -- it is exactly the objection the
     frontier family turned out to deserve.
   - **obc1: DO NOT implement** (user, 2026-07-30). Removed from R9's open items.
   - **R9 is STRICTLY obc0** (user, 2026-07-30). The pbc robustness material was
     excised from the report: sec.6.4 now states the frontier-charge caveat
     structurally (the charge is defined by reference to an END of the chain, so
     the question of whether it is bulk cannot be settled at obc0) and defers it
     to the pbc report. The task's two pbc-specified regression targets (156
     Lucas, 22 attractors) are noted as passing but deferred. The pbc facts
     themselves survive only in the test suite (3 tests, clearly labelled NOT
     the pbc sweep) and in this file, so they are not lost.
   - **60 and 102 are UNITARY frontier rules** -- established on obc0 evidence
     alone: same dyadic tower (verified N=10,12) and the extremal-excitation
     charge (verified N=9: lowest set bit for 60, highest for 102). So the
     frontier charge is NOT created by dissipation; it already exists in the
     unitary rule space, and is a destination reachable both directly (60, 102)
     and as a dissipative degradation of phi-fragmentation (110, 124, 188, 230
     from 108, 156, 198). This corrects their earlier filing beside 150/105.
   - **NAMING CORRECTION: do not say "phi-fragmented" (R9 sec.6.2a).** Wrong on
     two counts, both easy to fall into again:
       * ACROSS rules phi changes role. 156/198: n_wcc = 21,34,55,89,... is
         FIBONACCI (coeffs [1,1]), so phi is the base of the sector COUNT; their
         D_max = 6,9,12,16,20,27,... has NO integer recurrence and its base
         4^(1/5) comes from R2 sec.3's room-packing saddle point.
         108/201: n_wcc = 37,65,114,200,... has coeffs [2,-1,1], base 1.754878 =
         root of x^3 = 2x^2 - x + 1; their D_max = 8,13,21,34,55,... IS Fibonacci,
         so for THIS pair phi is the base of the largest sector SIZE.
       * WITHIN one rule the two quantities obey unrelated laws (Fibonacci word
         count vs a saddle-point optimisation over room length), so no single
         constant names "the fragmentation" of a rule.
     Use "the four exponentially fragmented parents" and always quote a base
     together with its observable. (a,b): 156/198 = (phi, 4^(1/5)), ab=2.1350;
     108/201 = (1.75488, phi), ab=2.8394 -- both above the curve but for
     different reasons, the first pair by having many sectors, the second by
     having large ones.
   - **The "reset raises the degree" claim holds only for the Fibonacci-count
     pair.** 156/198 count by x^2=x+1 and their rho children by x^3=x+1 (degree
     up); 201/108 already count by a cubic and their psi children stay cubic
     (x^3=x^2+1). No single statement covers all eight.
   - **pbc: DO NOT START the sweep yet** -- the user has further caveats to give
     first (2026-07-30). Held items for that report: the frontier-vs-bulk
     question, the 8 exponential rules' boundary robustness, 156-pbc Lucas and
     22-pbc.
   - **pbc is deliberately NOT done** (user directive: obc0 first, pbc as its
     own report). Note 156-pbc Lucas(N)+1 and 22-pbc were verified in passing.

15b. **R9 corner plots + the recurrent/transient map (2026-07-31).** Added on
   request, in R2 fig.6's layout: F7 (sectors) and F8 (terminal SCCs) put the
   base-base plane in the main panel with the two sub-leading powers in the
   margins, sharing axes, so one point carries (base, alpha) for both series.
   F7's bottom margin is the one that earns its place -- the column at a=1
   splits into alpha~0 (genuinely one sector) and alpha~1 (the pinned-frontier
   rules), which the base plane cannot distinguish.
   F9 is the new "terminal SCCs vs transient states" map: growth base of the
   number of terminal SCCs against the growth base of the recurrent MASS |Rec|.
   Results: the unitary rules sit at mass base exactly 2 with recurrent fraction
   1 (no transient at all), and **not one of the 233 non-unitary rules gets
   within 0.02 of 2** -- a single reset makes the recurrent set exponentially
   vanishing. Median recurrent fraction at N_max: 6.5e-3 (V+reset), 2.6e-4
   (V-free). The V-free family is the MORE contracting of the two (median mass
   base exactly 1, a bounded recurrent set, against psi = 1.46557 for V+reset):
   the Hadamard keeps exponentially more of the basis alive.
   Code: `sectors.attractor_point`, `sectors.recurrent_mass`,
   `sector_figure.fig_corner`, `sector_figure.fig_recurrent_transient`.

16. **R10 — X-gate PERMUTATION circuits — DONE (2026-07-30).** The same 256
   symbol tables with V = X instead of the Hadamard. Every symbol is then a
   FUNCTION on bits, so succ(x) is single-valued and the transition graph is a
   FUNCTIONAL GRAPH. Consequences, all verified over 4864 units (N=6..24, all
   256 rules, obc0):
   - **n_recurrent == n_wcc identically, 4864/4864, zero failures.** One cycle
     per weak component (F1), so basins ARE sectors and no state is shared. This
     is exactly the identity that fails badly for the Hadamard circuits (R9:
     rule 22 has 3 attractors inside 2 sectors), which makes the family a
     control on the machinery, not just a physics question.
   - **REVERSIBLE = the map is a bijection <=> every symbol is I or V, i.e. the
     16 rules 51,54,57,60,99,102,105,108,147,150,153,156,195,198,201,204.**
     Proved locally (each step is a controlled-X, so invertible; D/E destroy
     information), hence true at EVERY N and both bc -- checked N=7,8,9.
     **Do NOT conflate with the six reversible ECA** (15,51,85,170,204,240):
     different construction AND different quantifier -- the ECA six are those
     bijective for EVERY N (an intersection), and at fixed N the ECA bijective
     set is larger (N=7 also has 45,75,89,150). Only 51 and 204 are in both.
   - **Headline: large sectors, short attractors.** 239 of 256 rules have
     exponentially large sectors but only **8** have exponentially long cycles
     (207 bounded, 27 linear, 14 irregular); for the 80 V-free rules the
     exponential count is ZERO. Median cyclic fraction at N=24: 1.8e-7
     (V+reset), 1.5e-5 (V-free); median transient depth 13 and 12.
     [The count was 24 in the first draft; 16 were fitting artefacts, see the
     three guards in scaling/sectors.py added 2026-07-31.]
   - Sum rule and the finite-N hyperbola carry over unchanged (256/256, tightest
     exactly 1.0000). Sector map: 4 rules with ab<2 at N<=24, all the degenerate
     sub-exponential case of R9 sec.5 (X25, X74, X88, X182; 74 and 88 are V-free
     and are the same rules R9 lists). Cycle map: 223/241 below 2, as expected,
     since cycles do not partition.
   - Reversible anchors: 204 identity (2^N fixed points), 51 global flip
     (2^(N-1) 2-cycles), 150 = the linear map x_i -> x_i XOR x_{i-1} XOR x_{i+1}
     whose longest cycle is only N+1. Contrast 57/99 with 110 sectors at N=17,
     largest 36456.
   - Engine: `permutation/xca.py` ports the compiled step list from
     core/cycle.py (so the brick-wall order and boundaries are byte-identical to
     the validated engine; only the local action is swapped) and adds a numpy
     builder that lifts the loop over x out of Python -- N=22 went 30s -> 1.06s,
     which is what made N=24 affordable. Scalar and vectorised paths are checked
     against each other. Store `results_xgate/`, version 1x.1. 60 tests.
   - **Movement analyses (added 2026-07-31, R10 sec.5-8).** (i) WCC -> recurrent
     is a purely VERTICAL drop, by theorem: a is shared by F1 and L_k <= |S_k|
     pointwise. 16 reversible rules do not move; median drop exactly 1.000.
     (ii) parent -> children (resets off, D,E -> I): 16 clusters, 3^k-1 children
     each, the identity's cluster being the whole V-free family. For 177 of 240
     children the reset acts as the IDENTITY on the recurrent set, so the
     child's attractors are literally orbits of its reversible parent; the 63
     exceptions make only short cycles. An exponential cycle can only be
     inherited: 6 reversible parents have one and exactly 2 children keep it
     (73 under 201, 109 under 108, both with one reset symbol).
   - **Why V-free collapses.** The 16 all-reset rules ARE the 16 boolean
     functions f(l,r); 14 are constant/projection/monotone with cycle <= 2, and
     the only two that are neither are XOR and XNOR = rules 90 and 165, the
     GF(2)-affine ones, which are exactly the two V-free rules with unbounded
     cycles. Every other rule with a growing cycle contains a V.
   - **Three sizes, not two** (R10 sec.5): sectors partition (sum rule -> ab>=2),
     the recurrent set does not (|Rec|/2^N ~ 3.6e-7 median), and the cycle map's
     analogous bound degenerates to b>=1 because |Rec| ~ n_rec. Rule 1's
     recurrent set is EXACTLY the golden-mean shift (no 11), size F_{N+2}, while
     its longest cycle grows by 3 per site: an exponentially large attractor set
     made of fixed points. 21 more rules have |Rec| in {F_N, F_{N+1}, F_{N+2}}.
     Random-map yardstick at N=24: 5100 cyclic points and depth 2600 expected,
     against 6 and 13 observed.
   - **The eight rules above the curve in the cycle map** (R10 sec.3.1, added
     2026-07-31): 57/99 (ab=2.88, a single orbit through 41% of the space at
     N=24), 156/198 (2.55), 201 (2.34), 108 (2.34), 73 (2.08), 109 (2.08).
     They are exactly the 8 exponential-cycle rules, labelled in Fig X1 right
     with their sequences tabulated. Ten more sit exactly ON the curve at (2,1),
     all reversible.
   - **EXHIBITED long orbits (R10 sec.10, 2026-07-31).** The exponential-cycle
     claim was a FIT over N<=24 where the numbers are 391..561 (and 6.8M for
     57/99). Now proved by exhibition: `permutation/orbits.py` follows ONE orbit
     with a bitwise sweep -- at obc0 the sublattices are independent sets so the
     brick-wall sweep is two SIMULTANEOUS half-updates, ~20 integer ops, O(1)
     memory, 1.2M sweeps/s. Resets do NOT break this (an even site writes an
     even bit and reads only odd bits, whatever the local action); verified
     against xca.x_step for all 256 rules at N=6..11, every state, 0 mismatches.
     pbc is refused (sublattices are not independent at odd N).
     Results: 57/99 have ONE orbit through 203 million states at N=29 (38% of
     the space; 47% at N=26, 50% at N=27); the other six run from 391-561 at
     N=24 to 5460-21505 at N=40. Lengths stay lcm-composite (420, 840, 1260,
     1540, 2310, 4620, 6930, 9240 for 156). A sampled point is a LOWER BOUND and
     its tightness degrades with N, so its fitted base (1.18-1.23) understates
     the exact one (1.26-1.27) -- read it as a certificate, not a measurement.
   - **CORRECTION: the family DOES have decoherence-free subspaces (R10 sec.9).**
     The first draft's "no dark states / the Tier-1d certificate is vacuous" is
     withdrawn. A functional graph describes POPULATIONS; a reset has Kraus
     operators |0><0| and |0><1| which agree on basis states and differ on
     coherences. With J(x) = the sites where a reset fired on a bit not already
     at the reset value, |x><y| survives iff J(x) = J(y), and sec.8's
     inheritance property (J empty on Rec) is exactly a DFS criterion --
     inheritance flag == no-jump flag for 240/240 rules. Census at N=12: 127 of
     240 irreversible rules carry a DFS of dim>1, 121 of them the whole
     recurrent set. Dimensions are exponential and are constrained shifts:
     rule 1 = the golden-mean shift (no "11"), F_{N+2}, base phi (the PXP
     space); rules 76/73 = no "111", tribonacci base 1.8393; rule 22 = Lucas
     L_{N+1}; rule 232 (V-free, the user's example) base -> phi. Rules 90/165
     protect NOTHING (labels de-synchronise), matching R7 from the other side.
     Cross-checked against the exact superoperator: dim Fix(Phi) = n_rec^2
     (232 obc0: 49/121/289 at N=4,5,6; 90 pbc N=6: 16). Code
     `permutation/coherence.py`, `analytics/xgate_dfs_obc0.json`.
   - **pbc NOT run** (held with R9's).

17. **R11 — Hadamard vs X, one rule space two gates — DONE (2026-07-31).**
   `reports/tex/R11_gate_comparison.tex`, 8 pp, obc0, common window N<=16
   (R9 fits on 16 and R10 on 24, so the X descriptors are REBUILT on 16; never
   compare across windows).
   - **REFINEMENT THEOREM.** succ_X(x) is an element of succ_H(x) for every rule,
     state, N and bc -- the Hadamard produces both values of the target bit and
     the flip is one of them. So the X graph is a SUBGRAPH of the H graph and the
     X sectors REFINE the H sectors. Verified exhaustively (256/256 at N=8,9) and
     pointwise over 2816 units: n_wcc^X >= n_wcc^H and D_max^X <= D_max^H, zero
     failures. Superposition is a COARSENING operation on the sector lattice.
   - Corollary: no rule loses sector structure. Class cross-tab is strictly
     upper-triangular -- all 51 H-exponential rules stay exponential under X, and
     39 of the 173 H-constant rules gain structure (28 exponential).
   - Corollary: the 81 rules with no V are the SAME circuit under both gates.
     The two engines share no code below core/cycle.py::_compile and agree on
     all 916 shared units -- the strongest cross-implementation check we have.
   - Unitary/reversible baseline (the only like-for-like comparison, since
     A2 and F1 both give sectors = attractors): 11 of 16 sit at (1,2) under H;
     under X 9 of those 11 end at b=1, 7 at (2,1) exactly. 204 is the only fixed
     point (no V). Pearson r between the gates is +0.17 (a) and +0.03 (b) on the
     16; +0.47/+0.33 on the 175 rules with a V. Fragmentation transfers, the
     grammar producing it does not.
   - R9's headline sets: all 8 open-system-fragmented rules stay exponential,
     but 4 of them (140/196/206/220) are V-free -- the same circuit -- so the
     Hadamard was never doing any work there. The pinned frontier survives the
     gate for only 2 of 8 (188, 230 keep exactly N+1 sectors); 60/102 go to
     a=2. Combined with R9 sec.6.5 the frontier charge is fragile in BOTH the
     boundary condition and the gate.
   - **HSF reading (R11 sec.7, added 2026-07-31).** The (a,b) plane IS the
     fragmentation phase diagram: (1,2) unfragmented, (a>1, b=2) weak,
     (a>1, 1<b<2) strong, b=1 shattered; the corner a=1, b<2 is forbidden by
     the sum rule and always means a sub-exponential count ("degenerate").
     Counts: H 196 unfragmented / 50 strong / 1 shattered; X 152 / 75 / 12 plus
     2 weak (57, 99 -- the only weakly fragmented rules anywhere). Refinement
     forbids the lower-left: no rule becomes LESS fragmented under X.
   - **156/198 are the sharp contrast.** Under H a strongly fragmented
     CONFIGURATION SPACE: wall pattern frozen, rooms between walls, phi^N
     sectors of size up to 4^(1/5)^N (R2 sec.3). Under X the same sectors, cut
     into ORBITS by extra charges: at N=14 the 987 H sectors split into 2280 X
     orbits and EVERY sector factorises, dim = (#orbits) x (orbit length) --
     16x4, 4x4, 3x12, 6x6 ... Orbit lengths are lcms (35, 42, 60; the global
     maxima 105, 140, 210, 420 are lcm(3,5,7), lcm(4,5,7), lcm(2,3,5,7),
     lcm(3,4,5,7)), i.e. independent room cycles with the relative phases as
     the extra conserved charges. Holds for 156/198 at every N checked; FAILS
     for 108, 201, 54, 150, 105.
   - **Why H is the interesting end**: sectors big enough to thermalise inside
     (phi * 4^(1/5) = 2.135 > 2); visible in entanglement (R8 sec.10: rule-150
     saturation clusters by sector) whereas under X a basis state stays a basis
     state and S(t) = 0; and every X conserved quantity is a classical orbit
     indicator, which any bijection has for free.
   - **X54 IS Rule 54.** For all 16 reversible rules the local update is
     x_i -> x_i XOR f(l,r) and the induced ECA is the ECA of the SAME Wolfram
     number (checked for all 16). So the reversible family = the ECAs affine in
     the centre cell, brick-walled; X54 = IVVV = x_i XOR (l OR r) is the Rule 54
     automaton of the Klobas/Bertini/Prosen literature. R11 sec.7.5 also covers
     why automaton circuits are called quantum (unitarity is structural; the
     quantum content is in superposed initial states) and flags that the
     literature attributions there are from memory and need checking.
   - Code: `permutation/compare.py`, `compare_tables.py`, `compare_figures.py`;
     `analytics/gate_compare_obc0.json`; 17 tests in `tests/test_gate_compare.py`.

18. **Fitting-layer correction (2026-07-31), affects R9 and R10.** Three guards
   in `scaling/sectors.py`, all found by running the R9 fitting convention on the
   X-gate CYCLE series:
   - `_volume_fraction_base` (is D_max = c*2^N*N^alpha?) must now (a) see an
     eventually non-decreasing series, (b) have |alpha| <= 8, and (c) win on the
     FULL series and not only on a 6-point tail. Over a short window ln N is
     nearly affine in N, so a bounded or slowly growing series could win by
     accident.
   - An "exponential" class now requires the leave-one-out band of M2's kappa to
     exclude zero rate; BIC selects M2 for straight LINES too (X1's cycles run
     44,47,...,65 and were called exponential with base 1.0952).
   - Effect on R10: the exponential-cycle count 24 -> 8. Effect on R9: EIGHT
     D_max descriptors lose a spurious base 2 -- 28, 70, 78, 92, 132, 133, 164,
     222. 70/78 are now sqrt(3) exactly, matching 157/199 (D_max = 2*3^(N/2-1)
     on even N); 28/92 sit between their two parity branches (even sqrt 3, odd
     sqrt(1+sqrt 3)). No verdict changes: all eight were and remain above the
     curve, products now 2.20-2.33 instead of 2.65-3.24. R9 sec.6.3 carries the
     correction note; tab_r9_classes/exact/openfrag regenerated.

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

## R12 (C156 spy plots, 2026-08-04) --- DONE
17. **Report R12** draws the sector claim instead of counting it: two spy plots
    of the SAME rule-156 transition matrix at obc0, computational basis vs a
    basis permuted so each weak component is contiguous.
    - **Off-block nonzeros = 0** at N = 6..12, computed rather than asserted
      (`spy.check_block_diagonal`) --- the blocks are found by weak connectivity
      and the check then asks independently whether any matrix entry crosses.
    - **n_wcc = F_{N+2} exactly** (21, 34, 55, 89, 144, 233, 377), which is where
      R2's phi comes from. Under pbc the same rule gives Lucas + 1: same base,
      different sequence, so the bc must be stated with the constant.
    - **nnz obeys an exact order-4 recurrence** a_N = 2a_{N-1} + 2a_{N-3} +
      a_{N-4}, char poly (x^2-2x-1)(x^2+1), so base **1+sqrt2** (silver ratio)
      with a period-4 modulation from the +-i roots. So one rule carries three
      exact constants for three questions about one matrix: phi (how many
      blocks), 4^(1/5) (how big the largest, R2), 1+sqrt2 (how many nonzeros).
    - **The sparsity IS the block structure**: global density falls 1.81% ->
      0.24% from N=8 to N=12 while fill inside the blocks only drifts 74% ->
      62%, so the concentration factor grows 41x -> 259x.
    - Correction made while drawing: the computational-basis panel is NOT
      structureless (an earlier caption said so). It is strongly banded and
      self-similar, because integer order groups states by high bits and the
      rule is local --- but the bands run ACROSS sectors. Wrong structure, not
      no structure.
    - **Frobenius normal form (added 2026-08-04, R12 sec.4).** The same matrix
      reordered by SCC in reverse topological order (sinks first) for rules
      156, 157, 109, 150, 158 -- spanning exponential sector counts (phi, rho,
      psi) and polynomial ones, unitary and dissipative. Three panels each:
      computational / WCC block-diagonal / Frobenius block-upper-triangular.
      * **Nothing below the block diagonal** for any of the five, computed via
        `spy.check_frobenius`. The block order comes from a Kahn sweep on the
        condensation, not from Tarjan's internal numbering, so triangularity
        does not depend on a traversal detail.
      * **Unitary => the triangular form IS the diagonal form.** 156 and 150
        have n_scc = n_wcc = n_terminal, all states recurrent, drainage 0.
        This is R9's A2 as a picture.
      * **Dissipative => they are nothing alike.** 158 has 6 WCCs and 936 SCCs;
        157 has 16 and 758; 109 has 54 and 723. For 158, 7545 of 7919 nonzeros
        (95%) are strictly upper -- almost the whole matrix is transient flow.
      * **150 and 158 share the IDENTICAL sector partition** (same sets of
        states, not just sizes: 462,330,165,55,11,1 at N=10) at N=8,10,12, yet
        150 is unitary with 1024 recurrent states in 6 SCCs and 158 keeps 31 in
        6 terminal SCCs out of 936. No sector observable can separate them; the
        Frobenius form does at a glance. The sharpest concrete form of "the
        sector axis is uninformative for the V+reset family".
      * n_terminal == n_wcc for all five (none is among R9 sec.7.3's twelve).
    - **Multi-attractor examples (added 2026-08-04, R12 sec.4.3).** One per
      sector-count class from R9 sec.7.3's twelve: 203 (VEII, constant, n_wcc=1
      at every N, holding 28 attractors), 100 (IDIV, polynomial, n_wcc=N+3, 60
      attractors with 19 in one sector, and n_scc = 1024 = dim so every SCC is a
      singleton and every attractor a fixed point), 29 (EIVD, exponential, 16
      attractors in 7 sectors). Drawn as recurrent-corner figures: the
      |Rec|x|Rec| block with each terminal SCC coloured by its parent sector,
      bracketed where a sector holds several.
    - **29/157 is the DUAL of 150/158.** 29 and 157 have the IDENTICAL attractor
      set (same states, verified N=9,10,11: 12/41, 16/67, 21/99) but 157 has one
      attractor per sector and 29 merges them (16 attractors in 7 sectors). So:
      150/158 agree on the sector partition and differ in attractors; 29/157
      agree on the attractors and differ in sectors. Neither observable
      dominates -- both maps are needed.
    - **CORRECTION (user-reported, same day).** The first Frobenius figures used
      a GLOBAL Kahn sweep ordered by size, which interleaves SCCs from different
      sectors. Since no edge ever crosses a weak component, that scattered each
      sector's states across the index range and made intra-sector drainage look
      like flow BETWEEN sectors. Fixed: `spy.scc_blocks(nest_in_wcc=True)` is
      now the default -- one sweep per sector, sectors contiguous and largest
      first as in the WCC panel -- so the form is block-diagonal at the sector
      level and block-upper-triangular INSIDE each sector, with the sector
      outlines drawn on the panel. Both orderings are valid Frobenius forms and
      both are exactly triangular (tested); only the nested one is legible.
      Also fixed: spy.py's `__main__` guard sat mid-module, so `python -m
      ...spy` NameError'd after the first three figures -- the same defect as
      sector_figure.py had, now pinned by a test in both.
    - **Two more corrections, both user-reported (2026-08-04).** (i) The gold
      recurrent shading vanished when the sector outlines were added -- it was
      replaced rather than kept -- and under the nested ordering a single
      top-left corner would have been wrong anyway, since each sector holds its
      own sinks at the start of ITS range. Now drawn per sector. (ii) The Kahn
      key was `-size` alone, so a large predecessor became ready and jumped
      ahead of a smaller remaining sink: terminal SCCs were NOT a contiguous
      prefix of their sector for rules 203 and 100, and the gold rectangle would
      have painted transient states as recurrent. Key is now
      `(is-not-a-sink, -size)`, which emits every sink before any non-sink and
      is still a valid reverse-topological order because sinks have no
      successors. Pinned by a test asserting per-sector prefix contiguity.
    - Also clarified: a UNITARY rule's blocks are not triangular and should not
      be -- each sector is a single SCC, i.e. irreducible, and the Frobenius
      form of an irreducible matrix is the matrix itself. Triangularity is
      visible only where drainage exists to make it.
    - **How to read a block (R12 sec.5, added on a user question).** A recurrent
      SCC block is DIAGONAL only when the attractor is a single fixed point.
      "Recurrent" constrains the SET (closed + strongly connected), not the
      states: a state in an attractor keeps moving, it just never leaves. Three
      shapes: fixed point (1x1 self-loop, the only diagonal case), pure cycle
      (LxL permutation matrix, ZERO on the diagonal for L>1), branching (general
      irreducible block). Census over the eight rules drawn: only rule 100 has a
      diagonal recurrent part, and only because n_scc = 1024 = dim so the graph
      is acyclic and all 60 attractors are fixed points. **Not one pure cycle
      appears** -- every multi-state H-gate attractor branches, because a
      Hadamard puts the state into a superposition of successors inside the
      attractor. 203's six 2-state attractors are FULL 2x2 blocks, not
      anti-diagonal swaps. Pure cycles are the X-gate case (R10), where the
      successor is single-valued and every attractor is a permutation block.
      Column weight = branching number (up to 18 for 156, 243 for 150 at N=10).
    - **Frobenius normal form defined properly (R12 sec.4.1, 2026-08-04).**
      Reducible/irreducible, Frobenius' criterion (irreducible iff the digraph
      is strongly connected), and the normal form with irreducible diagonal
      blocks = the SCCs. Two criteria that decide what the figures can show:
      * NOTHING TO TRIANGULARISE when there is one component: an irreducible
        matrix has no strictly upper part and no permutation can give it one.
        Same block by block -- each M_ii is irreducible, so it has no internal
        triangular structure either. That is why the unitary panels look
        untriangular.
      * STRICTLY UPPER NONZERO is NOT the same as k >= 2. Two components with no
        edge between them give a block-DIAGONAL form. The criterion is that the
        condensation has an edge, i.e. some state is TRANSIENT. Verified as an
        equivalence over all eight rules. 156 and 150 have ZERO condensation
        edges at N=8,9,10.
      * Trivial blocks (1x1 zero = singleton SCC with no self-loop) dominate the
        dissipative rules: 669/758 for 157, 880/1024 for 100. And a self-loop is
        NOT recurrence -- rule 100 has 144 states with x->x but only 60 terminal
        components, the other 84 can also step away and never return.
    - Open: pbc is held. A dissipative rule at pbc, and the amplitude-resolved
      (not support-only) version, are both undone.
