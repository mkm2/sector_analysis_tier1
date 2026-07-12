# CONTEXT: Tier 2 — Rule-level analytic machinery for QCA fragmentation
# (wall grammars, sector partition functions, conserved-charge detection)

## Project summary

Companion to Tier 1 (exact graph numerics; see context_tier1_graph_numerics.md
for all model conventions, which are SHARED and not repeated in full here).
Goal: predict, directly from the rule and independent of system size N, the
quantities Tier 1 measures:

  a = scaling base of the number of sectors / recurrent classes,
  b = scaling base of the largest sector / basin,
  f(s) = sector free energy (full sector-size distribution via Legendre
         transform),
plus detection of conserved charges that produce polynomial (not exponential)
sector counts.

All computations here are exact/symbolic on small windows (sympy Rationals,
exact Z[1/sqrt2] engine imported from Tier 1 core), followed by root-finding
with mpmath for the free-energy curves.

Shared conventions: rule tuple (r_00, r_01, r_10, r_11) in {I,V,D,E}^4 decoded
from the Wolfram number as in Tier 1 Sec. 1.2 (including the two documented
draft typos). One cycle = even layer then odd layer, controls read current
values. V = Hadamard.

KEY STRUCTURAL FACT (monotonicity lemma), used throughout: destructive
interference occurs only WITHIN a single Kraus branch (amplitude sums), never
across branches (probability sums). Hence the support-level (nondeterministic
bitstring) dynamics is an over-approximation of the exact dynamics:
support-level frozen structures remain frozen exactly; interference can only
ADD walls and SPLIT sectors further. Support-level results are rigorous
one-sided bounds: a_supp <= a_exact, b_supp >= b_exact.

## Module A: wall grammar

### A.1 Definitions

A WALL is a finite word w in {0,1}^l such that, when present anywhere on the
chain (at a given sublattice offset), the word's sites are invariant under the
one-cycle dynamics for ALL exterior configurations, and consequently (radius-1
locality) the dynamics left and right of the word are independent given w.

Transparency amendment for dissipative rules: a D firing on a wall site is
admissible iff that site's value is 0 (reset target = current value); an E
firing iff the value is 1. Then the site is unchanged even though a channel
formally acts.

Two-site sufficient criterion (fast pre-scan): w = (alpha, beta) is a wall if
the update of the left-exposed neighbor never fires through beta and the
right-exposed never through alpha:
  B_U(., beta) == 0 identically AND B_U(alpha, .) == 0 identically,
with B_D/B_E firings allowed only transparently (value = reset target).
Here B_U(L,R) is the indicator that r_{LR} = V (and similarly B_D, B_E).

### A.2 Wall detection algorithm (exact, all lengths up to l_max)

for l in 1..l_max (default 6):
  for each word w in {0,1}^l:
    for each sublattice offset in {0, 1}:          # brickwork breaks
      for each exterior context (2 sites each side, all 16 combos):
        propagate ONE full cycle exactly on the (l+4)-site window with the
        Tier-1 branch engine, with the outer context sites held as boundary
        controls in both layers (light cone of one cycle = 2 sites).
        REQUIRE: in every branch and every output basis state, the l word
        sites equal w.
    if all pass at a given offset -> record (w, offset) as an exact wall.
Also record which walls already pass at SUPPORT level (ignoring amplitude
cancellation) vs only with exact interference ("cancellation walls") — this
distinction is a paper-worthy observable in itself.

Dedup: discard walls containing a shorter wall as a factor at compatible
offset (grammar minimality).

Validation: rule 156 -> unique minimal wall `01` (both offsets); rule 198 ->
`10`; rule 201 -> `11`; rule 108 -> `00`; rules 150, 54, 22 -> no walls up to
l_max.

### A.3 Interval spectroscopy

For each wall type and each interval length l (default l up to 14):
  Build the window [wall | l free sites | wall] at both offsets. Restrict to
  interior configurations. Using the exact succ() from Tier 1 within this
  window (walls act as fixed boundaries):
  - unitary rules: compute the partition of the 2^l interior configurations
    into SCC classes; record the multiset {d_j(l)} of class sizes.
  - dissipative rules: compute recurrent classes and transients inside the
    window; record {d_rec_j(l)} and the transient count.
If an interval sub-fragments (more than one class), each class becomes a
separate letter of the grammar with its own weight. Check for eventual
regularity of d_j(l) in l (constant / linear / polynomial / exponential fit on
exact integers); the growth law of d(l) is the key rule-level feature.

Validation: rule 156 intervals -> single class, d(l) = l + 1 (single-particle
box states 1^a 0^b). Certify by exact enumeration up to l = 12.

### A.4 Sector partition function and free energy

Definitions:
  Z_N(s) = sum over sectors K of (dim K)^s,
  f(s)   = lim_N (1/N) ln Z_N(s).
Renewal construction (PBC ring; single wall type of length |w|, interval
classes with dimensions d_j(l)):
  g_s(z) = sum_{l >= 0} sum_j d_j(l)^s * z^(l + |w|)
  f(s) = -ln z*(s), where z*(s) is the smallest positive root of g_s(z) = 1.
Multiple wall types: matrix renewal — transfer matrix over wall alphabet with
interval-weight entries; f(s) = ln(Perron root). Add the isolated wall-free
sectors (e.g., all-0 / all-1) separately at finite N; they do not affect f.

Outputs per rule:
  a = exp(f(0))                      # sector-count base
  b = exp(lim_{s->inf} f(s)/s)       # largest-sector base; closed form
      b = max over l, j of d_j(l)^(1/(l+|w|))   (integer maximum!)
  sum rule check: f(1) == ln 2       # COMPLETENESS CERTIFICATE
  sigma(kappa) = inf_s [f(s) - s*kappa]   # Legendre transform = large-N
      sector-size distribution; export the curve for comparison with Tier-1
      histograms (rule 156: compare against Fig. 4b-type data).
  f'(0) = typical sector exponent; f'(1) = typical state's sector exponent.

IF f(1) < ln 2: the grammar is incomplete — missing walls (raise l_max, check
cancellation walls) or a conserved charge (run Module B). Never silently
accept f(1) != ln 2.

Numerics: g_s and roots via mpmath (50 digits); d_j(l) exact ints; for
polylog-type sums use closed forms where available
(sum_l (l+1)^s z^(l+2) = z * polylog(-s, z) for rule 156) else truncate with
rigorous tail bounds (geometric domination).

Validation (rule 156, must reproduce):
  s = 0:   z* = 1/phi        -> a = phi = 1.618...
           (finite N, PBC: #sectors = Lucas L_N + 1)
  s = 1:   z* = 1/2          -> f(1) = ln 2 (sum rule)
  s -> inf: max_l (l+1)^(1/(l+2)) at l = 3 -> b = 4^(1/5) = 1.31951
  f'(0) ~ 0.199, f'(1) ~ 0.223 (report to 3 digits)
Rule 201 (walls `11`): reproduce Fibonacci largest-sector scaling (base phi)
consistent with the Floquet-PXP literature.
Dissipative prediction: rule 28 = (I, I, V, D) inherits rule 156's `01` walls
=> lower bound a >= phi for the attractor count; verify against Tier 1.

## Module B: conserved-charge detection (what walls miss)

Motivating example (must be reproduced as a test): rule 150 with Hadamard has
an exact U(1) charge — the domain-wall number
  W = sum_i (x_i XOR x_{i+1})   (boundary-inclusive with vacuum padding),
conserved gate-by-gate because the H at site i fires iff exactly one adjacent
bond carries a wall, and the X-branch of H = (X+Z)/sqrt2 then moves that wall
by one bond while the Z-branch changes nothing. Sectors = charge eigenspaces,
dimension C(N+1, w), w even. No wall grammar sees this: the invariant is
diagonal only in the BOND variables b_i = x_i XOR x_{i+1}.

### B.1 Detector 1 — edge-constraint null space (finds diagonal charges)

Ansatz: Q = sum_i q_e(x_i..x_{i+r-1}) over even i + sum_i q_o(...) over odd i
(two functions: brickwork reduces translation symmetry to shifts by 2).
Unknowns: 2 * 2^r rationals. For r = 1, 2, 3:
1. At small N (10..14, pbc), take the exact Tier-1 edge set
   { (x, y) : y in succ(x) } and impose Q(y) - Q(x) = 0 for every edge.
   Each is a linear constraint; assemble over sympy Rationals; null space =
   candidate charges (mod the trivial constant).
2. Run the SAME detector in the bond basis: define b_i = x_i XOR x_{i+1}
   (pbc: i mod N) and repeat the ansatz in b-variables. (Rule 150's charge is
   range-1 in b. Cheap and catches the known blind spot.)
3. PROMOTION TO THEOREM: each candidate must be verified gate-locally — for
   every layer and every configuration of a width-(r+4) window, propagate one
   layer exactly and check the charge difference telescopes to zero. Passing
   this check proves conservation for ALL N; report the certificate.

### B.2 Detector 2 — strong vs weak classification of each charge

For dissipative rules, classify every candidate Q:
- STRONG: Q(y) = Q(x) along every edge of every individual Kraus branch
  (equivalently [K_mu, Q] = 0 for all mu, checked on windows). Strong diagonal
  charges fragment the Tier-1 graph; predict #recurrent classes >= #occupied
  eigenvalues.
- WEAK: only <x| Phi^dag(Q) |x> = Q(x) for all x (channel-averaged; check by
  exact window propagation with Fraction probabilities). Weak charges do NOT
  fragment the graph; they organize coherence sectors and peripheral spectra —
  hand them to Tier 1b as block labels for the restricted superoperators.
Unitary rules: single Kraus operator, so strong == weak; skip the split.

### B.3 Detector 3 — commutant algebra (exhaustive backstop)

At N <= 12 (unitary) / N <= 8 (dissipative, strong commutant of the Kraus
set): numerically compute the commutant — all operators commuting with every
local gate (unitary case) or every local Kraus operator (strong case) — as the
common null space of the commutator superoperators (dense or iterative;
following Moudgalya & Motrunich, PRB 107, 224312 (2023)).
Diagnostic: growth of dim(commutant) with N:
  exponential -> fragmentation (walls should exist; if none found, escalate),
  polynomial  -> conventional symmetry (charges should exist; if none found,
                 escalate: non-diagonal or longer-range charge),
  O(1)        -> ergodic.
Cross-check: dim(commutant) must be consistent with the sector counts from
Tier 1 (for classical fragmentation, dim of the diagonal commutant = number of
sectors). Discrepancy = undiscovered structure -> flag rule for manual pass.

### B.4 Charge-resolved counting (closes the loop back to Module A)

Given a proven diagonal charge with local density, sector dimensions follow
from a fugacity-weighted transfer matrix: T(mu) with entries weighting each
local configuration by mu^(local charge); dimension of the charge-w sector =
coefficient of mu^w in Tr T(mu)^N (or the OBC analogue with boundary vectors).
Saddle point in mu gives the largest-sector correction to b = 2 (e.g., central
binomial: 2^N / sqrt(N) type prefactors — feed these into the Tier-1c fit
model with the alpha*ln N term).
Validation: rule 150 obc0 -> C(N+1, w), w even; reproduce the user's reference
table (Tier 1 file, Sec. 8).

## Module C: rule-space sweep and synthesis

1. For each of the 136 reflection representatives: run A.2 (walls), A.3
   (intervals), A.4 (f(s), a, b, sum rule), B.1-B.4 (charges).
2. Emit per-rule record:
   { rule, tuple, walls (words, offsets, support-vs-cancellation),
     interval_growth_law, f_curve (s grid), a_pred, b_pred, sum_rule_ok,
     charges (r, strong/weak, certificate), commutant_dims (N list),
     classification: {ergodic | charge(poly) | fragmented(exp) |
                      dissipative-multistable | mixed}, notes }
3. Comparison table generator: predicted (a, b) vs Tier-1 fitted (a, b) with
   discrepancy flags. This table is the backbone of the paper's final figure
   and of the supplement.
4. Known targets the sweep must reproduce (regression tests):
   - 156/198: fragmented, a = phi, b = 4^(1/5).
   - 150: charge (poly sectors, binomials), a-base = 1, b-base = 2 with
     -1/2 * ln N correction.
   - 54: ergodic (one frozen state), no walls, no charges r <= 3.
   - 22: no walls, no charges; 3 recurrent classes on pbc (from Tier 1);
     classify dissipative-multistable with O(1) attractors.
   - 204: trivial (everything frozen). 51: single sector.
   - 28, 29 (and reflections 70, 71): walls `01` survive with D/E transparent
     -> a >= phi lower bound; interval recurrent dimensions d_rec(l) from A.3
     dissipative mode; produce predicted f(s) and hand to Tier 1 for test.
   - 108, 201: `00` / `11` walls, Fibonacci-type counting.

## Deliverables and layout

  qca_analytics/
    walls.py        A.2 detection (imports Tier-1 exact engine)
    intervals.py    A.3 spectroscopy
    partition.py    A.4 g_s, roots, f(s), sigma(kappa), sum rule
    charges.py      B.1, B.2 detectors + gate-local certificates
    commutant.py    B.3
    counting.py     B.4 fugacity transfer matrices
    sweep.py        Module C orchestration
    tests/          all validation items above as pytest regressions
  Output: analytics/{rule}.json per Module C schema; predicted-vs-measured
  comparison CSV; f(s) and sigma(kappa) curves as CSV for plotting.

Pitfalls checklist:
- Always test walls at BOTH sublattice offsets; brickwork breaks single-site
  translation invariance.
- Never accept f(1) != ln 2 silently; it is the completeness certificate.
- The integer maximum in b (not the real-l saddle point) is the exact answer;
  the real-l saddle is only the N -> infinity mixture statement.
- Support-level results are bounds with a KNOWN direction (a_supp <= a,
  b_supp >= b); label them as such, never as exact values.
- Sector structure and even the attractor set depend on boundary conditions;
  all predictions here are PBC unless stated — convert explicitly (rule 150's
  binomial table is obc0) before comparing to Tier-1 obc0 data.
- Charges may be local only in transformed variables (bond basis); run
  Detector 1 in both bases before concluding absence.
