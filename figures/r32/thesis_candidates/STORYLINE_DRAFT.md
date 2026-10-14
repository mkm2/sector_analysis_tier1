# Storyline draft: ergodicity in the 16 unitary Hadamard QCA (R32)

**Status:** draft for discussion. Nothing has been handed off to the thesis figure project.
**Data:** open boundaries (obc0), N ≤ 14 (up to 20 for the fragmented rules). Rings (pbc): even N = 6 … 14, with
translation, the half-step Ω_T and the gate blocks resolved.

## One-sentence message
On the open chain, chaos needs the gate to fire on **three of the four neighbour patterns**. Exactly those rules are
GOE-chaotic, and each carries a fixed number of exact, gate-protected scar states, which do not grow with N. The
**one-sided** rules (two patterns, one neighbour) are the exception that proves the boundary matters: chaotic in the bulk
(on a ring), but fragmented into a boundary-anchored tower on the open chain. Every other rule is trivial, arithmetic, or
fragmented into Poissonian pieces, whatever the boundary.

## Proposed arc (one thesis section, about 4 pages)

1. **Question.** The Krylov analysis says *how* the Hilbert space falls apart. Does the dynamics *inside* the pieces
   thermalise? The tools are the quasienergy spectrum and the eigenstate entanglement of the Floquet operator U.

2. **What the tools mean for a circuit** (short, from the R32 literature section):
   - Floquet ETH is flat in quasienergy (infinite temperature), so there is no "middle of the spectrum".
   - U is real orthogonal and a product of two reflections, so the chaotic reference is β = 1, ⟨r⟩ = 0.531, on the half
     spectrum.
   - Symmetry resolution is mandatory. The relevant symmetries are the Krylov sectors, the hidden gate-algebra blocks,
     and the spatial symmetry, with the half-step Ω (Ω² = U) for even N.

3. **Main result: a classification by k, the number of firing patterns** (candidate figure A):
   - k = 0 or 4: trivial, U = 1 or H^⊗N.
   - k = 1: exponential fragmentation. Compared at one common N (19), two pairs: 108/201 (fire on 11 / 00, large
     blocks) are Poisson, as expected for tensor products of independent rooms; 156/198 (fire on 10 / 01, blocks of at
     most about 100 states) sit at ⟨r⟩ ≈ 0.42, above Poisson but far from GOE, consistent with structured sums of a few
     small rooms.
   - k = 2 XOR (150, 105): arithmetic spectra, 3^m distinct levels (connects to the earlier rule-150 chapter).
   - k = 2 one-sided: a boundary-driven tower of blocks; statistics near Poisson but drifting.
   - **k = 3: GOE chaos.**

4. **The chaotic rules up close** (candidate figure B, rule 57): thermal entanglement band, P(r) on the GOE surmise,
   and **exact scars**, the eigenstates of every individual gate.

5. **Why symmetry resolution matters** (candidate figure C): for rule 147, mixing the two reflection or Ω sectors makes
   a chaotic rule look integrable (0.42 against 0.53).

6. **Exact scars: how many, and how they scale** (candidate figure D): the count of gate singlets is constant for k = 3
   (2, 2, 4 and 1), linear for the one-sided rules (2N − 4), about one per sector for XOR, and exponential for k = 1.
   Scars in a chaotic spectrum versus a whole manifold of dark states in a fragmented one.

7. **Boundaries** (candidate figure E): on a ring, the k = 3 rules stay chaotic, now CUE (57/99, no mirror) or COE (147),
   as the symmetry dictates. The one-sided rules lose their tower and become chaotic with β = 1 statistics, an
   antiunitary symmetry we have not yet identified. The XOR and k = 1 classes are unchanged. The scars at θ = π are
   boundary states; one singlet at θ = 0 survives on the ring.

## Candidate figures (thesis width, thesis style; local drafts)
| | file | content | aspect |
|---|---|---|---|
| A | `../fig_r32_overview_H.pdf` | ⟨r⟩ per rule (unresolved / resolved) + distinct-level fraction, grouped by k | 0.75 |
| B | `cand_B_chaos_scars_W57.pdf` | rule 57: S vs θ with exact scars marked + P(r) vs Poisson / GOE | 0.42 |
| C | `cand_C_resolution.pdf` | ⟨r⟩ vs N for 147 and 54, Krylov-only vs fully resolved | 0.42 |
| D | `cand_D_singlets.pdf` | number of exact gate singlets vs N per class (log scale) | 0.55 |
| E | `../fig_r32_bc_compare_H.pdf` | open chain vs ring: ⟨r⟩ (ring: real / complex momentum sectors) and block counts | 0.75 |

Optional, if space allows: a per-rule appendix figure (4 panels) for each of the 16 rules
(`../fig_r32_rule_H_W*.pdf`).

## Open points before this can be a thesis claim
1. **The one-sided antiunitary symmetry.** On the ring, 153/195 are β = 1 in the shape of P(r) (χ² = 27 against GOE,
   385 for the best Poisson+GUE mixture at N = 14), but they have no mirror symmetry. The symmetry was not found by a
   brute-force search over reflections, translations and uniform or sublattice on-site X/Z/H. 60/102 are still rising
   (0.511 at N = 14).
2. **Rule 54** approaches its ensemble more slowly than the others on both boundaries (ring 0.483 at N = 14).
3. **The entangled singlet** of 54 (both boundaries) and of 60/102 (ring only) has the same entropy for all of them,
   0.4045 / 0.3447 / 0.2908 / 0.2430 for N = 8 … 14, and it decreases with N. It wants an analytic form.
4. **The 57 outliers** (N-stable, θ/π ≈ ±0.312, ±0.688) are unexplained.
