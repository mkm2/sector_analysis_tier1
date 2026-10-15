# Storyline draft: ergodicity in the 16 unitary Hadamard QCA (R32)

**Status:** draft for discussion (revised 2026-10-15). Nothing has been handed off to the thesis figure project.
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

8. **Thermalisation, directly** (candidate figures F, G): quenches from product states thermalise to the Page value
   only in the chaotic classes, and for the one-sided rules only on the ring (open chain: 0.72 / 0.91 of Page). The
   ETH fluctuations and the spectral form factor give the same ordering: 57/99 are fastest; 153, 147, 60 and 54 are
   progressively slower.
9. **One scar for all** (no new figure; candidate D covers the counts): every exact scar is |h+>^N, its projection off
   a frozen vacuum, or a boundary |h->. The 57 outliers are an edge mode on that vacuum; the 54/147 defect states relax
   slowly.
10. **Quantum vs classical** (candidate figure H): with X, 54 is the integrable RCA54 and X-102 is the solvable
    deterministic Floquet quantum East model. Only 57/99 have ergodic classical orbits, and only on the open chain
    (a domain-wall charge confines them on the ring). Chaos with H follows k, not the classical dynamics.
11. **Positioning**: the model is Hillberry's QECA (T_S = our HSF S). Known: Brody hints from entanglement spectra
    (2021), and free fermions for the Hadamard Goldilocks rule 150 (2024). New: the full symmetry-resolved
    classification, the boundary origin of the one-sided class and its hidden time reversal, the universal scar, the
    kink dispersion, and the thermalisation characterisation.

## Candidate figures (thesis width, thesis style; local drafts)
| | file | content | aspect |
|---|---|---|---|
| A | `../fig_r32_overview_H.pdf` | ⟨r⟩ per rule (unresolved / resolved) + distinct-level fraction, grouped by k | 0.75 |
| B | `cand_B_chaos_scars_W57.pdf` | rule 57: S vs θ with exact scars marked + P(r) vs Poisson / GOE | 0.42 |
| C | `cand_C_resolution.pdf` | ⟨r⟩ vs N for 147 and 54, Krylov-only vs fully resolved | 0.42 |
| D | `cand_D_singlets.pdf` | number of exact gate singlets vs N per class (log scale) | 0.55 |
| E | `../fig_r32_bc_compare_H.pdf` | open chain vs ring: ⟨r⟩ (ring: real / complex momentum sectors) and block counts | 0.75 |
| F | `../fig_r32_dynamics_N14.pdf` | quenches: S/Page vs t, open chain vs ring, and the scar defect | 0.36 |
| G | `../fig_r32_eth_sff.pdf` | ETH fluctuations vs dimension, SFF / RMT ramp, t_2 vs N | 0.36 |
| H | `../fig_r32_X_cycles.pdf` | X gate: longest cycle and <L>/2^N vs N, open chain vs ring | 0.42 |

Optional, if space allows: a per-rule appendix figure (4 panels) for each of the 16 rules
(`../fig_r32_rule_H_W*.pdf`).

## Open points before this can be a thesis claim (status 2026-10-15)
Resolved since the first draft:
- the one-sided antiunitary symmetry (a dressed bond reflection);
- the identity of the entangled singlet (|h+>^N off the vacuum, closed form);
- the 57 outliers (an edge mode);
- the ring check;
- one-sided on the open chain: no drift (plateau 0.424 for N = 13…16).

Still open:
1. Why the one-sided rules plateau at 0.424 on the open chain.
2. Why the SFF shows a large short-time excess and ETH converges slowly for 54, 60 and 147. Presumably both are tied to
   the |h+> vacuum.
3. An analytic form of the 57 edge mode.
4. Whether 54 reaches COE on the ring (0.510 at N = 16, still rising).
