# Storyline draft: ergodicity in the 16 unitary Hadamard QCA (R32)

**Status:** draft for discussion. Nothing has been handed off to the thesis figure project.
**Data:** open boundaries (obc0), N ≤ 13/14. The periodic-boundary check is still pending; see the open points below.

## One-sentence message
Among the 16 unitary rules, chaos needs the gate to fire on **three of the four neighbour patterns**. Exactly those rules
are GOE-chaotic, and each carries a fixed number of exact, gate-protected scar states, which do not grow with N. Every
other rule is trivial, free or arithmetic, or fragmented into Poissonian pieces.

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
   - k = 1: exponential fragmentation; inside each block, Poisson (tensor products of independent rooms).
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

## Candidate figures (thesis width, thesis style; local drafts)
| | file | content | aspect |
|---|---|---|---|
| A | `../fig_r32_overview_H.pdf` | ⟨r⟩ per rule (unresolved / resolved) + distinct-level fraction, grouped by k | 0.75 |
| B | `cand_B_chaos_scars_W57.pdf` | rule 57: S vs θ with exact scars marked + P(r) vs Poisson / GOE | 0.42 |
| C | `cand_C_resolution.pdf` | ⟨r⟩ vs N for 147 and 54, Krylov-only vs fully resolved | 0.42 |
| D | `cand_D_singlets.pdf` | number of exact gate singlets vs N per class (log scale) | 0.55 |

Optional, if space allows: a per-rule appendix figure (4 panels) for each of the 16 rules
(`../fig_r32_rule_H_W*.pdf`).

## Open points before this can be a thesis claim
1. **Periodic boundaries.** On a ring the one-sided rules lose their tower (seen at N = 8). Whether they then become
   chaotic, and whether the k = 3 rules remain GOE with the same scars, needs the pbc runs, which are pending.
2. **N = 14 for the k = 3 rules.** These runs were stopped by the memory reaper. The memory-lean pipeline is ready (see
   R32, section "Compute").
3. **One-sided drift.** ⟨r⟩ rises with N (60: 0.334 to 0.395 over N = 9 to 14). Larger N, or the pbc data, will show
   whether these rules are integrable or slowly chaotic.
4. **The gate singlets of rule 54 and the 57 outliers.** These want an analytic form; the entangled singlet of 54 has
   S = 0.29 at every N.
