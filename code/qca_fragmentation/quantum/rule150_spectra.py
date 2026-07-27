"""
Rule 150 obc0: what the sector graph does NOT determine.

The transition graph fixes the sector decomposition exactly (c150.py): sectors
are the wall-number shells, |S_w| = C(N+1, w) for even w.  A sector is the
MAXIMAL kinematically reachable set from any of its basis states -- it is not by
itself a statement about the long-time dynamics (R1 sec. "What a sector is").
This module computes, sector by sector, the quantities the graph is silent on:

  * U_w, the exact real-orthogonal one-cycle matrix restricted to the shell,
    built from the engine's exact Z[1/sqrt2] amplitudes (no tolerance anywhere
    in the construction);
  * its spectrum, the number of DISTINCT eigenvalues, and the largest
    degeneracy -- i.e. whether the sector supports dark superpositions;
  * dim K(x), the Krylov dimension of a single basis state, versus |S_w|:
    the graph only gives dim K(x) <= |S_w|;
  * the diagonal-ensemble effective dimension d_eff(x) = 1 / sum_a |<a|x>|^4
    and the long-time population distribution, which decides whether the
    unmonitored dynamics actually FILLS the sector (d_eff ~ |S_w|) or stays
    trapped (d_eff << |S_w|).  Under monitoring the answer is always "fills":
    M_w = |U_w|^2 is doubly stochastic, irreducible on S_w and aperiodic
    (<x|U|x> != 0 for every x, since both I and H have nonzero diagonal), so
    the monitored stationary state is exactly uniform on S_w.

Wall-language reading of U_w (derived in R8 sec.5).  In the bond basis the gate
at site i acts only on the pair of bonds (i, i+1); it is the identity when both
or neither carry a wall, and on the one-wall subspace -- basis ordered (wall at
bond i, wall at bond i+1) -- it is

      1/sqrt2 * [[-(-1)^c, 1], [1, (-1)^c]],    c = x_{i-1} = parity of the
                                                walls strictly left of bond i,

a Hadamard "coin" dressed by a Jordan-Wigner string.  After the JW transform the
string is absorbed and the gate is a two-mode number-conserving gate with
single-particle block H.  A number-conserving GAUSSIAN two-mode gate must act on
the doubly occupied state by det(u); here det(H) = -1 while the gate acts by
+1, so U is Gaussian times exp(i pi n_{i-1} n_i): free hard-core wall hopping
with a MAXIMAL nearest-neighbour interaction.  Rule 150 is therefore exactly
charge-resolved but not free -- gaussianity_defect() checks this numerically.
"""

from __future__ import annotations

import itertools
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.cycle import one_cycle_branches
from ..c150 import RULE, TUPLE, spin_from_walls, wall_number

SQRT2 = math.sqrt(2.0)


# --- sector basis ------------------------------------------------------------

def sector_states(N: int, w: int) -> List[int]:
    """
    Basis states of the wall-number-w shell of the obc0 chain, built directly
    from the bond side: choose which w of the N+1 bonds carry a wall and prefix
    XOR back to spins.  Cost C(N+1, w), never 2^N.
    """
    if w % 2:
        return []
    out = []
    for pos in itertools.combinations(range(N + 1), w):
        b = 0
        for p in pos:
            b |= 1 << p
        out.append(spin_from_walls(b, N))
    out.sort()
    return out


def sector_unitary(N: int, w: int, *, check: bool = True):
    """
    Exact U_w as a real numpy matrix, plus the state list.  Column x holds the
    amplitudes <y|U|x>; the engine returns a single branch (amps, m) with
    amplitude (a + b sqrt2) / 2^m.
    """
    states = sector_states(N, w)
    idx = {x: k for k, x in enumerate(states)}
    D = len(states)
    U = np.zeros((D, D), dtype=np.float64)
    for k, x in enumerate(states):
        branches = one_cycle_branches(x, N, TUPLE, "obc0")
        assert len(branches) == 1, "rule 150 is unitary: one Kraus branch"
        amps, m = branches[0]
        scale = 2.0 ** (-m)
        for y, (a, b) in amps.items():
            j = idx.get(y)
            assert j is not None, "one-cycle image left the wall shell (bug)"
            U[j, k] = (a + b * SQRT2) * scale
    if check:
        err = np.abs(U.T @ U - np.eye(D)).max()
        assert err < 1e-9, f"U_w not orthogonal: {err:.2e}"
    return U, states




# --- what the graph does not fix --------------------------------------------
#
# Eigenvalue clustering.  The degeneracies of U_w are EXACT and well separated
# from the rest of the spectrum: over the whole shell dataset the largest gap
# treated as a degeneracy is 8.8e-15 while the smallest gap treated as distinct
# is 4.8e-6.  A cluster tolerance of 1e-9 therefore has five orders of margin on
# the degenerate side and three on the distinct side.

CLUSTER_TOL = 1e-9


def _clusters(U: np.ndarray, tol: float = CLUSTER_TOL):
    """(eigenvalues sorted by phase, eigenvectors, list of (lo, hi) clusters)."""
    ev, evec = np.linalg.eig(U)
    order = np.argsort(np.angle(ev))
    ev, evec = ev[order], evec[:, order]
    bounds = [0]
    ph = np.angle(ev)
    for k in range(1, len(ev)):
        if ph[k] - ph[k - 1] >= tol:
            bounds.append(k)
    bounds.append(len(ev))
    return ev, evec, list(zip(bounds[:-1], bounds[1:]))


def spectral_gap_margin(U: np.ndarray, tol: float = CLUSTER_TOL) -> Tuple[float, float]:
    """(largest gap treated as a degeneracy, smallest gap treated as distinct)."""
    ph = np.sort(np.angle(np.linalg.eigvals(U)))
    g = np.diff(ph)
    deg = g[g < tol]
    dis = g[g >= tol]
    return (float(deg.max()) if deg.size else 0.0,
            float(dis.min()) if dis.size else float("inf"))


def reflection_matrix(N: int, states: List[int]) -> np.ndarray:
    """Site reflection i -> N-1-i as a permutation of the sector basis (the wall
    number is reflection invariant, so every sector is mapped to itself)."""
    idx = {x: k for k, x in enumerate(states)}
    D = len(states)
    P = np.zeros((D, D))
    for k, x in enumerate(states):
        r = 0
        for i in range(N):
            if (x >> i) & 1:
                r |= 1 << (N - 1 - i)
        P[idx[r], k] = 1.0
    return P


def spectral_portrait(N: int, w: int, *, seeds: Optional[List[int]] = None,
                      tol: float = CLUSTER_TOL) -> dict:
    """
    Everything the sector graph leaves open, for one wall-number shell.

    Returns, besides the bookkeeping:
      n_distinct                 number of distinct eigenvalues of U_w
      mult_hist                  {multiplicity: how many eigenvalues}
      mult_plus_one              dim ker(U_w - 1): stationary states in the shell
      dim_krylov                 dim K(x) for each seed = number of distinct
                                 eigenvalues the seed overlaps
      krylov_deficit             |S_w| - max_x dim K(x)  (dark directions)
      d_eff                      1 / Tr(rho_bar^2), rho_bar the Cesaro limit
      max_pop_over_uniform       max_y <y|rho_bar|y> * |S_w|  (1 == uniform)
      reflection                 "commutes" (N odd) or "conjugates U to U^-1"
    """
    U, states = sector_unitary(N, w)
    D = len(states)
    ev, evec, cl = _clusters(U, tol)
    mult = [hi - lo for lo, hi in cl]
    hist: Dict[int, int] = {}
    for m in mult:
        hist[m] = hist.get(m, 0) + 1

    P = reflection_matrix(N, states)
    comm = float(np.abs(P @ U - U @ P).max())
    conj = float(np.abs(P @ U @ P - U.T).max())
    refl = ("commutes" if comm < 1e-10 else
            ("conjugates U to U^-1" if conj < 1e-10 else "neither"))

    if seeds is None:
        seeds = sorted({0, D // 2, D - 1})
    dimk, d_eff, maxpop = [], [], []
    for s in seeds:
        wgt = np.abs(evec[s, :]) ** 2
        p = np.array([wgt[lo:hi].sum() for lo, hi in cl])
        p = p / p.sum()
        dimk.append(int(np.sum(p > 1e-12)))
        d_eff.append(float(1.0 / np.sum(p ** 2)))
        # rho_bar = sum_l P_l |x><x| P_l  =>  pop(y) = sum_l |<y|P_l|x>|^2
        pop = np.zeros(D)
        for (lo, hi), _ in zip(cl, p):
            V = evec[:, lo:hi]
            col = V @ (V.conj().T[:, s])
            pop += np.abs(col) ** 2
        pop = pop / pop.sum()
        maxpop.append(float(pop.max() * D))

    deg_gap, dis_gap = spectral_gap_margin(U, tol)
    return {
        "N": N, "w": w, "dim": D,
        "n_distinct": len(cl),
        "mult_hist": dict(sorted(hist.items())),
        "largest_multiplicity": max(mult),
        "mult_plus_one": int(sum(hi - lo for lo, hi in cl
                                 if abs(ev[lo] - 1.0) < 1e-9)),
        "mult_minus_one": int(sum(hi - lo for lo, hi in cl
                                  if abs(ev[lo] + 1.0) < 1e-9)),
        "seeds": seeds,
        "dim_krylov": dimk,
        "dim_krylov_over_dim": [k / D for k in dimk],
        "krylov_deficit": D - max(dimk),
        "sector_is_a_krylov_space": max(dimk) == D,
        "d_eff": d_eff,
        "d_eff_over_dim": [d / D for d in d_eff],
        "max_pop_over_uniform": maxpop,
        "reflection": refl,
        "degenerate_gap_max": deg_gap,
        "distinct_gap_min": dis_gap,
    }


def krylov_dim(N: int, w: int, seed: int, *, tol: float = CLUSTER_TOL) -> int:
    """dim K(x) = number of distinct eigenvalues of U_w that |x> overlaps."""
    U, states = sector_unitary(N, w)
    _, evec, cl = _clusters(U, tol)
    wgt = np.abs(evec[seed, :]) ** 2
    return int(sum(1 for lo, hi in cl if wgt[lo:hi].sum() > 1e-12 * wgt.sum()))


# --- the wall (bond) representation and the free comparison model ------------

def bond_cycle(N: int, *, string: bool = True,
               det_phase: bool = False) -> np.ndarray:
    """
    One cycle written on the N+1 BOND modes (dense, 2^(N+1) x 2^(N+1)).

    The gate at site i touches bonds i and i+1 only.  It is the identity when
    both bonds carry the same occupation, and on the one-wall subspace -- basis
    ordered (wall at bond i, wall at bond i+1) -- it is

        M(c) = 1/sqrt2 * [[-(-1)^c, 1], [1, (-1)^c]],     det M = -1,

    with c = x_{i-1} = the parity of the walls strictly left of bond i.  Two
    switches turn this into the free-fermion comparison model:

      string=False     drop the (-1)^c Jordan-Wigner string from the coin;
      det_phase=True   give the doubly occupied pair the phase det(M) = -1,
                       which is what a number-conserving Gaussian two-mode gate
                       with single-particle block M must do.

    Rule 150 is (string=True, det_phase=False); the Gaussian reference is
    (string=False, det_phase=True).  Only the latter has an additive many-body
    spectrum (see additivity_defect).
    """
    M = 1 << (N + 1)
    U = np.eye(M)
    for i in list(range(0, N, 2)) + list(range(1, N, 2)):
        L = np.zeros((M, M))
        bi_m, bp_m = 1 << i, 1 << (i + 1)
        for b in range(M):
            bi = (b >> i) & 1
            bp = (b >> (i + 1)) & 1
            if bi == bp:
                L[b, b] = -1.0 if (det_phase and bi == 1) else 1.0
                continue
            c = (bin(b & ((1 << i) - 1)).count("1") & 1) if string else 0
            s = (-1.0) ** c
            Mm = np.array([[-s, 1.0], [1.0, s]]) / SQRT2
            other = b ^ bi_m ^ bp_m
            if bi == 1:                       # column = |wall at bond i>
                L[b, b] += Mm[0, 0]
                L[other, b] += Mm[1, 0]
            else:                             # column = |wall at bond i+1>
                L[other, b] += Mm[0, 1]
                L[b, b] += Mm[1, 1]
        U = L @ U
    return U


def bond_weight_block(Ub: np.ndarray, N: int, w: int) -> np.ndarray:
    idx = [b for b in range(1 << (N + 1)) if bin(b).count("1") == w]
    return Ub[np.ix_(idx, idx)]


def bond_matches_engine(N: int) -> float:
    """
    max |<b'|U_bond|b> - <x'|U_engine|x>| over the even-weight bond sector,
    where x = spin_from_walls(b).  Zero to machine precision is the statement
    that the spin QCA IS the hard-core wall walk defined by bond_cycle.
    """
    Ub = bond_cycle(N)
    Us = _full_unitary(N)
    ev = [b for b in range(1 << (N + 1)) if bin(b).count("1") % 2 == 0]
    err = 0.0
    for b in ev:
        x = spin_from_walls(b, N)
        for b2 in ev:
            y = spin_from_walls(b2, N)
            err = max(err, abs(Ub[b2, b] - Us[y, x]))
    return err


def _circle_sort(z: np.ndarray, seam: float = 0.5) -> np.ndarray:
    """
    Sort unimodular numbers by angle measured from a generic seam.  Sorting by
    np.angle puts the seam at -1, where this model has genuine eigenvalues (and
    +1 is worse still: dim Fix(U) > 0 always), so a value straddling the cut
    would shift the whole comparison by one slot.  The offset moves the seam to
    a generic point of the circle.
    """
    ang = np.mod(np.angle(z) + seam, 2.0 * np.pi)
    return z[np.argsort(ang)]


def additivity_defect(N: int, *, string: bool = True, det_phase: bool = False,
                      w_max: int = 5) -> float:
    """
    max over w <= w_max of the largest phase mismatch between the spectrum of
    the w-wall block and the w-fold products of the ONE-wall block's
    eigenvalues.  Zero iff the model is free: a number-conserving Gaussian
    circuit acts on the w-particle sector as the w-th antisymmetric power of its
    single-particle block, so the w-particle eigenvalues are the products of w
    distinct single-particle eigenvalues (equivalently, eigenphases add).

    Compared as points on the unit circle -- see _circle_sort -- and reported as
    the largest |arg(z_k / z'_k)| in radians, which is branch-cut safe.
    """
    Ub = bond_cycle(N, string=string, det_phase=det_phase)
    one = np.linalg.eigvals(bond_weight_block(Ub, N, 1))
    worst = 0.0
    for w in range(2, min(w_max, N + 1) + 1):
        got = np.linalg.eigvals(bond_weight_block(Ub, N, w))
        pred = np.array([np.prod(one[list(c)])
                         for c in itertools.combinations(range(N + 1), w)])
        if got.shape != pred.shape:
            return float("nan")
        a, b = _circle_sort(got), _circle_sort(pred)
        worst = max(worst, float(np.abs(np.angle(a / b)).max()))
    return worst


def freeness_table(Ns: List[int], w_max: int = 5) -> List[dict]:
    """
    The four sign variants: rule 150, each defect removed alone, and both
    removed.  Only the last is free.
    """
    out = []
    for N in Ns:
        row = {"N": N, "bond_vs_engine": bond_matches_engine(N)}
        for tag, kw in (("rule150", dict(string=True, det_phase=False)),
                        ("no_string", dict(string=False, det_phase=False)),
                        ("det_phase_only", dict(string=True, det_phase=True)),
                        ("gaussian_ref", dict(string=False, det_phase=True))):
            row[tag] = additivity_defect(N, w_max=w_max, **kw)
        out.append(row)
    return out


def _full_unitary(N: int) -> np.ndarray:
    """Dense 2^N x 2^N one-cycle matrix in the spin basis (small N only)."""
    D = 1 << N
    U = np.zeros((D, D))
    for x in range(D):
        (amps, m), = one_cycle_branches(x, N, TUPLE, "obc0")
        sc = 2.0 ** (-m)
        for y, (a, b) in amps.items():
            U[y, x] = (a + b * SQRT2) * sc
    return U


# --- stationary states: Fix(U) = Fix(L_A) cap Fix(L_B) -----------------------

def _site_gate(N: int, i: int) -> np.ndarray:
    """The single-site gate at position i as a dense 2^N matrix."""
    D = 1 << N
    L = np.zeros((D, D))
    bit = 1 << i
    for x in range(D):
        left = (x >> (i - 1)) & 1 if i > 0 else 0
        right = (x >> (i + 1)) & 1 if i < N - 1 else 0
        if TUPLE[2 * left + right] == "I":
            L[x, x] = 1.0
        else:
            s = -1.0 if (x & bit) else 1.0
            L[x & ~bit, x] += 1.0 / SQRT2
            L[x | bit, x] += s / SQRT2
    return L


def half_layer(N: int, parity: int) -> np.ndarray:
    """L_A (parity 0, even sites) or L_B (parity 1, odd sites).  Both are
    products of commuting single-qubit involutions, hence Hermitian with
    L^2 = 1."""
    D = 1 << N
    L = np.eye(D)
    for i in range(parity, N, 2):
        L = _site_gate(N, i) @ L
    return L


def _fix_dim(M: np.ndarray, tol: float = 1e-9) -> int:
    D = M.shape[0]
    return D - int(np.linalg.matrix_rank(M - np.eye(D), tol=tol))


def fix_dim_full(N: int) -> int:
    """dim Fix(U) on the whole 2^N space (small N only)."""
    return _fix_dim(_full_unitary(N))


def fix_is_halflayer_intersection(N: int, tol: float = 1e-9) -> bool:
    """
    Is every stationary state of the cycle stationary under EACH half-layer?
    U = L_B L_A with L_A^2 = L_B^2 = 1, so U psi = psi iff L_A psi = L_B psi;
    the check is whether that already forces L_A psi = L_B psi = psi.
    """
    D = 1 << N
    LA, LB = half_layer(N, 0), half_layer(N, 1)
    stacked = np.vstack([LA - np.eye(D), LB - np.eye(D)])
    inter = D - int(np.linalg.matrix_rank(stacked, tol=tol))
    return inter == _fix_dim(LB @ LA, tol)


def involution_defect(N: int) -> dict:
    """
    Each half-layer is a real SYMMETRIC INVOLUTION, so the cycle is a product of
    two involutions, U = L_B L_A with L_A^2 = L_B^2 = 1.  Two consequences used
    in R8: L_A U L_A = U^{-1} (so the spectrum is closed under inversion, which
    for a real U is the same as conjugation, hence the +-theta pairing), and

        Fix(U) = [Fix(L_A) cap Fix(L_B)]  (+)  [Fix(-L_A) cap Fix(-L_B)] ,

    because on the 2-dimensional irreps of the dihedral algebra generated by the
    two involutions U has eigenvalues e^{+-i theta} != 1, and on the 1-dimensional
    ones U = L_B L_A = +1 exactly when the two signs agree.

    Returns the numerical defects, all of which are at machine precision.
    """
    LA, LB = half_layer(N, 0), half_layer(N, 1)
    D = 1 << N
    I = np.eye(D)
    U = LB @ LA
    return {
        "N": N,
        "LA_involution": float(np.abs(LA @ LA - I).max()),
        "LB_involution": float(np.abs(LB @ LB - I).max()),
        "LA_symmetric": float(np.abs(LA - LA.T).max()),
        "LB_symmetric": float(np.abs(LB - LB.T).max()),
        "LA_inverts_U": float(np.abs(LA @ U @ LA - U.T).max()),
        "matches_engine": float(np.abs(U - _full_unitary(N)).max()),
    }


def fix_pm_decomposition(N: int, tol: float = 1e-9) -> dict:
    """
    dim of the two 1-dimensional dihedral sectors and of Fix(U).  The (-1,-1)
    sector turns out to be EMPTY for every N computed, which is why
    Fix(U) = Fix(L_A) cap Fix(L_B) rather than merely containing it.
    """
    LA, LB = half_layer(N, 0), half_layer(N, 1)
    D = 1 << N
    I = np.eye(D)

    def kerdim(M):                      # domain dimension is the column count
        return M.shape[1] - int(np.linalg.matrix_rank(M, tol=tol))

    pp = kerdim(np.vstack([LA - I, LB - I]))
    mm = kerdim(np.vstack([LA + I, LB + I]))
    fu = kerdim(LB @ LA - I)
    return {"N": N, "dim_plus_plus": pp, "dim_minus_minus": mm,
            "dim_fix_U": fu, "adds_up": pp + mm == fu}


# --- how rich is the spectrum? -----------------------------------------------

def distinct_spectrum_count(N: int, tol: float = CLUSTER_TOL) -> int:
    """
    Number of DISTINCT eigenvalues of the one-cycle operator on the whole 2^N
    space.  Exponentially degenerate, and exactly so: see
    distinct_spectrum_closed_form.
    """
    ph = np.sort(np.angle(np.linalg.eigvals(_full_unitary(N))))
    return 1 + int((np.diff(ph) > tol).sum())


def distinct_spectrum_closed_form(N: int) -> int:
    """
    Exact law, verified for N = 4..14 (out of sample at N = 14):

        m = #{even sites} = ceil(N/2)
        N even:  3^m
        N odd :  (3^m + (-1)^m) / 2

    So the number of distinct eigenvalues grows like 3^(N/2) = (sqrt3)^N, while
    the Hilbert space grows like 2^N: the distinct fraction decays as
    (sqrt3/2)^N = 0.866^N.  A spectrum that degenerate is not something a
    chaotic Floquet operator does -- see R8 sec.7.
    """
    m = len(range(0, N, 2))
    if N % 2 == 0:
        return 3 ** m
    return (3 ** m + (-1) ** m) // 2


def universal_spectrum(N: int, tol: float = CLUSTER_TOL) -> np.ndarray:
    """The distinct eigenphases of the full-space operator, ascending."""
    ph = np.sort(np.angle(np.linalg.eigvals(_full_unitary(N))))
    keep: List[float] = []
    for p in ph:
        if not keep or p - keep[-1] >= tol:
            keep.append(float(p))
    return np.array(keep)


def universal_spectrum_check(N: int, tol: float = CLUSTER_TOL) -> dict:
    """
    Every wall shell's spectrum is a SUBSET of one universal set of
    distinct_spectrum_closed_form(N) angles, and the largest shell realises all
    of it.  So the spectral problem is not a random point process at all: there
    is one alphabet of 3^m angles and each shell picks a subset with
    multiplicities.  See R8 sec.7.
    """
    import itertools as _it
    from math import comb

    full = universal_spectrum(N, tol)
    rows = []
    for w in range(0, N + 2, 2):
        D = comb(N + 1, w)
        if D < 2:
            continue
        U, _ = sector_unitary(N, w)
        ph = np.sort(np.angle(np.linalg.eigvals(U)))
        sh: List[float] = []
        for p in ph:
            if not sh or p - sh[-1] >= tol:
                sh.append(float(p))
        sh_arr = np.array(sh)
        idx = np.clip(np.searchsorted(full, sh_arr), 1, len(full) - 1)
        dev = float(np.minimum(np.abs(full[idx] - sh_arr),
                               np.abs(full[idx - 1] - sh_arr)).max())
        rows.append({"w": w, "dim": D, "n_distinct": len(sh),
                     "is_subset": dev < tol, "max_deviation": dev,
                     "coverage": len(sh) / len(full)})
    d_max_w = max(rows, key=lambda r: r["dim"])["w"]
    return {
        "N": N,
        "universal_size": len(full),
        "closed_form": distinct_spectrum_closed_form(N),
        "all_shells_are_subsets": all(r["is_subset"] for r in rows),
        "largest_shell_w": d_max_w,
        "largest_shell_covers_all":
            next(r["coverage"] for r in rows if r["w"] == d_max_w) == 1.0,
        "shells": rows,
    }


# --- entanglement entropy, and when it is well defined -----------------------

def half_chain_entropy(vec_full: np.ndarray, N: int) -> float:
    """von Neumann entropy of the half-chain cut, sites 0..N//2-1 versus the
    rest.  vec_full is a normalised amplitude vector on the full 2^N space."""
    NA = N // 2
    NB = N - NA
    M = vec_full.reshape(2 ** NB, 2 ** NA)
    s = np.linalg.svd(M, compute_uv=False)
    p = s ** 2
    p = p[p > 1e-24]
    return float(-np.sum(p * np.log(p)))


def eigenspace_ee_spread(N: int, w: int, *, tol: float = CLUSTER_TOL,
                         max_clusters: int = 6, seed: int = 0) -> dict:
    """
    Show that the per-eigenstate entanglement entropy is NOT well defined on a
    degenerate eigenvalue: take two orthonormal bases of the same eigenspace
    (one from the eigensolver, one rotated by a random orthogonal matrix) and
    compare the entropies.  On a nondegenerate eigenvalue the two agree.

    This matters for the HSF eigendata, where 43%-74% of the stored entropies
    belong to states in a degenerate eigenspace (R8 sec.8).
    """
    U, states = sector_unitary(N, w)
    ev, evec, cl = _clusters(U, tol)
    D = 1 << N
    rng = np.random.default_rng(seed)

    def entropies(B):
        out = []
        for k in range(B.shape[1]):
            f = np.zeros(D)
            f[states] = B[:, k]
            f /= np.linalg.norm(f)
            out.append(half_chain_entropy(f, N))
        return np.array(out)

    deg, nondeg = [], []
    for lo, hi in cl:
        m = hi - lo
        V = np.real(evec[:, lo:hi])
        Q, _ = np.linalg.qr(V)
        R, _ = np.linalg.qr(rng.standard_normal((m, m)))
        e1, e2 = entropies(Q), entropies(Q @ R)
        spread = float(np.abs(np.sort(e1) - np.sort(e2)).max())
        (deg if m > 1 else nondeg).append(spread)
        if len(deg) >= max_clusters and len(nondeg) >= max_clusters:
            break
    return {
        "N": N, "w": w,
        "max_spread_degenerate": max(deg) if deg else 0.0,
        "max_spread_nondegenerate": max(nondeg) if nondeg else 0.0,
        "n_degenerate_clusters_tested": len(deg),
        "n_nondegenerate_clusters_tested": len(nondeg),
    }
