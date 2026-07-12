"""
Tier 1b: per-recurrent-class quantum structure (context Tier 1 sec.6).

For a recurrent class R (a terminal SCC, closed under the channel) we build the
exact restricted superoperator Phi_R on span(R) (x) span(R)*, cast to float, and
read its peripheral spectrum to classify the attractor:

  - unique eigenvalue 1, diagonal positive fixed point, no other peripheral
    eigenvalue                              -> MIXING (a pointer / dark state)
  - peripheral eigenvalues are roots of unity, all eigenoperators basis-diagonal
                                            -> CLASSICAL LIMIT CYCLE (period q)
  - eigenvalue-1 multiplicity > (# diagonal fixed operators), or any off-diagonal
    peripheral eigenoperator                -> COHERENCE-CARRYING (records d_k)

The restricted superoperator is assembled exactly from the labelled Kraus
branches: each surviving branch of |x> is tagged by the set of sites where the
jump operator A1 fired, which uniquely names the global Kraus operator K_b, so
  Phi_R[(y,y'),(x,x')] = sum_b <y|K_b|x> <y'|K_b|x'>
with all matrix elements exact elements of Z[1/sqrt2] cast to float only here.

Floating point is permitted in this module (Tier 1b), unlike the graph stage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..core.cycle import one_cycle_branches_labeled
from ..core.rules import Tuple4

SQRT2 = math.sqrt(2.0)
PERIPHERAL_TOL = 1e-8


@dataclass
class AttractorType:
    size: int
    kind: str                       # "mixing" | "limit_cycle" | "coherent"
    n_peripheral: int               # # eigenvalues with |lambda| ~ 1
    mult_one: int                   # multiplicity of eigenvalue 1
    period: Optional[int] = None    # limit-cycle period (roots of unity)
    d_k: int = 1                    # coherence dimension (>=2 => coherent)
    peripheral_eigs: List[complex] = field(default_factory=list)


def _branch_amplitudes(x: int, N: int, t: Tuple4, bc: str) -> Dict[frozenset, Dict[int, float]]:
    """{Kraus label -> {y: float amplitude <y|K_b|x>}} for input basis state x."""
    out: Dict[frozenset, Dict[int, float]] = {}
    for amps, m, lab in one_cycle_branches_labeled(x, N, t, bc):
        denom = float(1 << m)
        fl = {y: (a + b * SQRT2) / denom for y, (a, b) in amps.items()}
        # a label is unique per input; if it recurs, accumulate defensively
        if lab in out:
            for y, v in fl.items():
                out[lab][y] = out[lab].get(y, 0.0) + v
        else:
            out[lab] = fl
    return out


def restricted_superoperator(R: List[int], N: int, t: Tuple4, bc: str) -> np.ndarray:
    """Dense float Phi_R of dimension |R|^2 (row/col index = idx(y)*k + idx(y'))."""
    R = sorted(R)
    idx = {x: i for i, x in enumerate(R)}
    k = len(R)
    inR = idx.__contains__
    perx = {x: _branch_amplitudes(x, N, t, bc) for x in R}

    Phi = np.zeros((k * k, k * k))
    for x in R:
        dx = perx[x]
        ix = idx[x]
        for xp in R:
            dxp = perx[xp]
            col = ix * k + idx[xp]
            for lab, flx in dx.items():
                flxp = dxp.get(lab)
                if not flxp:
                    continue
                for y, ay in flx.items():
                    if not inR(y):
                        continue
                    riy = idx[y] * k
                    for yp, ayp in flxp.items():
                        if not inR(yp):
                            continue
                        Phi[riy + idx[yp], col] += ay * ayp
    return Phi


def geometric_multiplicity(A: np.ndarray, lam: complex, tol: float = 1e-7) -> int:
    """dim ker(A - lam I) via SVD (robust for non-normal superoperators).  This
    is the number of independent eigenoperators at lam -- the physically
    meaningful multiplicity, unlike the algebraic count which is inflated by the
    transient->recurrent Jordan structure."""
    n = A.shape[0]
    s = np.linalg.svd(A - lam * np.eye(n), compute_uv=False)
    return int(np.sum(s < tol * max(s[0], 1.0)))


def _null_space(A: np.ndarray, tol: float = 1e-7) -> np.ndarray:
    """Orthonormal basis of ker(A) (columns)."""
    u, s, vh = np.linalg.svd(A)
    ns = vh[np.sum(s > tol * max(s[0] if s.size else 1.0, 1.0)):]
    return ns.conj().T


def _all_diagonal(basis: np.ndarray, k: int, tol=1e-6) -> bool:
    """Are all operators spanned by `basis` (columns, length k^2) diagonal in R?"""
    for j in range(basis.shape[1]):
        M = basis[:, j].reshape(k, k)
        off = M - np.diag(np.diag(M))
        scale = np.abs(M).max() or 1.0
        if np.abs(off).max() / scale > tol:
            return False
    return True


def classify_attractor(R: List[int], N: int, t: Tuple4, bc: str,
                       max_size: int = 32) -> AttractorType:
    """Classify one recurrent class by its peripheral spectrum.  Classes larger
    than max_size are reported 'unresolved' (the dense |R|^2 x |R|^2 superoperator
    and its eigensolve become expensive; small attractors are the physically
    interesting case)."""
    k = len(R)
    if k == 1:
        return AttractorType(size=1, kind="mixing", n_peripheral=1, mult_one=1,
                             period=1, d_k=1, peripheral_eigs=[1.0])
    if k > max_size:
        return AttractorType(size=k, kind="unresolved", n_peripheral=0,
                             mult_one=0)

    Phi = restricted_superoperator(R, N, t, bc)
    vals = np.linalg.eigvals(Phi)
    mags = np.abs(vals)
    # distinct peripheral eigenvalues (cluster by value)
    peri_vals = [complex(v) for v in vals if mags_ok(v)]
    # geometric multiplicities
    mult_one = geometric_multiplicity(Phi, 1.0)
    # sum of geometric mults over the DISTINCT peripheral eigenvalues
    distinct = _distinct(peri_vals)
    n_peripheral = sum(geometric_multiplicity(Phi, lv) for lv in distinct)
    period = _estimate_period(distinct)

    ker1 = _null_space(Phi - np.eye(k * k))
    diag1 = _all_diagonal(ker1, k) if ker1.size else True

    if mult_one == 1 and len(distinct) == 1:      # only lambda = 1
        kind = "mixing"
        period = 1
        d_k = 1
    elif mult_one == 1 and diag1:                 # unique steady state, rotating
        kind = "limit_cycle"
        d_k = 1
    else:                                         # protected coherence in-class
        kind = "coherent"
        d_k = int(round(math.sqrt(mult_one)))
    return AttractorType(size=k, kind=kind, n_peripheral=n_peripheral,
                         mult_one=mult_one, period=period,
                         d_k=max(d_k, 1), peripheral_eigs=distinct)


def mags_ok(v) -> bool:
    return abs(v) > 1 - PERIPHERAL_TOL


def _distinct(vals: List[complex], tol=1e-5) -> List[complex]:
    out: List[complex] = []
    for v in vals:
        if not any(abs(v - u) < tol for u in out):
            out.append(v)
    return out


def full_channel_superoperator(N: int, t: Tuple4, bc: str) -> np.ndarray:
    """
    Dense 4^N x 4^N superoperator of the full one-cycle channel (validation only,
    N <= ~6).  Built from per-site Kraus A0,A1 as S = prod_sites (A0(x)A0* + A1 A1*).
    """
    from ..core.cycle import neighbor_bits, even_sites, odd_sites
    dim = 1 << N
    Hm = (1 / SQRT2) * np.array([[1., 1.], [1., -1.]])

    def site_kraus(i):
        A0 = np.zeros((dim, dim)); A1 = np.zeros((dim, dim)); use1 = False
        bit = 1 << i
        for x in range(dim):
            m, n = neighbor_bits(x, i, N, bc); s = t[2 * m + n]
            xi = (x >> i) & 1; x0 = x & ~bit; x1 = x | bit
            if s == "I":
                A0[x, x] += 1
            elif s == "V":
                if xi == 0:
                    A0[x0, x] += Hm[0, 0]; A0[x1, x] += Hm[1, 0]
                else:
                    A0[x0, x] += Hm[0, 1]; A0[x1, x] += Hm[1, 1]
            elif s == "D":
                if xi == 0:
                    A0[x, x] += 1
                else:
                    A1[x0, x] += 1; use1 = True
            elif s == "E":
                if xi == 1:
                    A0[x, x] += 1
                else:
                    A1[x1, x] += 1; use1 = True
        return A0, (A1 if use1 else None)

    S = np.eye(dim * dim)
    for layer in (even_sites(N), odd_sites(N)):
        for i in layer:
            A0, A1 = site_kraus(i)
            Si = np.kron(A0, A0.conj())
            if A1 is not None:
                Si = Si + np.kron(A1, A1.conj())
            S = Si @ S
    return S


def cesaro_rank(N: int, t: Tuple4, bc: str) -> int:
    """Exact dimension of the full channel's fixed-point space (geometric
    multiplicity of eigenvalue 1) -- the Cesaro projection rank.  N <= ~6."""
    S = full_channel_superoperator(N, t, bc)
    return geometric_multiplicity(S, 1.0)


def _estimate_period(eigs: List[complex]) -> int:
    """Smallest q such that all peripheral phases are q-th roots of unity."""
    phases = []
    for z in eigs:
        ang = math.atan2(z.imag, z.real) / (2 * math.pi)  # in (-0.5, 0.5]
        phases.append(ang % 1.0)
    for q in range(1, 64 + 1):
        if all(abs((p * q) - round(p * q)) < 1e-4 for p in phases):
            return q
    return 0
