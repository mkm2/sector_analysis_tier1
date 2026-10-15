"""
R32 step 9: block-first spectra beyond the dense-sector limit (relevant cases only).

The production pipeline (HSF/generate_spectral_data_v2.jl) diagonalises each Krylov
sector densely and resolves the gate-algebra blocks inside it, so its memory is set
by the largest SECTOR.  Here an exactly known symmetry splits the space first, by a
sparse construction, and only the pieces are treated densely:

  ring (pbc, even N):  T2-momentum j first (orbit basis, ~2^N / (N/2) states), then the
                       half step Omega_T = T1 L_A inside momentum j.  The level variable
                       is its eigenphase phi, as in the production pipeline.
  open chain, one-sided rules:  the boundary tower.  For 153, the projectors
                       Q_j = prod_{s<j} P+_s(H) * P-_j(H) commute with U (verified,
                       spectral tests), so block j is |h+>^j |h-> (x) (free sites), with
                       |h+->  the eigenvectors of the Hadamard.
                       The largest block (j = 0) is built in that product basis.

Only eigenvalues are computed (statistics), never eigenvectors:
  * unitary complex blocks: Cayley transform K = i (1 - A)(1 + A)^-1 (A = e^{i a} Omega_j)
    is Hermitian with eigenvalues tan(psi / 2);
  * real orthogonal blocks: cos(theta) from eigvalsh((U + U^T) / 2) (half spectrum).
Not resolved here (stated in the report): the gate-algebra search inside a momentum
sector.  At N <= 14 it found only momentum sectors plus ONE singlet for these rules;
a single extra level in ~8000 does not change <r> at the quoted precision.

Site convention: site 0 = least significant bit (as core.cycle); U = L_B L_A with
L_A = the gates on even sites, applied first.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg  # noqa: F401  (sp.linalg.norm)

from ..core import rules

SQ2 = 1.0 / np.sqrt(2.0)


def _bits(x, i, N):
    return (x >> (i % N)) & 1


def gate_sparse(w: int, N: int, i: int, bc: str, gate: str = "H") -> sp.csr_matrix:
    """The controlled gate on site i, as a sparse 2^N x 2^N matrix (columns = inputs)."""
    t = rules.wolfram_to_tuple(w)
    fire_pat = np.array([s == "V" for s in t])          # index 2 m + n
    n = 1 << N
    x = np.arange(n, dtype=np.int64)
    if bc == "pbc":
        m, r = _bits(x, i - 1, N), _bits(x, i + 1, N)
    else:                                                 # obc0: ghost neighbours fixed to 0
        m = _bits(x, i - 1, N) if i - 1 >= 0 else np.zeros_like(x)
        r = _bits(x, i + 1, N) if i + 1 < N else np.zeros_like(x)
    fire = fire_pat[2 * m + r]
    b = _bits(x, i, N)
    xf = x[fire]; bf = b[fire]
    flip = xf ^ (1 << i)
    if gate == "H":    # H|b> = (|0> + (-1)^b |1>)/sqrt2 ; column xf -> rows xf (same bit) and flip
        same = np.where(bf == 0, SQ2, -SQ2)
        rows = np.concatenate([xf, flip, x[~fire]])
        cols = np.concatenate([xf, xf, x[~fire]])
        data = np.concatenate([same, np.full(xf.size, SQ2), np.ones((~fire).sum())])
    else:              # X
        rows = np.concatenate([flip, x[~fire]]); cols = np.concatenate([xf, x[~fire]])
        data = np.ones(rows.size)
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))


def layer_sparse(w, N, parity, bc, gate="H"):
    L = sp.identity(1 << N, format="csr")
    for i in range(parity, N, 2):
        L = gate_sparse(w, N, i, bc, gate) @ L
    return L.tocsr()


def rot(x, s, N):
    """Cyclic shift of the site labels by s (site i -> i + s)."""
    s %= N
    return ((x << s) | (x >> (N - s))) & ((1 << N) - 1)


def momentum_basis(N: int, j: int) -> sp.csc_matrix:
    """Orthonormal basis of the T2 eigenspace with eigenvalue exp(2 pi i j / (N/2)),
    T2 = shift by two sites, as a sparse 2^N x d_j isometry (one column per orbit)."""
    n, L = 1 << N, N // 2
    x = np.arange(n, dtype=np.int64)
    orb = [x]
    for k in range(1, L):
        orb.append(rot(x, 2 * k, N))
    orb = np.stack(orb)                                   # orb[k] = T2^k x
    rep = orb.min(axis=0)
    reps = np.unique(rep)
    # period p of each representative
    period = np.full(reps.size, L)
    for p in range(1, L):
        if L % p == 0:
            same = (orb[p][reps] == reps) & (period == L)
            period[same] = p
    t = np.exp(2j * np.pi * j / L)
    keep = (j * period) % L == 0                          # t^p = 1 needed
    reps, period = reps[keep], period[keep]
    rows, cols, data = [], [], []
    for c, (r0, p) in enumerate(zip(reps, period)):
        for k in range(p):
            rows.append(int(rot(r0, 2 * k, N))); cols.append(c)
            data.append(t ** (-k) / np.sqrt(p))          # T2 |psi> = t |psi>
    return sp.csc_matrix((np.array(data), (rows, cols)), shape=(n, reps.size))


def cayley_phases(A: np.ndarray, rng=None) -> np.ndarray:
    """Eigenphases of a unitary matrix A in (-pi, pi], from a Hermitian eigenvalue problem."""
    rng = rng or np.random.default_rng(1)
    a = rng.uniform(0, 2 * np.pi)
    B = np.exp(1j * a) * A
    d = B.shape[0]
    K = 1j * np.linalg.solve(np.eye(d) + B, np.eye(d) - B)
    K = (K + K.conj().T) / 2
    psi = 2 * np.arctan(np.linalg.eigvalsh(K))
    return np.angle(np.exp(1j * (psi - a)))


def ring_phases(w: int, N: int, gate: str = "H") -> dict:
    """phi per momentum j: eigenphases of Omega_T = T1 L_A restricted to momentum j."""
    n = 1 << N
    LA = layer_sparse(w, N, 0, "pbc", gate)
    x = np.arange(n, dtype=np.int64)
    T1 = sp.csr_matrix((np.ones(n), (rot(x, 1, N), x)), shape=(n, n))
    Om = (T1 @ LA).tocsr()
    out = {}
    for j in range(N // 2):
        P = momentum_basis(N, j)
        Oj = (P.conj().T @ (Om @ P)).toarray()
        # the momentum sector must be invariant under Omega_T
        resid = (sp.linalg.norm(Om @ P - P @ sp.csc_matrix(Oj)) if P.shape[1] < 3000 else None)
        # Krylov sectors inside momentum j: connected components of Omega_T,j
        from scipy.sparse.csgraph import connected_components
        ncomp, lab = connected_components(sp.csr_matrix(np.abs(Oj) > 1e-12), directed=False)
        comps = [np.flatnonzero(lab == c) for c in range(ncomp)]
        out[j] = {"phi": [cayley_phases(Oj[np.ix_(c, c)]) if c.size > 1 else np.angle(Oj[c, c])
                          for c in comps],
                  "dim": P.shape[1], "comp_dims": [int(c.size) for c in comps], "resid": resid}
        del Oj
    return out


def tower_block_153(N: int, j: int = 0) -> np.ndarray:
    """U of rule 153 (obc0) restricted to tower block j: sites < j in |h+>, site j in |h->
    (the eigenstates of H with eigenvalue +1 and -1),
    sites > j free (computational basis).  Returned dense, real, dimension 2^(N-1-j)."""
    w = 153
    LA, LB = layer_sparse(w, N, 0, "obc0"), layer_sparse(w, N, 1, "obc0")
    # eigenvectors of the Hadamard H = (X + Z)/sqrt2 (NOT |+->, which diagonalise X)
    c8, s8 = np.cos(np.pi / 8), np.sin(np.pi / 8)
    plus, minus = np.array([c8, s8]), np.array([-s8, c8])
    # product basis: kron over sites (site 0 = LSB = last factor in np.kron order)
    free = N - 1 - j
    fixed = np.ones(1)
    for s in range(j + 1):                        # sites 0..j, built LSB-last
        fixed = np.kron(minus if s == j else plus, fixed)
    # basis vector for free configuration y: |y>_{sites j+1..} (x) fixed
    d = 1 << free
    Bm = sp.kron(sp.identity(d, format="csr"), sp.csr_matrix(fixed.reshape(-1, 1)), format="csc")
    UB = LB @ (LA @ Bm)
    return (Bm.T @ UB).toarray()


H_PLUS = np.array([np.cos(np.pi / 8), np.sin(np.pi / 8)])     # H |h+> = +|h+>
H_MINUS = np.array([-np.sin(np.pi / 8), np.cos(np.pi / 8)])   # H |h-> = -|h->
KET0, KET1 = np.array([1.0, 0.0]), np.array([0.0, 1.0])

#: the largest tower block of the one-sided rules on the open chain (verified against
#: the pipeline's gate-algebra blocks at small N, tests): fixed single-site states
TOWER_BLOCKS = {
    153: lambda N: {0: H_MINUS},                     # left end: Q_0 = P-_0(H)
    60: lambda N: {0: KET1, N - 1: H_MINUS},         # sector "site 0 = 1", right-end tower
}


def fixed_site_block(w: int, N: int, fixed: dict, bc: str = "obc0") -> np.ndarray:
    """U restricted to the span of (fixed single-site states) (x) (computational basis of the
    other sites).  The span must be U-invariant; the caller checks orthogonality."""
    LA, LB = layer_sparse(w, N, 0, bc), layer_sparse(w, N, 1, bc)
    free = [i for i in range(N) if i not in fixed]
    d = 1 << len(free)
    # basis vectors as a sparse 2^N x d matrix, built factor by factor (site 0 = LSB)
    cols = sp.identity(1, format="csr")
    fi = 0
    for i in range(N):                                   # np.kron order: later = more significant
        f = fixed.get(i)
        fac = sp.csr_matrix(f.reshape(-1, 1)) if f is not None else sp.identity(2, format="csr")
        cols = sp.kron(fac, cols, format="csr")
    Bm = cols.tocsc()
    return (Bm.T @ (LB @ (LA @ Bm))).toarray()


def half_spectrum_from_real_orthogonal(Ub: np.ndarray) -> np.ndarray:
    """|theta| in [0, pi] of a real orthogonal matrix, from cos(theta) (eigenvalues only)."""
    C = (Ub + Ub.T) / 2
    c = np.clip(np.linalg.eigvalsh(C), -1, 1)
    return np.arccos(c)


OUT_DIR = None


def _out_dir():
    import os
    from .floquet import SPECTRA_DIR
    d = os.path.join(SPECTRA_DIR, "blockfirst")
    os.makedirs(d, exist_ok=True)
    return d


def run_ring(w: int, N: int) -> str:
    import os, time
    t0 = time.time()
    R = ring_phases(w, N)
    path = os.path.join(_out_dir(), f"ring_N{N}_W{w}.npz")
    arrs = {}
    for j, v in R.items():
        for c, ph in enumerate(v["phi"]):
            arrs[f"phi_j{j}_c{c}"] = np.atleast_1d(ph)
    np.savez(path, N=N, rule=w, bc="pbc", runtime=time.time() - t0,
             dims=np.array([R[j]["dim"] for j in sorted(R)]), **arrs)
    return path


def run_tower(w: int, N: int) -> str:
    import os, time
    t0 = time.time()
    Ub = fixed_site_block(w, N, TOWER_BLOCKS[w](N))
    orth = float(np.abs(Ub[:, :64].T @ Ub[:, :64] - np.eye(64)).max())   # invariance spot check
    theta = np.angle(np.linalg.eigvals(Ub))
    path = os.path.join(_out_dir(), f"tower_N{N}_W{w}.npz")
    np.savez(path, N=N, rule=w, bc="obc0", theta=theta, dim=Ub.shape[0], orth_check=orth,
             runtime=time.time() - t0)
    return path




# --- verification of the block-first assumptions -------------------------------------

def _components_in_eigenbasis(K: np.ndarray, K2: np.ndarray, tol: float = 1e-9):
    """Generic-K trick: diagonalise Hermitian K, express K2 in that eigenbasis and return the
    connected components (= invariant subspaces of the algebra generated by K, K2) and the
    smallest eigenvalue gap of K (the test needs a non-degenerate K)."""
    from scipy.sparse.csgraph import connected_components
    import scipy.linalg as sla
    ev, Q = sla.eigh(K, overwrite_a=True, driver="evr", check_finite=False)
    del K
    M = K2 @ Q
    M = Q.conj().T @ M
    nc, lab = connected_components(sp.csr_matrix(np.abs(M) > tol * np.abs(M).max()), directed=False)
    return nc, np.bincount(lab), float(np.diff(ev).min()) if ev.size > 1 else np.inf


def verify_ring_sector(w: int, N: int, j: int, seed: int = 7) -> dict:
    """
    Is momentum sector j irreducible under T2-invariant elements of the gate algebra?
    K = a S_A + b S_B + c {S_A, S_B} + e (U + U^dag) (S_A, S_B: sums of the gates of layer A,
    B; all terms Hermitian, commute with T2, lie in the gate algebra).  One component per Krylov sector => no hidden
    block (any block of the full symmetry algebra would also split this subalgebra).
    Also: invariance of the sector under Omega_T, by random vectors.
    """
    rng = np.random.default_rng(seed)
    n = 1 << N
    gates = [gate_sparse(w, N, i, "pbc") for i in range(N)]
    SA = sum(gates[i] for i in range(0, N, 2)); SB = sum(gates[i] for i in range(1, N, 2))
    P = momentum_basis(N, j)
    def restr(Op):
        return (P.conj().T @ (Op @ P)).toarray()
    a, b, c, e, a2, b2, c2, e2 = rng.uniform(0.5, 1.5, 8)
    A_, B_ = restr(SA), restr(SB)
    K = a * A_ + b * B_; K2 = a2 * A_ + b2 * B_
    AB = A_ @ B_                                   # P^H S_A S_B P = (P^H S_A P)(P^H S_B P): S_B keeps sector j
    del A_, B_
    AB += AB.conj().T
    K += c * AB; K2 += c2 * AB
    del AB
    # invariance of the sector under Omega_T (random vectors), and U_j = Omega_j^2 / t_j
    LA = layer_sparse(w, N, 0, "pbc"); x = np.arange(n, dtype=np.int64)
    T1 = sp.csr_matrix((np.ones(n), (rot(x, 1, N), x)), shape=(n, n)); Om = T1 @ LA
    Oj = restr(Om)
    UH = (Oj @ Oj) * np.exp(-2j * np.pi * j / (N // 2))
    UH += UH.conj().T                              # U + U^dag: in the gate algebra, T2-invariant
    K += e * UH; K2 += e2 * UH
    del UH
    nc, sizes, gap = _components_in_eigenbasis(K, K2)
    v = rng.normal(size=(P.shape[1], 3)) + 1j * rng.normal(size=(P.shape[1], 3))
    inv = float(np.abs(Om @ (P @ v) - P @ (Oj @ v)).max())
    # Krylov components of Omega_T,j for comparison
    from scipy.sparse.csgraph import connected_components
    nk, labk = connected_components(sp.csr_matrix(np.abs(Oj) > 1e-12), directed=False)
    return {"rule": w, "N": N, "j": j, "dim": P.shape[1], "n_components_gate_test": int(nc),
            "component_sizes": sorted(sizes.tolist(), reverse=True)[:6],
            "krylov_components": sorted(np.bincount(labk).tolist(), reverse=True)[:6],
            "K_min_gap": gap, "omega_invariance": inv}


def verify_tower_block(w: int, N: int, seed: int = 7) -> dict:
    """Is the largest tower block irreducible under the gate algebra?  Generic K = sum a_i g_i
    restricted to the block (the gates must leave it invariant: checked)."""
    rng = np.random.default_rng(seed)
    fixed = TOWER_BLOCKS[w](N)
    gates = [gate_sparse(w, N, i, "obc0") for i in range(N)]
    # block basis (as in fixed_site_block)
    cols = sp.identity(1, format="csr")
    for i in range(N):
        f = fixed.get(i)
        fac = sp.csr_matrix(f.reshape(-1, 1)) if f is not None else sp.identity(2, format="csr")
        cols = sp.kron(fac, cols, format="csr")
    Bm = cols.tocsc()
    a, b = rng.uniform(0.5, 1.5, N), rng.uniform(0.5, 1.5, N)
    Ksp = sum(a[i] * gates[i] for i in range(N)); K2sp = sum(b[i] * gates[i] for i in range(N))
    v = rng.normal(size=(Bm.shape[1], 3))
    inv = float(np.abs(Ksp @ (Bm @ v) - Bm @ ((Bm.T @ (Ksp @ Bm)) @ v)).max())   # gates keep the block
    K = (Bm.T @ (Ksp @ Bm)).toarray(); K2 = (Bm.T @ (K2sp @ Bm)).toarray()
    nc, sizes, gap = _components_in_eigenbasis(K, K2)
    return {"rule": w, "N": N, "dim": Bm.shape[1], "n_components": int(nc),
            "component_sizes": sorted(sizes.tolist(), reverse=True)[:6], "K_min_gap": gap,
            "gate_invariance": inv}


def run_verify(kind: str, w: int, N: int, j: int = 1) -> str:
    import json, os, time
    t0 = time.time()
    r = verify_ring_sector(w, N, j) if kind == "vring" else verify_tower_block(w, N)
    r["runtime"] = time.time() - t0
    path = os.path.join(_out_dir(), f"verify_{kind}_N{N}_W{w}" + (f"_j{j}" if kind == "vring" else "") + ".json")
    with open(path, "w") as fh:
        json.dump(r, fh, indent=1, default=float)
    return path


if __name__ == "__main__":
    import sys
    kind, w, N = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    if kind in ("vring", "vtower"):
        print(run_verify(kind, w, N, int(sys.argv[4]) if len(sys.argv) > 4 else 1), flush=True)
    else:
        print((run_ring if kind == "ring" else run_tower)(w, N), flush=True)
