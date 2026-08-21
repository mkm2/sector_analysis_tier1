r"""
The asymptotic manifold of the channel, split into its two factors  (Report R22).

R22 sec.6 listed, as a limitation, that the support graph gives the DIMENSION of
an enclosure but cannot say how that dimension divides between a decoherence-free
factor and a mixing factor.  This module settles it exactly.  By the structure
theorem of Baumgartner-Narnhofer and Albert-Jiang every state converges into

    rho(inf)  =  (+)_k  p_k  rho_k (x) sigma_k ,

with `sigma_k` a FIXED full-rank state on an m_k-dimensional factor -- whatever
is written there is erased -- and `rho_k` an arbitrary state on an n_k-dimensional
factor evolving unitarily, rho_k -> U_k rho_k U_k^dagger.  So n_k counts qubits
that survive forever and m_k counts dimensions that are junk.

METHOD (exact; no time-stepping, no sampling).

 1. S = the dense one-cycle superoperator (`peripheral.full_channel_superoperator`).
 2. The Cesaro projector at eigenvalue 1, built from the left/right kernels of
    S - 1, applied to the maximally mixed state gives rho_inf, whose support IS
    the asymptotic space:  D = dim supp rho_inf = sum_k n_k m_k.
 3. Restrict the channel to that support.  It is invariant, and the restricted
    channel now has a FULL RANK fixed state, which is what makes the next step
    legitimate: with a faithful fixed state the PERIPHERAL eigenoperators of the
    dual map (|lambda| = 1) close into a *-algebra
        A  =  (+)_k  M_{n_k} (x) 1_{m_k},
    on which the dual map acts as an automorphism.  (The FIXED points alone are
    too few whenever U_k is non-trivial: for any unitary rule they give only the
    commutant of U, which is why this module uses the peripheral span.)
 4. Read off dim A = sum_k n_k^2, split the blocks with the centre Z(A) = A /\ A',
    and get n_k from the rank of the block subalgebra, m_k from the block support.

Two generic elements generate a matrix algebra, so A' is obtained as the
commutant of two random elements of A and, by the bicommutant theorem, Z as the
commutant of two random elements of each.  That keeps every linear system at
D^2 x D^2 instead of the (dim A)-fold stack, which is what makes N = 5 reachable.

Cost is the 4^N superoperator, so N <= 5 is comfortable and N = 6 is the edge.

CLI
    python -m qca_fragmentation.quantum.asymptotic --rules 232,91,189,19 --N 4
    python -m qca_fragmentation.quantum.asymptotic --table
    python -m qca_fragmentation.quantum.asymptotic --census 4
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import results_io
from ..core import rules as rules_mod
from ..core.cycle import succ
from . import peripheral as PE

BC_DEFAULT = "obc0"
TOL = 1e-7

CENSUS_PATH = os.path.join(results_io.REPO_ROOT, "analytics",
                           "asymptotic_blocks_{bc}_N{N}.json")

#: The rules R22 sec.6 tabulates: the two pure cases, the mixed case, and the
#: two rules sec.4 had classified as multistable from populations alone.
R22_BLOCK_ROWS: Tuple[Tuple[int, int], ...] = (
    (232, 4), (91, 4), (189, 4), (19, 4), (203, 4), (36, 4), (150, 4),
)


@dataclass
class Block:
    n: int                     #: decoherence-free dimension (qubits live here)
    m: int                     #: mixing dimension (erased and replaced by sigma_k)
    support: int               #: n * m


@dataclass
class Asymptotics:
    rule: int
    word: str
    N: int
    bc: str
    dim: int                   #: 2^N
    dim_asym: int              #: D = sum n_k m_k
    dim_alg: int               #: dim A = sum n_k^2
    dim_fix: int               #: dim of the fixed points of Phi-dagger
    blocks: List[Block] = field(default_factory=list)
    consistent: bool = True

    @property
    def n_blocks(self) -> int:
        return len(self.blocks)

    @property
    def logical_qubits(self) -> float:
        """log2 of the largest decoherence-free factor."""
        return max((np.log2(b.n) for b in self.blocks), default=0.0)

    @property
    def is_purely_classical(self) -> bool:
        return all(b.n == 1 for b in self.blocks)

    @property
    def is_purely_coherent(self) -> bool:
        return all(b.m == 1 for b in self.blocks)

    def describe(self) -> str:
        return " (+) ".join(f"M_{b.n}(x)sigma_{b.m}" for b in self.blocks) or "-"


# --- linear algebra helpers -------------------------------------------------

def nullspace(A: np.ndarray, tol: float = TOL) -> np.ndarray:
    """Orthonormal basis (columns) of ker A.  Economy SVD unless underdetermined."""
    full = A.shape[0] < A.shape[1]
    _, s, vh = np.linalg.svd(A, full_matrices=full)
    smax = s[0] if s.size else 1.0
    k = int(np.sum(s > tol * max(smax, 1.0)))
    return vh[k:].conj().T


def cesaro_projector(S: np.ndarray, tol: float = TOL) -> Tuple[np.ndarray, int]:
    """Spectral projector of S at eigenvalue 1, and its rank."""
    n = S.shape[0]
    Vr = nullspace(S - np.eye(n), tol)
    Vl = nullspace(S.T - np.eye(n), tol)
    G = Vl.conj().T @ Vr
    return Vr @ np.linalg.solve(G, Vl.conj().T), Vr.shape[1]


def peripheral_span(M: np.ndarray, tol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """Orthonormal columns spanning the |lambda| = 1 invariant subspace of M.

    Via a SORTED SCHUR form, not eigenvectors.  `np.linalg.eig` on a heavily
    degenerate peripheral spectrum returns nearly parallel eigenvectors -- for
    the unitary rule W51 at N = 5 all 1024 eigenvalues sit on the circle but the
    eigenvector matrix has numerical rank 1001 -- whereas the Schur basis of an
    invariant subspace is orthonormal by construction.
    """
    from scipy.linalg import schur
    T, Z, sdim = schur(M, output="complex",
                       sort=lambda z: abs(abs(z) - 1.0) < tol)
    return Z[:, :sdim], np.diag(T)[:sdim]


def _hermitian_samples(basis: np.ndarray, D: int, rng, k: int = 2) -> List[np.ndarray]:
    out = []
    for _ in range(k):
        c = rng.normal(size=basis.shape[1]) + 1j * rng.normal(size=basis.shape[1])
        M = (basis @ c).reshape(D, D)
        out.append((M + M.conj().T) / 2)
        out.append(1j * (M - M.conj().T) / 2)
    return out


def _commutant(ops: Sequence[np.ndarray], D: int, tol: float = TOL) -> np.ndarray:
    rows = [np.kron(M, np.eye(D)) - np.kron(np.eye(D), M.T) for M in ops]
    return nullspace(np.vstack(rows), tol)


# --- the decomposition ------------------------------------------------------

def decompose(rule: int, N: int, bc: str = BC_DEFAULT, *,
              seed: int = 0, tol: float = TOL) -> Asymptotics:
    """Exact  (+)_k M_{n_k} (x) sigma_{m_k}  split of the asymptotic manifold."""
    t = rules_mod.wolfram_to_tuple(rule)
    d = 1 << N
    S = PE.full_channel_superoperator(N, t, bc)
    P, dim_fix = cesaro_projector(S, tol)

    rho = (P @ (np.eye(d).reshape(-1) / d)).reshape(d, d)
    rho = (rho + rho.conj().T) / 2
    w, U = np.linalg.eigh(rho)
    W = U[:, w > 1e-9 * max(w.max(), 1e-30)]
    D = W.shape[1]

    Sr = np.kron(W.conj().T, W.T) @ S @ np.kron(W, W.conj())
    Abas, _ = peripheral_span(Sr.conj().T)
    dim_alg = Abas.shape[1]

    rng = np.random.default_rng(seed)
    gens = _hermitian_samples(Abas, D, rng)
    Apr = _commutant(gens, D, tol)                       # A'
    Zb = _commutant(gens + _hermitian_samples(Apr, D, rng), D, tol)   # Z(A)

    blocks: List[Block] = []
    if Zb.shape[1]:
        Zc = _hermitian_samples(Zb, D, rng, k=1)[0]
        ev, EV = np.linalg.eigh(Zc)
        groups, cur = [], [0]
        for i in range(1, D):
            if abs(ev[i] - ev[i - 1]) < 1e-6:
                cur.append(i)
            else:
                groups.append(cur)
                cur = [i]
        groups.append(cur)
        for gidx in groups:
            Q = EV[:, gidx]
            nm = Q.shape[1]
            V = np.array([(Q.conj().T @ Abas[:, j].reshape(D, D) @ Q).reshape(-1)
                          for j in range(dim_alg)])
            n = int(round(np.sqrt(int(np.linalg.matrix_rank(V, tol=1e-6)))))
            blocks.append(Block(n=n, m=nm // n if n else 0, support=nm))

    res = Asymptotics(rule=rule, word="".join(t), N=N, bc=bc, dim=d, dim_asym=D,
                      dim_alg=dim_alg, dim_fix=dim_fix, blocks=blocks)
    res.consistent = (sum(b.support for b in blocks) == D and
                      sum(b.n * b.n for b in blocks) == dim_alg and
                      all(b.n * b.m == b.support for b in blocks))
    return res


def census(N: int, bc: str = BC_DEFAULT, rules: Optional[Sequence[int]] = None,
           verbose: bool = True) -> Dict:
    """All 256 rules at one N."""
    out: Dict[str, Dict] = {}
    for r in (range(256) if rules is None else rules):
        z = decompose(r, N, bc)
        out[str(r)] = dict(word=z.word, dim_asym=z.dim_asym, dim_alg=z.dim_alg,
                           dim_fix=z.dim_fix, consistent=z.consistent,
                           blocks=[[b.n, b.m] for b in z.blocks])
        if verbose and r % 32 == 31:
            print(f"  ... {r + 1}/256", flush=True)
    return {"bc": bc, "N": N, "rules": out}


def write_census(d: Dict, path: Optional[str] = None) -> str:
    path = path or CENSUS_PATH.format(bc=d["bc"], N=d["N"])
    with open(path, "w") as fh:
        json.dump(d, fh, indent=1, sort_keys=True)
    return path


def load_census(N: int, bc: str = BC_DEFAULT) -> Optional[Dict]:
    path = CENSUS_PATH.format(bc=bc, N=N)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


# --- the two algebras themselves: C and F -----------------------------------

def kraus_operators(rule: int, N: int, bc: str = BC_DEFAULT) -> List[np.ndarray]:
    """Dense Kraus operators of one full brick-wall cycle.

    One per choice of jump/no-jump at every reset site, so the count is
    2^(number of sites whose gate can fire), and each is a product of the
    per-site operators in the project's even-then-odd order.
    """
    from ..core.cycle import even_sites, neighbor_bits, odd_sites
    t = rules_mod.wolfram_to_tuple(rule)
    dim = 1 << N
    H = (1 / np.sqrt(2.0)) * np.array([[1.0, 1.0], [1.0, -1.0]])

    def site(i: int) -> List[np.ndarray]:
        A0 = np.zeros((dim, dim))
        A1 = np.zeros((dim, dim))
        jumps = False
        bit = 1 << i
        for x in range(dim):
            m, n = neighbor_bits(x, i, N, bc)
            s = t[2 * m + n]
            xi = (x >> i) & 1
            x0, x1 = x & ~bit, x | bit
            if s == "I":
                A0[x, x] += 1.0
            elif s == "V":
                A0[x0, x] += H[0, xi]
                A0[x1, x] += H[1, xi]
            elif s == "D":
                if xi == 0:
                    A0[x, x] += 1.0
                else:
                    A1[x0, x] += 1.0
                    jumps = True
            else:                                  # "E"
                if xi == 1:
                    A0[x, x] += 1.0
                else:
                    A1[x1, x] += 1.0
                    jumps = True
        return [A0, A1] if jumps else [A0]

    ops = [np.eye(dim)]
    for layer in (even_sites(N), odd_sites(N)):
        for i in layer:
            ops = [A @ B for A in site(i) for B in ops]
    return [K for K in ops if np.abs(K).max() > 1e-12]


def commutant_basis(rule: int, N: int, bc: str = BC_DEFAULT,
                    tol: float = TOL) -> np.ndarray:
    """Basis (columns, vectorised) of C = {A : [A, K_j] = 0 for all j}.

    Intersected one Kraus operator at a time, so the largest matrix ever formed
    is d^2 x d^2.  Stacking all of them at once is what runs out of memory: with
    a reset at every site the stack is 2^N * d^2 rows.
    """
    dim = 1 << N
    B = np.eye(dim * dim)
    for K in kraus_operators(rule, N, bc):
        M = (np.kron(K, np.eye(dim)) - np.kron(np.eye(dim), K.T)) @ B
        B = B @ nullspace(M, tol)
        if B.shape[1] == 0:
            break
    return B


def conserved_basis(rule: int, N: int, bc: str = BC_DEFAULT,
                    tol: float = TOL) -> np.ndarray:
    """Basis of F = {A : Phi-dagger(A) = A}, the conserved quantities."""
    t = rules_mod.wolfram_to_tuple(rule)
    S = PE.full_channel_superoperator(N, t, bc)
    return nullspace(S.conj().T - np.eye(S.shape[0]), tol)


def diagonal_dimension(basis: np.ndarray, dim: int, tol: float = 1e-7) -> int:
    """dim of the subspace of span(basis) consisting of DIAGONAL operators.

    This is n_wcc for C and n_rec for F (R22 Props. 3 and 4).
    """
    if basis.shape[1] == 0:
        return 0
    off = [i * dim + j for i in range(dim) for j in range(dim) if i != j]
    return nullspace(basis[off, :], tol).shape[1]


def algebra_report(rule: int, N: int, bc: str = BC_DEFAULT) -> Dict:
    """dim C, dim F and their diagonal parts, next to the block structure."""
    dim = 1 << N
    C = commutant_basis(rule, N, bc)
    F = conserved_basis(rule, N, bc)
    z = decompose(rule, N, bc)
    PF = F @ F.conj().T
    inside = max((float(np.linalg.norm(PF @ C[:, j] - C[:, j]))
                  for j in range(C.shape[1])), default=0.0)
    return dict(rule=rule, word=z.word, N=N, bc=bc,
                dim_C=C.shape[1], dim_F=F.shape[1],
                n_wcc=diagonal_dimension(C, dim),
                n_rec=diagonal_dimension(F, dim),
                dim_asym=z.dim_asym, dim_alg=z.dim_alg, dim_fix=z.dim_fix,
                blocks=[(b.n, b.m) for b in z.blocks],
                logical_qubits=z.logical_qubits,
                C_inside_F=inside)


# --- the code space, and whether it protects anything -----------------------

PAULI = {
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "Y": np.array([[0.0, -1j], [1j, 0.0]], dtype=complex),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}


def code_space(rule: int, N: int, bc: str = BC_DEFAULT) -> np.ndarray:
    """Isometry onto the asymptotic support -- the subspace the channel keeps."""
    t = rules_mod.wolfram_to_tuple(rule)
    d = 1 << N
    S = PE.full_channel_superoperator(N, t, bc)
    P, _ = cesaro_projector(S)
    rho = (P @ (np.eye(d).reshape(-1) / d)).reshape(d, d)
    rho = (rho + rho.conj().T) / 2
    w, U = np.linalg.eigh(rho)
    return U[:, w > 1e-9 * max(w.max(), 1e-30)]


def basis_spanned(rule: int, N: int, bc: str = BC_DEFAULT,
                  W: Optional[np.ndarray] = None) -> Tuple[bool, List[int]]:
    """Is the asymptotic support spanned by computational basis states, and which?"""
    W = code_space(rule, N, bc) if W is None else W
    diag = np.diag(W @ W.conj().T).real
    words = [x for x in range(1 << N) if diag[x] > 1 - 1e-8]
    return len(words) == W.shape[1], words


def asymptotic_unitary(rule: int, N: int, bc: str = BC_DEFAULT,
                       W: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float, int]:
    """The channel restricted to the recurrent space, if it is a unitary.

    Returns (A, isometry error, number of OTHER Kraus operators that do not
    vanish there).  A zero in the last slot means every jump operator annihilates
    the code space: the dissipation is transient-only and the asymptotic
    evolution is closed.
    """
    W = code_space(rule, N, bc) if W is None else W
    k = W.shape[1]
    best, err, live = None, np.inf, 0
    for K in kraus_operators(rule, N, bc):
        M = W.conj().T @ K @ W
        if np.linalg.norm(M) < 1e-10:
            continue
        live += 1
        e = float(np.abs(M.conj().T @ M - np.eye(k)).max())
        if e < err:
            best, err = M, e
    return best, err, max(live - 1, 0)


def knill_laflamme(W: np.ndarray, N: int, weight: int, tol: float = 1e-8) -> bool:
    """Does P E_a^dagger E_b P = c_{ab} P hold for all Pauli errors of this weight?"""
    from itertools import combinations, product
    k = W.shape[1]
    es = [np.eye(1 << N, dtype=complex)]
    for sites in combinations(range(N), weight):
        for ps in product("XYZ", repeat=weight):
            E = np.eye(1 << N, dtype=complex)
            for i, p in zip(sites, ps):
                op = np.array([[1.0]], dtype=complex)
                for q in range(N):
                    op = np.kron(PAULI[p] if q == i else np.eye(2), op)
                E = op @ E
            es.append(E)
    for a in range(len(es)):
        Ma = es[a].conj().T @ W
        for b in range(a, len(es)):
            M = Ma.conj().T @ (es[b] @ W)
            if np.abs(M - (np.trace(M) / k) * np.eye(k)).max() > tol:
                return False
    return True


def code_distance(rule: int, N: int, bc: str = BC_DEFAULT, max_weight: int = 1) -> int:
    """Largest w with the Knill-Laflamme condition satisfied at every weight <= w.

    Zero means a single Pauli error is not even DETECTABLE.  For a basis-spanned
    code space that is forced: the operators W^dagger Z_i W are diagonal in the
    codeword basis with entries +-1, distinct codewords being distinct bit
    strings, so every joint eigenspace of them is one-dimensional and no
    subspace of dimension two can have all of them act as scalars.  So no
    SUBCODE helps either -- the whole family is distance one against dephasing.
    """
    W = code_space(rule, N, bc)
    d = 0
    for w in range(1, max_weight + 1):
        if not knill_laflamme(W, N, w):
            return d
        d = w
    return d


# --- what the census is for: memory and error correction --------------------

def fates(rule: int, N: int, bc: str = BC_DEFAULT) -> List[int]:
    """fate[x] = representative of the cycle x lands in.  Deterministic rules only."""
    t = rules_mod.wolfram_to_tuple(rule)
    if "V" in t:
        raise ValueError(f"rule {rule} branches; fates() needs a functional graph")
    dim = 1 << N
    nxt = [succ(x, N, t, bc)[0] for x in range(dim)]
    fate = [-1] * dim
    state = [0] * dim                        # 0 unseen, 1 on stack, 2 done
    for x0 in range(dim):
        path, x = [], x0
        while state[x] == 0:
            state[x] = 1
            path.append(x)
            x = nxt[x]
        if state[x] == 1:                    # closed a fresh cycle
            i = path.index(x)
            rep = min(path[i:])
            for y in path[i:]:
                fate[y] = rep
                state[y] = 2
            path = path[:i]
        f = fate[x]
        for y in reversed(path):
            fate[y] = f
            state[y] = 2
    return fate


def autonomous_distance(rule: int, N: int, codeword: int,
                        bc: str = BC_DEFAULT, fate: Optional[List[int]] = None) -> int:
    """Largest w such that EVERY bit-flip error of weight <= w on `codeword` is
    undone by the automaton itself -- the distance of the measurement-free code."""
    from itertools import combinations
    fate = fate if fate is not None else fates(rule, N, bc)
    target = fate[codeword]
    d = 0
    for w in range(1, N + 1):
        for sites in combinations(range(N), w):
            e = 0
            for i in sites:
                e |= 1 << i
            if fate[codeword ^ e] != target:
                return d
        d = w
    return d


def memory_profile(rule: int, N: int, bc: str = BC_DEFAULT) -> Dict:
    """Basins and autonomous distances of the two repetition codewords."""
    fate = fates(rule, N, bc)
    dim = 1 << N
    zero, one = 0, dim - 1
    b0 = sum(1 for x in range(dim) if fate[x] == fate[zero])
    b1 = sum(1 for x in range(dim) if fate[x] == fate[one])
    return dict(rule=rule, N=N, bc=bc, n_attractors=len(set(fate)),
                basin_zero=b0, basin_one=b1, code_basin=(b0 + b1) / dim,
                distance_zero=autonomous_distance(rule, N, zero, bc, fate),
                distance_one=autonomous_distance(rule, N, one, bc, fate),
                repetition_distance=(N - 1) // 2)


# --- R24 sec.6: the dissipative pair is the unitary pair, one letter changed --

#: (dissipative rule, unitary parent, which bit the constraint forbids adjacent)
PARENTS = ((73, 201, 1), (109, 108, 0))


def constrained_states(N: int, forbid: int) -> List[int]:
    """Basis states with no two ADJACENT sites both equal to `forbid`."""
    out = []
    for x in range(1 << N):
        b = [(x >> i) & 1 for i in range(N)]
        if all(not (b[i] == b[i + 1] == forbid) for i in range(N - 1)):
            out.append(x)
    return out


def cycle_on(rule: int, N: int, states: Sequence[int],
             bc: str = BC_DEFAULT) -> Tuple[np.ndarray, int, int]:
    """<y|K|x> summed over Kraus labels, restricted to `states`.

    Returns the matrix, the number of amplitudes that leave the set, and the
    number of DISTINCT Kraus labels seen -- one label means no jump ever fired.
    """
    t = rules_mod.wolfram_to_tuple(rule)
    idx = {x: i for i, x in enumerate(states)}
    M = np.zeros((len(states), len(states)))
    escaped, labels = 0, set()
    for x in states:
        for lab, fl in PE._branch_amplitudes(x, N, t, bc).items():
            labels.add(lab)
            for y, a in fl.items():
                if y in idx:
                    M[idx[y], idx[x]] += a
                elif abs(a) > 1e-12:
                    escaped += 1
    return M, escaped, len(labels)


def parent_agreement(rule: int, N: int, bc: str = BC_DEFAULT) -> Dict:
    """R24 Prop. 3: on the constrained space the reset never fires, so the
    dissipative rule acts exactly as its unitary parent."""
    parent, forbid = next((p, f) for r, p, f in PARENTS if r == rule)
    S = constrained_states(N, forbid)
    Md, esc_d, lab_d = cycle_on(rule, N, S, bc)
    Mu, esc_u, _ = cycle_on(parent, N, S, bc)
    return dict(rule=rule, parent=parent, N=N, dim=len(S),
                difference=float(np.abs(Md - Mu).max()),
                escaped=esc_d + esc_u, kraus_labels=lab_d,
                unitarity=float(np.abs(Md.T @ Md - np.eye(len(S))).max()))


def code_growth(rule: int, Ns: Sequence[int], bc: str = BC_DEFAULT) -> List[Dict]:
    """Growth of the protected space, from the graph engine (R24 sec.2.3).

    Prop. 3 makes D the number of recurrent basis states, so this needs no
    superoperator and reaches N ~ 18.  For W73/W109 the three columns carry
    three different exponents: n_rec at the supergolden 1.4656, d_max at the
    golden 1.6180, and D at the tribonacci constant 1.8393 -- the total
    decoherence-free space grows faster than any single enclosure because the
    coherences BETWEEN enclosures survive.
    """
    from ..graph import scc as _scc, wcc as _wcc
    t = rules_mod.wolfram_to_tuple(rule)
    out = []
    for N in Ns:
        g = _scc.analyze(rule, N, bc, t, detect_ergodic=False)
        w = _wcc.weak_components(rule, N, bc, t)
        out.append(dict(N=N, n_wcc=w.n_wcc, n_rec=g.n_recurrent,
                        D=sum(g.sizes_recurrent), d_max=max(g.sizes_recurrent)))
    return out


# --- CLI --------------------------------------------------------------------

def _print_rows(rows: Sequence[Asymptotics]) -> None:
    print(f"{'rule':>4} {'word':>5} {'N':>2} {'D':>4} {'dimA':>5} {'dimFix':>6} "
          f"{'ok':>3}  blocks")
    for z in rows:
        print(f"{z.rule:>4} {z.word:>5} {z.N:>2} {z.dim_asym:>4} {z.dim_alg:>5} "
              f"{z.dim_fix:>6} {'y' if z.consistent else 'N':>3}  {z.describe()}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--rules", default="232,91,189,19,203,36")
    ap.add_argument("--N", type=int, default=4)
    ap.add_argument("--bc", default=BC_DEFAULT)
    ap.add_argument("--table", action="store_true", help="the R22 sec.6 table")
    ap.add_argument("--census", type=int, metavar="N", help="all 256 rules at N")
    ap.add_argument("--memory", default="", metavar="RULE",
                    help="basins and autonomous distance of a V-free rule")
    a = ap.parse_args(argv)

    if a.census:
        d = census(a.census, a.bc)
        print("wrote", write_census(d))
        return 0
    if a.memory:
        r = int(a.memory)
        print(f"{'N':>3} {'n_att':>7} {'basin(0^N)':>11} {'basin(1^N)':>11} "
              f"{'code basin':>11} {'d(0^N)':>7} {'d(1^N)':>7} {'rep d':>6}")
        for N in range(4, 17):
            p = memory_profile(r, N, a.bc)
            print(f"{N:>3} {p['n_attractors']:>7} {p['basin_zero']:>11} "
                  f"{p['basin_one']:>11} {100 * p['code_basin']:>10.3f}% "
                  f"{p['distance_zero']:>7} {p['distance_one']:>7} "
                  f"{p['repetition_distance']:>6}")
        return 0

    rows = ([decompose(r, N, a.bc) for r, N in R22_BLOCK_ROWS] if a.table else
            [decompose(int(r), a.N, a.bc) for r in a.rules.split(",")])
    _print_rows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
