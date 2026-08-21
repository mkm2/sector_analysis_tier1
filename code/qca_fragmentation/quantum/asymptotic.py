"""
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
