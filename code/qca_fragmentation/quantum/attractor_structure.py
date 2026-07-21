"""
Is the channel's attractor SPANNED BY BASIS STATES?  (Tier 1b, structural test)

`peripheral.graph_faithfulness` answers a weaker question.  It asks whether
population leaks off the graph's recurrent set, starting from 1/d.  That test
misses the more common failure: a rule whose attractor is a genuine SUBSPACE
which happens to spread its populations over exactly the graph's recurrent
states.  Rule 27 (VEVD) is the sharp case -- the graph reports 9 recurrent
states, the truth is ONE pure stationary superposition, and leak_flow is 0.

The structural test asks the right question directly.  The graph's vertices ARE
basis states, so whatever it calls the attractor is a SET of basis states; the
channel's attractor is a subspace.  The graph can be faithful only when that
subspace is basis-spanned, i.e.

    dim_att == basis_in_att,      dim_att      = dim supp Cesaro-lim Phi^n(1/d)
                                  basis_in_att = # basis states inside it.

`coh_dirs = dim_att - basis_in_att > 0` is exactly the structure no
state-space graph can represent.  Unlike leak_flow this is initial-state
independent and not normalised by 2^N (leak_flow starting from 1/d gives rules
104/233 a spurious 2/2^N that vanishes with N for purely dilutional reasons).

CONVERGENCE.  The Cesaro average must be iterated to a genuine SPECTRAL GAP, not
to a fixed burn-in.  With burn-in 600 rule 19 at N=6 reports dim_att = 35 with
its smallest kept eigenvalue 1.9e-9 against a largest dropped 4.8e-10 -- no gap
at all, so the "attractor" is mostly still-decaying transient.  Converged
(burn-in 4000) it is 5, with a gap of 1.8e-2 against 1e-15.  We therefore double
the burn-in until  smallest_kept / largest_dropped > GAP_MIN,  and mark the rule
NOT CONVERGED if that never happens.

Cost is 2^N dense rho with sparse per-site Kraus operators, so N <= 8-9 is
reachable (the 4^N superoperator of peripheral.py caps out near N = 6).
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp

from .. import results_io
from ..core.cycle import even_sites, neighbor_bits, odd_sites
from ..core.rules import Tuple4, is_unitary, wolfram_to_tuple

SQRT2 = math.sqrt(2.0)

# Eigenvalues of the Cesaro limit above this count as attractor support...
CUT = 1e-9
# ...and the split is trusted only with this much spectral gap around it.
GAP_MIN = 1e4

CENSUS_PATH = os.path.join(results_io.REPO_ROOT, "analytics",
                           "coherent_attractor_census.json")


def site_kraus_sparse(i: int, N: int, t: Tuple4, bc: str):
    """(A0, A1) for the local channel at site i, as sparse 2^N matrices.

    A1 is None when the local symbol never jumps on any configuration.
    """
    dim = 1 << N
    H = (1 / SQRT2) * np.array([[1.0, 1.0], [1.0, -1.0]])
    r0, c0, v0 = [], [], []
    r1, c1, v1 = [], [], []
    bit = 1 << i
    for x in range(dim):
        m, n = neighbor_bits(x, i, N, bc)
        s = t[2 * m + n]
        xi = (x >> i) & 1
        x0, x1 = x & ~bit, x | bit
        if s == "I":
            r0.append(x), c0.append(x), v0.append(1.0)
        elif s == "V":
            r0.append(x0), c0.append(x), v0.append(H[0, xi])
            r0.append(x1), c0.append(x), v0.append(H[1, xi])
        elif s == "D":
            if xi == 0:
                r0.append(x), c0.append(x), v0.append(1.0)
            else:
                r1.append(x0), c1.append(x), v1.append(1.0)
        elif s == "E":
            if xi == 1:
                r0.append(x), c0.append(x), v0.append(1.0)
            else:
                r1.append(x1), c1.append(x), v1.append(1.0)
    A0 = sp.csr_matrix((v0, (r0, c0)), shape=(dim, dim))
    A1 = sp.csr_matrix((v1, (r1, c1)), shape=(dim, dim)) if r1 else None
    return A0, A1


def kraus_list(N: int, t: Tuple4, bc: str):
    """Per-site (A0, A1) pairs in the brickwork order (even sites, then odd)."""
    return [site_kraus_sparse(i, N, t, bc)
            for layer in (even_sites(N), odd_sites(N)) for i in layer]


def apply_cycle(rho: np.ndarray, ks) -> np.ndarray:
    """One full brickwork cycle of the channel applied to a dense rho."""
    for A0, A1 in ks:
        new = A0 @ rho @ A0.getH()
        if A1 is not None:
            new = new + A1 @ rho @ A1.getH()
        rho = np.asarray(new)
    return rho


def attractor_structure(N: int, t: Tuple4, bc: str, burn0: int = 1000,
                        avg: int = 720, max_burn: int = 64000) -> Dict:
    """dim / basis-content of the attractor, with a spectral-gap guarantee.

    Returns {dim_att, basis_in_att, coh_dirs, converged, burn, gap}.  The
    Cesaro window `avg` kills limit-cycle oscillation (720 = lcm of the small
    periods that occur here).
    """
    dim = 1 << N
    ks = kraus_list(N, t, bc)
    rho = np.eye(dim, dtype=complex) / dim
    burn, target = 0, burn0
    while True:
        for _ in range(target - burn):
            rho = apply_cycle(rho, ks)
        burn = target
        acc = np.zeros((dim, dim), dtype=complex)
        r2 = rho.copy()
        for _ in range(avg):
            r2 = apply_cycle(r2, ks)
            acc += r2
        acc /= avg

        w = np.sort(np.linalg.eigvalsh(acc))[::-1]
        k = int((w > CUT).sum())
        smallest_kept = w[k - 1] if k else 1.0
        largest_drop = w[k] if k < len(w) else 0.0
        gap = (float("inf") if largest_drop <= 0
               else smallest_kept / max(largest_drop, 1e-300))
        if gap > GAP_MIN or target >= max_burn:
            U = np.linalg.eigh(acc)[1][:, ::-1][:, :k]
            P = U @ U.conj().T
            nb = int(np.sum(np.abs(np.real(np.diag(P)) - 1.0) < 1e-6))
            return {"dim_att": k, "basis_in_att": nb, "coh_dirs": k - nb,
                    "converged": bool(gap > GAP_MIN), "burn": burn,
                    "gap": gap}
        target *= 2


def basis_spanned(N: int, t: Tuple4, bc: str, **kw) -> bool:
    """True iff the graph CAN in principle name this rule's attractor."""
    return attractor_structure(N, t, bc, **kw)["coh_dirs"] == 0


def census(Ns: List[int], bc: str = "pbc", verbose: bool = True) -> Dict:
    """Run the structural test over all 240 dissipative rules at each N."""
    from ..graph.scc import recurrent_classes

    by_N: Dict[str, Dict] = {}
    for N in Ns:
        t0 = time.time()
        entry: Dict = {"bc": bc, "rules": {}, "not_converged": 0}
        for r in range(256):
            t = wolfram_to_tuple(r)
            if is_unitary(t):
                continue
            st = attractor_structure(N, t, bc)
            if not st["converged"]:
                entry["not_converged"] += 1
            if st["coh_dirs"] != 0:
                rec = sorted(x for R in recurrent_classes(r, N, bc, t) for x in R)
                entry["rules"][str(r)] = {
                    "tuple": "".join(t), "n_rec": len(rec),
                    "dim_att": st["dim_att"], "basis_in_att": st["basis_in_att"],
                    "coh_dirs": st["coh_dirs"], "burn": st["burn"]}
        entry["coherent_rules"] = sorted(int(r) for r in entry["rules"])
        entry["n_coherent"] = len(entry["coherent_rules"])
        by_N[str(N)] = entry
        if verbose:
            print(f"N={N} {bc}: n_coherent_attractor={entry['n_coherent']}/240  "
                  f"not_converged={entry['not_converged']}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    return by_N


def write_census(by_N: Dict, path: str = CENSUS_PATH, bc: str = "pbc"):
    from .. import ENGINE_VERSION
    os.makedirs(os.path.dirname(path), exist_ok=True)
    doc = {
        "description":
            "Coherent-attractor census: rules whose channel attractor is NOT "
            "spanned by computational-basis states (dim_att > basis_in_att), "
            "i.e. rules for which the Tier-1a transition graph cannot name the "
            "attractor.",
        "method":
            "Cesaro average of Phi^n(1/d); eigenvalues kept above 1e-9 and the "
            "burn-in doubled until smallest_kept/largest_dropped > 1e4 "
            "(spectral-gap convergence guarantee); basis_in_att = # "
            "computational-basis states inside the attractor support.",
        "source": "qca_fragmentation.quantum.attractor_structure",
        "engine_version": ENGINE_VERSION,
        "bc": bc, "n_dissipative": 240, "by_N": by_N,
    }
    with open(path, "w") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
    return path


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, nargs="+", default=[4, 6, 8])
    ap.add_argument("--bc", default="pbc")
    ap.add_argument("--out", default=CENSUS_PATH)
    ap.add_argument("--merge", action="store_true",
                    help="keep sizes already present in the census file")
    args = ap.parse_args(argv)

    by_N = census(args.N, args.bc)
    if args.merge and os.path.exists(args.out):
        with open(args.out) as f:
            old = json.load(f).get("by_N", {})
        old.update(by_N)
        by_N = old
    print("wrote", write_census(by_N, args.out, args.bc))


if __name__ == "__main__":
    main()
