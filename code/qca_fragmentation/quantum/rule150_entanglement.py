"""
C150 obc0: why the same rule gives area-law and volume-law entanglement
depending on the initial state (QCA_Circuits Fig. 3a), and how much of that the
sector structure alone decides.

Fig. 3a starts C150 from three product states: a single excitation
|0...010...0>, the Neel state |0101...>, and the pair state |001100...>.  In
wall language (c150.py) these are not three states of one big sector, they are
three states of three very differently sized sectors:

    single excitation   w = 2        |S_w| = C(N+1,2)  ~ N^2/2   (polynomial)
    Neel                w = N or N-1 |S_w| = N+1 or C(N+1,2)     (linear!)
    pair                w ~ N/2      |S_w| = C(N+1,w)            (exponential)

The binomial C(N+1,w) is exponentially large only near w = (N+1)/2; at both ends
it is small.  The Neel state, which looks like the most excited state in the
chain, is in wall language a ONE-HOLE state: a wall on every bond but one.  Walls
are hard core, so nothing can move except at that single hole -- it is jammed.

Kinematic ceiling.  A basis state factorises as |x_A>|x_B> across a cut at site
c, and the wall count of x_A is determined by x_A alone (the bond map is a prefix
XOR, so x_A -> (b_0..b_{c-1}) is a bijection on c bits).  Hence in shell w only
the x_A with at most w internal walls occur, and the Schmidt rank obeys

    rank <= min( |S_w|, sum_{k<=min(w,c)} C(c,k), sum_{k<=min(w,N-c)} C(N-c,k) )

which is exactly the size of the compressed Schmidt matrix built by
rule150_spectra.schmidt_index (verified in the tests).  So

    S(t) <= ln(that bound)  for every t and every state of the shell.

That ceiling already separates the regimes: log-law for fixed w, volume law only
for extensive w.  What the sector does NOT fix is whether the ceiling is
approached -- see dome() (basis states reach ~0.3-0.5 of it, roughly uniformly
in w) and dark_state_entropy() (the stationary states of the same shell have
S(t) constant in time).

    python -m qca_fragmentation.quantum.rule150_entanglement --run
"""

from __future__ import annotations

import argparse
import json
import os
import time
from math import comb, log
from typing import Dict, List, Optional, Tuple

import numpy as np

from .. import c150
from ..results_io import REPO_ROOT
from . import rule150_spectra as rs

ENT_PATH = os.path.join(REPO_ROOT, "analytics", "c150_entanglement.json")


# --- the three Fig. 3a initial states ---------------------------------------

def _from_string(s: str) -> int:
    """Site 0 first -> bitmask with site i in bit i."""
    return sum(int(ch) << i for i, ch in enumerate(s))


def fig3_states(N: int) -> Dict[str, int]:
    return {
        "single": _from_string("0" * (N // 2) + "1" + "0" * (N - N // 2 - 1)),
        "neel": _from_string("".join("01"[i % 2] for i in range(N))),
        "pair": _from_string("".join("0011"[i % 4] for i in range(N))),
    }


# --- the kinematic ceiling ---------------------------------------------------

def schmidt_bound(N: int, w: int, cut: Optional[int] = None) -> int:
    """
    Exact upper bound on the Schmidt rank of any state of the w-wall shell,
    across the cut between sites cut-1 and cut.

    Bond bookkeeping: x_A (sites 0..cut-1) determines the `cut` bonds
    b_0..b_{cut-1}; x_B (sites cut..N-1) determines the N-cut bonds
    b_{cut+1}..b_N; the cut bond b_cut is shared.  So

        w = w_A + b_cut + w_B,   w_A in [0, cut],  w_B in [0, N-cut],

    and the number of left patterns that can occur is the number of x_A whose
    internal wall count w_A lies in the admissible window -- the UPPER limit
    min(w, cut) is the obvious one, but there is also a LOWER limit, because the
    right side can absorb at most N-cut+1 walls.  Forgetting the lower limit
    over-counts badly at high filling: it is what makes the Neel state look less
    constrained than it is.
    """
    c = N // 2 if cut is None else cut
    lo_a, hi_a = max(0, w - 1 - (N - c)), min(w, c)
    lo_b, hi_b = max(0, w - 1 - c), min(w, N - c)
    left = sum(comb(c, k) for k in range(lo_a, hi_a + 1))
    right = sum(comb(N - c, k) for k in range(lo_b, hi_b + 1))
    return min(comb(N + 1, w), left, right)


def kinematic_ceiling(N: int, w: int, cut: Optional[int] = None) -> float:
    b = schmidt_bound(N, w, cut)
    return log(b) if b > 0 else 0.0


def volume_law_value(N: int, cut: Optional[int] = None) -> float:
    c = N // 2 if cut is None else cut
    return min(c, N - c) * log(2.0)


# --- dynamics ----------------------------------------------------------------

def entropy_trace(N: int, w: int, v0: np.ndarray, *, t_max: int = 200,
                  U=None, states=None) -> np.ndarray:
    """S(t) for t = 0..t_max, using the compressed Schmidt matrix."""
    if U is None:
        U, states = rs.sector_unitary(N, w, check=False)
    R, C, nb, na = rs.schmidt_index(N, states)
    v = v0.astype(np.float64).copy()
    v /= np.linalg.norm(v)
    out = np.empty(t_max + 1)
    for t in range(t_max + 1):
        out[t] = rs.entropy_from_amplitudes(v, R, C, nb, na)
        v = U @ v
    return out


def basis_trace(N: int, x: int, *, t_max: int = 200) -> Tuple[int, np.ndarray]:
    """S(t) starting from the product state |x>."""
    w = c150.wall_number(x, N, "obc0")
    U, states = rs.sector_unitary(N, w, check=False)
    k = states.index(x)
    v = np.zeros(len(states))
    v[k] = 1.0
    return w, entropy_trace(N, w, v, t_max=t_max, U=U, states=states)


def plateau(trace: np.ndarray, burn: int = 100) -> float:
    return float(np.mean(trace[burn:]))


# --- the three experiments ---------------------------------------------------

def fig3_scaling(Ns_small=(10, 14, 18, 22, 26, 30, 34, 38),
                 Ns_pair=(8, 10, 12, 14)) -> List[dict]:
    """S_plateau against N for the three Fig. 3a states."""
    out = []
    for name in ("single", "neel", "pair"):
        for N in (Ns_pair if name == "pair" else Ns_small):
            x = fig3_states(N)[name]
            w, tr = basis_trace(N, x)
            out.append({
                "state": name, "N": N, "w": w,
                "sector_dim": comb(N + 1, w),
                "schmidt_bound": schmidt_bound(N, w),
                "ceiling": kinematic_ceiling(N, w),
                "volume_law": volume_law_value(N),
                "s_plateau": plateau(tr),
                "s_final": float(tr[-1]),
            })
            print(f"  {name:>6} N={N:3d} w={w:3d} |S_w|={comb(N+1,w):8d} "
                  f"ceiling={out[-1]['ceiling']:.3f} "
                  f"S_plateau={out[-1]['s_plateau']:.3f}", flush=True)
    return out


def dome(N: int, n_seeds: int = 4, seed: int = 2) -> List[dict]:
    """S_plateau against wall filling, averaged over random basis seeds."""
    rng = np.random.default_rng(seed)
    out = []
    for w in range(0, N + 2, 2):
        D = comb(N + 1, w)
        if D < 2:
            out.append({"N": N, "w": w, "filling": w / (N + 1.0), "sector_dim": D,
                        "ceiling": 0.0, "s_plateau": 0.0, "frac_of_ceiling": None,
                        "frozen": True})
            continue
        U, states = rs.sector_unitary(N, w, check=False)
        vals = []
        for k in rng.choice(len(states), min(n_seeds, len(states)), replace=False):
            v = np.zeros(len(states))
            v[k] = 1.0
            vals.append(plateau(entropy_trace(N, w, v, t_max=150, U=U,
                                              states=states), burn=100))
        ceil_ = kinematic_ceiling(N, w)
        m = float(np.mean(vals))
        out.append({"N": N, "w": w, "filling": w / (N + 1.0), "sector_dim": D,
                    "ceiling": ceil_, "s_plateau": m,
                    "s_plateau_min": float(np.min(vals)),
                    "s_plateau_max": float(np.max(vals)),
                    "frac_of_ceiling": m / ceil_ if ceil_ > 0 else None,
                    "frozen": False})
        print(f"  N={N} w={w:3d} nu={w/(N+1):.2f} dim={D:6d} ceiling={ceil_:.3f} "
              f"S={m:.3f} frac={out[-1]['frac_of_ceiling']:.2f}", flush=True)
    return out


def dark_state_entropy(N: int, w: int, n: int = 3) -> dict:
    """
    The counterexample to 'the sector decides': the stationary states of the SAME
    shell have S(t) constant, however large the shell is.
    """
    U, states = rs.sector_unitary(N, w, check=False)
    ev, evec, cl = rs._clusters(U)
    fix = [(lo, hi) for lo, hi in cl if abs(ev[lo] - 1.0) < 1e-9]
    if not fix:
        return {"N": N, "w": w, "dim_fix": 0}
    lo, hi = fix[0]
    Q, _ = np.linalg.qr(np.real(evec[:, lo:hi]))
    rows = []
    for j in range(min(n, Q.shape[1])):
        tr = entropy_trace(N, w, Q[:, j], t_max=60, U=U, states=states)
        rows.append({"s_mean": float(tr.mean()), "s_time_std": float(tr.std())})
    # a product state of the SAME shell, for comparison
    v = np.zeros(len(states))
    v[0] = 1.0
    basis = plateau(entropy_trace(N, w, v, t_max=200, U=U, states=states))
    return {"N": N, "w": w, "dim_fix": hi - lo,
            "dim_fix_closed_form": comb((N + 1) // 2, w // 2),
            "dark": rows, "basis_state_plateau": basis,
            "ceiling": kinematic_ceiling(N, w)}


def run(*, force: bool = False) -> dict:
    if os.path.exists(ENT_PATH) and not force:
        with open(ENT_PATH) as f:
            return json.load(f)
    t0 = time.time()
    print("Fig.3a scaling:")
    scal = fig3_scaling()
    print("entanglement dome:")
    domes = {str(N): dome(N) for N in (12, 14)}
    print("dark states:")
    dark = [dark_state_entropy(N, w) for (N, w) in ((12, 6), (14, 6))]
    traces = {}
    for N in (14,):
        for name, x in fig3_states(N).items():
            w, tr = basis_trace(N, x, t_max=60)
            traces[f"{name}_N{N}"] = {"w": w, "S": tr.tolist()}
    out = {"scaling": scal, "dome": domes, "dark": dark, "traces": traces,
           "runtime": time.time() - t0,
           "generated": time.strftime("%Y-%m-%dT%H:%M:%S")}
    os.makedirs(os.path.dirname(ENT_PATH), exist_ok=True)
    with open(ENT_PATH, "w") as f:
        json.dump(out, f)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="C150 entanglement vs sector")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    if args.run:
        d = run(force=args.force)
        print(f"\nsaved {ENT_PATH} [{d['runtime']:.0f}s]")


if __name__ == "__main__":
    main()
