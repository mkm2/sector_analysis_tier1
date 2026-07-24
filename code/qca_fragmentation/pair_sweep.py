"""
Memory-guarded Tier-1d pair-graph sweep.

Runs the pair-graph certificate + weak charges over a family of rules, walking N
upward so an interrupt still leaves a COMPLETE dataset at every size reached.
Each unit is a subprocess with an address-space rlimit and a wall-clock timeout,
so a near-ergodic rule (whose WCC block is |S|^2 and can blow up) dies alone and
is simply recorded bounded-only.  Fragmented rules have small WCCs, so the pair
graph stays cheap far beyond the dense-rho ceiling of N = 8.

    python -m qca_fragmentation.pair_sweep --which coherent --n-min 4 --n-max 12

Idempotent via the results_tier1d store: a cached unit is skipped, so re-running
is how a bounded-only unit is retried under a larger cap.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

from . import results_io
from .core import rules
from .quantum.weak_charges import WALL_HADAMARD_CORE

# tier -> (workers, address-space cap GB, timeout s), split by previous runtime.
SMALL_CUTOFF_S = 3.0
TIERS: Dict[int, Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = {
    # N : (small units),         (large units)
    10: ((8, 3, 1800), (3, 8, 3600)),
    11: ((6, 4, 3600), (2, 12, 7200)),
    12: ((4, 6, 7200), (2, 16, 14400)),
    13: ((3, 8, 14400), (1, 26, 28800)),
    14: ((2, 12, 28800), (1, 28, 57600)),
}
DEFAULT_TIERS = ((8, 3, 1800), (2, 12, 7200))


def coherent_rules() -> List[int]:
    """Union of the 92 coherent-attractor rules (census), else all dissipative."""
    path = os.path.join(results_io.REPO_ROOT, "analytics",
                        "coherent_attractor_census.json")
    if os.path.exists(path):
        with open(path) as f:
            doc = json.load(f)
        rs = set()
        for entry in doc.get("by_N", {}).values():
            rs.update(entry.get("coherent_rules", []))
        if rs:
            return sorted(rs)
    return [r for r in range(256)
            if not rules.is_unitary(rules.wolfram_to_tuple(r))]


def _units(N: int, bcs, rule_list) -> List[Tuple[float, int, str]]:
    out = []
    for rule in rule_list:
        for bc in bcs:
            if results_io.has_pair_unit(rule, bc, N):
                continue
            recs = results_io.load_pair_results(rule, bc)
            prev = recs[max(recs)] if recs else {}
            out.append((prev.get("runtime") or 0.0, rule, bc))
    out.sort()
    return out


def _run_one(rule: int, bc: str, N: int, mem_gb: int, timeout: int,
             r: int, cesaro_max_N: int, node_budget: int) -> Dict:
    def _limit():
        lim = mem_gb * (1 << 30)
        resource.setrlimit(resource.RLIMIT_AS, (lim, lim))

    cmd = [sys.executable, "-m", "qca_fragmentation.pair_run",
           "--rule", str(rule), "--N", str(N), "--bc", bc, "--r", str(r),
           "--cesaro-max-N", str(cesaro_max_N), "--node-budget", str(node_budget)]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(__file__)),
                           capture_output=True, text=True, timeout=timeout,
                           preexec_fn=_limit)
        ok = p.returncode == 0
        why = (p.stderr or "").strip().splitlines()
        return {"rule": rule, "bc": bc, "N": N, "ok": ok,
                "seconds": time.time() - t0,
                "why": ("" if ok else (why[-1] if why else f"rc={p.returncode}"))}
    except subprocess.TimeoutExpired:
        return {"rule": rule, "bc": bc, "N": N, "ok": False,
                "seconds": time.time() - t0, "why": f"timeout>{timeout}s"}


def sweep(n_min, n_max, bcs, rule_list, r, cesaro_max_N, node_budget,
          log_path=None):
    failures: List[Dict] = []
    for N in range(n_min, n_max + 1):
        cfgs = TIERS.get(N, DEFAULT_TIERS)
        units = _units(N, bcs, rule_list)
        small = [(rr, bc) for c, rr, bc in units if c < SMALL_CUTOFF_S]
        large = [(rr, bc) for c, rr, bc in units if c >= SMALL_CUTOFF_S]
        print(f"\n=== N={N}: {len(units)} units ({len(small)} small, "
              f"{len(large)} large)", flush=True)
        if not units:
            continue
        t0 = time.time()
        for todo, cfg in ((small, cfgs[0]), (large, cfgs[1])):
            if not todo:
                continue
            workers, mem_gb, timeout = cfg
            done = 0
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(_run_one, rr, bc, N, mem_gb, timeout, r,
                                  cesaro_max_N, node_budget) for rr, bc in todo]
                for f in futs:
                    res = f.result()
                    done += 1
                    if not res["ok"]:
                        failures.append(res)
                        print(f"     FAIL W{res['rule']} {res['bc']} N{N}: "
                              f"{res['why']}", flush=True)
                    if done % 25 == 0 or done == len(todo):
                        print(f"     {done}/{len(todo)} "
                              f"[{time.time() - t0:.0f}s]", flush=True)
        print(f"=== N={N} complete, {time.time() - t0:.0f}s", flush=True)
        if log_path:
            with open(log_path, "w") as fh:
                json.dump({"failures": failures}, fh, indent=1)
    return failures


def main(argv=None):
    ap = argparse.ArgumentParser(description="Tier 1d pair-graph sweep")
    ap.add_argument("--which", choices=["coherent", "wallcore", "dissipative",
                                        "all256"], default="coherent")
    ap.add_argument("--n-min", type=int, default=4)
    ap.add_argument("--n-max", type=int, default=12)
    ap.add_argument("--bc", choices=["pbc", "obc0", "both"], default="pbc")
    ap.add_argument("--r", type=int, default=3)
    ap.add_argument("--cesaro-max-N", type=int, default=6)
    ap.add_argument("--node-budget", type=int, default=6_000_000)
    ap.add_argument("--log", default=None)
    args = ap.parse_args(argv)

    bcs = ["pbc", "obc0"] if args.bc == "both" else [args.bc]
    if args.which == "coherent":
        rule_list = coherent_rules()
    elif args.which == "wallcore":
        rule_list = list(WALL_HADAMARD_CORE)
    elif args.which == "dissipative":
        rule_list = [r for r in range(256)
                     if not rules.is_unitary(rules.wolfram_to_tuple(r))]
    else:
        rule_list = list(range(256))

    print(f"pair sweep: {len(rule_list)} rules x {len(bcs)} bc, "
          f"N={args.n_min}..{args.n_max}", flush=True)
    fails = sweep(args.n_min, args.n_max, bcs, rule_list, args.r,
                  args.cesaro_max_N, args.node_budget, args.log)
    print(f"\nfinished with {len(fails)} failed units")


if __name__ == "__main__":
    main()
