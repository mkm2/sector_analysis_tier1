"""
Memory-guarded deep sweep: push every rule to the largest N this box can hold.

`sweep.py` runs units in-process, which is fine while a unit costs megabytes.
Past N ~ 14 a single dissipative unit costs gigabytes (rule 19, pbc, N=14:
1.09 GB and 108 s; both roughly double per step), so one runaway unit would OOM
the whole overnight run.  Here every unit is a separate subprocess with

  * an address-space rlimit, so it dies alone instead of taking the box down,
  * a wall-clock timeout, so a pathological rule cannot eat the night,

and the sweep walks N upwards, finishing each size for all rules before
starting the next.  That way an interrupted run still leaves a COMPLETE dataset
at every size it got through -- which is what the scaling fits need, since a
series with holes in it cannot be fitted or checked against a recurrence.

Units are ordered cheapest-first (by the time the same rule took at the previous
N) so the easy majority is banked before the expensive tail starts.

Budgets are per unit, in three tiers, because cost is wildly uneven: at N=16
most units finish in seconds while the three-Hadamard rules need the box to
themselves.  Nothing here overrides the idempotence of run_rule -- a cached unit
is skipped, so re-running the sweep is exactly how a failed unit is retried
(under whatever budget the tiers give it now).

    python -m qca_fragmentation.deep_sweep --n-min 14 --n-max 17
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

# Cost is wildly uneven -- at N=14 most units finish in under a second while the
# three-Hadamard rules take 1.09 GB and 108 s -- so budgeting per SIZE would
# force the whole of N=17 to run single-threaded for the sake of a handful of
# rules.  We budget per UNIT instead, in two tiers split by the runtime the same
# rule took at the previous size (which doubles-ish per step, so a unit under
# ~5 s at N-1 is nowhere near the memory ceiling at N).
#
# A third tier is needed for the handful of three-Hadamard rules: W55 obc0 took
# 415 s at N=15 and then died on a 8 GB cap at N=16.  The cap is on ADDRESS
# SPACE, which numpy reserves well beyond its resident set, so these want the
# box to themselves rather than a slightly larger share of it.
#
# tier -> (parallel workers, address-space cap per worker in GB, timeout in s)
SMALL_CUTOFF_S = 5.0
HUGE_CUTOFF_S = 250.0
TIERS: Dict[int, Tuple[Tuple[int, int, int], ...]] = {
    #      small units          large units       huge units
    14: ((10, 3, 1800), (6, 4, 1800), (2, 12, 7200)),
    15: ((10, 3, 3600), (5, 5, 3600), (2, 12, 14400)),
    16: ((8, 3, 7200), (3, 8, 7200), (1, 26, 28800)),
    17: ((6, 4, 21600), (2, 12, 21600), (1, 28, 57600)),
    18: ((4, 6, 43200), (1, 26, 43200), (1, 28, 86400)),
    # N >= 19 is unitary territory: sectors are found by union-find (SCC == WCC
    # for a permutation), which is far leaner than the directed Tarjan the
    # dissipative rules need, so several heavy units fit side by side.
    19: ((6, 4, 43200), (4, 6, 43200), (2, 12, 86400)),
    20: ((5, 5, 43200), (4, 6, 43200), (2, 12, 86400)),
    21: ((4, 6, 86400), (3, 8, 86400), (1, 26, 172800)),
    22: ((3, 8, 86400), (2, 12, 86400), (1, 28, 172800)),
}
DEFAULT_TIERS = ((2, 8, 43200), (1, 26, 43200), (1, 28, 86400))


def _units(N: int, bcs, rule_list) -> List[Tuple[int, str]]:
    """Uncached (rule, bc) units at this N, cheapest first.

    Cost is ranked by the wall time the same unit took at N-1, which is the
    best predictor available and costs nothing to read.
    """
    out = []
    for rule in rule_list:
        for bc in bcs:
            if results_io.has_unit(rule, bc, N):
                continue
            recs = results_io.load_results(rule, bc)
            prev = recs.get(N - 1) or recs.get(N - 2) or {}
            if prev.get("ergodic_flag"):
                continue          # fragmentation cannot reappear at larger N
            out.append((prev.get("runtime") or 0.0, rule, bc))
    out.sort()
    return out


def _run_one(rule: int, bc: str, N: int, mem_gb: int, timeout: int,
             f_erg: float, node_budget: int) -> Dict:
    def _limit():
        lim = mem_gb * (1 << 30)
        resource.setrlimit(resource.RLIMIT_AS, (lim, lim))

    cmd = [sys.executable, "-m", "qca_fragmentation.run_rule",
           "--rule", str(rule), "--N", str(N), "--bc", bc,
           "--f-erg", str(f_erg), "--node-budget", str(node_budget)]
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(__file__)),
                           capture_output=True, text=True, timeout=timeout,
                           preexec_fn=_limit)
        ok, why = p.returncode == 0, (p.stderr or "").strip().splitlines()
        return {"rule": rule, "bc": bc, "N": N, "ok": ok,
                "seconds": time.time() - t0,
                "why": ("" if ok else (why[-1] if why else f"rc={p.returncode}"))}
    except subprocess.TimeoutExpired:
        return {"rule": rule, "bc": bc, "N": N, "ok": False,
                "seconds": time.time() - t0, "why": f"timeout>{timeout}s"}


def sweep(n_min: int, n_max: int, bcs, rule_list, f_erg: float,
          node_budget: int, log_path: Optional[str] = None):
    failures: List[Dict] = []
    for N in range(n_min, n_max + 1):
        cfgs = TIERS.get(N, DEFAULT_TIERS)
        units = _units(N, bcs, rule_list)
        buckets = {
            "small": [(r, bc) for c, r, bc in units if c < SMALL_CUTOFF_S],
            "large": [(r, bc) for c, r, bc in units
                      if SMALL_CUTOFF_S <= c < HUGE_CUTOFF_S],
            "huge": [(r, bc) for c, r, bc in units if c >= HUGE_CUTOFF_S],
        }
        print(f"\n=== N={N}: {len(units)} units ("
              + ", ".join(f"{len(v)} {k}" for k, v in buckets.items())
              + ")", flush=True)
        if not units:
            continue
        t0 = time.time()
        # Tiers run cheapest-first and in sequence, so by the time the huge tier
        # starts it has the box to itself and its high cap is honourable.
        for (tier, todo), cfg in zip(buckets.items(), cfgs):
            if not todo:
                continue
            workers, mem_gb, timeout = cfg
            print(f"  -- {tier}: {len(todo)} units, {workers} workers, "
                  f"{mem_gb} GB cap, {timeout} s timeout", flush=True)
            done = 0
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(_run_one, r, bc, N, mem_gb, timeout, f_erg,
                                  node_budget) for r, bc in todo]
                for f in futs:
                    res = f.result()
                    done += 1
                    if not res["ok"]:
                        failures.append(res)
                        print(f"     FAIL W{res['rule']} {res['bc']} N{N}: "
                              f"{res['why']}", flush=True)
                    if done % 25 == 0 or done == len(todo):
                        print(f"     {done}/{len(todo)}  "
                              f"[{time.time() - t0:.0f}s elapsed]", flush=True)
        nf = sum(1 for x in failures if x["N"] == N)
        print(f"=== N={N} complete: {len(units) - nf}/{len(units)} ok, "
              f"{time.time() - t0:.0f}s", flush=True)
        if log_path:
            with open(log_path, "w") as fh:
                json.dump({"failures": failures}, fh, indent=1)
    return failures


def main(argv=None):
    ap = argparse.ArgumentParser(description="memory-guarded deep sweep")
    ap.add_argument("--n-min", type=int, default=14)
    ap.add_argument("--n-max", type=int, default=17)
    ap.add_argument("--bc", choices=["pbc", "obc0", "both"], default="both")
    ap.add_argument("--which", choices=["dissipative", "unitary", "all256"],
                    default="dissipative")
    ap.add_argument("--f-erg", type=float, default=0.9)
    ap.add_argument("--node-budget", type=int, default=5_000_000)
    ap.add_argument("--log", default=None)
    args = ap.parse_args(argv)

    bcs = ["pbc", "obc0"] if args.bc == "both" else [args.bc]
    if args.which == "dissipative":
        rule_list = [r for r in range(256)
                     if not rules.is_unitary(rules.wolfram_to_tuple(r))]
    elif args.which == "unitary":
        rule_list = list(rules.UNITARY_RULES)
    else:
        rule_list = list(range(256))

    fails = sweep(args.n_min, args.n_max, bcs, rule_list, args.f_erg,
                  args.node_budget, args.log)
    print(f"\nfinished with {len(fails)} failed units")
    for f in fails:
        print(f"  W{f['rule']} {f['bc']} N{f['N']}: {f['why']}")


if __name__ == "__main__":
    main()
