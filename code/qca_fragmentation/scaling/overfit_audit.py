"""
Is the growth-law machinery overfitting?  (run before believing any base)

Two claims in the Tier-1c pipeline are easy to fake on a short series and both
would be silent failures:

  RECURRENCE  `find_integer_recurrence` solves for up to 4 coefficients over Q.
              With ~12 terms an order-4 fit is determined by 8 of them and
              verified on 4 -- not obviously enough to trust an "exact"
              algebraic base.
  PERIOD      `fit_with_period` may split a series into p residue classes.  At
              p=4 with 12 sizes that is 3 points per class against a 2-parameter
              log-linear model, which fits nearly anything.

Three independent checks, in increasing order of how much they prove:

  1. NULL (recurrence).  Feed surrogate sequences with no linear-recurrence
     structure and count false "ok"s.  This bounds the false-positive rate of
     the search itself.
  2. NULL (period).  Feed smooth exponentials plus multiplicative noise and
     count spurious period>1 verdicts.
  3. HOLD-OUT (the real test).  Refit on the first n-k sizes and PREDICT the
     last k, on the actual data.  A recurrence that is real extrapolates
     exactly; an overfitted one does not.  Prediction of unseen larger N is a
     far stronger statement than any in-sample residual.

    python -m qca_fragmentation.scaling.overfit_audit
    python -m qca_fragmentation.scaling.overfit_audit --json analytics/overfit_audit.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Dict, List

from .. import results_io
from ..core import rules
from .dissipative import SERIES_KEYS, dissipative_rules, fit_with_period
from .fits import find_integer_recurrence, find_recurrence_by_period
from .summary import load_series

HOLDOUT = 3          # predict this many largest sizes from the rest
N_SURROGATE = 4000


# --------------------------------------------------------------------------
# 1 + 2.  Null models
# --------------------------------------------------------------------------
def null_recurrence_rate(n_terms: int = 12, seed: int = 0,
                         trials: int = N_SURROGATE) -> Dict:
    """False-positive rate of the exact-recurrence search on structureless data.

    Surrogates are drawn log-uniformly over the range the real series occupy
    (1 .. 2^17) and rounded, so they have the same "smallish integers, growing"
    character without obeying any linear recurrence.
    """
    rng = random.Random(seed)
    hits = 0
    for _ in range(trials):
        seq = sorted(rng.randint(1, 1 << 17) for _ in range(n_terms))
        if rng.random() < 0.5:                   # half monotone, half shuffled
            rng.shuffle(seq)
        if find_integer_recurrence(seq)["ok"]:
            hits += 1
    return {"n_terms": n_terms, "trials": trials, "false_positives": hits,
            "rate": hits / trials}


def null_period_rate(n_sizes: int = 12, noise: float = 0.25, seed: int = 0,
                     trials: int = 600) -> Dict:
    """How often a smooth exponential + noise is falsely called period-split."""
    rng = random.Random(seed)
    Ns = list(range(6, 6 + n_sizes))
    counts: Dict[int, int] = {}
    for _ in range(trials):
        kappa = rng.uniform(0.05, 0.6)
        ys = [max(1, int(round(2.718281828 ** (kappa * n)
                               * (1 + rng.gauss(0, noise))))) for n in Ns]
        p = fit_with_period(Ns, ys)["period"]
        counts[p] = counts.get(p, 0) + 1
    return {"n_sizes": n_sizes, "noise": noise, "trials": trials,
            "periods": dict(sorted(counts.items())),
            "false_split_rate": 1 - counts.get(1, 0) / trials}


# --------------------------------------------------------------------------
# 3.  Hold-out extrapolation on the real data
# --------------------------------------------------------------------------
def _predict(seq: List[int], rec: Dict, k: int) -> List[int]:
    """Roll a recurrence forward k steps from the tail of `seq`."""
    c = rec["coeffs"]
    out = list(seq)
    for _ in range(k):
        out.append(sum(ci * out[-i - 1] for i, ci in enumerate(c)))
    return out[len(seq):]


def holdout_unit(rule: int, bc: str, key: str, k: int = HOLDOUT) -> Dict:
    """Fit the recurrence on all but the last k sizes; predict those k.

    Only step-1 (unsplit) series are extrapolated here: a period-p recurrence
    predicts each residue class separately and the bookkeeping obscures what is
    being tested.  Those are covered by `holdout_period`.
    """
    s = load_series(rule, bc)
    seq, Ns = s[key], s["N"]
    row = {"rule": rule, "bc": bc, "key": key, "n": len(seq)}
    if len(seq) < 2 * k or not all(y for y in seq):
        return {**row, "status": "too short"}

    full = find_integer_recurrence(seq)
    if not full["ok"]:
        return {**row, "status": "no recurrence on full series"}

    train = seq[:-k]
    part = find_integer_recurrence(train)
    if not part["ok"]:
        # The law is only visible with all the data -- honest, but it means the
        # fit was never tested out of sample.
        return {**row, "status": "not recoverable from the training part",
                "full_order": full["order"]}
    pred = _predict(train, part, k)
    ok = pred == seq[-k:]
    return {**row, "status": "predicted" if ok else "MISPREDICTED",
            "train_order": part["order"], "full_order": full["order"],
            "predicted": pred, "actual": seq[-k:],
            "same_base": abs(part["base"] - full["base"]) < 1e-9}


def holdout_period(rule: int, bc: str, key: str, k: int = HOLDOUT) -> Dict:
    """Is the chosen period stable when the largest k sizes are withheld?

    An overfitted period is one that only appears because of the specific sizes
    present; a real commensurability is visible in any window.
    """
    s = load_series(rule, bc)
    seq, Ns = s[key], s["N"]
    if len(seq) < 2 * k:
        return {"rule": rule, "bc": bc, "key": key, "status": "too short"}
    p_full = fit_with_period(Ns, seq)["period"]
    p_part = fit_with_period(Ns[:-k], seq[:-k])["period"]
    return {"rule": rule, "bc": bc, "key": key, "period_full": p_full,
            "period_train": p_part, "stable": p_full == p_part}


def audit(bcs=("pbc", "obc0"), keys=("n_recurrent", "d_max")) -> Dict:
    recs, pers = [], []
    for rule in dissipative_rules():
        for bc in bcs:
            for key in keys:
                recs.append(holdout_unit(rule, bc, key))
                pers.append(holdout_period(rule, bc, key))
    return {"holdout_recurrence": recs, "holdout_period": pers,
            "null_recurrence": null_recurrence_rate(),
            "null_period": null_period_rate()}


def report(a: Dict, out=sys.stdout) -> int:
    def _p(*x):
        print(*x, file=out)

    nr, npd = a["null_recurrence"], a["null_period"]
    _p("NULL MODELS")
    _p(f"  recurrence on structureless integers: "
       f"{nr['false_positives']}/{nr['trials']} = {nr['rate']:.4%} false positives")
    _p(f"  period split on smooth exp + {npd['noise']:.0%} noise: "
       f"{npd['false_split_rate']:.2%} falsely split  {npd['periods']}")

    from collections import Counter
    st = Counter(r["status"] for r in a["holdout_recurrence"])
    _p("\nHOLD-OUT: fit on N<=N_max-3, predict the 3 largest sizes")
    for k, v in st.most_common():
        _p(f"  {k:38s} {v}")
    mis = [r for r in a["holdout_recurrence"] if r["status"] == "MISPREDICTED"]
    tested = st["predicted"] + len(mis)
    if tested:
        _p(f"  => of {tested} testable, {st['predicted']} extrapolate EXACTLY "
           f"({st['predicted'] / tested:.1%})")
    for r in mis[:12]:
        _p(f"     W{r['rule']} {r['bc']} {r['key']}: "
           f"pred {r['predicted']} vs {r['actual']}")

    ps = [r for r in a["holdout_period"] if "stable" in r]
    split = [r for r in ps if r["period_full"] > 1]
    _p(f"\nPERIOD STABILITY: {sum(r['stable'] for r in ps)}/{len(ps)} unchanged "
       f"when the 3 largest sizes are withheld")
    _p(f"  among the {len(split)} split series: "
       f"{sum(r['stable'] for r in split)}/{len(split)} stable")
    return 0 if not mis else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    a = audit()
    rc = report(a)
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(a, f, indent=1)
        print(f"\nwrote {args.json}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
