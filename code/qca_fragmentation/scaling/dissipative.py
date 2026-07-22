"""
Tier-1c scaling extraction for the 240 DISSIPATIVE rules (PLAN.md T6/T7).

The unitary counterpart (`scaling/summary.py`) fits the sector count and the
largest sector.  For a dissipative rule the same two series mean something
different -- the graph is directed, so

    n_recurrent  = number of ATTRACTORS (recurrent classes),
    d_max        = size of the largest attractor,
    max_basin    = size of the largest basin of attraction,
    transient_depth = longest condensation-DAG path (relaxation-time proxy),

and we extract the exponential base of each, exactly as in the unitary case:
BIC selection over  ln y = c (+ alpha ln N) (+ kappa N),  base = e^kappa.

COMMENSURABILITY.  Unlike the unitary sweep, many dissipative rules oscillate
strongly with N, because the attractor is a spatial pattern that only fits a
ring of the right length.  The familiar case is parity: a pbc chain of odd
length cannot host the two Neel states, so e.g. rule 22 has 3 attractors at even
N and 2 at odd N, while its largest attractor is a single state at even N but
grows like 2N at odd N.  A single fit then reports a meaningless average.

Parity is NOT the only period.  Rules whose attractor is a period-3 density wave
oscillate with that period instead -- rule 28 (IIVD) has D_max =
4,2,4,8,4,8,16,8,16,32,16,32 over N=6..17, three interleaved doubling sequences
with base 2^(1/3), not noise.  While the fitter only knew about parity, 32 of
the 240 pbc D_max series were stamped "irregular"; generalising the period
brings that to 20.

CHOOSING THE PERIOD IS WHERE OVERFITTING LIVES, so the criterion is calibrated
against a null, not tuned by eye (see scaling/overfit_audit.py):

  * a split must clear the unsplit model by _BIC_MARGIN in BIC, with at least
    _MIN_PER_CLASS points per class.  The obvious cheaper rules both fail the
    null: "smallest period that cuts the residual by 2x" splits 21% of smooth
    exponentials with 25% noise, and plain argmin-BIC splits 33%.  The
    calibrated rule splits 3%.
  * OR an exact integer recurrence holds on every residue class, which fired
    0/4000 times on the same surrogates and so is allowed to certify a period
    whose classes are too small for BIC to afford.
  * and in either case only when the joint fit is actually failing
    (r_all > _RESID_BAD) -- otherwise a constant series, which satisfies
    a(n)=a(n-p) for every p, gets a meaningless period.

The consequence worth stating: at N<=17 NO period-4 claim survives.  Rule 11
(VEDD) looks period-4 by eye -- 9,11,2,2,15,17,2,2,21,23,2,2 -- but that leaves
3 points per class against a 2-parameter line and no recurrence certifies it, so
it is reported irregular.  Period 4 needs N>=21 to be honest.

`kappa_eff` is the number to plot: the joint fit when the series is
period-clean, otherwise the residue class with the larger |kappa| (the dominant
growth).  Note the exact-base search needs none of this -- a period-p
oscillation is itself a recurrence of order >= p, so `find_integer_recurrence`
already caught these on the whole series.  The period split fixes the growth
CLASS, not the base, and the exact-base counts are identical either way.

Outputs
    figures/dissipative_summary.csv     one row per (rule, bc)
    reports/tex/tab_diss_scaling.tex    the representative-rule table for R5
"""

from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from .. import results_io
from ..core import rules
from .fits import (find_integer_recurrence, find_recurrence_by_period,
                   fit_pure_exponential, fit_series)
from .summary import load_series

FIGURES_DIR = os.path.join(results_io.REPO_ROOT, "figures")
TEX_DIR = os.path.join(results_io.REPO_ROOT, "reports", "tex")
ANALYTICS_DIR = os.path.join(results_io.REPO_ROOT, "analytics")

SERIES_KEYS = ("n_recurrent", "d_max", "max_basin", "transient_depth")

# A joint fit whose ln-residuals exceed this (rms) is suspect, and is the
# PRECONDITION for considering any period split at all.
_RESID_BAD = 0.15
# Scatter (rms of ln y) that survives the split: the series is then not
# described by any smooth growth law and is reported as "irregular".
_RESID_NOISY = 0.35

# Candidate commensurability periods.  Period 2 is the Neel splitting (an odd
# pbc ring cannot host the two Neel states); 3 and 4 are equally real and were
# being mislabelled "irregular" until N reached 17 -- e.g. W28 IIVD has
# D_max = 4,2,4,8,4,8,16,8,16,32,16,32, which is three interleaved doubling
# sequences (base 2^(1/3)), not noise.  Periods beyond 4 are never admitted, and
# at N<=17 even 4 is unreachable -- see the module docstring.
_PERIODS = (1, 2, 3, 4)
# Splitting at period p costs 2p parameters against the same n points, so it
# must be paid for.  A ratio-of-residuals rule cannot do that job: measured
# against a smooth exponential with 25% multiplicative noise it split 21% of
# structureless series (mostly to p=3), which is straightforward overfitting.
# BIC over the pooled log-residuals is the principled version and is what the
# growth-class selection already uses; _MIN_PER_CLASS keeps a class from being
# fitted by a 2-parameter line through 3 points.
_MIN_PER_CLASS = 4
# ...and BIC alone is not enough at n ~ 12: plain argmin-BIC split 33% of the
# smooth-plus-noise surrogates, because with 12 points the RSS drop from 2 to 6
# parameters beats 2p ln n by chance a third of the time.  A split must clear
# the unsplit model by a decisive margin.  10 is the Kass-Raftery "very strong
# evidence" threshold; the genuine period-3/4 series clear it by hundreds,
# because their residue classes fit EXACTLY (RSS at the floor), so the margin
# costs nothing real.
_BIC_MARGIN = 10.0


def dissipative_rules() -> List[int]:
    return [r for r in range(256)
            if not rules.is_unitary(rules.wolfram_to_tuple(r))]


def _rms_resid(Ns: Sequence[float], ys: Sequence[float], fit: Dict) -> float:
    """rms residual of the SELECTED model on ln y."""
    if not fit.get("ok"):
        return float("nan")
    Ns = np.asarray(Ns, float)
    ys = np.asarray(ys, float)
    good = ys > 0
    Ns, ys = Ns[good], ys[good]
    beta = np.asarray(fit["params"][fit["best_model"]], float)
    cols = [np.ones_like(Ns)]
    if len(beta) > 1:
        cols.append(np.log(Ns))
    if len(beta) > 2:
        cols.append(Ns)
    pred = np.column_stack(cols) @ beta
    r = np.log(ys) - pred
    return float(np.sqrt((r * r).mean()))


def _split_by(Ns: Sequence[int], ys: Sequence[int], p: int):
    """The p residue classes of N, as (Ns, ys) pairs."""
    return [([n for n in Ns if n % p == k],
             [y for n, y in zip(Ns, ys) if n % p == k]) for k in range(p)]


# Relative floor on the pooled RSS, so an exactly-fitting split (which the real
# period-3/4 series are) does not produce log(0) and win by an infinite margin.
_RSS_FLOOR = 1e-12


def _period_bic(Ns: Sequence[int], ys: Sequence[int], p: int) -> float:
    """BIC of the model "log-linear in N, separately on each class mod p".

    n ln(RSS/n) + k ln n with k = 2p, the honest parameter count: splitting
    buys a fresh intercept and slope per class and must pay for both.
    """
    tot, npts = 0.0, 0
    for sn, sy in _split_by(Ns, ys, p):
        x = np.asarray(sn, float)
        y = np.asarray(sy, float)
        good = y > 0
        x, y = x[good], y[good]
        if len(x) < _MIN_PER_CLASS:
            return float("inf")          # not enough data to earn this period
        ly = np.log(y)
        X = np.column_stack([np.ones_like(x), x])
        beta, *_ = np.linalg.lstsq(X, ly, rcond=None)
        r = ly - X @ beta
        tot += float(r @ r)
        npts += len(x)
    if npts == 0:
        return float("inf")
    rss = max(tot, _RSS_FLOOR * npts)
    return npts * np.log(rss / npts) + 2 * p * np.log(npts)


def fit_with_period(Ns: Sequence[int], ys: Sequence[int]) -> Dict:
    """Fit a series jointly and split by N mod p, choosing the smallest p that
    works.

    The growth CLASS comes from the BIC selection (`fit_series`); the reported
    RATE comes from the two-parameter `fit_pure_exponential`, because the
    alpha*ln N term of M2 is unstable on the ragged dissipative series (see
    fits.fit_pure_exponential).  A series that is still scattered after the best
    split is labelled "irregular" and carries no rate at all.

    The period is chosen by BIC (`_period_bic`), NOT by comparing residuals.
    A larger p always fits at least as well -- it has 2p free parameters for the
    same n points -- so any criterion that does not charge for parameters will
    drift to p=4.  A ratio-of-residuals rule was measured splitting 21% of pure
    smooth-exponential-plus-noise surrogates; under BIC that falls to ~0 while
    the real period-3/4 series, whose classes fit exactly, are kept.
    Ties go to the smaller period.
    """
    Ns = list(Ns)
    ys = list(ys)
    out: Dict = {"all": fit_series(Ns, ys), "all_exp": fit_pure_exponential(Ns, ys)}

    # Per-class fits for every period that can be fitted at all.  The
    # _MIN_PER_CLASS gate lives in _period_bic, not here: the recurrence route
    # below may legitimately select a period with small classes, and it still
    # needs the per-class rates to report.
    cand: Dict[int, Dict] = {}
    for p in _PERIODS:
        cls = _split_by(Ns, ys, p)
        if any(len(sn) < 2 for sn, _ in cls):
            continue
        fits = [fit_series(sn, sy) for sn, sy in cls]
        exps = [fit_pure_exponential(sn, sy) for sn, sy in cls]
        rs = [_rms_resid(sn, sy, f) for (sn, sy), f in zip(cls, fits)]
        rs = [r for r in rs if r == r]          # drop nan
        cand[p] = {"fits": fits, "exps": exps,
                   "resid": max(rs) if rs else float("nan")}

    r_all = _rms_resid(Ns, ys, out["all"])
    # Period by BIC, but only against a decisive margin over the unsplit model
    # (see _BIC_MARGIN).  Smallest qualifying period wins.
    bics = {p: _period_bic(Ns, ys, p) for p in _PERIODS}
    out["period_bic"] = bics
    period = 1
    out["period_via"] = None

    # PRECONDITION for any split: the unsplit model must actually be failing.
    # Without this the recurrence route below fires on trivially constant
    # series -- a run of 3s satisfies a(n)=a(n-p) for every p -- and labels
    # them "period 2", which is true and useless.  A split has to buy
    # something.
    splittable = r_all == r_all and r_all > _RESID_BAD
    if splittable:
        for p in _PERIODS[1:]:
            if bics[p] < bics[1] - _BIC_MARGIN:
                period = p
                out["period_via"] = "bic"
                break

    # Second, independent route to a period: an EXACT integer recurrence holding
    # on every residue class.  This licenses periods that BIC cannot afford --
    # p=4 leaves only 3 points per class at N<=17, too few for a 2-parameter
    # line -- because it is a far stricter certificate than any residual test:
    # measured on 4000 smooth-plus-noise surrogates it fired 0 times at every
    # period, since noise does not satisfy a small-coefficient integer
    # recurrence exactly on every remaining term.  A series that passes neither
    # route stays unsplit and will usually be reported irregular, which is the
    # honest verdict when the sizes cannot settle the question.
    if splittable and period == 1 and len(ys) >= 6:
        for p in _PERIODS[1:]:
            if p in cand and find_recurrence_by_period(Ns, ys, period=p).get("ok"):
                period = p
                out["period_via"] = "recurrence"
                break

    # Parity is reported by name whatever the winning period, because the
    # even/odd rates are what the Neel story is told in.
    for name, keep in (("even", 0), ("odd", 1)):
        sn = [n for n in Ns if n % 2 == keep]
        sy = [y for n, y in zip(Ns, ys) if n % 2 == keep]
        out[name] = (fit_series(sn, sy) if len(sn) >= 3
                     else {"ok": False, "reason": "too few"})
        out[name + "_exp"] = fit_pure_exponential(sn, sy)

    out["period"] = period
    out["parity_split"] = period == 2
    out["split"] = period > 1
    out["resid_all"] = r_all
    out["resid_eff"] = cand[period]["resid"] if period > 1 else r_all

    order = {"exponential": 0, "polynomial": 1, "constant": 2}
    if period > 1:
        exps = [e for e in cand[period]["exps"] if e.get("ok")]
        classes = [f["growth_class"] for f in cand[period]["fits"] if f.get("ok")]
    else:
        exps = [out["all_exp"]] if out["all_exp"].get("ok") else []
        classes = [out["all"]["growth_class"]] if out["all"].get("ok") else []

    growth = min(classes, key=lambda c: order.get(c, 9)) if classes else "n/a"
    kappa = max((e["kappa"] for e in exps), key=abs) if exps else None

    # Guards: an unbounded rate, or residual scatter that survives the best
    # split, means the series has no exponential description at all.
    unbounded = any(not e["bounded"] for e in exps)
    noisy = out["resid_eff"] == out["resid_eff"] and out["resid_eff"] > _RESID_NOISY
    if unbounded or noisy:
        growth, kappa = "irregular", None

    out["growth_eff"] = growth
    out["kappa_eff"] = kappa
    out["irregular"] = growth == "irregular"
    return out


# The period-2 name the rest of the codebase grew up with.
fit_with_parity = fit_with_period


def attractor_class(n_rec: Optional[int], d_max: Optional[int]) -> str:
    """Coarse graph-level attractor structure at the largest computed N."""
    if not n_rec or not d_max:
        return "n/a"
    if d_max == 1:
        return "unique-point" if n_rec == 1 else "multistable"
    return "extended" if n_rec == 1 else "multi-extended"


def load_coherent_census(path: Optional[str] = None) -> Dict:
    """The Tier-1b coherent-attractor census (analytics/coherent_attractor_census.json).

    Returns {} if the census has not been generated; callers must degrade
    gracefully (the scaling extraction itself does not depend on it).
    """
    path = path or os.path.join(ANALYTICS_DIR, "coherent_attractor_census.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def summarize_rule(rule: int, bc: str, coherent: Optional[set] = None) -> Dict:
    s = load_series(rule, bc)
    t = "".join(rules.wolfram_to_tuple(rule))
    row = {
        "rule": rule,
        "tuple": t,
        "bc": bc,
        "n_points": len(s["N"]),
        "N_min": s["N"][0] if s["N"] else None,
        "N_max": s["N"][-1] if s["N"] else None,
        "n_att_at_Nmax": s["n_recurrent"][-1] if s["n_recurrent"] else None,
        "d_max_at_Nmax": s["d_max"][-1] if s["d_max"] else None,
        "depth_at_Nmax": s["transient_depth"][-1] if s["transient_depth"] else None,
    }
    row["attractor_class"] = attractor_class(row["n_att_at_Nmax"],
                                             row["d_max_at_Nmax"])
    row["coherent_attractor"] = (rule in coherent) if coherent is not None else None
    row["wall_hadamard"] = (t[1] == "V" and t[2] == "V")
    for key in SERIES_KEYS:
        f = fit_with_period(s["N"], s[key])
        row[f"{key}_growth"] = f["growth_eff"]
        row[f"{key}_kappa"] = f["kappa_eff"]
        row[f"{key}_base"] = (np.exp(f["kappa_eff"])
                              if f["kappa_eff"] is not None else None)
        row[f"{key}_period"] = f["period"]
        # "irregular" conflates two very different situations, so name which.
        # A V-free rule is DETERMINISTIC on basis states (D and E are resets --
        # both Kraus branches land on the same basis state -- and I is the
        # identity; only V branches), so its graph is a map and d_max is the
        # length of a cycle.  Cycle lengths are multiplicative orders: they are
        # arithmetic functions of N and no growth law exists to be found.  That
        # is a RESULT.  A rule containing a V that comes out irregular is
        # instead a rule whose period the available sizes cannot settle, which
        # is a statement about the dataset, not about the rule.
        row[f"{key}_irregular_kind"] = (
            None if not f["irregular"]
            else ("arithmetic" if not rules.has_V(rules.wolfram_to_tuple(rule))
                  else "undetermined"))
        row[f"{key}_parity_split"] = f["parity_split"]
        row[f"{key}_irregular"] = f["irregular"]
        row[f"{key}_resid"] = f["resid_eff"]
        row[f"{key}_model"] = f["all"].get("best_model")
        for p in ("even", "odd"):
            e = f[p + "_exp"]
            row[f"{key}_kappa_{p}"] = e["kappa"] if e.get("ok") else None
        # EXACT base: an integer linear recurrence satisfied by the whole
        # series pins the growth base as an algebraic number.  A split series
        # obeys no single recurrence, but each residue class usually does, so
        # it gets the per-class search at the period chosen above.
        if not s[key]:
            rec = {"ok": False}
        elif f["period"] > 1:
            rec = find_recurrence_by_period(s["N"], s[key], period=f["period"])
        else:
            rec = find_integer_recurrence(s[key])
        row[f"{key}_rec_order"] = rec["order"] if rec["ok"] else None
        row[f"{key}_rec_coeffs"] = (",".join(map(str, rec["coeffs"]))
                                    if rec["ok"] else None)
        row[f"{key}_base_exact"] = rec["base"] if rec["ok"] else None
        row[f"{key}_base_name"] = rec.get("name") if rec["ok"] else None
    return row


def summarize(bcs=("pbc", "obc0")) -> List[Dict]:
    census = load_coherent_census()
    # a rule counts as "coherent" if its attractor is non-basis-spanned at ANY
    # of the censused sizes (the census is per-N; see analytics/ JSON)
    coherent = set()
    for _N, entry in census.get("by_N", {}).items():
        coherent.update(int(r) for r in entry.get("coherent_rules", []))
    # ...and only for the bc the census was actually run under: it is a pbc
    # census, and pasting its verdicts onto obc0 rows would be a fabrication.
    census_bc = census.get("bc")
    rows = []
    for rule in dissipative_rules():
        for bc in bcs:
            if results_io.load_results(rule, bc):
                rows.append(summarize_rule(
                    rule, bc, coherent if (census and bc == census_bc) else None))
    return rows


def write_csv(rows: List[Dict], path: str):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _fmt(x, nd=3):
    if x is None:
        return "--"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


# Rules shown explicitly in R5's scaling table: the ground-truth cases, the
# fastest growers, and the wall-Hadamard core.
TABLE_RULES = [0, 4, 22, 28, 50, 146, 178, 200, 232, 3, 19, 35]


def latex_table(rows: List[Dict], path: str, bc: str = "pbc",
                only: Optional[Sequence[int]] = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    only = TABLE_RULES if only is None else only
    sel = [r for r in rows if r["bc"] == bc and r["rule"] in only]
    sel.sort(key=lambda r: only.index(r["rule"]))
    lines = [
        r"\begin{tabular}{@{}rl rr l l l@{}}",
        r"\toprule",
        r"W & tuple & \#att & $D_{\max}$ & \#att growth (base) & "
        r"$D_{\max}$ growth (base) & attractor class \\",
        r"\midrule",
    ]
    for r in sel:
        def g(key):
            cls = r[f"{key}_growth"]
            s = cls
            if cls == "exponential":
                exact, nm, b = (r[f"{key}_base_exact"], r[f"{key}_base_name"],
                                r[f"{key}_base"])
                if exact:
                    s += f" ({exact:.4f}" + (f", {nm}" if nm else "") + ")"
                elif b:
                    s += f" ({b:.3f})"
            per = int(r[f"{key}_period"] or 1)
            if per > 1:
                s += rf"$^{{\ast{per}}}$"
            return s
        lines.append(
            f"{r['rule']} & \\texttt{{{r['tuple']}}} & "
            f"{_fmt(r['n_att_at_Nmax'])} & {_fmt(r['d_max_at_Nmax'])} & "
            f"{g('n_recurrent')} & {g('d_max')} & {r['attractor_class']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(FIGURES_DIR,
                                                  "dissipative_summary.csv"))
    args = ap.parse_args(argv)
    rows = summarize()
    write_csv(rows, args.out)
    latex_table(rows, os.path.join(TEX_DIR, "tab_diss_scaling.tex"))

    import collections
    for bc in ("pbc", "obc0"):
        sub = [r for r in rows if r["bc"] == bc]
        gn = collections.Counter(r["n_recurrent_growth"] for r in sub)
        gd = collections.Counter(r["d_max_growth"] for r in sub)
        per = collections.Counter()
        for r in sub:
            p_ = max(int(r["n_recurrent_period"] or 1), int(r["d_max_period"] or 1))
            if p_ > 1:
                per[p_] += 1
        print(f"{bc}: {len(sub)} rules | #att growth {dict(gn)} | "
              f"Dmax growth {dict(gd)} | period-split {dict(sorted(per.items()))}")
        exact = collections.Counter()
        for r in sub:
            for key in ("n_recurrent", "d_max"):
                if r[f"{key}_growth"] == "exponential" and r[f"{key}_base_exact"]:
                    exact[round(r[f"{key}_base_exact"], 6)] += 1
        print(f"    exact algebraic bases ({sum(exact.values())} series): "
              + ", ".join(f"{b} x{c}" for b, c in exact.most_common()))
    print(f"wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
