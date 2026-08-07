"""
The rules that appear BELOW the ab >= 2 hyperbola in R9 Fig. 1 (left panel).

Two entirely different things put a marker under that dotted line, and only one
of them is about the data:

  (1) THE OFFSET.  F1 nudges the three families apart by +-0.019 in a, without
      which the 144 V+reset rules at (1,2) sit exactly under the V-free ones and
      one family is invisible.  The V-free nudge is to the LEFT, and the curve
      b = 2/a rises as a falls, so EVERY V-free rule sitting exactly ON the
      curve is drawn just under it -- including the single biggest marker in the
      figure, the 35-rule cell at (1,2), which lands at a = 0.981 where the curve
      is 2.0387.  Nothing about those 35 rules is below anything; they are on the
      curve to machine precision.  This is the artefact most likely to catch the
      eye, because it is the largest marker on the panel.

  (2) SEVEN RULES whose FITTED product really is below 2:
      W6, W14, W20, W84   (a = 1, b = 1.9226,  ab = 1.9226)
      W74                 (a = 1, b = 1.9322,  ab = 1.9322)
      W88                 (a = 1, b = 1.9368,  ab = 1.9368)
      W229                (a = 1, b = 1.9188,  ab = 1.9188)

This module is about (2).  The answer is that it IS a fitting problem, and an
unusually clean one, because the theorem that draws the curve also tells us the
right answer:

  * n_wcc is an exact quasi-linear staircase for all seven -- ceil(N/2)+1 for
    the first four, floor((N+4)/3) for the other three -- verified at EVERY N
    computed, with no exception.  So a = 1 and n_wcc = Theta(N).

  * D_max >= 2^N / n_wcc always (some sector must hold at least the average).
    With n_wcc = Theta(N),

        b = lim D_max^(1/N) >= lim (2^N / cN)^(1/N) = 2,

    and D_max <= 2^N gives b <= 2.  Hence b = 2 EXACTLY, with a power-law
    prefactor, and the true coordinate of all seven is (1, 2) -- on the curve,
    beside rule 150.

  The pair (a, b) = (1, 1.92) is therefore not a violated theorem but an
  INTERNALLY INCONSISTENT descriptor: a = 1 and b < 2 cannot both be right.

Why the fitter missed it.  sectors._volume_fraction_base exists precisely to
catch D_max = c * 2^N * N^alpha and hand back base 2; it is what puts the
binomial family on the curve.  It requires the power model to beat the
exponential on the 6-point tail AND on the full series.  On these seven the tail
comparison is a coin flip -- the two residual sums differ by 1.3% for W6/14/20/84
-- and the power model wins the full series for five of the seven.  The
discriminator is being asked to separate c*2^N*N^-0.41 from 1.9226^N over
N = 6..16, and those two curves agree to better than 2% everywhere in that
window.  No estimator can do it; the certified control proves as much: rule 134's
D_max is the central binomial, base 2 with alpha = -1/2 by derivation, and its
rolling pure-exponential base over the SAME window runs 1.904 -> 1.938, i.e.
inside the spread of the seven.

The fix is therefore not a better fit but the analytic override the module
already has for rule 150, and sectors.ANALYTIC now carries it.
"""

from __future__ import annotations

import json
import math
import os
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .. import results_io
from . import sectors

OUT_JSON = os.path.join(sectors.ANALYTICS, "below_curve_{bc}.json")

#: the family offset F1 uses, copied from sector_figure so this module can be
#: read on its own; test_below_curve pins the two together.
FAM_DX = {"unitary": 0.0, "mixed": 0.019, "classical": -0.019}

#: window length of the rolling pure-exponential base
ROLL_WIN = 6

#: certified base-2-with-power-prefactor control: W134's D_max is exactly rule
#: 150's central binomials (R9 sec. on rates).
CONTROL_RULE = 134


# --- 1. who is where ---------------------------------------------------------

class without_r16_override:
    """
    Context manager that removes exactly the analytic entries R16 installed, so
    the PRE-correction fit can be reconstructed on demand.

    Without this the module would be self-erasing: once sectors.ANALYTIC pins
    these seven at (1, 2) the sector map has no rule below the curve at all, and
    a report about the seven would have to hard-code the numbers it is
    explaining.  Everything below is therefore recomputed from the series.
    """

    def __enter__(self):
        self.saved = {}
        for key in sectors.R16_OVERRIDES:
            if key in sectors.ANALYTIC:
                self.saved[key] = sectors.ANALYTIC.pop(key)
        return self

    def __exit__(self, *exc):
        sectors.ANALYTIC.update(self.saved)
        return False


def raw_below(bc: str = "obc0", n_cap: Optional[int] = None) -> List[Dict]:
    """
    Rules whose fitted a*b < 2, recomputed from the series with R16's analytic
    override lifted -- i.e. the state R9 Fig. 1 was drawn in.  `corrected_below`
    asks the same question of the map as it stands now, and must return [].
    """
    n_cap = sectors.UNIFORM_N_CAP if n_cap is None else n_cap
    out = []
    with without_r16_override():
        for r in range(256):
            pt = sectors.rule_point(r, bc, n_cap)
            if pt is None:
                continue
            h = sectors.hyperbola_check(pt)
            if h.get("raw_below"):
                h["tuple"] = pt["tuple"]
                h["family"] = pt["family"]
                out.append(h)
    return out


def corrected_below(bc: str = "obc0") -> List[int]:
    """The same question asked of the sector map as it stands.  Empty once the
    override is in place; this is the regression check that it took effect."""
    d = sectors.load(bc)
    if d is None:
        raise FileNotFoundError(f"no sector map for {bc}; run sectors.main first")
    return [h["rule"] for h in d["hyperbola"] if h.get("raw_below")]


def _visual_from_points(points: List[Dict]) -> Dict:
    out = []
    for p in points:
        a, b = p["n_wcc"]["base"], p["d_max_wcc"]["base"]
        if a is None or b is None:
            continue
        if a * b < 2.0 - 1e-9:
            continue                      # genuinely below, case (2)
        if (a + FAM_DX[p["family"]]) * b < 2.0 - 1e-9:
            out.append({"rule": p["rule"], "family": p["family"], "a": a, "b": b,
                        "ab": a * b, "a_drawn": a + FAM_DX[p["family"]],
                        "ab_drawn": (a + FAM_DX[p["family"]]) * b})
    return {"n": len(out), "rules": sorted(r["rule"] for r in out),
            "detail": out,
            "families": sorted({r["family"] for r in out})}


def visual_below(bc: str = "obc0", pre: bool = False) -> Dict:
    """
    Markers drawn under the dotted line by the family offset ALONE, i.e. rules
    with a*b >= 2 but (a + dx)*b < 2.  A rendering artefact of F1, nothing to do
    with the fits.

    `pre=True` reconstructs the count for the figure as it was BEFORE R16's
    correction (35 rules); afterwards the five V-free members of the seven join
    the same cell and it is 40.
    """
    if not pre:
        return _visual_from_points(sectors.load(bc)["points"])
    with without_r16_override():
        pts = [p for p in (sectors.rule_point(r, bc, sectors.UNIFORM_N_CAP)
                           for r in range(256)) if p is not None]
    return _visual_from_points(pts)


# --- 2. the sector count is an exact staircase -------------------------------

def quasilinear_formula(Ns: Sequence[int], ys: Sequence[int],
                        max_period: int = 4) -> Optional[Dict]:
    """
    The smallest p for which y is EXACTLY affine in N on each residue class
    mod p, i.e. y = A_r N + B_r for N = r (mod p) with rational A_r, B_r.

    This is what a staircase like ceil(N/2)+1 is, and it is the statement that
    makes a = 1 airtight: no exponential can agree with an exact affine law on
    every residue class over thirteen consecutive N.  Returns None if no period
    up to `max_period` fits exactly.
    """
    for p in range(1, max_period + 1):
        slopes, offsets, ok = {}, {}, True
        for r in range(p):
            pts = [(N, y) for N, y in zip(Ns, ys) if N % p == r]
            if len(pts) < 3:                     # not enough to claim a law
                ok = False
                break
            (n0, y0), (n1, y1) = pts[0], pts[1]
            A = Fraction(y1 - y0, n1 - n0)
            B = Fraction(y0) - A * n0
            if any(A * N + B != y for N, y in pts):
                ok = False
                break
            slopes[r], offsets[r] = A, B
        if ok:
            return {"period": p,
                    "slope": {r: str(slopes[r]) for r in slopes},
                    "offset": {r: str(offsets[r]) for r in offsets},
                    "slope_common": (str(next(iter(slopes.values())))
                                     if len(set(slopes.values())) == 1 else None),
                    "n_points": len(Ns), "N_range": [min(Ns), max(Ns)]}
    return None


#: the two closed forms, for the report; checked against quasilinear_formula
CLOSED_FORM = {
    (6, 14, 20, 84): ("ceil(N/2) + 1", lambda N: (N + 1) // 2 + 1),
    (74, 88, 229): ("floor((N+4)/3)", lambda N: (N + 4) // 3),
}


#: LaTeX rendering of the same two forms (kept apart from the callable so the
#: table generator never has to put a backslash inside an f-string expression --
#: a syntax error before Python 3.12).
CLOSED_FORM_TEX = {
    "ceil(N/2) + 1": r"$\lceil N/2\rceil+1$",
    "floor((N+4)/3)": r"$\lfloor (N+4)/3\rfloor$",
}


def closed_form_for(rule: int) -> Optional[Tuple[str, object]]:
    for group, cf in CLOSED_FORM.items():
        if rule in group:
            return cf
    return None


def staircase_check(rule: int, bc: str = "obc0") -> Dict:
    """Does the closed form reproduce n_wcc at every N on record (no cap)?"""
    recs = results_io.load_wcc_results(rule, bc)
    Ns = sorted(recs)
    ys = [recs[N]["n_wcc"] for N in Ns]
    cf = closed_form_for(rule)
    exact = cf is not None and all(cf[1](N) == y for N, y in zip(Ns, ys))
    return {"rule": rule, "N": Ns, "n_wcc": ys,
            "closed_form": cf[0] if cf else None,
            "closed_form_exact": bool(exact),
            "quasilinear": quasilinear_formula(Ns, ys)}


# --- 3. the squeeze that fixes b = 2 -----------------------------------------

def squeeze(rule: int, bc: str = "obc0") -> Dict:
    """
    D_max >= 2^N / n_wcc at every N (pigeonhole on the partition), so the finite-N
    lower bound on the base is (2^N / n_wcc)^(1/N).  It converges to 2 from below
    at the rate (cN)^(-1/N), which is why no accessible N makes the bound bite:
    at N = 16 with n_wcc = 9 it is only 1.743.  The ASYMPTOTIC statement is
    nevertheless exact, and it is the whole answer.
    """
    recs = results_io.load_wcc_results(rule, bc)
    Ns = sorted(recs)
    rows = []
    for N in Ns:
        n, dm = recs[N]["n_wcc"], recs[N]["d_max_wcc"]
        avg = (1 << N) / n
        rows.append({"N": N, "n_wcc": n, "d_max": dm,
                     "average_sector": avg,
                     "d_max_over_average": dm / avg,
                     "b_lower_bound_at_N": avg ** (1.0 / N),
                     "finite_theorem": n * dm / (1 << N)})
    return {"rule": rule, "bc": bc, "rows": rows,
            "min_finite_theorem": min(r["finite_theorem"] for r in rows),
            "b_bound_at_Nmax": rows[-1]["b_lower_bound_at_N"],
            "asymptotic_base": 2.0,
            "argument": "n_wcc = Theta(N) => (2^N/n_wcc)^(1/N) -> 2, and "
                        "D_max <= 2^N, so b = 2 exactly"}


def secant_bases(Ns: Sequence[int], ys: Sequence[float],
                 moduli: Sequence[int] = (1, 2, 3),
                 min_span: int = 6, tail_span: int = 8) -> Dict:
    """
    Growth rate between two points of the SAME residue class,
    (y_j / y_i) ** (1 / (N_j - N_i)).

    This is the one diagnostic here that needs no asymptotics and no model.  A
    pure exponential c*b^N gives exactly b on every pair, so a secant ABOVE 2 is
    a finite-N refutation of any base at or below 2 -- and combined with
    D_max <= 2^N, which caps the base at 2, it pins b = 2 from the data alone.

    The residue class matters.  n_wcc is a staircase with period 2 or 3, D_max
    inherits the modulation, and a secant taken across residues measures the
    modulation as much as the growth: for W74 the residual alternates by nearly
    50% between odd and even N, which is larger than the entire effect being
    argued about.  Within a class the modulation cancels.
    """
    def sec(pts):
        (n0, y0), (n1, y1) = pts[0], pts[-1]
        return {"N_from": n0, "N_to": n1,
                "base": float((y1 / y0) ** (1.0 / (n1 - n0)))}

    n_max = max(Ns)
    best = best_tail = None
    per_class = []
    for m in moduli:
        for k in range(m):
            pts = [(N, y) for N, y in zip(Ns, ys) if N % m == k]
            if len(pts) < 2 or pts[-1][0] - pts[0][0] < min_span:
                continue
            rec = {"modulus": m, "residue": k}
            rec.update(sec(pts))
            # The same secant restricted to the tail.  The early points are the
            # ones where the power-law prefactor is still moving fast, so the
            # widest secant is systematically dragged down by them; the tail is
            # where the two models have to disagree if they disagree at all.
            tp = [p for p in pts if p[0] >= n_max - tail_span]
            rec["tail"] = sec(tp) if len(tp) >= 2 else None
            per_class.append(rec)
            if best is None or rec["base"] > best["base"]:
                best = rec
            if rec["tail"] and (best_tail is None
                                or rec["tail"]["base"] > best_tail["tail"]["base"]):
                best_tail = rec
    return {"per_class": per_class, "best": best, "best_tail": best_tail,
            "tail_span": tail_span,
            # b <= 2 is a theorem (D_max <= 2^N).  A pure exponential c*b^N has
            # EVERY secant equal to b, so a secant above 2 does not merely favour
            # base 2 -- it refutes the fitted model outright on that stretch.
            "refutes_sub_two": bool(best_tail is not None
                                    and best_tail["tail"]["base"] > 2.0)}


def secant(rule: int, bc: str = "obc0") -> Dict:
    e = extended_series(rule, bc)
    return secant_bases(e["N"], e["d_max"])


# --- 4. why the discriminator refused ----------------------------------------

def discriminator_margin(rule: int, bc: str = "obc0",
                         n_cap: int = 16, tail: int = 6) -> Dict:
    """
    The two residual sums sectors._volume_fraction_base compares, reported as a
    margin instead of a boolean.  `power_wins` on BOTH windows is what the
    branch requires; the point of this function is that the tail decision is a
    coin flip, not a considered rejection.
    """
    s = sectors.load_series(rule, bc, n_cap)
    Ns, ys = s["N"], s["d_max_wcc"]
    ratio = np.log([y / (1 << N) for N, y in zip(Ns, ys)])
    allN = np.asarray(Ns, float)

    def window(k: int) -> Dict:
        n, r = allN[-k:], ratio[-k:]

        def rss(x):
            A = np.column_stack([np.ones_like(x), x])
            beta, *_ = np.linalg.lstsq(A, r, rcond=None)
            return float(((r - A @ beta) ** 2).sum()), beta
        rp, bp = rss(np.log(n))
        re, be = rss(n)
        return {"k": k, "rss_power": rp, "rss_exp": re,
                "power_wins": bool(rp < re),
                "relative_margin": (re - rp) / max(re, rp),
                "alpha": float(bp[1]), "base_from_exp": 2 * math.exp(float(be[1]))}

    w_tail, w_full = window(tail), window(len(ys))
    return {"rule": rule, "bc": bc, "tail": w_tail, "full": w_full,
            "branch_fires": bool(w_tail["power_wins"] and w_full["power_wins"]
                                 and abs(w_tail["alpha"]) <= sectors._ALPHA_MAX),
            "alpha_at_base_2": sectors._alpha_at_base(Ns, ys, 2.0)[0]}


# --- 5. the rolling base, against two controls -------------------------------

def rolling_base(Ns: Sequence[int], ys: Sequence[float],
                 win: int = ROLL_WIN) -> List[Tuple[int, float]]:
    """Pure-exponential base fitted on a sliding window of `win` points, keyed by
    the LAST N in the window.  A genuine base is flat in N; a power-law prefactor
    on base 2 drifts upward towards 2, and never gets there at any reachable N."""
    out = []
    for i in range(len(ys) - win + 1):
        n = np.asarray(Ns[i:i + win], float)
        y = np.log(np.asarray(ys[i:i + win], float))
        A = np.column_stack([np.ones_like(n), n])
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        out.append((int(n[-1]), float(math.exp(beta[1]))))
    return out


def cumulative_base(Ns: Sequence[int], ys: Sequence[float],
                    start: int = 5) -> List[Tuple[int, float]]:
    """
    Pure-exponential base fitted on N_min..M, as M grows.  Smoother than
    `rolling_base` -- these series oscillate with parity and a short window is
    dominated by it -- and it is the estimator that actually answers the
    question: with a genuine base it is flat in M, with c*2^N*N^alpha it climbs
    monotonically toward 2 and never arrives.
    """
    out = []
    for k in range(start, len(ys) + 1):
        n = np.asarray(Ns[:k], float)
        y = np.log(np.asarray(ys[:k], float))
        A = np.column_stack([np.ones_like(n), n])
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        out.append((int(n[-1]), float(math.exp(beta[1]))))
    return out


def controls(Ns: Sequence[int], null_base: float = 1.9226) -> Dict:
    """
    The two hypotheses, evaluated on the same grid:
      * ALTERNATIVE  c * 2^N * N^-1/2 (central binomials) -- base 2, certified;
      * NULL         null_base^N -- a genuine sub-2 base, flat by construction.
    """
    binom = [math.comb(N, N // 2) for N in Ns]
    geo = [null_base ** N for N in Ns]
    return {"N": list(Ns),
            "binomial": binom, "geometric": geo,
            "rolling_binomial": rolling_base(Ns, binom),
            "rolling_geometric": rolling_base(Ns, geo),
            "cumulative_binomial": cumulative_base(Ns, binom),
            "cumulative_geometric": cumulative_base(Ns, geo)}


def extended_series(rule: int, bc: str = "obc0") -> Dict:
    """Every N on record, ignoring UNIFORM_N_CAP.  The cap keeps the headline map
    like-for-like across 256 rules; here we want the longest lever arm there is."""
    recs = results_io.load_wcc_results(rule, bc)
    Ns = sorted(recs)
    return {"N": Ns,
            "n_wcc": [recs[N]["n_wcc"] for N in Ns],
            "d_max": [recs[N]["d_max_wcc"] for N in Ns]}


# --- 6. assemble -------------------------------------------------------------

def census(bc: str = "obc0") -> Dict:
    below = raw_below(bc)
    rules_below = [h["rule"] for h in below]
    out = {
        "bc": bc,
        "n_below": len(below),
        "rules": rules_below,
        "visual_offset_artefact": visual_below(bc, pre=True),
        "visual_offset_artefact_after": visual_below(bc, pre=False),
        "still_below_after_override": corrected_below(bc),
        "per_rule": [],
    }
    for h in below:
        r = h["rule"]
        fam, tup = {r: h["family"]}, {r: h["tuple"]}
        st = staircase_check(r, bc)
        sq = squeeze(r, bc)
        dm = discriminator_margin(r, bc)
        s16 = sectors.load_series(r, bc, sectors.UNIFORM_N_CAP)
        ext = extended_series(r, bc)
        out["per_rule"].append({
            "rule": r, "tuple": tup[r], "family": fam[r],
            "fitted_a": h["a"], "fitted_b": h["b"], "fitted_ab": h["product"],
            "verdict_in_R9": h["verdict"],
            "staircase": st,
            "squeeze": {k: sq[k] for k in ("min_finite_theorem",
                                           "b_bound_at_Nmax", "asymptotic_base",
                                           "argument")},
            "discriminator": dm,
            "secant": secant_bases(ext["N"], ext["d_max"]),
            "corrected_a": 1.0, "corrected_b": 2.0,
            "corrected_alpha_b": dm["alpha_at_base_2"],
            "rolling": rolling_base(s16["N"], s16["d_max_wcc"]),
            "extended": ext,
            "cumulative": cumulative_base(ext["N"], ext["d_max"]),
        })
    ctrl_N = list(range(6, max(p["extended"]["N"][-1]
                               for p in out["per_rule"]) + 1))
    out["controls"] = controls(ctrl_N)
    cs = extended_series(CONTROL_RULE, bc)
    out["control_rule"] = {"rule": CONTROL_RULE, "N": cs["N"],
                           "d_max": cs["d_max"],
                           "rolling": rolling_base(cs["N"], cs["d_max"]),
                           "cumulative": cumulative_base(cs["N"], cs["d_max"])}
    out["all_finite_theorem_ok"] = all(
        p["squeeze"]["min_finite_theorem"] >= 1.0 - 1e-9 for p in out["per_rule"])
    out["all_staircases_exact"] = all(
        p["staircase"]["closed_form_exact"] for p in out["per_rule"])
    return out


def build(bc: str = "obc0") -> Dict:
    c = census(bc)
    with open(OUT_JSON.format(bc=bc), "w") as f:
        json.dump(c, f, indent=1)
    return c


def load(bc: str = "obc0") -> Optional[Dict]:
    p = OUT_JSON.format(bc=bc)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


# --- 7. figures --------------------------------------------------------------

_C_BELOW = "#c62828"
_C_CTRL = "#1f4e9c"
_C_NULL = "#8a8a85"
_C_ART = "#6f9f6f"


def _style(ax):
    ax.grid(True, color="#e9e9e6", linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _distinct(c: Dict) -> List[Dict]:
    """The seven collapse onto FOUR distinct series -- W6/14/20/84 share one
    exactly -- and plotting the duplicates four times just thickens a line and
    repeats a legend entry."""
    groups: Dict[tuple, List[Dict]] = {}
    for p in c["per_rule"]:
        groups.setdefault(tuple(p["extended"]["d_max"][:11]), []).append(p)
    out = []
    for _, ps in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        rs = sorted(x["rule"] for x in ps)
        lab = "$W_{" + "}, W_{".join(str(x) for x in rs) + "}$"
        out.append({"label": lab, "rules": rs, "rep": ps[0]})
    return out


def fig_below(bc: str, out: str, c: Optional[Dict] = None):
    """
    Three panels: where the markers actually are, what the residual looks like,
    and the base estimate against both controls.
    """
    c = c or load(bc) or build(bc)
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.4))
    shades = ["#c62828", "#e05c3a", "#a8324a", "#d98324"]
    dis = _distinct(c)

    # (a) the corner of F1, drawn twice: as plotted and as the data says
    ax = axes[0]
    _style(ax)
    a = np.linspace(0.90, 1.42, 300)
    ax.plot(a, 2.0 / a, color="#444444", lw=1.0, ls=":", zorder=2)
    ax.annotate("$ab=2$", (1.34, 2.0 / 1.34), xytext=(4, 6),
                textcoords="offset points", fontsize=7.6, color="#444444")
    art = c["visual_offset_artefact"]
    ax.scatter([1.0 - 0.019], [2.0], s=330, facecolor=_C_ART,
               edgecolor="white", linewidth=1.0, alpha=0.8, zorder=4)
    ax.annotate(str(art["n"]), (1.0 - 0.019, 2.0), fontsize=7.5, ha="center",
                va="center", color="white", fontweight="bold", zorder=9)
    ax.annotate("$" + str(art["n"]) + "$ V-free rules exactly ON the\n"
                "curve, drawn at $a-0.019$ and so\n"
                "under it: a rendering artefact",
                (1.0 - 0.019, 2.0), xytext=(0.075, 0.86),
                textcoords="axes fraction", fontsize=7.4, ha="left",
                color="#3d6b3d",
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#3d6b3d"))
    for p in c["per_rule"]:
        dx = FAM_DX[p["family"]]
        ax.scatter([p["fitted_a"] + dx], [p["fitted_b"]], s=44,
                   facecolor="white", edgecolor=_C_BELOW, linewidth=1.5,
                   zorder=6)
    ax.annotate("", xy=(1.0, 1.985), xytext=(1.0, 1.932),
                arrowprops=dict(arrowstyle="->", lw=1.2, color=_C_BELOW),
                zorder=8)
    ax.scatter([1.0], [2.0], s=70, marker="X", color=_C_BELOW, zorder=7)
    ax.annotate("the seven, as fitted:\n$a=1$, $b=1.919$–$1.937$",
                (1.0, 1.9226), xytext=(0.40, 0.10), textcoords="axes fraction",
                fontsize=7.4, color=_C_BELOW,
                arrowprops=dict(arrowstyle="->", lw=0.8, color=_C_BELOW))
    ax.annotate("$b=2$ exactly,\nby the squeeze", (1.0, 2.0),
                xytext=(0.56, 0.70), textcoords="axes fraction", fontsize=7.4,
                color=_C_BELOW)
    ax.set_xlim(0.945, 1.42)
    ax.set_ylim(1.78, 2.16)
    ax.set_xlabel("$a$  (base of $n_{\\rm wcc}$)")
    ax.set_ylabel("$b$  (base of $D_{\\max}$)")
    ax.set_title("(a)  the corner of F1: two ways to sit low", fontsize=9.5)

    # (b) the residual D_max / 2^N, normalised at N=6, log-log.
    #     A power-law prefactor is a STRAIGHT LINE here; a genuine base < 2 bends.
    ax = axes[1]
    _style(ax)

    def even(Ns, ys):
        """Even N only.  W74's residual alternates by nearly 50% between odd and
        even N -- a separate finite-size feature -- and plotting both branches
        turns every curve into a zigzag that hides the trend being compared."""
        pairs = [(N, y) for N, y in zip(Ns, ys) if N % 2 == 0]
        r = [y / (1 << N) for N, y in pairs]
        return [p[0] for p in pairs], [x / r[0] for x in r]
    for col, g in zip(shades, dis):
        e = g["rep"]["extended"]
        ax.plot(*even(e["N"], e["d_max"]), marker="o", ms=3.4, lw=1.2,
                color=col, alpha=0.9, label=g["label"])
    cr = c["control_rule"]
    ax.plot(*even(cr["N"], cr["d_max"]), marker="s", ms=3.8, lw=1.7,
            color=_C_CTRL, label="$W_{134}$: $c\\,2^N N^{-1/2}$, base $2$")
    ctl = c["controls"]
    ax.plot(*even(ctl["N"], ctl["geometric"]), ls="--", lw=1.7,
            color=_C_NULL, label="null: a genuine base $1.9226$")
    # W229's ODD branch stops decaying near N = 15 and then climbs.  For
    # c*b^N with b < 2 the residual c*(b/2)^N must decay geometrically forever,
    # so this one branch refutes the fitted model without any asymptotics --
    # it is the strongest model-free statement in the figure and deserves to be
    # drawn rather than described.
    p229 = next(p for p in c["per_rule"] if p["rule"] == 229)
    e = p229["extended"]
    odd = [(N, y) for N, y in zip(e["N"], e["d_max"]) if N % 2 == 1]
    r0 = odd[0][1] / (1 << odd[0][0])
    ax.plot([N for N, _ in odd], [y / (1 << N) / r0 for N, y in odd],
            marker="^", ms=3.6, lw=1.4, color="#7b3fa0", ls="-.",
            label="$W_{229}$, odd $N$: stops decaying")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([6, 8, 10, 12, 16, 20])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("$N$  (even branch, except where marked)")
    ax.set_ylabel("$D_{\\max}/2^N$, normalised at $N=6$")
    ax.set_title("(b)  the residual: a power of $N$, or an exponential?",
                 fontsize=9.5)
    ax.legend(fontsize=6.8, frameon=False, loc="lower left")

    # (c) cumulative base
    ax = axes[2]
    _style(ax)
    for col, g in zip(shades, dis):
        cu = g["rep"]["cumulative"]
        ax.plot([x[0] for x in cu], [x[1] for x in cu], "-o", ms=3, lw=1.1,
                color=col, alpha=0.9, label=g["label"])
    ax.plot([x[0] for x in cr["cumulative"]], [x[1] for x in cr["cumulative"]],
            "-s", ms=3.6, lw=1.6, color=_C_CTRL, label="$W_{134}$ (base $2$)")
    # W134's own data stops at N=17; the same series continued analytically runs
    # the whole window, so the comparison with the seven is like-for-like at the
    # right-hand edge, where it matters.
    cbi = ctl["cumulative_binomial"]
    ax.plot([x[0] for x in cbi], [x[1] for x in cbi], ":", lw=1.6,
            color=_C_CTRL, label=r"$\binom{N}{N/2}$ continued (base $2$)")
    cg = ctl["cumulative_geometric"]
    ax.plot([x[0] for x in cg], [x[1] for x in cg], "--", lw=1.6, color=_C_NULL,
            label="null: a genuine base $1.9226$")
    ax.axhline(2.0, color="#444444", lw=0.9, ls=":")
    ax.annotate("$b=2$", (0.02, 2.0), xycoords=("axes fraction", "data"),
                fontsize=7.6, va="bottom", color="#444444")
    ax.set_xlabel("largest $N$ used in the fit")
    ax.set_ylabel("pure-exponential base on $N=6\\ldots M$")
    # NOT "climbing towards 2": only the ceil(N/2)+1 group climbs monotonically.
    # The floor((N+4)/3) group carries a period-3 modulation that a base fitted
    # across residues partly absorbs, so its trace wanders.  The honest reading
    # is the band, not the slope.
    ax.set_title("(c)  fitting a base on a short window", fontsize=9.5)
    ax.legend(fontsize=6.8, frameon=False, loc="lower right")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


# --- 8. tables ---------------------------------------------------------------

def write_tables(bc: str, outdir: str, c: Optional[Dict] = None):
    c = c or load(bc) or build(bc)
    os.makedirs(outdir, exist_ok=True)
    rows = []
    for p in sorted(c["per_rule"], key=lambda x: x["rule"]):
        cf = CLOSED_FORM_TEX.get(p["staircase"]["closed_form"], "---")
        won = "yes" if p["discriminator"]["full"]["power_wins"] else "no"
        t = p["secant"]["best_tail"]
        sec = (f"{t['tail']['base']:.4f}", f"{t['tail']['N_from']}"
               f"\\ldots{t['tail']['N_to']}")
        rows.append(
            f"$W_{{{p['rule']}}}$ & \\rt{{{p['tuple']}}} & {cf} & "
            f"{p['fitted_b']:.4f} & {p['fitted_ab']:.4f} & "
            f"{p['discriminator']['tail']['relative_margin'] * 100:+.1f}\\% & "
            f"{won} & {sec[0]} & {sec[1]} & {p['corrected_alpha_b']:+.3f} \\\\")
    body = "\n".join(rows)
    tab = ("\\begin{tabular}{llcccrccrc}\n\\toprule\n"
           "rule & tuple & $n_\\wcc$ exactly & fitted $b$ & fitted $ab$ & "
           "tail margin & power wins full & tail secant & over $N$ & "
           "$\\alpha$ at $b=2$ \\\\\n\\midrule\n"
           + body + "\n\\bottomrule\n\\end{tabular}\n")
    with open(os.path.join(outdir, f"tab_below_curve_{bc}.tex"), "w") as f:
        f.write(tab)

    #: The controls belong in a table of their own: they are the reason the
    #: correction is derived rather than fitted, and burying them in prose makes
    #: the claim unfalsifiable.
    ctl_rows = []
    for lab, rule, note in (
            ("$W_{134}$", CONTROL_RULE,
             "central binomials, $b=2$ by derivation"),
            ("$W_{28}$", 28, "a genuine sub-$2$ base"),
    ):
        s = secant(rule, bc)
        t = s["best_tail"]["tail"]
        ext = extended_series(rule, bc)
        ctl_rows.append(f"{lab} & {note} & {ext['N'][-1]} & "
                        f"{s['best']['base']:.4f} & {t['base']:.4f} & "
                        f"{'yes' if s['refutes_sub_two'] else 'no'} \\\\")
    ctl = ("\\begin{tabular}{llrrrc}\n\\toprule\n"
           "control & what it is & $N_{\\max}$ & widest secant & tail secant & "
           "secant $>2$ \\\\\n\\midrule\n" + "\n".join(ctl_rows)
           + "\n\\bottomrule\n\\end{tabular}\n")
    with open(os.path.join(outdir, f"tab_below_controls_{bc}.tex"), "w") as f:
        f.write(ctl)
    return tab


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="rules below the ab>=2 hyperbola")
    ap.add_argument("--bc", default="obc0")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args(argv)
    c = build(args.bc) if args.rebuild else (load(args.bc) or build(args.bc))
    print(f"raw below the curve (override lifted) : {c['n_below']}  {c['rules']}")
    print(f"still below with the override in place : "
          f"{c['still_below_after_override']}")
    print(f"drawn below by the family offset alone : "
          f"{c['visual_offset_artefact']['n']} before the correction, "
          f"{c['visual_offset_artefact_after']['n']} after "
          f"({', '.join(c['visual_offset_artefact']['families'])})")
    print(f"finite-N theorem holds everywhere : {c['all_finite_theorem_ok']}")
    print(f"all sector staircases exact       : {c['all_staircases_exact']}")
    for p in c["per_rule"]:
        d = p["discriminator"]
        print(f"  W{p['rule']:<4d} {p['tuple']}  fitted ab={p['fitted_ab']:.4f}"
              f"  n_wcc={p['staircase']['closed_form']}"
              f"  tail margin={d['tail']['relative_margin'] * 100:+.1f}%"
              f"  full power wins={d['full']['power_wins']}"
              f"  tail secant={p['secant']['best_tail']['tail']['base']:.4f}"
              f"  -> (1, 2) with alpha={p['corrected_alpha_b']:+.3f}")
    figdir = os.path.join(results_io.REPO_ROOT, "figures")
    os.makedirs(figdir, exist_ok=True)
    fig_below(args.bc, os.path.join(figdir, f"fig_below_curve_{args.bc}"), c)
    write_tables(args.bc, os.path.join(results_io.REPO_ROOT, "reports", "tex"), c)
    print("figure + table written")


if __name__ == "__main__":
    main()
