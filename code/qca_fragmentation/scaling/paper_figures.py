"""R19: the exact laws behind the paper figures, and the tables that quote them.

`notebooks/paper_figures.ipynb` draws three figures -- sector count vs N, D_max
vs N, and the sector-size distribution of W156 and W108.  All the *drawing* code
lives in the notebook, where it is meant to be edited.  What lives here is the
part that must NOT drift: the closed forms annotated on the curves.

Scope: **all sixteen unitary rules** at obc0, ergodic ones included.  The
sixteen fall into four behaviours and nothing else --

    ergodic      51, 57, 99, 147, 153, 195   one sector, the whole space
                 54                          one sector plus one frozen state
    polynomial   60, 102, 105, 150           Theta(N) sectors, each huge
    exponential  108, 156, 198, 201          a^N sectors with 1 < a < 2
    frozen       204                         2^N sectors, all of size 1

Every law in `LAWS` was read off the integer series and is then checked at every
computed N by `verify`.  The notebook calls it before drawing and the test suite
calls it independently, so a figure can never carry a tag the data has stopped
supporting.  Nothing here is fitted -- contrast scaling/sectors.py, where the
`source` field tells you which bases are derived and which are regressions.

Two structural facts organise the table.  Reflection (r01 <-> r10) IS a symmetry
of obc0, so the sixteen rules collapse to twelve distinct series; the pairs
{57,99}, {60,102}, {153,195}, {156,198} are term-for-term identical and
`reflection_identities` checks it.  The spin flip (r00 <-> r11, r01 <-> r10) is
NOT: it maps the vacuum-0 padding to vacuum-1.  Its five pairs straddle the
classification -- (108,201) stays exponential and (156,198) is identical, but
(60,195) and (102,153) put a polynomially fragmented rule opposite an ergodic
one.  `spinflip_break` records exactly that.

The one asymptotic statement, W156's D_max ~ 4^{N/5}, is deliberately NOT in
`LAWS`: at finite N the largest sector is a mixture of rooms of length 2 and 3
(see the derivation in scaling/sectors.py), so no exact recurrence holds over
the computed range.  `room_base_check` reports the five-step ratio instead,
which converges to 4^{1/5} to seven digits by N = 21.
"""

from __future__ import annotations

import os
from math import comb
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .. import results_io
from ..core import rules
from .summary import load_series  # noqa: F401  (kept for callers/back-compat)

TEX_DIR = os.path.join(results_io.REPO_ROOT, "reports", "tex")

#: the sixteen rules with no D/E channel, in Wolfram order
UNITARY: Tuple[int, ...] = tuple(sorted(rules.UNITARY_RULES))

#: the six that thermalise completely: one sector of size 2^N
ERGODIC_SIX: Tuple[int, ...] = (51, 57, 99, 147, 153, 195)

#: rule -> behaviour class.  Every unitary rule has exactly one.
CLASS_OF: Dict[int, str] = {
    **{r: "ergodic" for r in ERGODIC_SIX},
    54: "ergodic",
    60: "polynomial", 102: "polynomial", 105: "polynomial", 150: "polynomial",
    108: "exponential", 156: "exponential", 198: "exponential",
    201: "exponential",
    204: "frozen",
}

#: the three integer series a unit record supports
SERIES_KEYS: Tuple[str, ...] = ("n_recurrent", "d_max", "n_frozen")

#: the two exponentially fragmented rules whose size distribution Figure 3 draws
HIST_RULES = (156, 108)

#: the N shown in Figure 3, light -> dark
HIST_NS = (13, 17, 21)

_F: List[int] = [0, 1]
while len(_F) < 64:
    _F.append(_F[-1] + _F[-2])


def fib(k: int) -> int:
    """Fibonacci with F_1 = F_2 = 1."""
    return _F[k]


def central(N: int, parity: int) -> int:
    """Largest binomial C(N+1, w) over w of the given parity.

    R8 sec.4: rule 150's sectors are the even-weight levels of the domain-wall
    number on a chain of N+1 bonds, so its largest sector is the largest even
    binomial rather than the central one.  They differ whenever the central
    coefficient sits at odd w.
    """
    return max(comb(N + 1, w) for w in range(N + 2) if w % 2 == parity)


# ---------------------------------------------------------------- the series

def _tier1a(rule: int, bc: str) -> Dict[int, Tuple[int, int, Optional[int]]]:
    out = {}
    for N, rec in results_io.load_results(rule, bc).items():
        # An early-exit ("ergodic bound") record has no decomposition at all.
        if rec.get("n_recurrent") is None:
            continue
        sizes = results_io.sizes_from_record(rec, "sizes_recurrent")
        if not sizes:
            continue
        hist = rec.get("size_hist")
        # A histogram with no size-1 key means zero frozen states, not an
        # unknown: W105 at N = 1 (mod 4) is exactly that case.
        frozen = None if not hist else int(hist.get("1", hist.get(1, 0)))
        out[N] = (int(rec["n_recurrent"]), int(sizes[0]), frozen)
    return out


def _tier1e(rule: int, bc: str) -> Dict[int, Tuple[int, int, Optional[int]]]:
    out = {}
    for N, rec in results_io.load_wcc_results(rule, bc).items():
        if rec.get("aborted") or rec.get("n_wcc") is None:
            continue
        out[N] = (int(rec["n_wcc"]), int(rec["d_max_wcc"]),
                  None if rec.get("n_frozen") is None else int(rec["n_frozen"]))
    return out


def unit_series(rule: int, bc: str = "obc0") -> Dict[str, List[int]]:
    """The three integer series for one rule, Tier-1a and Tier-1e merged.

    Two sweeps computed these units.  Tier-1a (`results/`) ran Tarjan and
    reaches N = 21-22 on the fragmented rules but *stops* once a rule is
    classified ergodic; Tier-1e (`results_tier1e/`) ran union-find over all 256
    rules to N = 16-17 and therefore carries the ergodic ones in full.  For a
    unitary rule the weak and strong components coincide, so the two agree
    wherever they overlap -- `wcc_scc_agreement` asserts it -- and the union is
    the honest N range.

    Unlike `summary.load_series`, ergodic units are kept.  Dropping them is
    right when fitting a growth law and wrong when drawing a census: "one
    sector of size 2^N" is a measurement, not a missing value.
    """
    a, e = _tier1a(rule, bc), _tier1e(rule, bc)
    Ns, n, d, fr = [], [], [], []
    for N in sorted(set(a) | set(e)):
        va, ve = a.get(N), e.get(N)
        v = va if va is not None else ve
        Ns.append(N)
        n.append(v[0])
        d.append(v[1])
        f = v[2]
        if f is None and ve is not None:
            f = ve[2]
        if f is None and va is not None:
            f = va[2]
        fr.append(f)
    return {"N": Ns, "n_recurrent": n, "d_max": d, "n_frozen": fr}


def series(rule: int, bc: str = "obc0"):
    """(N grid, sector counts, D_max) as plain integer lists."""
    s = unit_series(rule, bc)
    return s["N"], s["n_recurrent"], s["d_max"]


def wcc_scc_agreement(bc: str = "obc0") -> bool:
    """The two sweeps never disagree on a unitary unit they both computed."""
    for r in UNITARY:
        a, e = _tier1a(r, bc), _tier1e(r, bc)
        for N in sorted(set(a) & set(e)):
            if a[N][:2] != e[N][:2]:
                return False
    return True


# ---------------------------------------------------------------- the laws

def _frozen_105(N: int) -> int:
    """1, 2, 1, 0 as N runs through 2, 3, 0, 1 (mod 4).

    The zero is the interesting entry: at N = 1 (mod 4) rule 105 has no frozen
    state at all, and that is the same residue class at which its sector count
    and D_max part company with rule 150's.
    """
    return {0: 1, 1: 0, 2: 1, 3: 2}[N % 4]


def _frozen_108(N: int) -> int:
    """F_{floor(N/2)+2} * F_{ceil(N/2)+2}: the frozen count factorises.

    A state is frozen iff nothing on either sublattice can move, and the two
    sublattices decouple, so the count is a product of two Fibonacci numbers of
    nearly equal index -- the same even/odd split that carries R18's two
    independent wall charges.
    """
    return fib(N // 2 + 2) * fib((N + 1) // 2 + 2)


def _frozen_201(N: int) -> int:
    """F_{m+1}^2 at N = 2m, F_m F_{m+3} at N = 2m+1.

    Same factorisation as W108 with the indices pulled down by one; at odd N
    the two factors sit three apart instead of one, which by Catalan's identity
    is W108's value at N-2 off by exactly (-1)^m.
    """
    m = N // 2
    return fib(m + 1) ** 2 if N % 2 == 0 else fib(m) * fib(m + 3)


def _pow2(N: int) -> int:
    return 1 << N


#: (rule, series) -> (closed form, human-readable name, LaTeX)
LAWS: Dict[Tuple[int, str], Tuple[Callable[[int], int], str, str]] = {
    (156, "n_recurrent"): (lambda N: fib(N + 2), "F_{N+2}", r"$F_{N+2}$"),
    (198, "n_recurrent"): (lambda N: fib(N + 2), "F_{N+2}", r"$F_{N+2}$"),
    (108, "d_max"): (lambda N: fib(N), "F_N", r"$F_{N}$"),
    (201, "d_max"): (lambda N: fib(N + 2), "F_{N+2}", r"$F_{N+2}$"),
    (60, "n_recurrent"): (lambda N: N + 1, "N+1", r"$N+1$"),
    (60, "d_max"): (lambda N: 2 ** (N - 1), "2^(N-1)", r"$2^{N-1}$"),
    (102, "n_recurrent"): (lambda N: N + 1, "N+1", r"$N+1$"),
    (102, "d_max"): (lambda N: 2 ** (N - 1), "2^(N-1)", r"$2^{N-1}$"),
    (150, "n_recurrent"): (lambda N: (N + 1) // 2 + 1,
                           "floor((N+1)/2)+1",
                           r"$\lfloor (N{+}1)/2\rfloor+1$"),
    (150, "d_max"): (lambda N: central(N, 0),
                     "max over even w of C(N+1,w)",
                     r"$\max_{w\ \mathrm{even}}\binom{N+1}{w}$"),
    (105, "n_recurrent"): (lambda N: N // 2 + 1 + (N % 4 == 3),
                           "floor(N/2)+1+[N=3 mod 4]",
                           r"$\lfloor N/2\rfloor+1+[N\equiv3\ (4)]$"),
    (105, "d_max"): (lambda N: central(N, 1 if N % 4 == 1 else 0),
                     "max C(N+1,w), odd w at N=1 mod 4",
                     r"$\max_{w}\binom{N+1}{w}$, odd $w$ at $N\equiv1\ (4)$"),

    # --- the trivial extremes, which are measurements like any other ---------
    **{(r, "n_recurrent"): (lambda N: 1, "1", r"$1$") for r in ERGODIC_SIX},
    **{(r, "d_max"): (_pow2, "2^N", r"$2^{N}$") for r in ERGODIC_SIX},
    **{(r, "n_frozen"): (lambda N: 0, "0", r"$0$") for r in ERGODIC_SIX},
    (54, "n_recurrent"): (lambda N: 2, "2", r"$2$"),
    (54, "d_max"): (lambda N: (1 << N) - 1, "2^N - 1", r"$2^{N}-1$"),
    (54, "n_frozen"): (lambda N: 1, "1", r"$1$"),
    (204, "n_recurrent"): (_pow2, "2^N", r"$2^{N}$"),
    (204, "d_max"): (lambda N: 1, "1", r"$1$"),
    (204, "n_frozen"): (_pow2, "2^N", r"$2^{N}$"),

    # --- frozen-state counts of the fragmented rules -------------------------
    (60, "n_frozen"): (lambda N: 2, "2", r"$2$"),
    (102, "n_frozen"): (lambda N: 2, "2", r"$2$"),
    (105, "n_frozen"): (_frozen_105, "1,2,1,0 for N=2,3,0,1 mod 4",
                        r"$1,2,1,0$ for $N\equiv2,3,0,1\ (4)$"),
    (150, "n_frozen"): (lambda N: 1 if N % 2 == 0 else 2,
                        "1 if N even else 2", r"$1$ ($N$ even), $2$ (odd)"),
    (156, "n_frozen"): (lambda N: (N + 1) // 2 + 1, "floor((N+1)/2)+1",
                        r"$\lfloor (N{+}1)/2\rfloor+1$"),
    (198, "n_frozen"): (lambda N: (N + 1) // 2 + 1, "floor((N+1)/2)+1",
                        r"$\lfloor (N{+}1)/2\rfloor+1$"),
    (108, "n_frozen"): (_frozen_108, "F_{floor(N/2)+2} F_{ceil(N/2)+2}",
                        r"$F_{\lfloor N/2\rfloor+2}F_{\lceil N/2\rceil+2}$"),
    (201, "n_frozen"): (_frozen_201, "F_{m+1}^2 / F_m F_{m+3}, m=floor(N/2)",
                        r"$F_{m+1}^{2}$ ($N{=}2m$), $F_{m}F_{m+3}$ ($N{=}2m{+}1$)"),
}

#: series label -> how it prints
_SERIES_TEX = {"n_recurrent": r"$n_{\wcc}$", "d_max": r"$D_{\max}$",
               "n_frozen": r"$n_{1}$"}


def check_law(rule: int, key: str, bc: str = "obc0") -> Dict:
    """Evaluate one closed form at every computed N."""
    f = LAWS[(rule, key)][0]
    s = unit_series(rule, bc)
    Ns = [N for N, v in zip(s["N"], s[key]) if v is not None]
    y = [v for v in s[key] if v is not None]
    bad = [N for N, v in zip(Ns, y) if f(N) != v]
    return {"rule": rule, "key": key, "N_min": Ns[0] if Ns else None,
            "N_max": Ns[-1] if Ns else None, "n_points": len(Ns),
            "mismatch": bad, "ok": not bad and bool(Ns)}


def plastic_recurrence(bc: str = "obc0") -> bool:
    """W108's sector count obeys a_N = 2a_{N-1} - a_{N-2} + a_{N-3}.

    The dominant root is rho^2 = 1.754878, rho the plastic number; R18 derives
    the same recurrence from the wall grammar rather than from the series.
    """
    _, n, _ = series(108, bc)
    return len(n) > 3 and all(n[i] == 2 * n[i - 1] - n[i - 2] + n[i - 3]
                              for i in range(3, len(n)))


def shifted_partner(bc: str = "obc0") -> bool:
    """W201's sector count is W108's, evaluated one site shorter."""
    N108, n108, _ = series(108, bc)
    N201, n201, _ = series(201, bc)
    prev = dict(zip(N108, n108))
    overlap = [(N, v) for N, v in zip(N201, n201) if N - 1 in prev]
    return bool(overlap) and all(v == prev[N - 1] for N, v in overlap)


def room_base_check(bc: str = "obc0") -> Dict:
    """W156's D_max grows like 4^{N/5}; report the five-step ratio.

    D_max(N+5)/D_max(N) is 4 asymptotically, so its fifth root estimates the
    base without fitting anything.  Not an exact law -- see the module
    docstring -- which is why it is reported rather than asserted.
    """
    _, _, d = series(156, bc)
    if len(d) < 6:
        return {"ok": False}
    est = (d[-1] / d[-6]) ** 0.2
    return {"ok": True, "estimate": est, "target": 4 ** 0.2,
            "abs_error": abs(est - 4 ** 0.2)}


# ------------------------------------------------- symmetries and the census

def _same_series(a: int, b: int, bc: str) -> bool:
    sa, sb = unit_series(a, bc), unit_series(b, bc)
    common = sorted(set(sa["N"]) & set(sb["N"]))
    if not common:
        return False
    for key in ("n_recurrent", "d_max"):
        da = dict(zip(sa["N"], sa[key]))
        db = dict(zip(sb["N"], sb[key]))
        if any(da[N] != db[N] for N in common):
            return False
    return True


def reflection_identities(bc: str = "obc0") -> Dict[Tuple[int, int], bool]:
    """Reflection is an obc0 symmetry, so each pair must match term for term.

    Not a claim about the drawing: it is why sixteen rules give twelve curves.
    """
    out = {}
    for r in UNITARY:
        rr = rules.reflect_wolfram(r)
        if rr != r and r < rr:
            out[(r, rr)] = _same_series(r, rr, bc)
    return out


def ergodic_six_coincide(bc: str = "obc0") -> bool:
    """All six thermalising rules give literally the same two series."""
    return all(_same_series(ERGODIC_SIX[0], r, bc) for r in ERGODIC_SIX[1:])


def spinflip_partner(rule: int) -> int:
    """(r00,r01,r10,r11) -> (r11,r10,r01,r00): global 0<->1 relabelling.

    X^{tensor N} is a basis permutation and XHX has the same matrix magnitudes,
    so this IS a symmetry of the transition graph under pbc.  obc0 pins the
    padding to |0>, which the relabelling sends to |1>, so it is not one here --
    `spinflip_break` is the measurement of how badly.
    """
    t = rules.wolfram_to_tuple(rule)
    return rules.tuple_to_wolfram((t[3], t[2], t[1], t[0]))


def spinflip_break(bc: str = "obc0") -> List[Dict]:
    """One row per spin-flip pair: the classes it connects, and whether the
    two series survive the broken symmetry."""
    rows, seen = [], set()
    for r in UNITARY:
        p = spinflip_partner(r)
        if p == r or (p, r) in seen:
            continue
        seen.add((r, p))
        rows.append({"pair": (r, p),
                     "classes": (CLASS_OF[r], CLASS_OF[p]),
                     "is_reflection": rules.reflect_wolfram(r) == p,
                     "identical": _same_series(r, p, bc)})
    return rows


def flip_survives_only_via_reflection(bc: str = "obc0") -> bool:
    """A spin-flip pair keeps its series exactly when it is also a reflection
    pair -- i.e. never on the flip's own account.

    (57,99) and (156,198) are both; their series match. (54,147), (60,195),
    (102,153) and (108,201) are not; every one of them parts company, and two
    of those four put a fragmented rule opposite an ergodic one.
    """
    return all(row["identical"] == row["is_reflection"]
               for row in spinflip_break(bc))


def saturation(rule: int, bc: str = "obc0") -> Dict:
    """n_wcc * D_max / 2^N at the largest computed N.

    The hyperbola n_wcc * D_max >= 2^N is an identity-plus-inequality: equality
    holds exactly when every sector has the same size.  Both trivial extremes
    do -- one sector of 2^N, or 2^N sectors of one -- so they sit ON the
    hyperbola and every fragmented rule sits strictly above it.
    """
    s = unit_series(rule, bc)
    N = s["N"][-1]
    n, d = s["n_recurrent"][-1], s["d_max"][-1]
    return {"rule": rule, "N": N, "n": n, "d_max": d,
            "ratio": n * d / float(1 << N), "saturates": n * d == (1 << N)}


def census(bc: str = "obc0") -> List[Dict]:
    """One row per unitary rule: class, N range, endpoint values, saturation."""
    out = []
    for r in UNITARY:
        s = unit_series(r, bc)
        sat = saturation(r, bc)
        out.append({
            "rule": r, "tuple": "".join(rules.wolfram_to_tuple(r)),
            "class": CLASS_OF[r], "reflect": rules.reflect_wolfram(r),
            "flip": spinflip_partner(r),
            "N_min": s["N"][0], "N_max": s["N"][-1], "n_points": len(s["N"]),
            "n_recurrent": s["n_recurrent"][-1], "d_max": s["d_max"][-1],
            "n_frozen": s["n_frozen"][-1], "ratio": sat["ratio"],
            "laws": [k for (rr, k) in LAWS if rr == r],
        })
    return out


def verify(bc: str = "obc0") -> Dict:
    """Every claim the figure annotations make, checked against the data."""
    checks = {(r, k): check_law(r, k, bc) for (r, k) in LAWS}
    refl = reflection_identities(bc)
    hyper = {r: saturation(r, bc) for r in UNITARY}
    sat_ok = all(h["saturates"] == (h["rule"] in ERGODIC_SIX or h["rule"] == 204)
                 for h in hyper.values())
    return {"laws": checks,
            "plastic_recurrence": plastic_recurrence(bc),
            "shifted_partner": shifted_partner(bc),
            "room_base": room_base_check(bc),
            "reflection": refl,
            "ergodic_six_coincide": ergodic_six_coincide(bc),
            "spinflip": spinflip_break(bc),
            "flip_needs_reflection": flip_survives_only_via_reflection(bc),
            "wcc_scc": wcc_scc_agreement(bc),
            "hyperbola": hyper,
            "ok": all(c["ok"] for c in checks.values())
                  and plastic_recurrence(bc) and shifted_partner(bc)
                  and all(refl.values()) and ergodic_six_coincide(bc)
                  and wcc_scc_agreement(bc) and sat_ok
                  and flip_survives_only_via_reflection(bc)}


def size_stats(rule: int, N: int, bc: str = "obc0") -> Dict:
    """Sector-size summary for one (rule, N), from the complete `size_hist`.

    `sizes_recurrent` in the record is truncated at 2048 entries for the large
    N of these rules; `size_hist` is not.  The two asserts are what make it
    safe to prefer the histogram, and they are the reason Figure 3 can run out
    to N = 21 at all.
    """
    rec = results_io.load_results(rule, bc)[N]
    h = {int(k): int(v) for k, v in rec["size_hist"].items()}
    assert sum(h.values()) == rec["n_recurrent"], (rule, N, "class count")
    assert sum(s * c for s, c in h.items()) == 2 ** N, (rule, N, "mass")

    sizes = sorted(h)
    total = float(2 ** N)
    cum, median = 0.0, sizes[-1]
    for s in sizes:
        cum += s * h[s] / total
        if cum >= 0.5:
            median = s
            break
    d_max = sizes[-1]
    n_sec = sum(h.values())
    return {"rule": rule, "N": N, "n_sectors": n_sec, "d_max": d_max,
            "d_max_fraction": d_max / total, "mean": total / n_sec,
            "median": median, "n_distinct_sizes": len(sizes),
            "n_singletons": h.get(1, 0)}


# ------------------------------------------------------------------ tables

#: the twelve reflection classes, in class order, as (rules, representative)
LAW_ROWS: Tuple[Tuple[Tuple[int, ...], int], ...] = (
    (ERGODIC_SIX[:1], 51), ((57, 99), 57), ((147,), 147), ((153, 195), 153),
    ((54,), 54),
    ((60, 102), 60), ((105,), 105), ((150,), 150),
    ((108,), 108), ((156, 198), 156), ((201,), 201),
    ((204,), 204),
)


def _law_cell(rule: int, key: str, v: Dict) -> str:
    if (rule, key) not in LAWS:
        return "---"
    tex = LAWS[(rule, key)][2]
    return tex if v["laws"][(rule, key)]["ok"] else r"\textbf{FAILS}"


def _tex_laws(bc: str) -> str:
    v = verify(bc)
    rows = []
    prev_class = None
    for group, rep in LAW_ROWS:
        cls = CLASS_OF[rep]
        if prev_class is not None and cls != prev_class:
            rows.append("\\midrule")
        prev_class = cls
        c = [v["laws"][(rep, k)] for k in SERIES_KEYS if (rep, k) in LAWS]
        rng = f"${min(x['N_min'] for x in c)}$--${max(x['N_max'] for x in c)}$"
        ok = "yes" if all(x["ok"] for x in c) else "NO"
        rows.append(
            f"{', '.join(str(r) for r in group)} & "
            f"\\rt{{{''.join(rules.wolfram_to_tuple(rep))}}} & {cls} & "
            + " & ".join(_law_cell(rep, k, v) for k in SERIES_KEYS)
            + f" & {rng} & {ok} \\\\")
    body = "\n".join(rows)
    return (
        "\\begin{table}[htbp]\n\\centering\n\\footnotesize\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        # All three closed-form columns wrap: W105's D_max carries a residue
        # qualifier and W201's frozen count is case-split, either of which is
        # wider than the text block on its own.  Plain `p` rather than an
        # array-package variant, so the table needs no preamble beyond booktabs.
        "\\begin{tabular}{lll p{0.155\\textwidth} p{0.165\\textwidth} "
        "p{0.165\\textwidth} lc}\n"
        "\\toprule\n"
        "rules & tuple & class & $n_{\\wcc}$ & $D_{\\max}$ & frozen $n_1$ & "
        "$N$ & holds \\\\\n\\midrule\n" + body + "\n\\bottomrule\n"
        "\\end{tabular}\n"
        "\\caption{Every unitary rule at \\rt{" + bc + "}, one row per "
        "reflection class (reflection is an \\rt{obc0} symmetry, so the "
        "sixteen rules give twelve distinct series). Each closed form was read "
        "off the integer series and then evaluated at \\emph{every} computed "
        "$N$; the last column is that check, not a goodness of fit. "
        "``---'' marks a series with no exact law: W156/W198's largest sector "
        "mixes rooms of two lengths and only approaches $4^{N/5}$, and "
        "W108/W201's sector counts satisfy a recurrence rather than a closed "
        "form. $F_k$ is the Fibonacci number with $F_1=F_2=1$ and $m=\\lfloor "
        "N/2\\rfloor$.}\n"
        "\\label{tab:r19laws}\n\\end{table}\n")


def _tex_census(bc: str) -> str:
    rows = []
    prev = None
    for row in sorted(census(bc),
                      key=lambda r: (["ergodic", "polynomial", "exponential",
                                      "frozen"].index(r["class"]), r["rule"])):
        if prev is not None and row["class"] != prev:
            rows.append("\\midrule")
        prev = row["class"]
        rows.append(
            f"{row['rule']} & \\rt{{{row['tuple']}}} & {row['class']} & "
            f"{row['reflect']} & {row['flip']} & "
            f"${row['N_min']}$--${row['N_max']}$ & "
            f"{row['n_recurrent']:,} & {row['d_max']:,} & "
            f"{row['n_frozen']:,} & {row['ratio']:.4g} \\\\")
    body = "\n".join(rows).replace(",", "\\,")
    return (
        "\\begin{table}[htbp]\n\\centering\n\\small\n"
        "\\begin{tabular}{rllrrlrrrr}\n\\toprule\n"
        "rule & tuple & class & refl. & flip & $N$ & $n_{\\wcc}$ & "
        "$D_{\\max}$ & $n_1$ & $n_{\\wcc}D_{\\max}/2^N$ \\\\\n\\midrule\n"
        + body + "\n\\bottomrule\n\\end{tabular}\n"
        "\\caption{The unitary census at \\rt{" + bc + "}: all sixteen rules, "
        "the reflection and spin-flip partner of each, and the three counts at "
        "the largest computed $N$. The last column is the hyperbola "
        "$n_{\\wcc}D_{\\max}\\ge 2^N$; it equals one exactly for the six "
        "ergodic rules and for the identity, the only rules whose sectors all "
        "have the same size, and exceeds it by up to three orders of magnitude "
        "for the fragmented ones. Reflection preserves the series; the spin "
        "flip does not, and it is what puts an ergodic rule opposite a "
        "fragmented one in the pairs $(60,195)$ and $(102,153)$.}\n"
        "\\label{tab:r19census}\n\\end{table}\n")


def _tex_sizes(bc: str) -> str:
    rows = []
    for rule in HIST_RULES:
        for N in HIST_NS:
            s = size_stats(rule, N, bc)
            rows.append(
                f"{rule} & {N} & {s['n_sectors']:,} & {s['d_max']:,} & "
                f"{s['d_max_fraction']:.2e} & {s['mean']:.1f} & "
                f"{s['median']:,} & {s['n_singletons']:,} \\\\")
    body = "\n".join(rows).replace(",", "\\,")
    return (
        "\\begin{table}[htbp]\n\\centering\n\\small\n"
        "\\begin{tabular}{rrrrrrrr}\n\\toprule\n"
        "rule & $N$ & sectors & $D_{\\max}$ & $D_{\\max}/2^N$ & mean & "
        "median & singletons \\\\\n\\midrule\n" + body +
        "\n\\bottomrule\n\\end{tabular}\n"
        "\\caption{The numbers behind Figure~\\ref{fig:r19hist} (\\rt{" + bc +
        "}). \\emph{mean} is $2^N/n_{\\wcc}$; \\emph{median} is the size of "
        "the sector a uniformly random computational-basis state lands in, "
        "i.e.\\ where the cumulative curve crosses $1/2$. Both are far below "
        "$D_{\\max}$, which is why the largest sector is a poor description of "
        "the typical one.}\n"
        "\\label{tab:r19sizes}\n\\end{table}\n")


def write_tables(bc: str = "obc0") -> List[str]:
    os.makedirs(TEX_DIR, exist_ok=True)
    out = []
    for name, tex in (("laws", _tex_laws(bc)), ("census", _tex_census(bc)),
                      ("sizes", _tex_sizes(bc))):
        path = os.path.join(TEX_DIR, f"tab_r19_{name}_{bc}.tex")
        with open(path, "w") as f:
            f.write(tex)
        out.append(path)
    return out


def main(argv=None):
    bc = "obc0"
    v = verify(bc)
    for (rule, key) in sorted(LAWS, key=lambda t: (t[1], t[0])):
        c = v["laws"][(rule, key)]
        print(f"{'ok  ' if c['ok'] else 'FAIL'} W{rule:<4d} {key:<12s} "
              f"= {LAWS[(rule, key)][1]}"
              + (f"   mismatch at N={c['mismatch']}" if c["mismatch"] else ""))
    print()
    print(f"{'ok  ' if v['plastic_recurrence'] else 'FAIL'} W108  "
          f"a_N = 2a_(N-1) - a_(N-2) + a_(N-3)   (root rho^2 = 1.754878)")
    print(f"{'ok  ' if v['shifted_partner'] else 'FAIL'} W201  "
          f"n_wcc(N) = W108 n_wcc(N-1)")
    rb = v["room_base"]
    print(f"ok   W156  D_max five-step ratio^(1/5) = {rb['estimate']:.7f} "
          f"vs 4^(1/5) = {rb['target']:.7f}")
    for (a, b), same in sorted(v["reflection"].items()):
        print(f"{'ok  ' if same else 'FAIL'} W{a} = W{b} term for term "
              f"(reflection is an {bc} symmetry)")
    print(f"{'ok  ' if v['ergodic_six_coincide'] else 'FAIL'} the six ergodic "
          f"rules give one and the same series")
    print(f"{'ok  ' if v['wcc_scc'] else 'FAIL'} Tier-1a and Tier-1e agree on "
          f"every overlapping unitary unit")
    print()
    for row in v["spinflip"]:
        a, b = row["pair"]
        ca, cb = row["classes"]
        why = ("  (identical -- but they are reflection partners too, so "
               "reflection is doing the work)" if row["is_reflection"]
               else "  (series differ -- obc0 breaks the flip)")
        print(f"     spin flip W{a} <-> W{b}: {ca} vs {cb}{why}")
    print(f"{'ok  ' if v['flip_needs_reflection'] else 'FAIL'} the spin flip "
          f"preserves the {bc} series only where it coincides with a "
          f"reflection")
    print()
    for p in write_tables(bc):
        print("wrote", os.path.relpath(p, results_io.REPO_ROOT))


if __name__ == "__main__":
    main()
