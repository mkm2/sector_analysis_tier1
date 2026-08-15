"""
Which of our integer sequences are already in the OEIS, and under what name?

Every growth law this project reports was derived here -- from an exact linear
recurrence over the wall grammar (R18), from a closed form (R19), or, in the
cases R16 flagged, from a fit.  That is enough to state an exponent but it says
nothing about whether the sequence is *known*.  This module asks the second
question: is the integer sequence a named entry in the On-Line Encyclopedia of
Integer Sequences, and if so at which index shift?

Three things come out of it that a growth constant alone cannot give.

  1. AN EXTERNAL ORACLE.  An OEIS entry carries a b-file with hundreds of terms.
     Where our frontier stops at N = 17 or N = 22, the named sequence continues,
     so a future extension has something to be checked against that was not
     produced by this code.

  2. A NAME FOR A FALSIFICATION.  R9's task sec.6 predicted a_wcc >= phi for
     rule 28, inheriting rule 156's '01' wall grammar.  test_wcc pins the
     failure: the obc0 sector count obeys a(n) = a(n-1) + a(n-2) - a(n-4),
     characteristic polynomial (x-1)(x^3-x-1), plastic number rho = 1.324718.
     That recurrence is the DEFINITION of A023434, "dying rabbits" -- Fibonacci
     rabbits with a finite lifetime.  The falsifying sequence is a classical
     combinatorial object, not a numerical accident.

  3. A COST FUNCTION FOR THE 4^(N/5).  W156/W198's D_max is A178715, the
     "Select All, Copy, Paste" problem (Rowell, J. Integer Seq. 18 (2015)
     15.10.7): the largest product of factors whose costs sum to a budget, where
     a factor m costs m + 1.  The optimum is all factors 4 at cost 5, which is
     exactly where 4^(1/5) = 1.3195 comes from, and the entry's own recurrence
     a(n) = 4 a(n-5) for n >= 16 reproduces our ladder term for term.

The identifications below are all EXACT CONTIGUOUS BLOCKS: every term we have,
in order, appearing consecutively in the OEIS entry, with the index shift
recorded.  Nothing is matched on a truncated prefix and nothing is matched up to
a constant factor.

WHAT IS NOT IN THE OEIS.  The pbc series of the dissipative rho-group -- both
n_recurrent branches (28/70/157/199 and 29/71) and their common D_max ladder --
return no hits at all.  They are recorded here as unmatched rather than omitted,
because "we searched and there is nothing" is the useful statement.

Provenance of the local side.  Unitary series come from scaling.paper_figures's
merged Tier-1a + Tier-1e loader (N = 6..22).  Dissipative series come from the
Tier-1a archive for N >= 6 and are recomputed from graph.scc for N <= 12, which
covers the archive's lower end; the overlap is asserted to agree, so no term is
taken on trust from a single code path.  Reflection partners are NEVER inferred
from each other -- W28 and W70 agree because both were computed.

The OEIS terms themselves are cached in analytics/oeis_terms.json so that
verification is offline and reproducible; `--refresh` re-fetches them.
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .. import results_io
from ..core.rules import wolfram_to_tuple
from . import sectors

CACHE = os.path.join(sectors.ANALYTICS, "oeis_terms.json")
OUT_JSON = os.path.join(sectors.ANALYTICS, "oeis_identifications.json")

#: N range over which the dissipative series are assembled.  The archive starts
#: at N = 6; 1..5 are recomputed, and 6..12 are recomputed as a cross-check.
DISS_NS = tuple(range(1, 18))
RECOMPUTE_UPTO = 12

#: Growth constants, to the precision the report quotes them.
RHO = 1.324717957244746      # x^3 = x + 1,   plastic number
PSI = 1.465571231876768      # x^3 = x^2 + 1, supergolden
PHI = 1.618033988749895      # x^2 = x + 1,   golden
QUARTIC = 4.0 ** 0.2         # 1.3195079..., the copy-paste optimum
CBRT2 = 2.0 ** (1.0 / 3.0)   # 1.2599210..., the 2^floor(N/3) ladder


# --- the table ----------------------------------------------------------------

@dataclass(frozen=True)
class Ident:
    """One claim: this series of ours is that OEIS entry, at this index shift."""
    rules: Tuple[int, ...]
    bc: str
    key: str
    aid: str
    shift: int                    # ours(N) = aid(N + shift)
    growth: Optional[float] = None
    closed: str = ""              # closed form in our own index N, if known
    note: str = ""


IDENTIFICATIONS: Tuple[Ident, ...] = (
    # -- dissipative, obc0: four recurrent-class families ---------------------
    Ident((28, 70), "obc0", "n_recurrent", "A023434", 2, RHO,
          "a(N) = a(N-1) + a(N-2) - a(N-4)",
          "dying rabbits; the sequence that falsifies the phi prediction"),
    Ident((29, 71, 157, 199), "obc0", "n_recurrent", "A000931", 6, RHO,
          "a(N) = a(N-2) + a(N-3)", "Padovan"),
    Ident((73,), "obc0", "n_recurrent", "A000930", 1, PSI,
          "a(N) = a(N-1) + a(N-3)", "Narayana's cows"),
    Ident((109,), "obc0", "n_recurrent", "A179070", 2, PSI,
          "a(N) = a(N-1) + a(N-3)", "same recurrence as 73, different seed"),

    # -- dissipative, obc0: the largest recurrent class ------------------------
    Ident((28, 29, 70, 71, 157, 199), "obc0", "d_max", "A173862", 2, CBRT2,
          "2^floor((N+1)/3)",
          "the only OEIS home for 2^floor(n/3), and an obscure one"),
    Ident((73,), "obc0", "d_max", "A000045", 2, PHI, "F(N+2)", "Fibonacci"),
    Ident((109,), "obc0", "d_max", "A000045", 0, PHI, "F(N)", "Fibonacci"),

    # -- dissipative, pbc ------------------------------------------------------
    Ident((73, 109), "pbc", "n_recurrent", "A001609", 0, PSI,
          "a(N) = a(N-1) + a(N-3)", "the two coincide on the ring"),
    Ident((73, 109), "pbc", "d_max", "A169985", 0, PHI,
          "round(phi^N)", "Lucas numbers, shifted"),

    # -- unitary, obc0: sector counts -----------------------------------------
    Ident((108,), "obc0", "n_recurrent", "A005251", 3, RHO ** 2,
          "a(N) = a(N-1) + a(N-2) + a(N-4)",
          "R18's wall-set recurrence, root rho^2 = 1.754878"),
    Ident((201,), "obc0", "n_recurrent", "A005251", 2, RHO ** 2,
          "a(N) = a(N-1) + a(N-2) + a(N-4)", "W108's series shifted by one"),
    Ident((156, 198), "obc0", "n_recurrent", "A000045", 2, PHI,
          "F(N+2)", "Fibonacci -- R18's a(N) = a(N-1) + a(N-2)"),
    Ident((150,), "obc0", "n_recurrent", "A008619", 1, 1.0,
          "floor((N+1)/2) + 1", "positive integers repeated"),
    Ident((105,), "obc0", "n_recurrent", "A004525", 2, 1.0,
          "(N + 1 - eps)/2 + 1, eps = ceil(N/2) mod 2",
          "one even followed by three odd; the period-4 gap is R21's parity "
          "class"),

    # -- unitary, obc0: largest sector ----------------------------------------
    Ident((108,), "obc0", "d_max", "A000045", 0, PHI, "F(N)", "Fibonacci"),
    Ident((201,), "obc0", "d_max", "A000045", 2, PHI, "F(N+2)", "Fibonacci"),
    Ident((156, 198), "obc0", "d_max", "A178715", 0, QUARTIC,
          "a(N) = 4 a(N-5) for N >= 16",
          "Select All, Copy, Paste -- where the 4^(N/5) comes from"),
    Ident((150,), "obc0", "d_max", "A214282", 2, 2.0,
          "", "largest Euler characteristic of a downset on the N-cube"),
    Ident((105,), "obc0", "d_max", "A001405", 1, 2.0,
          "binomial(N+1, floor((N+1)/2))", "central binomial coefficient"),

    # -- unitary, obc0: frozen states -----------------------------------------
    Ident((108,), "obc0", "n_frozen", "A006498", 2, PHI,
          "F(floor(N/2)+2) * F(ceil(N/2)+2)",
          "the entry states the Fibonacci-product factorisation itself"),
    Ident((201,), "obc0", "n_frozen", "A195971", -1, PHI,
          "F(m+1)^2 (N=2m even), F(m) F(m+3) (N=2m+1 odd)", ""),
    Ident((156, 198), "obc0", "n_frozen", "A008619", 1, 1.0,
          "floor(N/2) + 1", "positive integers repeated"),
    Ident((150,), "obc0", "n_frozen", "A000034", -6, 1.0,
          "1 + (N mod 2)", "period 2"),
    Ident((105,), "obc0", "n_frozen", "A007877", -5, 1.0,
          "period 4: 1, 0, 1, 2 for N = 1, 2, 3, 0 (mod 4)",
          "period-4 zigzag; the zero at N = 1 (mod 4) is R19's"),
)

#: Series we searched for and did not find.  Recorded so that the absence is a
#: result rather than an omission.  The two branches are kept apart because the
#: ring splits the rho-group that obc0 holds together: the D_max ladders of the
#: two branches agree only from N = 2 (at N = 1 the "ring" is a single site).
NO_MATCH: Tuple[Tuple[Tuple[int, ...], str, str], ...] = (
    ((28, 70, 157, 199), "pbc", "n_recurrent"),
    ((29, 71), "pbc", "n_recurrent"),
    ((28, 70, 157, 199), "pbc", "d_max"),
    ((29, 71), "pbc", "d_max"),
)


# --- our side of the comparison ----------------------------------------------

_DISS_CACHE: Dict[Tuple[int, str], Dict[str, List[int]]] = {}


def _recompute(rule: int, N: int, bc: str) -> Dict[str, int]:
    """One unit straight from the Tarjan engine, ergodic early-exit disabled so
    that the small-N units report a class count rather than a bail-out."""
    from ..graph import scc
    r = scc.analyze(rule, N, bc, wolfram_to_tuple(rule), detect_ergodic=False)
    sizes = r.sizes_recurrent
    return {"n_recurrent": r.n_recurrent,
            "d_max": max(sizes) if sizes else 0,
            "n_frozen": sum(1 for s in sizes if s == 1)}


def _archive(rule: int, bc: str) -> Dict[int, Dict[str, Optional[int]]]:
    out: Dict[int, Dict[str, Optional[int]]] = {}
    for N, rec in results_io.load_results(rule, bc).items():
        h = rec.get("size_hist") or {}
        out[N] = {"n_recurrent": rec["n_recurrent"],
                  "d_max": max(int(k) for k in h) if h else None,
                  "n_frozen": int(h.get("1", h.get(1, 0))) if h else None}
    return out


def diss_series(rule: int, bc: str,
                Ns: Sequence[int] = DISS_NS) -> Dict[str, List[int]]:
    """Merged series for a dissipative rule, with the overlap asserted.

    N <= RECOMPUTE_UPTO comes from the engine; the rest from the Tier-1a
    archive.  Where both exist they must agree -- that is the check which makes
    the low-N terms usable for fixing an OEIS offset.
    """
    ck = (rule, bc)
    if ck in _DISS_CACHE:
        return _DISS_CACHE[ck]
    arc = _archive(rule, bc)
    got: Dict[int, Dict[str, Optional[int]]] = {}
    for N in Ns:
        fresh = _recompute(rule, N, bc) if N <= RECOMPUTE_UPTO else None
        old = arc.get(N)
        if fresh and old:
            for k in ("n_recurrent", "d_max"):
                if old[k] is not None and fresh[k] != old[k]:
                    raise AssertionError(
                        f"W{rule} {bc} N={N} {k}: engine {fresh[k]} vs "
                        f"archive {old[k]}")
        if fresh or old:
            got[N] = fresh or old
    out = {"N": sorted(got)}
    for k in ("n_recurrent", "d_max", "n_frozen"):
        out[k] = [got[N][k] for N in out["N"]]
    _DISS_CACHE[ck] = out
    return out


def local_series(rule: int, bc: str, key: str) -> Tuple[List[int], List[int]]:
    """(N grid, values) for one rule/bc/key, unitary or dissipative."""
    from ..core.rules import is_unitary
    if is_unitary(wolfram_to_tuple(rule)) and bc == "obc0":
        from . import paper_figures as pf
        s = pf.unit_series(rule, bc)
        return list(s["N"]), list(s[key])
    s = diss_series(rule, bc)
    return list(s["N"]), list(s[key])


# --- the OEIS side ------------------------------------------------------------

def _fetch(aid: str) -> Dict:
    url = f"https://oeis.org/search?fmt=text&q=id:{aid}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        txt = r.read().decode("utf-8", "replace")
    terms: List[int] = []
    offset, name = None, ""
    for line in txt.splitlines():
        if line[:3] in ("%S ", "%T ", "%U "):
            terms += [int(v) for v in line.split(None, 2)[2].split(",")
                      if v.strip()]
        if line.startswith("%O ") and offset is None:
            offset = int(line.split(None, 2)[2].split(",")[0])
        if line.startswith("%N ") and not name:
            name = line.split(None, 2)[2].strip()
    if not terms:
        raise RuntimeError(f"no terms returned for {aid}")
    return {"terms": terms, "offset": offset or 0, "name": name}


def load_cache() -> Dict[str, Dict]:
    if not os.path.exists(CACHE):
        return {}
    with open(CACHE) as f:
        return json.load(f)


def refresh(aids: Optional[Sequence[str]] = None, *, pause: float = 1.6) -> Dict:
    """Re-fetch the OEIS entries and rewrite the cache."""
    aids = aids or sorted({i.aid for i in IDENTIFICATIONS})
    cache = load_cache()
    for aid in aids:
        cache[aid] = _fetch(aid)
        time.sleep(pause)
    cache["_fetched"] = {"utc": time.strftime("%Y-%m-%d", time.gmtime())}
    os.makedirs(sectors.ANALYTICS, exist_ok=True)
    with open(CACHE, "w") as f:
        json.dump(cache, f, indent=1)
    return cache


def entry(aid: str, cache: Optional[Dict] = None) -> Dict:
    cache = cache if cache is not None else load_cache()
    if aid not in cache:
        raise KeyError(f"{aid} is not cached; run `--refresh`")
    return cache[aid]


# --- the check ----------------------------------------------------------------

def check(ident: Ident, cache: Optional[Dict] = None) -> Dict:
    """Is every one of our terms an exact contiguous block of the entry, at the
    declared shift, for every rule the claim covers?"""
    e = entry(ident.aid, cache)
    terms, offset = e["terms"], e["offset"]
    rows = []
    for rule in ident.rules:
        Ns, ys = local_series(rule, ident.bc, ident.key)
        ys = [v for v in ys if v is not None]
        Ns = Ns[:len(ys)]
        pos = next((i for i in range(len(terms) - len(ys) + 1)
                    if terms[i:i + len(ys)] == ys), None)
        shift = None if pos is None else pos + offset - Ns[0]
        rows.append({"rule": rule, "n_terms": len(ys),
                     "N_lo": Ns[0], "N_hi": Ns[-1],
                     "contiguous": pos is not None,
                     "shift": shift,
                     "shift_ok": shift == ident.shift})
    return {"aid": ident.aid, "name": e["name"], "bc": ident.bc,
            "key": ident.key, "rules": list(ident.rules),
            "declared_shift": ident.shift, "closed": ident.closed,
            "note": ident.note, "growth": ident.growth,
            "rows": rows,
            "ok": all(r["contiguous"] and r["shift_ok"] for r in rows)}


def verify(cache: Optional[Dict] = None) -> List[Dict]:
    cache = cache if cache is not None else load_cache()
    return [check(i, cache) for i in IDENTIFICATIONS]


def partners_agree() -> List[Dict]:
    """The rules an identification groups together must have been measured to
    agree, not assumed to.  Reflection is not a guaranteed sector-size symmetry
    for dissipative rules, so W28 = W70 is a result, not a shortcut.
    """
    out = []
    for i in IDENTIFICATIONS:
        if len(i.rules) < 2:
            continue
        ref = local_series(i.rules[0], i.bc, i.key)
        same = all(local_series(r, i.bc, i.key) == ref for r in i.rules[1:])
        out.append({"aid": i.aid, "bc": i.bc, "key": i.key,
                    "rules": list(i.rules), "identical": same,
                    "n_terms": len(ref[1])})
    return out


def build() -> Dict:
    d = {"identifications": verify(),
         "partners": partners_agree(),
         "no_match": [{"rules": list(r), "bc": bc, "key": k}
                      for r, bc, k in NO_MATCH],
         "fetched": load_cache().get("_fetched", {})}
    os.makedirs(sectors.ANALYTICS, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(d, f, indent=1)
    return d


# --- tables -------------------------------------------------------------------

def _rules_tex(rules: Sequence[int]) -> str:
    return ", ".join(f"$W_{{{r}}}$" for r in rules)


_KEY_TEX = {"n_recurrent": r"$n_{\rm rec}$",
            "d_max": r"$D_{\max}$",
            "n_frozen": r"$n_{\rm froz}$"}


def _esc(s: str) -> str:
    return s.replace("_", r"\_").replace("^", r"\textasciicircum{}")


def write_tables(outdir: str, d: Optional[Dict] = None) -> List[str]:
    d = d or build()
    os.makedirs(outdir, exist_ok=True)
    paths = []

    rows = []
    for e in d["identifications"]:
        span = e["rows"][0]
        g = "---" if e["growth"] is None else f"{e['growth']:.6f}"
        rows.append(
            f"{_rules_tex(e['rules'])} & \\rt{{{e['bc']}}} & "
            f"{_KEY_TEX[e['key']]} & \\rt{{{e['aid']}}} & "
            f"$a(N{e['declared_shift']:+d})$ & "
            f"{span['n_terms']} & {span['N_lo']}--{span['N_hi']} & "
            f"{g} \\\\")
    tab = ("\\setlength{\\tabcolsep}{4pt}\n"
           "\\begin{tabular}{lllllrrl}\n\\toprule\n"
           "rules & bc & series & OEIS & index & terms & $N$ & growth \\\\\n"
           "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    p = os.path.join(outdir, "tab_r20_oeis.tex")
    with open(p, "w") as f:
        f.write(tab)
    paths.append(p)

    nrows = []
    for e in d["identifications"]:
        nrows.append(
            f"\\rt{{{e['aid']}}} & \\parbox[t]{{0.44\\textwidth}}{{\\raggedright "
            f"{_esc(e['name'])}}} & "
            f"\\parbox[t]{{0.25\\textwidth}}{{\\raggedright "
            f"{_esc(e['closed']) or '---'}}} \\\\")
    ntab = ("\\begin{tabular}{lll}\n\\toprule\n"
            "OEIS & entry name & closed form in our $N$ \\\\\n\\midrule\n"
            + "\n".join(nrows) + "\n\\bottomrule\n\\end{tabular}\n")
    p = os.path.join(outdir, "tab_r20_oeis_names.tex")
    with open(p, "w") as f:
        f.write(ntab)
    paths.append(p)
    return paths


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(
        description="OEIS identifications for the Tier-1 growth sequences")
    ap.add_argument("--refresh", action="store_true",
                    help="re-fetch the OEIS entries (needs network)")
    ap.add_argument("--tables", action="store_true")
    args = ap.parse_args(argv)

    if args.refresh:
        refresh()
        print("cache refreshed")
    d = build()
    bad = [e for e in d["identifications"] if not e["ok"]]
    for e in d["identifications"]:
        span = e["rows"][0]
        mark = "ok " if e["ok"] else "FAIL"
        print(f"{mark} {e['aid']}  {_rules_tex(e['rules']):28s} "
              f"{e['bc']:5s} {e['key']:12s} "
              f"a(N{e['declared_shift']:+d})  {span['n_terms']:2d} terms "
              f"N={span['N_lo']}..{span['N_hi']}  {e['name'][:52]}")
    for p in d["partners"]:
        if not p["identical"]:
            print(f"WARNING  {p['rules']} {p['bc']} {p['key']} differ")
    print(f"\n{len(d['identifications']) - len(bad)}/"
          f"{len(d['identifications'])} identifications verified; "
          f"{len(d['no_match'])} series searched with no OEIS hit")
    if args.tables:
        for p in write_tables(os.path.join(results_io.REPO_ROOT,
                                           "reports", "tex"), d):
            print("wrote", p)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
