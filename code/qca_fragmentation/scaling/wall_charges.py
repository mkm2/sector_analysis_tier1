"""
Two conservation laws, or frozen domain walls?  Rules 108/201 and 156/198.

Hilbert-space fragmentation is often explained by the interaction of two
conservation laws -- charge and dipole moment, or domain walls and magnetisation
(Khemani, Hermele, Nandkishore, arXiv:1912.04300).  The four rules that R9 puts
off both axes of its attractor map are the natural place to ask whether that
story applies here.  All four are unitary, so their terminal SCCs ARE their
Krylov sectors (R17), and all four are one-Hadamard rules whose gate fires on a
single neighbour pattern:

    W_201 = VIII   flip x_i iff (x_{i-1}, x_{i+1}) = (0,0)     the PXP constraint
    W_108 = IIIV   flip x_i iff (x_{i-1}, x_{i+1}) = (1,1)     its spin-flip image
    W_156 = IIVI   flip x_i iff (x_{i-1}, x_{i+1}) = (1,0)     chiral
    W_198 = IVII   flip x_i iff (x_{i-1}, x_{i+1}) = (0,1)     its reflection

THE ANSWER IS THAT IT IS FULLY CONSTRUCTIVE, and the conservation laws are a
consequence of the wall grammar rather than an independent ingredient.

  1. There ARE conserved charges, and they are domain-wall numbers -- exactly
     the flavour the question suggests.  Writing b_i = x_i XOR x_{i+1},

         W_108, W_201:  the STAGGERED wall number  S = sum_i (-1)^i b_i
         W_156, W_198:  the TOTAL wall number      D = sum_i b_i

     and in each case the other one is NOT conserved.  Exactly two independent
     local charges exist per rule (the rank of the certified charge values on
     2^N is 3, including the constant), and they are the SUBLATTICE-RESOLVED
     WALL COUNTS |W_even| and |W_odd|.  D and S are functions of that pair:
     D = 2(|W_even| + |W_odd|) for the chiral rules, S their difference up to
     sign.  So even the "two conservation laws" are wall data in disguise.

     TERMINOLOGY, because it is easy to get wrong and this module got it wrong
     once.  ALL of these are diagonal in the computational basis: b_i is
     (1 - Z_i Z_{i+1})/2, a two-site diagonal operator, and the whole sector
     decomposition is a statement about basis states, so nothing here is
     off-diagonal.  Bond variables are a change of VARIABLES, not of basis.
     What is actually true is a statement about RANGE: there is no conserved
     quantity of the form sum_i q(x_i), i.e. no single-site charge, so
     magnetisation is not conserved.  The conserved densities are range 2 and
     are certified as such directly in the site basis -- each of the four rules
     has a 6-dimensional space of certified range-2 site-basis charges, and D
     (chiral pair) or S (PXP-like pair) is among them.  Tier-2's
     `bond=True/False` flag selects which local variables the ansatz is built
     from; its docstring's "computational OR bond basis" is the same shorthand.

  1b. The reason the charge/dipole mechanism does not apply is therefore NOT a
     missing diagonal charge.  It is that the DIPOLE of the conserved density is
     not conserved: P = sum_i i*b_i fails for all four rules at both boundary
     conditions, because a move hops a wall by one site -- leaving the wall
     number fixed and shifting its dipole by +-1.  Forbidding exactly that hop
     is what a dipole-conserving model does.

  2. The charges are nowhere near enough.  Their joint level sets number O(N^2)
     against rho^2 ^N or phi^N Krylov sectors: for W_108 the sectors-per-level
     ratio grows 3.7 -> 7.6 -> 16.7 -> 38.6 -> 92.5 over N = 6..14.  This is the
     standard fragmentation signature -- exponentially many Krylov sectors
     inside polynomially many symmetry sectors -- and two conservation laws
     cannot be the explanation, because they do not have the cardinality to be
     one.

  3. The wall grammar IS enough.  Each rule has exactly ONE minimal wall word,

         W_108: '00'    W_201: '11'    W_156: '01'    W_198: '10'

     and the SET OF ITS OCCURRENCES is a complete invariant: two states lie in
     the same sector if and only if the word occurs in the same places.  This is
     checked exhaustively to N = 14.  It is a strictly local label computed from
     a single state, with no reference to the sector -- which is what makes it a
     constructive account rather than a restatement.

  4. Both exponents drop out of it.  The number of realisable wall sets obeys a
     linear recurrence -- a(N) = 2a(N-1) - a(N-2) + a(N-3) for W_108/W_201 with
     dominant root rho^2 = 1.754878, and a(N) = a(N-1) + a(N-2) for W_156/W_198
     with root phi -- reproducing R9's sector-count bases exactly.  The largest
     preimage of the wall map reproduces R9's D_max series term for term, so the
     wall partition gives the whole size distribution and not merely the count.

  5. The conserved charge is a FUNCTION of the wall set, at every N and both
     boundary conditions.  For W_156/W_198 it is simply D = 2|W|.  So the charge
     carries strictly less information than the walls: it is a shadow of the
     grammar, not a second ingredient interacting with it.

The one blemish is on the ring.  For W_156/W_198 at pbc the wall set is complete
except that the two uniform states 0^N and 1^N -- both frozen singletons, both
containing no '01' -- share the empty wall set, so n_sectors = n_wallsets + 1
exactly, at every N.  W_108/W_201 have no such defect because 0^N and 1^N are
not both frozen for them.

Boundary note: at obc0 the chain is padded with a frozen 0 at each end, and the
wall word must be searched on the PADDED string.  Skipping that is not a detail
-- W_108's wall word is '00', the padding supplies '0's, and searching the bare
chain finds only 616 of the 1081 sectors at N = 12.

Provenance: the wall grammar and the charge null-space detector are Tier-2
modules (qca/walls.py, qca/charges.py).  The wall detector below is an
INDEPENDENT Tier-1 reimplementation, and `test_wall_charges` pins its output
against the four words Tier-2 reports.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .. import results_io
from ..core.cycle import succ
from ..core.rules import wolfram_to_tuple
from ..graph.wcc import make_succ
from . import sectors

BC_DEFAULT = "obc0"
OUT_JSON = os.path.join(sectors.ANALYTICS, "wall_charges_{bc}.json")

#: The four rules R9 places with both attractor-map coordinates above 1 (R17).
RULES: Tuple[int, ...] = (108, 201, 156, 198)

#: Bases R9 reports for the sector count, for the closing comparison.
R9_SECTOR_BASE = {108: 1.754878, 201: 1.754878, 156: 1.618034, 198: 1.618034}

NS_DEFAULT = (6, 8, 10, 12, 14)

#: The two independent charges, spelled for LaTeX and for matplotlib mathtext.
#: A bare "|W_even|" is neither -- the underscore subscripts a single character
#: and the bars are not delimiters outside math mode.
_TEX_CHARGE = r"$|W_{\rm even}|,\ |W_{\rm odd}|$"


# --- 1. an independent wall detector -----------------------------------------

def _is_wall(rule: int, word: str, offset: int) -> bool:
    """
    Is `word`, placed at sublattice parity `offset`, invariant for every exterior?

    One cycle has light cone 2 -- site i reads i-2..i+2 of the ORIGINAL state --
    so two context sites on each side exhaust the exterior dependence, and the
    16 configurations of those four sites are the whole test.  The window is
    padded beyond them, which is harmless: the padded sites only ever feed the
    context sites' first-layer values, never the word's.
    """
    l = len(word)
    start = 2 + offset                      # word sits at start .. start+l-1
    width = start + l + 2
    t = wolfram_to_tuple(rule)
    wbits = sum(int(c) << (start + k) for k, c in enumerate(word))
    wmask = ((1 << l) - 1) << start
    ctx = [start - 2, start - 1, start + l, start + l + 1]
    for cfg in range(16):
        x = wbits
        for bit, idx in enumerate(ctx):
            if (cfg >> bit) & 1:
                x |= 1 << idx
        for y in succ(x, width, t, "obc0"):
            if (y & wmask) != wbits:
                return False
    return True


def wall_words(rule: int, l_max: int = 4) -> List[str]:
    """Minimal wall words: shortest words invariant at BOTH sublattice offsets,
    dropping any that contains a shorter one as a factor."""
    kept: List[str] = []
    for l in range(1, l_max + 1):
        for v in range(1 << l):
            w = "".join(str((v >> k) & 1) for k in range(l))
            if not all(_is_wall(rule, w, o) for o in (0, 1)):
                continue
            if any(s in w for s in kept):
                continue
            kept.append(w)
    return kept


# --- 2. states, walls, charges -----------------------------------------------

def padded(x: int, N: int, bc: str) -> List[int]:
    """The string the dynamics actually sees.  obc0 has a frozen 0 outside each
    end, and those zeros are wall material -- see the module docstring."""
    s = [(x >> i) & 1 for i in range(N)]
    return [0] + s + [0] if bc == "obc0" else s


def wall_set(x: int, N: int, word: str, bc: str) -> frozenset:
    s = padded(x, N, bc)
    L, l = len(s), len(word)
    rng = range(L - l + 1) if bc == "obc0" else range(L)
    return frozenset(i for i in rng
                     if all(s[(i + j) % L] == int(c) for j, c in enumerate(word)))


def bond_bits(x: int, N: int, bc: str) -> List[int]:
    s = padded(x, N, bc)
    if bc == "pbc":
        return [s[i] ^ s[(i + 1) % N] for i in range(N)]
    return [s[i] ^ s[i + 1] for i in range(len(s) - 1)]


def charge_total(x: int, N: int, bc: str) -> int:
    """D = sum_i b_i, the domain-wall number."""
    return sum(bond_bits(x, N, bc))


def charge_staggered(x: int, N: int, bc: str) -> int:
    """S = sum_i (-1)^i b_i, the staggered domain-wall number."""
    return sum((-1) ** i * b for i, b in enumerate(bond_bits(x, N, bc)))


def charge_dipole(x: int, N: int, bc: str) -> int:
    """P = sum_i i * b_i, the DIPOLE MOMENT of the domain-wall density.

    This is the charge the reference construction would need alongside D.  It is
    not conserved for any of the four rules: a move hops a wall by one site, so
    D survives and P shifts by +-1.  That, and not any absence of a diagonal
    charge, is why the charge/dipole mechanism does not apply here.
    """
    return sum(i * b for i, b in enumerate(bond_bits(x, N, bc)))


def n_word(word: str):
    """Count of a given 2-site pattern -- the two wall TYPES separately.

    For the chiral pair both n01 and n10 are conserved: '01' counts the frozen
    walls and '10' the mobile ones, which is where the second independent charge
    lives.  D = n01 + n10, and on a ring n01 = n10, whence D = 2|W|.
    """
    return lambda x, N, bc: len(wall_set(x, N, word, bc))


CHARGES = {"D": charge_total, "S": charge_staggered, "P": charge_dipole,
           "n01": n_word("01"), "n10": n_word("10")}

#: The charges whose joint level sets are the "symmetry sectors" the
#: two-conservation-law hypothesis would have to work with.  P is included so
#: that its failure is recorded rather than assumed.
_ORDER = ("D", "S", "P", "n01", "n10")


# --- 3. sectors ---------------------------------------------------------------

def sector_ids(rule: int, N: int, bc: str) -> List[int]:
    """Union-find over the streamed successor graph, returning a label per state.
    graph.wcc gives sizes only; here the labels themselves are the object."""
    nxt = make_succ(rule, N, bc)
    parent = list(range(1 << N))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for x in range(1 << N):
        for y in nxt(x):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx
    return [find(x) for x in range(1 << N)]


def _groups(rule: int, N: int, bc: str) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    for x, cid in enumerate(sector_ids(rule, N, bc)):
        out.setdefault(cid, []).append(x)
    return out


# --- 4. the three questions ---------------------------------------------------

def charge_status(rule: int, N: int, bc: str) -> Dict[str, bool]:
    """Which candidate charges are constant on every sector."""
    g = _groups(rule, N, bc)
    return {name: all(len({CHARGES[name](x, N, bc) for x in mem}) == 1
                      for mem in g.values())
            for name in _ORDER}


def resolving_power(rule: int, N: int, bc: str,
                    word: Optional[str] = None) -> Dict:
    """
    Krylov sectors against the level sets of the conserved charges -- the
    polynomial-vs-exponential comparison that decides the question.

    Tier-2's null-space detector finds the rank of the certified local charges to
    be 3 including the constant, i.e. exactly TWO independent conserved
    quantities.  They are the SUBLATTICE-RESOLVED WALL COUNTS,

        |W_even| = #{i in W : i even},   |W_odd| = #{i in W : i odd},

    for all four rules and both boundary conditions -- which is why Tier-2's
    detector needs its parity split to see two of them.  Every other charge in
    play is a function of this pair: D = 2(|W_even| + |W_odd|) for the chiral
    rules, and S is (up to sign) their difference.  Using the pair rather than
    one of its shadows is the point: testing the two-conservation-law hypothesis
    against a single charge would understate the symmetry by a factor of N.

    These joint level sets reproduce Tier-2's full certified charge set exactly
    -- 13, 18, 25 at N = 8, 10, 12 (pbc) for W108, and 12, 17, 23 for W156 --
    which is the cross-check that nothing conserved has been left out.
    """
    g = _groups(rule, N, bc)
    st = charge_status(rule, N, bc)
    word = word or wall_words(rule)[0]

    def wpar(x):
        w = wall_set(x, N, word, bc)
        return (sum(1 for i in w if i % 2 == 0), sum(1 for i in w if i % 2))

    one = {sum(wpar(x)) for x in range(1 << N)}
    both = {wpar(x) for x in range(1 << N)}
    return {"N": N, "n_sectors": len(g),
            "conserved": ["|W_even|", "|W_odd|"],
            "also_conserved": [n for n in _ORDER if st[n]],
            "n_charge_levels": len(both),
            "n_levels_one_charge": len(one),
            "sectors_per_level": len(g) / max(1, len(both))}


def wall_completeness(rule: int, N: int, bc: str, word: str) -> Dict:
    """Is the wall-occurrence set a complete invariant for the sectors?

    Two separate claims, reported separately: CONSTANT on each sector (the wall
    set is conserved) and INJECTIVE across sectors (it separates them).  Only
    both together make it a complete label.
    """
    g = _groups(rule, N, bc)
    const = True
    lab: Dict[frozenset, List[int]] = {}
    for cid, mem in g.items():
        ws = {wall_set(x, N, word, bc) for x in mem}
        if len(ws) != 1:
            const = False
        lab.setdefault(sorted(ws, key=sorted)[0], []).append(cid)
    collisions = {w: v for w, v in lab.items() if len(v) > 1}
    # size distribution from the wall partition alone
    pred = Counter(wall_set(x, N, word, bc) for x in range(1 << N))
    actual = sorted((len(m) for m in g.values()), reverse=True)
    predicted = sorted(pred.values(), reverse=True)
    return {"N": N, "n_sectors": len(g), "n_wall_sets": len(lab),
            "constant_on_sectors": const,
            "complete": bool(const and not collisions),
            "n_collisions": len(collisions),
            "defect": len(g) - len(lab),
            "sizes_match": actual == predicted,
            "d_max_actual": actual[0], "d_max_predicted": predicted[0]}


def charge_from_walls(rule: int, N: int, bc: str, word: str) -> Dict:
    """Is the conserved charge a function of the wall set -- i.e. does the
    symmetry carry strictly less information than the grammar?"""
    st = charge_status(rule, N, bc)
    out = {}
    for name, ok in st.items():
        if not ok:
            out[name] = None
            continue
        fn: Dict[frozenset, int] = {}
        determined = True
        for x in range(1 << N):
            w = wall_set(x, N, word, bc)
            q = CHARGES[name](x, N, bc)
            if fn.setdefault(w, q) != q:
                determined = False
                break
        affine = None
        if determined:
            by: Dict[int, set] = {}
            for w, q in fn.items():
                by.setdefault(len(w), set()).add(q)
            if all(len(v) == 1 for v in by.values()) and len(by) > 1:
                ks = sorted(by)
                a = (next(iter(by[ks[1]])) - next(iter(by[ks[0]]))) / (ks[1] - ks[0])
                b0 = next(iter(by[ks[0]])) - a * ks[0]
                if all(abs(next(iter(by[k])) - (a * k + b0)) < 1e-9 for k in ks):
                    affine = [a, b0]
        out[name] = {"determined_by_walls": determined, "affine_in_wall_count": affine}
    return out


# --- 5. counting --------------------------------------------------------------

def wall_set_counts(rule: int, bc: str, word: str,
                    Ns: Sequence[int]) -> List[int]:
    return [len({wall_set(x, N, word, bc) for x in range(1 << N)}) for N in Ns]


def linear_recurrence(seq: Sequence[int], max_order: int = 5):
    """Smallest homogeneous linear recurrence with rational coefficients that the
    whole sequence satisfies, plus the dominant root of its characteristic
    polynomial.  Exact arithmetic: these are integer sequences and a
    least-squares fit would invite exactly the base/prefactor confusion R16 is
    about."""
    from fractions import Fraction as F
    for k in range(1, max_order + 1):
        if len(seq) < 2 * k + 1:
            break
        A = [[F(seq[i + j]) for j in range(k)] + [F(seq[i + k])]
             for i in range(k)]
        piv = 0
        for c in range(k):
            r = next((rr for rr in range(piv, k) if A[rr][c] != 0), None)
            if r is None:
                continue
            A[piv], A[r] = A[r], A[piv]
            pv = A[piv][c]
            A[piv] = [v / pv for v in A[piv]]
            for rr in range(k):
                if rr != piv and A[rr][c] != 0:
                    f = A[rr][c]
                    A[rr] = [a - f * b for a, b in zip(A[rr], A[piv])]
            piv += 1
        if piv < k:
            continue
        coef = [A[i][k] for i in range(k)]
        if all(sum(coef[j] * F(seq[i + j]) for j in range(k)) == F(seq[i + k])
               for i in range(len(seq) - k)):
            poly = [1.0] + [-float(c) for c in reversed(coef)]
            root = float(max(abs(r) for r in np.roots(poly)))
            return {"order": k, "coefficients": [str(c) for c in coef],
                    "dominant_root": root}
    return None


# --- 6. assemble --------------------------------------------------------------

def analyse(rule: int, bc: str = BC_DEFAULT,
            Ns: Sequence[int] = NS_DEFAULT) -> Dict:
    words = wall_words(rule)
    word = words[0]
    rows = [dict(wall_completeness(rule, N, bc, word),
                 **{"charges": charge_status(rule, N, bc)}) for N in Ns]
    cnt_Ns = list(range(4, max(Ns) + 1))
    counts = wall_set_counts(rule, bc, word, cnt_Ns)
    return {
        "rule": rule, "tuple": "".join(wolfram_to_tuple(rule)), "bc": bc,
        "wall_words": words, "word": word,
        "rows": rows,
        "resolving": [resolving_power(rule, N, bc, word) for N in Ns],
        "charge_from_walls": charge_from_walls(rule, max(Ns[:3]), bc, word),
        "wall_set_counts": {"N": cnt_Ns, "counts": counts},
        "recurrence": linear_recurrence(counts),
        "r9_sector_base": R9_SECTOR_BASE[rule],
    }


def build(bc: str = BC_DEFAULT, Ns: Sequence[int] = NS_DEFAULT) -> Dict:
    d = {"bc": bc, "Ns": list(Ns),
         "rules": [analyse(r, bc, Ns) for r in RULES]}
    with open(OUT_JSON.format(bc=bc), "w") as f:
        json.dump(d, f, indent=1, default=str)
    return d


def load(bc: str = BC_DEFAULT) -> Optional[Dict]:
    p = OUT_JSON.format(bc=bc)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


# --- 7. figure ----------------------------------------------------------------

_COL = {108: "#1f4e9c", 201: "#3d7fd1", 156: "#c62828", 198: "#e0725c"}
_MK = {108: "o", 201: "s", 156: "^", 198: "D"}


def _style(ax):
    ax.grid(True, color="#e9e9e6", linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_walls(bc: str, out: str, d: Optional[Dict] = None):
    d = d or load(bc) or build(bc)
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.3))

    ax = axes[0]
    _style(ax)
    # The two curves coincide exactly, which is the whole point and also makes
    # them impossible to tell apart: sectors are drawn as a thick line and wall
    # sets as hollow markers sitting on it, so the reader sees the agreement
    # rather than one line hiding the other.
    for e in d["rules"]:
        Ns = [r["N"] for r in e["rows"]]
        ax.plot(Ns, [r["n_sectors"] for r in e["rows"]], lw=3.0, alpha=0.45,
                color=_COL[e["rule"]], solid_capstyle="round",
                label=f"$W_{{{e['rule']}}}$ sectors")
        ax.plot(Ns, [r["n_wall_sets"] for r in e["rows"]], ls="none",
                marker=_MK[e["rule"]], ms=5.5, markerfacecolor="white",
                markeredgecolor=_COL[e["rule"]], markeredgewidth=1.4)
    for e in d["rules"][:1]:
        Ns = [r["N"] for r in e["resolving"]]
        ax.plot(Ns, [r["n_charge_levels"] for r in e["resolving"]], ":",
                lw=2.0, color="#444444", label="charge level sets")
    ax.set_yscale("log")
    ax.set_xlabel("$N$")
    ax.set_ylabel("count")
    ax.set_title("(a)  walls track the sectors; charges do not", fontsize=9.5)
    ax.legend(fontsize=6.9, frameon=False, loc="upper left")
    ax.annotate("hollow markers: number of wall sets\n"
                "(exactly on the sector line)",
                (0.97, 0.06), xycoords="axes fraction", ha="right",
                fontsize=7.0, color="#555555")

    ax = axes[1]
    _style(ax)
    for e in d["rules"]:
        rs = e["resolving"]
        ax.plot([r["N"] for r in rs], [r["sectors_per_level"] for r in rs],
                marker=_MK[e["rule"]], ms=5, lw=1.5, color=_COL[e["rule"]],
                label=f"$W_{{{e['rule']}}}$")
    ax.axhline(1.0, color="#444444", ls=":", lw=1.0)
    ax.annotate("$1$ = symmetry explains everything", (0.04, 1.0),
                xycoords=("axes fraction", "data"), fontsize=7.2,
                va="bottom", color="#444444")
    ax.set_yscale("log")
    ax.set_xlabel("$N$")
    ax.set_ylabel("Krylov sectors per charge level set")
    ax.set_title(r"(b)  beyond the symmetry: level sets of "
                 r"$|W_{\rm even}|,|W_{\rm odd}|$", fontsize=9.5)
    ax.legend(fontsize=7.0, frameon=False, loc="upper left")

    ax = axes[2]
    _style(ax)
    labels, got, want = [], [], []
    for e in d["rules"]:
        labels.append(f"$W_{{{e['rule']}}}$\n'{e['word']}'")
        got.append(e["recurrence"]["dominant_root"] if e["recurrence"] else 0.0)
        want.append(e["r9_sector_base"])
    xs = np.arange(len(labels))
    ax.bar(xs - 0.18, want, 0.36, color="#bfbfbf", label="R9 fitted base")
    ax.bar(xs + 0.18, got, 0.36, color="#1f4e9c",
           label="wall-grammar recurrence")
    for i, (g, w) in enumerate(zip(got, want)):
        ax.annotate(f"{g:.6f}", (i + 0.18, g), ha="center", va="bottom",
                    fontsize=6.6, rotation=90, color="#1f4e9c")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 2.35)
    ax.set_ylabel("sector-count growth base")
    ax.set_title("(c)  the exponent comes out of the grammar", fontsize=9.5)
    ax.legend(fontsize=7.0, frameon=False, loc="upper right")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


# --- 8. tables ----------------------------------------------------------------

def write_tables(bc: str, outdir: str, d: Optional[Dict] = None):
    d = d or load(bc) or build(bc)
    os.makedirs(outdir, exist_ok=True)

    rows = []
    for e in d["rules"]:
        last = e["rows"][-1]
        st = last["charges"]
        cons = ", ".join(n for n, ok in st.items() if ok) or "---"
        rec = e["recurrence"]
        rows.append(
            f"$W_{{{e['rule']}}}$ & \\rt{{{e['tuple']}}} & \\rt{{{e['word']}}} & "
            f"{cons} & {last['N']} & {last['n_sectors']} & "
            f"{last['n_wall_sets']} & "
            f"{'yes' if last['complete'] else 'no'} & "
            f"{'yes' if last['sizes_match'] else 'no'} & "
            f"{rec['order']} & {rec['dominant_root']:.6f} & "
            f"{e['r9_sector_base']:.6f} \\\\")
    tab = ("\\begin{tabular}{lllcrrrccrrr}\n\\toprule\n"
           "rule & tuple & wall & conserved & $N$ & sectors & wall sets & "
           "complete & sizes & rec.\\ order & root & R9 base \\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    with open(os.path.join(outdir, f"tab_r18_walls_{bc}.tex"), "w") as f:
        f.write(tab)

    rrows = []
    for e in d["rules"]:
        for r in e["resolving"]:
            rrows.append(
                f"$W_{{{e['rule']}}}$ & {_TEX_CHARGE} & {r['N']} & "
                f"{r['n_levels_one_charge']} & {r['n_charge_levels']} & "
                f"{r['n_sectors']} & {r['sectors_per_level']:.1f} \\\\")
    rtab = ("\\begin{tabular}{llrrrrr}\n\\toprule\n"
            "rule & conserved & $N$ & levels (wall charge) & levels (both) & "
            "Krylov sectors & ratio \\\\\n\\midrule\n" + "\n".join(rrows)
            + "\n\\bottomrule\n\\end{tabular}\n")
    with open(os.path.join(outdir, f"tab_r18_resolving_{bc}.tex"), "w") as f:
        f.write(rtab)
    return tab


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(
        description="conservation laws vs frozen walls for 108/201/156/198")
    ap.add_argument("--bc", default=BC_DEFAULT)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--n-max", type=int, default=max(NS_DEFAULT))
    args = ap.parse_args(argv)
    Ns = tuple(n for n in NS_DEFAULT if n <= args.n_max)
    d = build(args.bc, Ns) if args.rebuild else (load(args.bc) or
                                                 build(args.bc, Ns))
    for e in d["rules"]:
        last = e["rows"][-1]
        st = last["charges"]
        rec = e["recurrence"]
        print(f"W{e['rule']:<4d} {e['tuple']}  wall='{e['word']}'  "
              f"conserved={[n for n, ok in st.items() if ok]}  "
              f"N={last['N']} sectors={last['n_sectors']} "
              f"wallsets={last['n_wall_sets']} complete={last['complete']} "
              f"sizes_match={last['sizes_match']}  "
              f"root={rec['dominant_root']:.6f} (R9 {e['r9_sector_base']:.6f})")
        cf = e["charge_from_walls"]
        for name, v in cf.items():
            if v:
                print(f"      {name} determined by walls={v['determined_by_walls']}"
                      + (f", = {v['affine_in_wall_count'][0]:g}*|W|"
                         f" + {v['affine_in_wall_count'][1]:g}"
                         if v["affine_in_wall_count"] else ""))
        r = e["resolving"][-1]
        print(f"      charge level sets={r['n_charge_levels']} vs "
              f"sectors={r['n_sectors']}  ratio={r['sectors_per_level']:.1f}")
    figdir = os.path.join(results_io.REPO_ROOT, "figures")
    os.makedirs(figdir, exist_ok=True)
    fig_walls(args.bc, os.path.join(figdir, f"fig_wall_charges_{args.bc}"), d)
    write_tables(args.bc, os.path.join(results_io.REPO_ROOT, "reports", "tex"), d)
    print("figure + tables written")


if __name__ == "__main__":
    main()
