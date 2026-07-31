"""
Coherence in the permutation circuits: where R10's first draft was wrong.

THE ERROR.  R10 said of the X-gate family that there is "no interference, so no
destructive cancellation in succ; no dark states; no distinction between the
monitored and the unmonitored channel at the level of the support".  The clause
after the last semicolon is right and the ones before it are not, and the reason
is that a functional graph is a statement about POPULATIONS only.

A reset is not a unitary.  Reset-to-|0> has two Kraus operators, |0><0| and
|0><1|, and BOTH send a basis state to a state whose bit is 0 -- which is why
succ(x) is single-valued and the graph is functional -- but they act completely
differently on a coherence.  For rho = |x><y| the channel gives

    sum_mu K_mu |x><y| K_mu^dagger ,

and the only surviving terms are those where the SAME mu has a non-zero matrix
element on both sides.  At a reset site that means: the coherence survives iff x
and y agree on the target bit at the moment the reset fires, and is destroyed
otherwise.  So label the Kraus operator of one sweep by

    J(x) = the set of sites at which a reset fired on a bit that did NOT already
           hold the reset value                                  ("jump sites")

and the rule is: |x><y| -> |Phi(x)><Phi(y)| if J(x) = J(y), and 0 otherwise.
This is exactly the same-label join of the Tier-1d pair graph (R7), specialised
to a family where succ is single-valued, and it is invisible to the functional
graph because J does not affect where a basis state goes.

CONSEQUENCE.  Being recurrent is not enough for a coherence to survive; the
labels have to stay synchronised around the cycle.  Where they do, the span of
the synchronised states is a DECOHERENCE-FREE SUBSPACE: the channel acts on it as
the permutation, which is unitary.  The relation

    x ~ y   iff   J(Phi^t x) = J(Phi^t y) for every t >= 0

is an equivalence relation on the recurrent set, and its classes are exactly the
DFS blocks.  R10 sec.8's "inheritance" measurement -- the reset acts as the
IDENTITY on the recurrent set for 177 of 240 rules -- is precisely the statement
J = 0 there, so for those 177 rules the WHOLE recurrent set is one DFS.  That
reading was missed in the first draft.

The type case is rule 232 (DIIE), which is V-free and therefore the same circuit
under both gates: its recurrent states are all fixed points, no reset ever jumps
on them, and the exact Cesaro rank is n_rec^2 (49, 121, 289 at N = 4, 5, 6 obc0)
-- a full matrix algebra M_d, i.e. a decoherence-free subspace of dimension
d = n_rec ~ phi^N.  The monitored description sees d classical pointer states and
misses all d^2 - d coherences.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import results_io
from ..core import rules as rules_mod
from ..core.cycle import _compile
from . import analysis, xca

DFS_PATH = os.path.join(analysis.ANALYTICS, "xgate_dfs_{bc}.json")


def jump_sites(x: int, N: int, rule: int, bc: str = "obc0",
               t=None) -> Tuple[int, int]:
    """
    (successor, jump label) for one brick-wall sweep.

    The label is a bitmask of the sites where a reset fired on a bit that did not
    already hold the reset value -- the sites where the Kraus branch is the
    off-diagonal |0><1| (or |1><0|) rather than the diagonal one.  A rule with no
    reset always returns label 0, which is the statement that a unitary circuit
    decoheres nothing.
    """
    if t is None:
        t = rules_mod.wolfram_to_tuple(rule)
    lab = 0
    for (i, lpos, rpos, tt) in _compile(N, t, bc):
        left = (x >> lpos) & 1 if lpos >= 0 else 0
        right = (x >> rpos) & 1 if rpos >= 0 else 0
        s = tt[2 * left + right]
        if s == "I":
            continue
        bit = 1 << i
        cur = (x >> i) & 1
        if s == "V":
            x ^= bit
        elif s == "D":
            if cur:
                lab |= bit            # |0><1| fired
            x &= ~bit
        else:                         # "E"
            if not cur:
                lab |= bit            # |1><0| fired
            x |= bit
    return x, lab


def recurrent_states(rule: int, N: int, bc: str = "obc0") -> List[List[int]]:
    """The cycles of the functional graph, as lists of states."""
    T = xca.step_table_np(rule, N, bc)
    seen = np.zeros(1 << N, np.int8)
    out: List[List[int]] = []
    for x in range(1 << N):
        if seen[x]:
            continue
        path = []
        y = x
        while seen[y] == 0:
            seen[y] = 1
            path.append(y)
            y = int(T[y])
        if seen[y] == 1:
            out.append(path[path.index(y):])
        for z in path:
            seen[z] = 2
    return out


def _label_word(x: int, N: int, rule: int, bc: str, length: int, t) -> Tuple:
    """The first `length` Kraus labels along the trajectory of x."""
    w = []
    for _ in range(length):
        x, lab = jump_sites(x, N, rule, bc, t)
        w.append(lab)
    return tuple(w)


def dfs_blocks(rule: int, N: int, bc: str = "obc0",
               max_rec: int = 200_000) -> Dict:
    """
    Partition the recurrent set into decoherence-free blocks.

    Two recurrent states carry a surviving mutual coherence iff their Kraus
    label words agree for ever.  Both lie on cycles, so the words are periodic
    and it is enough to compare them over the least common multiple of the two
    periods; comparing over lcm(all periods) is simpler and no less correct, but
    can be large, so we compare over a window of L_max * 2 and record the window.

    Returns the block sizes.  A block of size d means a DFS of dimension d: the
    channel restricted to its span is the permutation, hence unitary.
    """
    from math import gcd
    t = rules_mod.wolfram_to_tuple(rule)
    cycles = recurrent_states(rule, N, bc)
    n_rec = sum(len(c) for c in cycles)
    if n_rec > max_rec:
        return {"rule": rule, "N": N, "bc": bc, "n_rec": n_rec,
                "skipped": True}
    lengths = sorted({len(c) for c in cycles})
    lcm = 1
    for L in lengths:
        lcm = lcm * L // gcd(lcm, L)
    window = min(max(2 * max(lengths), 8), 4096)
    groups: Dict[Tuple, List[int]] = {}
    for c in cycles:
        for x in c:
            groups.setdefault(_label_word(x, N, rule, bc, window, t),
                              []).append(x)
    sizes = sorted((len(v) for v in groups.values()), reverse=True)
    all_quiet = all(k == tuple([0] * window) for k in groups)
    return {"rule": rule, "N": N, "bc": bc, "tuple": "".join(t),
            "n_rec": n_rec, "n_cycles": len(cycles),
            "cycle_lengths": lengths[-6:], "lcm_of_periods": lcm,
            "window": window,
            "n_blocks": len(sizes), "dfs_max": sizes[0] if sizes else 0,
            "dfs_sizes": sizes[:12],
            "no_jump_anywhere": bool(all_quiet),
            "skipped": False}


def census(Ns: Sequence[int] = (10, 12), bc: str = "obc0",
           rules_in: Optional[Sequence[int]] = None) -> Dict:
    """dfs_blocks for every rule, at a couple of sizes."""
    rows = []
    for rule in (range(256) if rules_in is None else rules_in):
        for N in Ns:
            r = dfs_blocks(rule, N, bc)
            rows.append(r)
    out = {"bc": bc, "Ns": list(Ns), "rows": rows}
    os.makedirs(analysis.ANALYTICS, exist_ok=True)
    with open(DFS_PATH.format(bc=bc), "w") as f:
        json.dump(out, f)
    return out


def load(bc: str = "obc0") -> Optional[Dict]:
    p = DFS_PATH.format(bc=bc)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def summarise(d: Dict) -> Dict:
    """How many rules carry a DFS of dimension > 1, by family."""
    from ..core import rules as R
    N = max(d["Ns"])
    fam_of = {}
    for rule in range(256):
        t = R.wolfram_to_tuple(rule)
        fam_of[rule] = ("reversible" if all(s in "IV" for s in t)
                        else ("V+reset" if "V" in t else "V-free"))
    per: Dict[str, Dict[str, int]] = {}
    big = []
    for r in d["rows"]:
        if r["N"] != N or r.get("skipped"):
            continue
        f = fam_of[r["rule"]]
        row = per.setdefault(f, {"rules": 0, "dfs": 0, "whole_rec": 0})
        row["rules"] += 1
        if r["dfs_max"] > 1:
            row["dfs"] += 1
            big.append((r["dfs_max"], r["rule"], r["tuple"], r["n_rec"]))
        if r["dfs_max"] == r["n_rec"] and r["n_rec"] > 1:
            row["whole_rec"] += 1
    big.sort(reverse=True)
    return {"N": N, "by_family": per, "largest": big[:20]}


def main(argv=None):
    ap = argparse.ArgumentParser(description="X-gate decoherence-free blocks")
    ap.add_argument("--bc", default="obc0", choices=["obc0", "pbc"])
    ap.add_argument("--n", type=int, nargs="*", default=[10, 12])
    a = ap.parse_args(argv)
    d = census(tuple(a.n), a.bc)
    s = summarise(d)
    print(f"DFS census at N={s['N']} ({a.bc})")
    for f, row in sorted(s["by_family"].items()):
        print(f"  {f:<11} {row['rules']} rules; DFS of dim>1: {row['dfs']}; "
              f"whole recurrent set one block: {row['whole_rec']}")
    print("\n  largest blocks:")
    for dim, rule, tup, nrec in s["largest"]:
        print(f"    X{rule:<4}{tup}  dfs {dim:<8} of n_rec {nrec}")


if __name__ == "__main__":
    main()
