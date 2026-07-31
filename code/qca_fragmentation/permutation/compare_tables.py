"""Tables for R11 (Hadamard vs X gate)."""

from __future__ import annotations

import argparse
import os
from typing import Dict

from .. import results_io
from ..core import rules
from . import compare

TEXDIR = os.path.join(results_io.REPO_ROOT, "reports", "tex")

_CLS = ("constant", "polynomial", "exponential", "irregular")


def _n(v, fmt="{:.4f}"):
    return "--" if v is None else fmt.format(v)


def tab_structure(bc: str, d: Dict) -> str:
    """The refinement checks: the backbone of the whole comparison."""
    rc = d["refinement"]
    pw = d["pointwise"]
    vf = d["vfree_identity"]
    Ns = ", ".join(f"$N={r['N']}$" for r in rc)
    rows = [
        f"$\\operatorname{{succ}}_X(x)\\in\\operatorname{{succ}}_H(x)$ & "
        f"{Ns} & ${min(r['edges_ok'] for r in rc)}/256$ & rules, all states \\\\",
        f"X sectors refine H sectors & {Ns} & "
        f"${min(r['refines_ok'] for r in rc)}/256$ & rules, exhaustive \\\\",
        f"$n_\\wcc^X\\ge n_\\wcc^H$ & $N\\le{d['n_cap']}$ & "
        f"${pw['units'] - len(pw['n_wcc_failures'])}/{pw['units']}$ & "
        f"units \\\\",
        f"$D_{{\\max}}^X\\le D_{{\\max}}^H$ & $N\\le{d['n_cap']}$ & "
        f"${pw['units'] - len(pw['d_max_failures'])}/{pw['units']}$ & "
        f"units \\\\",
        f"V-free rules identical in both stores & all $N$ & "
        f"${vf['units_same']}/{vf['units_same'] + vf['units_diff']}$ & "
        f"units, ${vf['rules']}$ rules \\\\",
    ]
    return ("\\begin{tabular}{llrl}\n\\hline\n"
            "check & window & holds & unit \\\\\n\\hline\n"
            + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n")


def tab_baseline(bc: str, d: Dict) -> str:
    """The 16 rules that are unitary under H and reversible under X."""
    rows = []
    for r in compare.baseline(d):
        rows.append(
            f"${r['rule']}$ & \\texttt{{{r['tuple']}}} & "
            f"${_n(r['a_h'])}$ & ${_n(r['b_h'])}$ & "
            f"${_n(r['a_x'])}$ & ${_n(r['b_x'])}$ & "
            f"{r['cls_a_h']} & {r['cls_a_x']} \\\\")
    return ("\\begin{tabular}{rlrrrrll}\n\\hline\n"
            " & & \\multicolumn{2}{c}{Hadamard} & \\multicolumn{2}{c}{$X$} & "
            "\\multicolumn{2}{c}{sector-count class} \\\\\n"
            "rule & tuple & $a$ & $b$ & $a$ & $b$ & $H$ & $X$ \\\\\n\\hline\n"
            + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n")


def tab_classes(bc: str, d: Dict) -> str:
    """Cross-tabulation of the sector-count growth class under the two gates."""
    ct = d["class_sector_count"]
    rows = []
    for h in _CLS:
        if h not in ct:
            continue
        cells = [f"${ct[h].get(x, 0)}$" for x in _CLS]
        rows.append(f"{h} & " + " & ".join(cells) +
                    f" & ${sum(ct[h].values())}$ \\\\")
    tot = [f"${sum(ct.get(h, {}).get(x, 0) for h in _CLS)}$" for x in _CLS]
    rows.append("\\hline\nall & " + " & ".join(tot) +
                f" & ${sum(sum(v.values()) for v in ct.values())}$ \\\\")
    return ("\\begin{tabular}{lrrrrr}\n\\hline\n"
            " & \\multicolumn{4}{c}{under $X$} & \\\\\n"
            "under $H$ & constant & linear & exponential & irregular & all "
            "\\\\\n\\hline\n" + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n")


def tab_survivors(bc: str, d: Dict) -> str:
    """R9's headline sets, and what the gate change does to them."""
    sv = compare.survivors(d)
    label = {"open_fragmented": "open-system fragmented (R9 \\S6.3)",
             "linear_count": "strictly linear sector count (R9 \\S6.4)",
             "frontier": "pinned frontier (R9 \\S6.5)"}
    rows = []
    for name, rowset in sv.items():
        rows.append("\\multicolumn{7}{l}{\\emph{%s}} \\\\" % label[name])
        for r in rowset:
            rows.append(
                f"\\quad ${r['rule']}$ & \\texttt{{{r['tuple']}}} & "
                f"{r['cls_a_h']} & ${r['n_wcc_h']:,}$ & {r['cls_a_x']} & "
                f"${r['n_wcc_x']:,}$ & {r['cls_b_rec_x']} \\\\"
                .replace(",", r"\,"))
    return ("\\begin{tabular}{rlrrrrl}\n\\hline\n"
            " & & \\multicolumn{2}{c}{$H$: sector count} & "
            "\\multicolumn{2}{c}{$X$: sector count} & $X$ cycle \\\\\n"
            "rule & tuple & class & at $N=16$ & class & at $N=16$ & class "
            "\\\\\n\\hline\n" + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n")


def tab_correlation(bc: str, d: Dict) -> str:
    co = d["correlation"]
    label = {"all": "all 256", "has_V": "the 175 with a \\rt{V}",
             "unitary": "the 16 unitary/reversible",
             "irreversible": "the 240 others"}
    rows = []
    for tag in ("all", "has_V", "unitary", "irreversible"):
        for coord, nm in (("a", "$a$ (sector count)"),
                          ("b", "$b$ (largest sector)")):
            v = co.get(f"{tag}_{coord}")
            if not v:
                continue
            pr = "--" if v["pearson"] is None else f"{v['pearson']:+.3f}"
            rows.append(f"{label[tag]} & {nm} & ${v['n']}$ & ${pr}$ & "
                        f"${v['mean_abs_diff']:.3f}$ & ${v['identical']}$ \\\\")
    return ("\\begin{tabular}{llrrrr}\n\\hline\n"
            "subset & coordinate & rules & Pearson $r$ & mean $|\\Delta|$ & "
            "unchanged \\\\\n\\hline\n"
            + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n")


def write_all(bc: str = "obc0") -> None:
    d = compare.load(bc) or compare.build(bc)
    os.makedirs(TEXDIR, exist_ok=True)
    for name, txt in ((f"tab_r11_structure_{bc}", tab_structure(bc, d)),
                      (f"tab_r11_baseline_{bc}", tab_baseline(bc, d)),
                      (f"tab_r11_classes_{bc}", tab_classes(bc, d)),
                      (f"tab_r11_survivors_{bc}", tab_survivors(bc, d)),
                      (f"tab_r11_correlation_{bc}", tab_correlation(bc, d))):
        p = os.path.join(TEXDIR, f"{name}.tex")
        with open(p, "w") as f:
            f.write(txt)
        print("wrote", p)


def main(argv=None):
    ap = argparse.ArgumentParser(description="R11 tables")
    ap.add_argument("--bc", default="obc0", choices=["obc0", "pbc"])
    write_all(ap.parse_args(argv).bc)


if __name__ == "__main__":
    main()
