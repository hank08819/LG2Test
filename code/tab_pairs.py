"""Rows of the table of reversal tests beyond PCA: UMAP, t-SNE and SR-tSNE2 (map A, local) against PaCMAP, and t-SNE and
SR-tSNE2 against NPCA (map B, global), on the twelve large datasets; p of the declaration (bold: declared at level 0.05)
and crossover s^x in % of n. Last pair: NPCA against PCA, global end only: G^PCA - G^NPCA and p_G (bold below 0.05).
usage: python3 tab_pairs.py <results dir> <tex dir>"""
import sys, json
R, T = sys.argv[1:3]
r = json.load(open(f"{R}/pairwise_lgtest.json")); r.update(json.load(open(f"{R}/pairwise_lgtest_sr.json"))); r.update(json.load(open(f"{R}/pairwise_lgtest_npca.json"))); r.update(json.load(open(f"{R}/pairwise_lgtest_pcanpca.json")))
DS = [("path100k", "Pathology-100K"), ("seaad1m", "SEA-AD"), ("instacart", "Instacart"), ("marco1m", "MS MARCO"), None,
      ("mnist", "MNIST"), ("cifar10", "CIFAR-10"), ("fmnist", "Fashion-MNIST"), ("speech", "Speech Commands"),
      ("renal", "Renal-Cortex"), ("brain", "Brain-Cortex"), ("covertype", "Covertype"), ("agnews", "AG News")]
def fp(p):
    if p >= 0.1: return f"{p:.2f}"
    if p >= 0.01:                                        # enough digits that the shown value falls on the same side of 0.05
        for d in (3, 4, 5, 6):
            s = f"{p:.{d}f}"
            if (p < 0.05) == (float(s) < 0.05): return s
    m, e = f"{p:.1e}".split("e"); return f"{m}\\times10^{{{int(e)}}}"
out = []
for d in DS:
    if d is None: out.append("\\midrule"); continue
    cells = []
    for A, B in (("UMAP", "PaCMAP"), ("t-SNE", "PaCMAP"), ("SR-tSNE2", "PaCMAP"), ("t-SNE", "NPCA"), ("SR-tSNE2", "NPCA")):
        x = r[f"{d[0]}|{A}|{B}"]; p = fp(x["p_reversal"])
        cells += [f"$\\mathbf{{{p}}}$" if x["reversal"] else f"${p}$", f"{100 * x['crossover']:.1f}"]
    x = r[f"{d[0]}|NPCA|PCA"]; p = fp(x["pG"])
    cells += [f"${x['dG']:+.3f}$".replace("+", ""), f"$\\mathbf{{{p}}}$" if x["pG"] < 0.05 else f"${p}$"]
    out.append(" & ".join([d[1]] + cells) + " \\\\")
open(f"{T}/tab_pairs_rows.tex", "w").write("\n".join(out) + "\n")
print("\n".join(out))
