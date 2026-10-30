"""Table of the author's methods against the method each one modifies, from own_*.json (exp_own.py) and paper_*.json.
usage: python3 tab_own.py <own results dir> <paper results dir> <out .tex>"""
import sys, json, numpy as np
O, P, out = sys.argv[1], sys.argv[2], sys.argv[3]
DS = ["Iris", "Wine", "Breast cancer", "Digits", "MNIST-5k", "Swiss roll", "S-curve", "Tree"]
rows, summ = [], dict(dL=[], dG=[], up_L=0, up_G=0, n=0)
for d in DS:
    o = json.load(open(f"{O}/own_{d.replace(' ', '_')}.json"))["methods"]; p = json.load(open(f"{P}/paper_{d.replace(' ', '_')}.json"))["methods"]
    pca = p["PCA"][0]; npca = o["NPCA"][0]
    st = o["t-SNE (PCA init, start)"]; sr = o["SR-tSNE2"]
    Ls, Gs = np.array([r["L"] for r in st]), np.array([r["G"] for r in st]); Lr, Gr = np.array([r["L"] for r in sr]), np.array([r["G"] for r in sr])
    dL, dG = Lr - Ls, Gr - Gs; summ["dL"] += dL.tolist(); summ["dG"] += dG.tolist(); summ["up_L"] += int((dL > 0).sum()); summ["up_G"] += int((dG > 0).sum()); summ["n"] += len(dL)
    rows.append(f"{d} & {pca['L']:.2f}/{pca['G']:.2f} & {npca['L']:.2f}/{npca['G']:.2f} & {Ls.mean():.3f}/{Gs.mean():.3f} & {Lr.mean():.3f}/{Gr.mean():.3f} & ${dL.mean():+.3f}$ & ${dG.mean():+.3f}$ \\\\")
    print(d, "PCA %.3f/%.3f NPCA %.3f/%.3f rho %.3f/%.3f | start %.3f/%.3f SR %.3f/%.3f (sd %.4f/%.4f) dL %+.4f dG %+.4f" % (
        pca["L"], pca["G"], npca["L"], npca["G"], pca["rho"], npca["rho"], Ls.mean(), Gs.mean(), Lr.mean(), Gr.mean(), Lr.std(ddof=1), Gr.std(ddof=1), dL.mean(), dG.mean()))
open(out, "w").write("\n".join(rows) + "\n")
print("SR-tSNE2 vs its start over all runs: L up %d/%d, G up %d/%d; dL range %+.4f..%+.4f, dG range %+.4f..%+.4f" % (
    summ["up_L"], summ["n"], summ["up_G"], summ["n"], min(summ["dL"]), max(summ["dL"]), min(summ["dG"]), max(summ["dG"])))
