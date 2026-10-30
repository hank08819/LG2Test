"""Four stores on one figure: pathology (CRC-VAL-HE-7K tiles, Phikon), single cell (PBMC 3k), supply chain (Online Retail II) and
finance agent memory (SEC 8-K filings, BGE-base). Row per store: (profile) nu_k; (plane) L against G; (order) O-DLP@10;
(NPCA) the NPCA map with its axes named; (SR-tSNE2) the SR-tSNE2 map. NPCA axes: the top loadings where the inputs are genes or
products; where the inputs are encoder coordinates, the most common class among the 5% of points scoring highest on the axis.
usage: python3 fig_apps.py <fig dir> <stem=dir> ..."""
import os, sys, json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from collections import Counter
F = sys.argv[1]; ROWS = [a.split("=") for a in sys.argv[2:]]
TITLE = {"path": "Pathology-7K\n7,180 H&E tiles", "pbmc": "Single cell\n2,638 PBMCs", "retail": "Online Retail (supply chain)\n5,459 customers", "8k": "8-K (agent memory)\n5,000 filings"}
NAME = {"2.02": "earnings", "5.02": "officers", "7.01": "Reg FD", "8.01": "other events", "1.01": "agreement", "5.07": "votes", "5.03": "bylaws", "2.03": "debt", "none": "no item",
        "ADI": "adipose", "BACK": "background", "DEB": "debris", "LYM": "lymphocytes", "MUC": "mucus", "MUS": "muscle", "NORM": "normal mucosa", "STR": "stroma", "TUM": "tumor"}
RETAIL_K8 = ["heart T-light holders", "ceramic trinket boxes", "cake cases", "cake stands, baking", "ornaments, tea glasses", "metal signs", "vintage games, cards", "jumbo and lunch bags"]
LABKEY = {"path": "label", "pbmc": "label", "8k": "item"}
plt.rcParams.update({"font.size": 6.5, "axes.spines.top": False, "axes.spines.right": False})
MS = ["PCA", "NPCA", "t-SNE", "SR-tSNE2", "UMAP", "PaCMAP", "PHATE", "DREAMS"]
C = {"PCA": "0.55", "NPCA": "black", "t-SNE": "#e08a8a", "SR-tSNE2": "#8b0000", "UMAP": "#2c7fb8", "PaCMAP": "#41b6c4", "PHATE": "#7a5195", "DREAMS": "#d95f0e"}
OWN = ("SR-tSNE2", "NPCA")
nr = len(ROWS); fig, ax = plt.subplots(nr, 5, figsize=(7.2, 1.9 * nr + 0.4), gridspec_kw=dict(width_ratios=[1.05, 0.9, 0.95, 1.1, 1.1]))
ax = np.atleast_2d(ax); L_ = "abcdefghijklmnopqrstuvwxyz"; check = []
for row, (st, D) in enumerate(ROWS):
    res = json.load(open(f"{D}/{st}.json")); ms = [m for m in MS if m in res]; z = np.load(f"{D}/{st}.npz", allow_pickle=True)
    lab = np.array([NAME.get(str(v), str(v)) for v in z[LABKEY[st]]]) if st in LABKEY else None
    if st == "retail":   # k-means (k=8, seed 0) of the purchase vectors; each cluster named by the top products of its centroid
        lab = np.array(RETAIL_K8)[np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "retail_kmeans8.npy"))]
    maps = np.load(f"{D}/{st}_maps.npz"); P = lambda j: f"({L_[5 * row + j]})"
    a = ax[row, 0]
    for m in ms:
        nu = np.load(f"{D}/nu_{st}_{m}.npy"); s = np.arange(1, len(nu) + 1) / (len(nu) + 2)
        a.plot(s, nu, color=C[m], ls="--" if m == "NPCA" else "-", lw=1.3 if m in OWN else 0.8, label=m)
    a.set_xscale("log"); a.set_xticks([1e-3, 1e-1]); a.set_xticklabels(["$10^{-3}$", "$10^{-1}$"]); a.axvspan(0.3, 0.5, color="0.9", zorder=0); a.axvline(10 / (len(nu) + 2), color="0.6", ls=":", lw=0.7)
    a.set_ylabel(TITLE[st] + "\n" + r"$\nu_k$", fontsize=6.3); a.set_title(f"{P(0)} Profile", loc="left", fontsize=7)
    if row == nr - 1: a.set_xlabel("$k/n$ (log)")
    a = ax[row, 1]
    for m in ms:
        a.scatter(res[m]["L"], res[m]["G"], s=16, color=C[m], edgecolor="black" if m in OWN else "none", zorder=3)
        check.append((st, m, round(res[m]["L"], 4), round(res[m]["G"], 4), round(res[m]["odlp"], 4)))
    for m in OWN: a.annotate(m, (res[m]["L"], res[m]["G"]), xytext=(4, -9) if m == "SR-tSNE2" else (5, 0), textcoords="offset points", ha="right" if m == "SR-tSNE2" else "left", va="center", fontsize=5.5, fontweight="bold")
    a.xaxis.set_major_locator(plt.MaxNLocator(3)); a.yaxis.set_major_locator(plt.MaxNLocator(4))
    a.set_title(f"{P(1)} $G$ against $L$", loc="left", fontsize=7)
    if row == nr - 1: a.set_xlabel("$L$")
    a = ax[row, 2]; v = [res[m]["odlp"] for m in ms]
    a.barh(range(len(ms)), v, color=[C[m] for m in ms], edgecolor=["black" if m in OWN else "none" for m in ms], lw=0.5)
    a.set_yticks(range(len(ms))); a.set_yticklabels(ms, fontsize=5.3); a.invert_yaxis(); a.set_xlim(0, max(v) * 1.45); a.set_xticks([])
    for i, x in enumerate(v): a.text(x, i, f" {x:.3f}", va="center", fontsize=4.8)
    a.set_title(f"{P(2)} O-DLP@10", loc="left", fontsize=7)
    if row == nr - 1: a.set_xlabel("neighbor order")
    a = ax[row, 3]; Y = maps["NPCA"]; names = []
    if "names" in z.files:
        top = json.load(open(f"{D}/{st}_npca_loadings.json"))["top"]
        names = [", ".join(t[0] for t in top[f"axis{q + 1}"][:3]) for q in range(2)]
        if st == "retail":   # product descriptions shortened by hand; the full top loadings are printed below
            print("CHECK retail loadings", [[t[0] for t in top[f"axis{q + 1}"][:3]] for q in range(2)])
            names = ["cake cases, jumbo bags, lunch bags", "heart T-light holders, bird ornaments, wicker hearts"]
    else:
        for q in range(2):
            hi = np.argsort(-Y[:, q])[:max(1, len(Y) // 20)]; c, k = Counter(lab[hi]).most_common(1)[0]
            names.append(f"{c} ({100 * k / len(hi):.0f}%)"); check.append((st, f"NPCA axis {q + 1}", c, round(k / len(hi), 3)))
    if lab is not None:
        cl = [c for c, _ in Counter(lab).most_common(9)]
        a.scatter(Y[:, 0], Y[:, 1], s=0.6, color="0.85", lw=0)
        for i_, c in enumerate(cl): w = lab == c; a.scatter(Y[w, 0], Y[w, 1], s=0.6, color=plt.cm.tab10(i_), lw=0)
    else: a.scatter(Y[:, 0], Y[:, 1], s=0.6, color="0.35", alpha=0.4, lw=0)
    a.set_xticks([]); a.set_yticks([])
    import textwrap
    a.set_xlabel(textwrap.fill("axis 1: " + names[0], 34), fontsize=5, labelpad=1.5); a.set_ylabel(textwrap.fill("axis 2: " + names[1], 30), fontsize=5, labelpad=1.5)
    a.set_title(f"{P(3)} NPCA", loc="left", fontsize=7)
    a = ax[row, 4]; Y = maps["SR_tSNE2"]
    if lab is not None:
        a.scatter(Y[:, 0], Y[:, 1], s=0.6, color="0.85", lw=0)
        for i_, c in enumerate(cl): w = lab == c; a.scatter(Y[w, 0], Y[w, 1], s=0.6, color=plt.cm.tab10(i_), lw=0, label=c)
        a.legend(frameon=False, fontsize=4.6, markerscale=5, loc="center left", bbox_to_anchor=(1.0, 0.5), handletextpad=0.1, labelspacing=0.2)
    else: a.scatter(Y[:, 0], Y[:, 1], s=0.6, color="0.35", alpha=0.4, lw=0)
    a.set_xticks([]); a.set_yticks([]); a.set_title(f"{P(4)} SR-tSNE2", loc="left", fontsize=7)
from matplotlib.lines import Line2D
fig.legend([Line2D([], [], color=C[m], ls="--" if m == "NPCA" else "-", lw=1.3) for m in MS], MS, frameon=False, fontsize=6, ncol=8, loc="upper center", bbox_to_anchor=(0.5, 1.0))
fig.subplots_adjust(left=0.085, right=0.865, bottom=0.07, top=1 - 0.55 / fig.get_figheight(), wspace=0.42, hspace=0.48)
fig.savefig(f"{F}/fig_apps.pdf"); fig.savefig(f"{F}/fig_apps.png", dpi=200)
for c in check: print("CHECK", *c)
