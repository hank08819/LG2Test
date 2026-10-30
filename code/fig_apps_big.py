"""The four large stores on one figure: pathology (NCT-CRC-HE-100K, Phikon), single cell (SEA-AD MTG, 1,378,211 nuclei),
supply chain (Instacart, 195,827 customers), AI agent memory (MS MARCO, 1,000,000 passages, BGE-base).
Row per store: (profile) nu_k on the 2,000-point subsample; (plane) L (exact) against G (ten subsamples); (order) O-DLP@10;
(NPCA) the NPCA map, axes named by top loadings (Instacart products) or by the most common class among the 5% of points scoring
highest on the axis; (SR-tSNE2) the SR-tSNE2 map. Maps show a uniform subsample of 20,000 points; MS MARCO has no classes and is
shaded by point density.
usage: python3 fig_apps_big.py <large dir> <apps npz dir> <fig dir> <checks file>"""
import os, sys, json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import gaussian_kde
LD, AD, F, CK = sys.argv[1:5]
ROWS = [("path100k", "Pathology-100K\n100,000 tiles"), ("seaad1m", "SEA-AD (single cell)\n1,378,211 nuclei"), ("instacart", "Instacart (supply chain)\n195,827 customers"), ("marco1m", "MS MARCO (agent memory)\n1,000,000 passages")]
NAME = {"ADI": "adipose", "BACK": "background", "DEB": "debris", "LYM": "lymphocytes", "MUC": "mucus", "MUS": "muscle", "NORM": "normal mucosa", "STR": "stroma", "TUM": "tumor"}
MS = ["PCA", "NPCA", "t-SNE", "t-SNE-200", "SR-tSNE2", "UMAP", "PaCMAP"]
C = {"PCA": "0.55", "NPCA": "black", "t-SNE": "#e08a8a", "t-SNE-200": "#f2c4c4", "SR-tSNE2": "#8b0000", "UMAP": "#2c7fb8", "PaCMAP": "#41b6c4"}
LBL = {"t-SNE-200": "t-SNE$_{200}$"}
OWN = ("SR-tSNE2", "NPCA")
plt.rcParams.update({"font.size": 6.5, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(4, 5, figsize=(7.2, 8.0), gridspec_kw=dict(width_ratios=[1.05, 0.9, 0.95, 1.1, 1.1], hspace=0.6, wspace=0.8))
L_ = "abcdefghijklmnopqrst"; check = []
def short(s, w=26): return s if len(s) <= w else s[:w - 1] + "…"
for row, (st, title) in enumerate(ROWS):
    res = json.load(open(f"{LD}/{st}/scores.json")); z = np.load(f"{AD}/hh_apps_{st}.npz", allow_pickle=True); P = lambda j: f"({L_[5 * row + j]})"
    lab = np.array([NAME.get(v, v) for v in z["lab"]]) if "lab" in z.files else None
    if st == "marco1m":   # k-means (k=8) of the one million BGE embeddings; each cluster named by its top two TF-IDF words
        kc = np.load(os.path.join(AD, "marco1m_clusters8.npz")); lab = np.array([", ".join(str(n).split(", ")[:2]) for n in kc["names"]])[kc["lab"]]
    a = ax[row, 0]
    for m in MS:
        nu = np.load(f"{LD}/{st}/nu_{m}.npy"); s = np.arange(1, len(nu) + 1) / (len(nu) + 2); keep = s <= 0.6; nu, s = nu[keep], s[keep]
        a.plot(s, nu, color=C[m], ls="--" if m == "NPCA" else "-", lw=1.3 if m in OWN else 0.8, label=LBL.get(m, m))
    a.set_xscale("log"); a.set_xticks([1e-3, 1e-1]); a.set_xticklabels(["$10^{-3}$", "$10^{-1}$"]); a.minorticks_off()
    a.axvspan(0.3, 0.5, color="0.9", zorder=0); a.axvline(10 / 2000, color="0.6", ls=":", lw=0.7)
    a.set_ylabel(title + "\n" + r"$\nu_k$", fontsize=6.3); a.set_title(f"{P(0)} Profile", loc="left", fontsize=7)
    if row == 3: a.set_xlabel("$k/m$ (log), $m=2{,}000$")
    a = ax[row, 1]
    for m in MS:
        a.scatter(res[m]["L"], res[m]["G"], s=16, color=C[m], edgecolor="black" if m in OWN else "none", zorder=3)
        check.append((st, m, "L", round(res[m]["L"], 4), "G", round(res[m]["G"], 4), "odlp", round(res[m]["odlp"], 4)))
    OFF = {"seaad1m": {"SR-tSNE2": (-4, 9)}, "marco1m": {"NPCA": (5, -7)}}
    for m in OWN: a.annotate(m, (res[m]["L"], res[m]["G"]), xytext=OFF.get(st, {}).get(m, (-4, -9) if m == "SR-tSNE2" else (5, 5)), textcoords="offset points", ha="right" if m == "SR-tSNE2" else "left", va="center", fontsize=5.5, fontweight="bold")
    a.xaxis.set_major_locator(plt.MaxNLocator(3)); a.yaxis.set_major_locator(plt.MaxNLocator(4))
    a.set_title(f"{P(1)} $G$ against $L$", loc="left", fontsize=7)
    if row == 3: a.set_xlabel("$L$")
    a = ax[row, 2]; v = [res[m]["odlp"] for m in MS]
    a.barh(range(len(MS)), v, color=[C[m] for m in MS], edgecolor=["black" if m in OWN else "none" for m in MS], lw=0.5)
    a.set_yticks(range(len(MS))); a.set_yticklabels([LBL.get(m, m) for m in MS], fontsize=5.3); a.invert_yaxis(); a.set_xlim(0, max(v) * 1.55); a.set_xticks([])
    for i, x in enumerate(v): a.text(x, i, f" {x:.3f}", va="center", fontsize=4.8)
    a.set_title(f"{P(2)} O-DLP@10", loc="left", fontsize=7)
    if row == 3: a.set_xlabel("neighbor order")
    cl = [c for c, _ in Counter(lab).most_common(9)] if lab is not None else []
    for j, key in ((3, "npca"), (4, "sr")):
        a = ax[row, j]; Y = z[key]
        if lab is not None:
            a.scatter(Y[:, 0], Y[:, 1], s=0.15, color="0.85", lw=0, rasterized=True)
            for i_, c in enumerate(cl): w = lab == c; a.scatter(Y[w, 0], Y[w, 1], s=0.15, color=plt.cm.tab10(i_), lw=0, rasterized=True, label=c)
        else:
            d = gaussian_kde(Y[::4].T)(Y.T); o = np.argsort(d); a.scatter(Y[o, 0], Y[o, 1], s=0.15, c=d[o], cmap="Greys", vmin=-0.3 * d.max(), lw=0, rasterized=True)
        a.set_xticks([]); a.set_yticks([])
    a = ax[row, 3]; names = []
    for q, s_ in enumerate(z["axes"]):
        if "|" in s_ and st != "instacart":
            c, f_ = s_.split("|"); names.append(f"{NAME.get(c, c)} ({100 * float(f_):.0f}%)")
        elif st == "instacart": names.append(", ".join(t.strip().replace("Organic ", "org. ").replace("Bag of org. Bananas", "org. bananas") for t in s_.split("|")[:2]))
        else: names.append("")
        check.append((st, f"NPCA axis {q + 1}", s_))
    import textwrap
    if names[0]: a.set_xlabel("axis 1: " + "\n".join(textwrap.wrap(names[0], 34)), fontsize=5.5)
    if names[1]: a.set_ylabel("axis 2: " + "\n".join(textwrap.wrap(names[1], 34)), fontsize=5.5)
    a.set_title(f"{P(3)} NPCA", loc="left", fontsize=7)
    a = ax[row, 4]; a.set_title(f"{P(4)} SR-tSNE2", loc="left", fontsize=7)
    if lab is not None:
        h = [plt.Line2D([], [], ls="", marker="o", ms=3, color=plt.cm.tab10(i_)) for i_ in range(len(cl))]
        a.legend(h, cl, frameon=False, fontsize=5, loc="center left", bbox_to_anchor=(1.0, 0.5), handletextpad=0.1, labelspacing=0.25)
    else: a.text(1.03, 0.5, "no classes;\nshaded by\npoint density", transform=a.transAxes, fontsize=5.5, va="center")
h, l = ax[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=7, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 0.94))
for e in ("pdf", "png"): fig.savefig(f"{F}/fig_apps_big.{e}", bbox_inches="tight", dpi=220)
with open(CK, "a") as fh:
    fh.write("== fig_apps_big (four large stores; profiles on 2,000 subsample; maps show 20,000 points)\n")
    for c in check: fh.write("  " + " ".join(map(str, c)) + "\n")
print(*check, sep="\n")
