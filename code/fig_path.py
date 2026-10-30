"""Pathology figure: 7,180 H&E tiles (CRC-VAL-HE-7K), Phikon embeddings.
(a) SR-tSNE2 map by tissue; (b) the same map paved with the tiles; (c) NPCA map with the six highest-scoring tiles of each axis;
(d) retrieval for one tumor tile drawn at random (seed 0): its 5 nearest tiles in the Phikon space, the SR-tSNE2 map and the UMAP map,
with the number of the 10 true neighbors each map returns among its 10 nearest.
usage: python3 fig_path.py <path dir> <fig dir>"""
import sys, json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
D, F = sys.argv[1:3]
z = np.load(f"{D}/path.npz", allow_pickle=True); X = z["X"]; lab = z["label"]; T = np.load(f"{D}/thumbs.npz")["T"]; M = np.load(f"{D}/path_maps.npz")
res = json.load(open(f"{D}/path.json"))
NAME = {"ADI": "adipose", "BACK": "background", "DEB": "debris", "LYM": "lymphocytes", "MUC": "mucus", "MUS": "smooth muscle", "NORM": "normal mucosa", "STR": "stroma", "TUM": "tumor"}
CL = list(NAME); col = {c: plt.cm.tab10(i) for i, c in enumerate(CL)}
plt.rcParams.update({"font.size": 7})
fig = plt.figure(figsize=(7.2, 8.4)); gs = fig.add_gridspec(3, 2, height_ratios=[1, 0.9, 0.8], hspace=0.3, wspace=0.12)
Y = M["SR_tSNE2"]
a = fig.add_subplot(gs[0, 0]); [s_.set_visible(False) for s_ in a.spines.values()]
for c in CL: w = lab == c; a.scatter(Y[w, 0], Y[w, 1], s=0.8, color=col[c], lw=0, label=NAME[c])
a.legend(frameon=False, fontsize=5.6, markerscale=6, loc="lower left", ncol=5, bbox_to_anchor=(0, -0.13), columnspacing=0.5, handletextpad=0.1)
a.set_xticks([]); a.set_yticks([]); a.set_title(f"(a) SR-tSNE2 map of 7,180 tiles ($L$={res['SR-tSNE2']['L']:.3f}, $G$={res['SR-tSNE2']['G']:.3f})", loc="left", fontsize=7.5)
a = fig.add_subplot(gs[0, 1]); g = 36; [s_.set_visible(False) for s_ in a.spines.values()]
x0, x1 = Y[:, 0].min(), Y[:, 0].max(); y0, y1 = Y[:, 1].min(), Y[:, 1].max(); dx, dy = (x1 - x0) / g, (y1 - y0) / g
nn = cKDTree(Y)
for i in range(g):
    for j in range(g):
        cx, cy = x0 + (i + 0.5) * dx, y0 + (j + 0.5) * dy; d, k = nn.query([cx, cy]); d = np.array([[d]])
        if d[0, 0] < 0.6 * max(dx, dy): a.imshow(T[k], extent=(cx - dx / 2, cx + dx / 2, cy - dy / 2, cy + dy / 2), interpolation="lanczos")
a.set_xlim(x0, x1); a.set_ylim(y0, y1); a.set_xticks([]); a.set_yticks([]); a.set_aspect("auto")
a.set_title("(b) The same map paved with the tiles", loc="left", fontsize=7.5)
a = fig.add_subplot(gs[1, 0]); Yn = M["NPCA"]; a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
for c in CL: w = lab == c; a.scatter(Yn[w, 0], Yn[w, 1], s=0.8, color=col[c], lw=0)
from collections import Counter
names = []
for q in range(2):
    hi = np.argsort(-Yn[:, q])[:len(Yn) // 20]; c, k = Counter(lab[hi]).most_common(1)[0]; names.append(f"{NAME[c]} ({100 * k / len(hi):.0f}% of top 5%)"); print("CHECK NPCA axis", q + 1, c, k / len(hi))
a.set_xlabel("NPCA 1: " + names[0], fontsize=6.3); a.set_ylabel("NPCA 2: " + names[1], fontsize=6.3); a.set_xticks([]); a.set_yticks([])
a.set_title(f"(c) NPCA: each axis a tissue ($G$={res['NPCA']['G']:.3f}; PCA {res['PCA']['G']:.3f})", loc="left", fontsize=7.5)
sub = gs[1, 1].subgridspec(2, 1, hspace=0.15)
for q in range(2):
    s2 = sub[q].subgridspec(1, 6, wspace=0.05); top = np.argsort(-Yn[:, q])[:6]
    for j, k in enumerate(top):
        b = fig.add_subplot(s2[0, j]); b.imshow(T[k]); b.set_xticks([]); b.set_yticks([])
        for s_ in b.spines.values(): s_.set_visible(True); s_.set_color(col[lab[k]]); s_.set_linewidth(1.5)
        if j == 0: b.set_title(f"highest on NPCA axis {q + 1}", loc="left", fontsize=6.5)
rng = np.random.default_rng(0); qi = rng.choice(np.where(lab == "TUM")[0])
true10 = cKDTree(X).query(X[qi], 11)[1]; true10 = [k for k in true10 if k != qi][:10]
import json, os
RT = {}
if os.path.exists(f"{sys.argv[1]}/retrieval.json"):
    _r = json.load(open(f"{sys.argv[1]}/retrieval.json")); RT = {"SR-tSNE2 map": _r["SR_tSNE2"]["TUM"], "UMAP map": _r["UMAP"]["TUM"]}
rows = [("Phikon space", None), ("SR-tSNE2 map", M["SR_tSNE2"]), ("UMAP map", M["UMAP"])]
sub = gs[2, :].subgridspec(3, 7, wspace=0.05, hspace=0.25, width_ratios=[1.3, 1, 1, 1, 1, 1, 1])
for r, (nm, Z) in enumerate(rows):
    if Z is None: nb = true10[:5]; hit = 10
    else:
        nb10 = cKDTree(Z).query(Z[qi], 11)[1]; nb10 = [k for k in nb10 if k != qi][:10]
        nb = nb10[:5]; hit = len(set(nb10) & set(true10))
    t = fig.add_subplot(sub[r, 0]); t.axis("off"); t.text(0, 0.5, f"{nm}\nthis tile: {hit} of 10\n" + ("" if Z is None else f"all tumor tiles: {RT[nm]:.1f}"), va="center", fontsize=6.3)
    if r == 0: t.set_title("(d) Retrieval of tumor tissue; green: among the 10 true neighbors (mean over 1,233 tumor tiles at left)", loc="left", fontsize=7.5, x=0, pad=10)
    print("CHECK retrieval", nm, hit)
    if r == 0:
        b = fig.add_subplot(sub[r, 1]); b.imshow(T[qi]); b.set_xticks([]); b.set_yticks([]); b.set_xlabel("query", fontsize=6.3, labelpad=1)
        for s_ in b.spines.values(): s_.set_color("black"); s_.set_linewidth(2)
    for j, k in enumerate(nb):
        b = fig.add_subplot(sub[r, 2 + j]); b.imshow(T[k]); b.set_xticks([]); b.set_yticks([])
        ok = k in true10
        for s_ in b.spines.values(): s_.set_visible(True); s_.set_color("#1a9850" if ok else "#d73027"); s_.set_linewidth(1.8)
fig.savefig(f"{F}/fig_path.pdf", bbox_inches="tight", dpi=300); fig.savefig(f"{F}/fig_path.png", dpi=200, bbox_inches="tight")
