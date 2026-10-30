"""Fig. 1 (concept) on MNIST, n = 70,000: from DLP_k to L and G.
Maps fitted on all 70,000 points (results/large/mnist); (a) uses the exact 10 nearest neighbours of all points;
(b)-(e) use a uniform subsample of m = 2,000 points (seed 0), the subsample on which G is estimated; (f) the (L,G) plane of
the seven maps (L exact, G mean over ten subsamples, results/large/mnist/scores.json).
usage: python3 fig_concept_big.py <concept npz> <large dir> <fig dir> <checks file>"""
import sys, json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from scipy.stats import norm
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
sys.path.insert(0, __file__.rsplit("/", 1)[0]); from lgcore import ranks, profile
Z, LD, F, CK = sys.argv[1:5]
plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False})
RED, GREEN = "#c0392b", "#1a9850"; cols = {"PCA": "0.45", "t-SNE": RED}
def tag(ax, s): ax.set_title(s, loc="left", fontsize=7.3)
z = np.load(Z); y, knn, Yt_all, Yp_all = z["y"], z["knn"], z["t_SNE"], z["PCA"]; S = z["S"]; XS = z["XS"].astype(np.float64)
check = []
# (a) local overlap at k = 10 on all 70,000 points
NY = NearestNeighbors(n_neighbors=11).fit(Yt_all).kneighbors(Yt_all, return_distance=False)[:, 1:]
ov = np.array([len(set(a) & set(b)) for a, b in zip(knn, NY)])
c6 = np.where(ov == 6)[0]; i = int(c6[np.argsort(np.abs(Yt_all[c6] - np.median(Yt_all, 0)).sum(1))[len(c6) // 3]])
# (b)-(e) on the subsample
DX = cdist(XS, XS, "sqeuclidean"); Yt, Yp = Yt_all[S].astype(np.float64), Yp_all[S].astype(np.float64)
DT, DP = cdist(Yt, Yt, "sqeuclidean"), cdist(Yp, Yp, "sqeuclidean")
RX = ranks(DX); K, _, nut = profile(RX, ranks(DT)); _, _, nup = profile(RX, ranks(DP))
m = len(S); N1 = m - 1; f = K / m; band = (f >= 0.3) & (f <= 0.5); sel = f <= 0.5
dlp_p = (nup * (N1 - K) + K) / N1; dlp_t = (nut * (N1 - K) + K) / N1
fig, axs = plt.subplots(2, 3, figsize=(7.0, 5.0), gridspec_kw=dict(wspace=0.42, hspace=0.55))
ax = axs[0, 0]; pal = plt.get_cmap("tab10")
ax.scatter(Yt_all[:, 0], Yt_all[:, 1], s=0.15, c=[pal(v) for v in y], alpha=0.15, linewidths=0, rasterized=True)
both = np.intersect1d(knn[i], NY[i]); onlyX = np.setdiff1d(knn[i], NY[i])
# zoom inset around point i so that the neighbours are visible
dk = np.sqrt(((Yt_all[both] - Yt_all[i]) ** 2).sum(1)).max(); r = max(dk * 2.5, 1.0)
ins = ax.inset_axes([0.58, 0.58, 0.42, 0.42]); ins.set_xticks([]); ins.set_yticks([])
win = (np.abs(Yt_all - Yt_all[i]) < r).all(1)
ins.scatter(Yt_all[win, 0], Yt_all[win, 1], s=1.5, c=[pal(v) for v in y[win]], alpha=0.35, linewidths=0, rasterized=True)
for j in knn[i]: ins.plot([Yt_all[i, 0], Yt_all[j, 0]], [Yt_all[i, 1], Yt_all[j, 1]], color=RED if j in both else "#555555", lw=0.5)
ins.scatter(*Yt_all[both].T, s=12, facecolors="none", edgecolors=RED, linewidths=0.9, label="kept")
ins.scatter(*Yt_all[onlyX].T, s=14, marker="x", color="#333333", linewidths=0.9, label="lost")
ins.scatter(*Yt_all[i], s=36, marker="*", color="k", zorder=5, label="point $i$")
ins.set_xlim(Yt_all[i, 0] - r, Yt_all[i, 0] + r); ins.set_ylim(Yt_all[i, 1] - r, Yt_all[i, 1] + r)
ax.indicate_inset_zoom(ins, edgecolor="0.3", lw=0.6)
ax.set_xticks([]); ax.set_yticks([]); h, l = ins.get_legend_handles_labels(); ax.legend(h, l, loc="lower left", frameon=False, fontsize=6, handletextpad=0.2, borderaxespad=0.1)
tag(ax, f"(a) $k=10$: point $i$ keeps {ov[i]} of 10")
check += [("Fig1a overlap of point i (all 70,000 points)", int(ov[i])), ("Fig1a mean overlap/10 over all points (= DLP_10 of t-SNE, exact)", round(ov.mean() / 10, 4))]
ax = axs[0, 1]
ax.plot(f, dlp_p, color=cols["PCA"], lw=1.2, label="PCA"); ax.plot(f, dlp_t, color=cols["t-SNE"], lw=1.2, label="t-SNE")
ax.plot(f, K / N1, color="k", lw=0.8, ls="--", label="chance $k/(m-1)$")
ax.axvline(10 / m, color="k", lw=0.5, ls=":"); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
ax.set_xlabel("$k/m$"); ax.set_ylabel(r"$\mathrm{DLP}_k$"); ax.legend(frameon=False, fontsize=6, loc="lower right")
dl10 = dlp_t[K == 10][0]; ax.annotate(f"{dl10:.2f} at $k=10$", (10 / m, dl10), xytext=(0.12, 0.88), fontsize=6.3, arrowprops=dict(arrowstyle="-", lw=0.5))
tag(ax, r"(b) $\mathrm{DLP}_k$ rises to 1 by chance"); check.append(("Fig1b t-SNE DLP_10 on the 2,000 subsample", round(dl10, 3)))
ax = axs[0, 2]
ax.plot(f[sel], nup[sel], color=cols["PCA"], lw=1.2); ax.plot(f[sel], nut[sel], color=cols["t-SNE"], lw=1.2)
ax.axvspan(0.3, 0.5, color="#eeeeee", zorder=0); ax.axvline(10 / m, color="k", lw=0.5, ls=":"); ax.axhline(0, color="k", lw=0.4)
Lp, Lt = nup[K == 10][0], nut[K == 10][0]; Gp, Gt = nup[band].mean(), nut[band].mean()
ax.plot([10 / m] * 2, [Lp, Lt], "o", ms=3.2, color="k", zorder=5)
ax.text(10 / m * 1.4, Lt + 0.03, f"$\\nu_{{10}}$={Lt:.2f}", fontsize=6.3, va="bottom"); ax.text(10 / m * 1.4, Lp + 0.03, f"$\\nu_{{10}}$={Lp:.2f}", fontsize=6.3, va="bottom")
ax.hlines([Gp, Gt], 0.3, 0.5, colors=[cols["PCA"], cols["t-SNE"]], lw=2.4)
ax.text(0.39, Gp + 0.12, f"$G$={Gp:.2f}", fontsize=6.3, ha="center", color=cols["PCA"]); ax.text(0.39, Gt - 0.13, f"$G$={Gt:.2f}", fontsize=6.3, ha="center", color=cols["t-SNE"])
sx = f[np.argmax(nup >= nut)]; ax.axvline(sx, color=GREEN, lw=0.9); ax.text(sx * 0.85, 0.93, f"crossover\n$s^\\times$={sx:.2f}", color=GREEN, fontsize=6.5, ha="right", va="top")
ax.set_xscale("log"); ax.set_ylim(-0.05, 1.0); ax.set_xlabel("$k/m$"); ax.set_ylabel(r"$\nu_k$"); ax.xaxis.set_minor_formatter(NullFormatter())
tag(ax, r"(c) $\nu_k$: $G$ on $0.3$–$0.5$")
check += [("Fig1c subsample PCA nu_10 / G", round(Lp, 3), round(Gp, 3)), ("Fig1c subsample t-SNE nu_10 / G", round(Lt, 3), round(Gt, 3)), ("Fig1c crossover s_x (subsample)", round(float(sx), 3))]
# (d) near and far halves of one subsample point
ii = int(np.argmin(np.abs(np.median(DT, 1) - np.median(DT))))
ax = axs[1, 0]
rx = np.argsort(np.argsort(DX[ii])) + 1; ry = np.argsort(np.argsort(DT[ii])) + 1; mask = np.arange(m) != ii
ks = N1 / 2; nx, ny = rx[mask] <= ks, ry[mask] <= ks; agree = float(np.mean(nx == ny))
ax.fill_between([0, ks], 0, ks, color="#fde0dd", zorder=0); ax.fill_between([ks, m], ks, m, color="#deebf7", zorder=0)
ax.scatter(rx[mask], ry[mask], s=0.8, c=np.where(nx == ny, "#444444", "#bbbbbb"), linewidths=0, rasterized=True)
ax.axvline(ks, color="k", lw=0.6, ls="--"); ax.axhline(ks, color="k", lw=0.6, ls="--"); ax.set_xlim(0, m); ax.set_ylim(0, m)
ax.set_xlabel("rank in the data, $r^X_i(j)$"); ax.set_ylabel("rank in the map, $r^Y_i(j)$")
ax.text(0.97, 0.03, f"$h(i)$={agree:.2f}\n$2h(i)-1$={2 * agree - 1:.2f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.3, bbox=dict(fc="white", ec="none", alpha=0.85))
tag(ax, r"(d) Near and far halves, $k^\star$")
ax = axs[1, 1]; Ai, pred = [], []
for q in np.random.default_rng(0).choice(m, 400, replace=False):
    mm = np.arange(m) != q; a_ = np.argsort(np.argsort(DX[q][mm])); b_ = np.argsort(np.argsort(DT[q][mm]))
    Ai.append(np.mean((a_ < ks) == (b_ < ks)))
    za, zb = norm.ppf((a_ + 0.5) / N1), norm.ppf((b_ + 0.5) / N1); pred.append(2 / np.pi * np.arcsin(np.corrcoef(za, zb)[0, 1]))
Ai, pred = 2 * np.array(Ai) - 1, np.array(pred)
ax.scatter(pred, Ai, s=3, color=RED, alpha=0.6, linewidths=0); lim = [min(pred.min(), Ai.min()) - 0.05, max(pred.max(), Ai.max()) + 0.05]
ax.plot(lim, lim, "k:", lw=0.7); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel(r"$\frac{2}{\pi}\arcsin\gamma(i)$, $\gamma(i)$: normal-scores correlation"); ax.set_ylabel(r"$2h(i)-1$ (halves agreement)")
ci = np.corrcoef(pred, Ai)[0, 1]; tag(ax, f"(e) Halves vs. rank corr., $r$={ci:.2f}")
check.append(("Fig1e Pearson r, mean pred, mean 2A-1 (400 points, t-SNE)", round(ci, 3), round(pred.mean(), 3), round(Ai.mean(), 3)))
# (f) (L,G) plane of the seven maps, exact L and subsample G
ax = axs[1, 2]; sc = json.load(open(f"{LD}/mnist/scores.json"))
C7 = {"PCA": "0.45", "NPCA": "black", "t-SNE": RED, "t-SNE-200": "#e6a0a0", "SR-tSNE2": "#8b0000", "UMAP": "#2c7fb8", "PaCMAP": "#41b6c4"}
OFF = {"PCA": (3, 2), "NPCA": (3, -7), "t-SNE": (-12, -9), "t-SNE-200": (3, 3), "SR-tSNE2": (3, 2), "UMAP": (3, -2), "PaCMAP": (-14, 5)}
for mt, r_ in sc.items():
    ax.scatter(r_["L"], r_["G"], s=16, color=C7[mt], zorder=4, edgecolors="black" if mt in ("NPCA", "SR-tSNE2") else "white", linewidths=0.5)
    ax.annotate(mt.replace("t-SNE-200", "t-SNE$_{200}$"), (r_["L"], r_["G"]), fontsize=5.4, xytext=OFF[mt], textcoords="offset points")
a, b = sc["t-SNE"], sc["t-SNE-200"]; eta = -(b["G"] - a["G"]) / (b["L"] - a["L"])
ax.annotate("", xy=(b["L"], b["G"]), xytext=(a["L"], a["G"]), arrowprops=dict(arrowstyle="->", color=RED, lw=1.1))
ax.text(0.30, 0.28, f"perplexity 30→200: $\\eta$={eta:.2f}", transform=ax.transAxes, color=RED, fontsize=6.3, va="top")
ax.set_xlabel("$L$"); ax.set_ylabel("$G$"); tag(ax, r"(f) $(L,G)$ plane, $n=70{,}000$")
check += [("Fig1f " + mt + " L/G", round(r_["L"], 3), round(r_["G"], 3)) for mt, r_ in sc.items()] + [("Fig1f eta perplexity 30->200", round(eta, 3), "dL", round(b["L"] - a["L"], 3), "dG", round(b["G"] - a["G"], 3))]
for e in ("pdf", "png"): fig.savefig(f"{F}/fig_concept.{e}", bbox_inches="tight", dpi=200)
with open(CK, "a") as fh:
    fh.write("== fig_concept (MNIST 70,000; panels b-e on a uniform subsample of 2,000, seed 0)\n")
    for c in check: fh.write("  " + " ".join(map(str, c)) + "\n")
print(*check, sep="\n")
