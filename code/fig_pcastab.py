"""PCA as a reference map on MNIST: G over 40 random subsets of 5,000 points, with shared bins, and PCA G
against the second principal angle between the subset's PCA plane and the full-data plane. usage: python fig_pcastab.py <results dir> <out png>"""
import sys, json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R, out = sys.argv[1], sys.argv[2]
MS = [("PCA", "PCA, fitted on the subset", "#1f77b4"), ("PCA-full", "PCA, full-data plane", "#7f7f7f"), ("t-SNE", "t-SNE, perplexity 30", "#d62728"), ("t-SNE-200", "t-SNE, perplexity 200", "#ff7f0e")]
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
for j, (f, title) in enumerate((("dist_mnist_5k.json", "(d) MNIST, 5,000 points"),)):
    d = json.load(open(f"{R}/{f}")); rs = sorted(d, key=int); allg = [d[r][m]["G"] for r in rs for m, _, _ in MS]
    bins = np.linspace(min(allg) - 0.005, max(allg) + 0.005, 36)
    for m, lab, c in MS:
        v = np.array([d[r][m]["G"] for r in rs]); ax[j].hist(v, bins=bins, color=c, alpha=0.6, label=f"{lab}: {v.mean():.3f} ({v.std():.3f})")
    ax[j].set_xlabel("G"); ax[j].set_ylabel("subsets"); ax[j].set_title(f"{title}, {len(rs)} subsets", fontsize=10); ax[j].legend(fontsize=7, frameon=False, loc="upper left")
d = json.load(open(f"{R}/dist_mnist_5k.json")); rs = sorted(d, key=int)
cos = np.array([d[r]["pca_cos"][1] for r in rs]); g = np.array([d[r]["PCA"]["G"] for r in rs])
ax[1].scatter(cos, g, s=22, color="#1f77b4"); ax[1].axhline(np.mean([d[r]["PCA-full"]["G"] for r in rs]), color="#7f7f7f", ls="--", lw=1, label="full-data plane")
ax[1].axhline(np.mean([d[r]["t-SNE"]["G"] for r in rs]), color="#d62728", ls=":", lw=1, label="t-SNE, perplexity 30")
ax[1].set_xlabel(r"$\cos\theta_2$, subset PCA plane vs full-data plane"); ax[1].set_ylabel("G of PCA fitted on the subset"); ax[1].set_title("(e) MNIST: G follows the second axis", fontsize=10); ax[1].legend(fontsize=7, frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(out, dpi=200); fig.savefig(out.replace(".png", ".pdf")); print("saved", out)
