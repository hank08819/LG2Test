"""The LG reversal test (Section lgtest) on every dataset: A in {t-SNE, SR-tSNE2, UMAP} against B = PCA.
(a) the declaration p-value max(p_L, p_G); (b) the two mean differences with 95% jackknife intervals; (c) crossover scale with its
jackknife interval; (d) calibration: one-sided rejection rate at 0.05 under a centered null with the data's block dependence.
usage: python3 fig_lgtest.py <lgtest json> <fig dir>"""
import sys, json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import t as tdist
T = json.load(open(sys.argv[1])); F = sys.argv[2]; DS = list(T); DN = [{"Pathology, 7,180": "Pathology-7K", "Supply chain, 5,459": "Online Retail", "8-K, 5,000": "8-K"}.get(d, d) for d in DS]; AS = ["t-SNE", "SR-tSNE2", "UMAP"]
COL = {"t-SNE": "#e08a8a", "SR-tSNE2": "#8b0000", "UMAP": "#2c7fb8"}
plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False})
fig = plt.figure(figsize=(7.2, 5.0)); gs = fig.add_gridspec(2, 3, width_ratios=[0.9, 1.25, 1.0], hspace=0.62, wspace=0.55)
a = fig.add_subplot(gs[:, 0]); P = np.array([[T[d][A]["p_reversal"] for A in AS] for d in DS]); M = -np.log10(np.clip(P, 1e-4, 1))
a.imshow(M, cmap="Greens", vmin=0, vmax=4, aspect="auto")
for (i, j), p in np.ndenumerate(P):
    rev = T[DS[i]][AS[j]]["reversal"]; s = ("<.001" if p < 0.001 else f"{p:.3f}".replace("0.", ".")) if p < 0.995 else ">.99"
    a.text(j, i, ("✓ " if rev else "") + s, ha="center", va="center", fontsize=5.6, color="white" if M[i, j] > 2.2 else "black", fontweight="bold" if rev else "normal")
a.set_xticks(range(3)); a.set_xticklabels([f"{A}\nvs PCA" for A in AS], fontsize=6); a.set_yticks(range(len(DS))); a.set_yticklabels(DN, fontsize=6.5)
a.tick_params(length=0); [s.set_visible(False) for s in a.spines.values()]
a.set_title("(e) Reversal p-value\n(✓: declared at 0.05)", loc="left", fontsize=7.5)
q = tdist.ppf(0.975, 19)
for k, (key, se, lab) in enumerate([("dL", "seL", r"$L^A-L^{\rm PCA}$"), ("dG", "seG", r"$G^{\rm PCA}-G^A$")]):
    a = fig.add_subplot(gs[k, 1])
    for j, A in enumerate(AS):
        y = np.arange(len(DS)) + (j - 1) * 0.25; v = np.array([T[d][A][key] for d in DS]); e = q * np.array([T[d][A][se] for d in DS])
        a.errorbar(v, y, xerr=e, fmt="o", ms=2.5, color=COL[A], lw=0.8, label=f"{A} vs PCA")
    a.axvline(0, color="black", lw=0.6); a.set_yticks(range(len(DS))); a.set_yticklabels(DN, fontsize=5.5); a.invert_yaxis()
    a.set_xlabel(lab + " (95% interval)"); a.set_title(f"({'fg'[k]}) {'Local' if k == 0 else 'Global'} difference", loc="left", fontsize=7.5)
    if k == 0: a.legend(frameon=False, fontsize=5.5, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, handletextpad=0.1, columnspacing=0.6)
a = fig.add_subplot(gs[0, 2])
for j, A in enumerate(["t-SNE", "SR-tSNE2"]):
    for i, d in enumerate(DS):
        r = T[d][A]; y = i + (j - 0.5) * 0.3
        if r["sx"] is not None and r["sx_ci"] is not None: a.errorbar(r["sx"], y, xerr=[[r["sx"] - r["sx_ci"][0]], [r["sx_ci"][1] - r["sx"]]], fmt="o", ms=2.5, color=COL[A], lw=0.8)
        elif j == 0: a.text(0.97, i, "none", fontsize=5.5, color="black", va="center", ha="right")
a.set_yticks(range(len(DS))); a.set_yticklabels(DN, fontsize=5.5); a.invert_yaxis(); a.set_xlim(0, 1)
a.set_xlabel(r"crossover $s^\times$ against PCA"); a.set_title("(h) Crossover with 95% interval", loc="left", fontsize=7.5)
a = fig.add_subplot(gs[1, 2]); c = np.array([[T[d][A][k] for A in AS for k in ("calL", "calG")] for d in DS]).ravel()
a.hist(c, bins=np.arange(0, 0.105, 0.0075), color="0.5", edgecolor="white"); a.axvline(0.05, color="#8b0000", lw=1)
a.set(xlabel="rejection rate under the null", ylabel="tests"); a.yaxis.set_major_locator(plt.MaxNLocator(integer=True)); a.set_title("(i) Calibration at level 0.05", loc="left", fontsize=7.5)
print("CHECK calibration n %d mean %.4f min %.4f max %.4f" % (len(c), c.mean(), c.min(), c.max()))
fig.savefig(f"{F}/fig_lgtest.pdf", bbox_inches="tight"); fig.savefig(f"{F}/fig_lgtest.png", dpi=200, bbox_inches="tight")
