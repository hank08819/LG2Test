"""Worked example of the LG reversal test on the memory of a retrieval agent, which retrieves in the full embedding and is inspected on a map: one million MS MARCO passages (BGE-base),
SR-tSNE2 against PCA. (a) profiles with the crossover and its 95% jackknife interval, in passages; (b) the two one-sided tests
with 95% block-jackknife intervals; (c) the rule the test returns.
usage: python3 fig_memtest.py <large dir> <fig dir>"""
import sys, json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import t as tdist
LD, F = sys.argv[1:3]; st = "marco1m"; N = 1_000_000
T = json.load(open(f"{LD}/{st}/lgtest.json"))["SR-tSNE2"]; tq = tdist.ppf(0.975, T["n_blocks"] - 1)
plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False})
fig = plt.figure(figsize=(7.2, 2.5)); gs = fig.add_gridspec(1, 3, wspace=0.45, width_ratios=[1.3, 0.85, 1.3])
a = fig.add_subplot(gs[0])
C = {"PCA": "#7f7f7f", "t-SNE": "#e08080", "SR-tSNE2": "#8b0000"}
for m in ("SR-tSNE2", "PCA"):
    nu = np.load(f"{LD}/{st}/nu_{m}.npy").astype(float); s = np.arange(1, len(nu) + 1) / (len(nu) + 2); k = s <= 0.5
    a.plot(s[k], nu[k], color=C[m], lw=1.6 if m != "t-SNE" else 0.8, label=m)
cx, (lo, hi) = T["crossover"], T["crossover_ci"]
a.axvspan(lo, hi, color="#5e35b1", alpha=0.13, lw=0); a.axvline(cx, color="#5e35b1", lw=1)
a.axvspan(0.3, 0.5, color="0.92", zorder=0, lw=0)
a.text(cx / 2, 0.035, "near a query\nuse SR-tSNE2", ha="center", fontsize=6, color="#8b0000", fontweight="bold")
a.text(0.33, 0.035, "topics, consolidation\nuse PCA", ha="center", fontsize=6, color="0.3", fontweight="bold")
a.set_xlim(0, 0.5); a.set_ylim(0, 0.30); a.set_xlabel("$k/n$"); a.set_ylabel(r"$\nu_k$"); a.legend(frameon=False, fontsize=5.8, loc="upper right")
top = a.secondary_xaxis("top", functions=(lambda x: x * N / 1000, lambda y: y * 1000 / N)); top.set_xlabel("passages (thousands)", fontsize=6); top.tick_params(labelsize=5.5)
r = lambda v: f"{round(v * N, -2):,.0f}"; a.set_title(f"(a) crossover at {r(cx)} passages ({r(lo)}–{r(hi)})", loc="left", fontsize=7, pad=4, y=1.17)
a = fig.add_subplot(gs[1])
rows = [("local test\n$L^{\\rm SR}-L^{\\rm PCA}$", T["dL"], T["seL"], T["pL"], "#8b0000"), ("global test\n$G^{\\rm PCA}-G^{\\rm SR}$", T["dG"], T["seG"], T["pG"], "0.35")]
for i, (nm, d, se, p, c) in enumerate(rows):
    a.barh(i, d, 0.5, color=c); a.errorbar(d, i, xerr=tq * se, color="k", lw=0.9, capsize=2)
    e = int(np.floor(np.log10(p))); a.text(d + tq * se + 0.008, i, f"$p={p / 10 ** e:.1f}\\times10^{{{e}}}$", va="center", fontsize=6)
a.set_yticks([0, 1]); a.set_yticklabels([r[0] for r in rows], fontsize=6); a.invert_yaxis(); a.axvline(0, color="k", lw=0.5); a.set_xlim(0, 0.36)
a.set_xlabel("difference, 95% block-jackknife interval"); a.set_title(f"(b) reversal declared\n$p=\\max(p_L,p_G)={T['p_reversal'] * 1e5:.1f}\\times10^{{-5}}$", loc="left", fontsize=7)
a = fig.add_subplot(gs[2]); a.axis("off"); a.set_title("(c) which map to use", loc="left", fontsize=7)
lines = [("SR-tSNE2", "the passages near a query", "#8b0000"), ("SR-tSNE2", "a subtopic of under 79,000 passages", "#8b0000"),
         ("PCA", "which of the eight topics lie far apart", "0.3"), ("PCA", "merge or delete whole topics", "0.3"), ("both", "a group of 79,000 to 222,000 passages", "#5e35b1")]
for i, (m, q, c) in enumerate(lines):
    y = 0.88 - i * 0.19; a.text(0.0, y, m, fontsize=6.3, va="center", color=c, fontweight="bold", transform=a.transAxes)
    a.text(0.30, y, q, fontsize=6.3, va="center", transform=a.transAxes)
    a.plot([0, 1], [y - 0.095, y - 0.095], color="0.85", lw=0.5, transform=a.transAxes)
for e in ("pdf", "png"): fig.savefig(f"{F}/fig_memtest.{e}", bbox_inches="tight", dpi=220)
print("crossover", cx, lo * N, hi * N, "dL", T["dL"], T["seL"], T["pL"], "dG", T["dG"], T["seG"], T["pG"], "p", T["p_reversal"], "tq", tq)
