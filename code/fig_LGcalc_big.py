"""How L and G are computed, on MNIST (n = 70,000): PCA, NPCA, t-SNE and SR-tSNE2.
The profile nu_k is the mean profile over ten uniform subsamples of m = 2,000 points (nu_*.npy of exp_bigstore.py; its band mean equals the reported G); DLP_k is recovered from it by
inverting the chance correction. L = nu_10 is computed exactly over all 70,000 points (scores.json) and drawn at k/n = 10/70,000.
usage: python3 fig_LGcalc_big.py <large dir> <fig dir> <checks file>"""
import sys, json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
LD, F, CK = sys.argv[1:4]; n = 70000
M = [("PCA", "0.55", "-"), ("NPCA", "black", "--"), ("t-SNE", "#e08a8a", "-"), ("SR-tSNE2", "#8b0000", "-")]
sc = json.load(open(f"{LD}/mnist/scores.json"))
plt.rcParams.update({"font.size": 7.5, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.9)); check = []
YL = {"PCA": 0.06, "NPCA": 0.0, "t-SNE": 0.30, "SR-tSNE2": 0.44}; YG = {"PCA": 0.45, "NPCA": 0.40, "SR-tSNE2": 0.13, "t-SNE": 0.06}; Ltxt = []
for nm, c, ls in M:
    nu = np.load(f"{LD}/mnist/nu_{nm}.npy").astype(np.float64); m = len(nu) + 2; N = m - 1; k = np.arange(1, len(nu) + 1); s = k / m
    dlp = (nu * (N - k) + k) / N; band = (s >= 0.3) & (s <= 0.5); G1 = nu[band].mean(); L = sc[nm]["L"]
    check.append((nm, "exact L", round(L, 4), "G band of mean profile", round(G1, 4), "G ten subsamples", round(sc[nm]["G"], 4), "sub nu_10", round(nu[9], 4)))
    ax[0].plot(s, dlp, color=c, ls=ls, lw=1.2, label=nm)
    ax[1].plot(s, nu, color=c, ls=ls, lw=1.2); ax[1].plot(10 / n, L, "o", color=c, ms=4, zorder=5)
    Ltxt.append(f"{nm} {L:.3f}")
    ax[2].plot(s, nu, color=c, ls=ls, lw=1.2); ax[2].hlines(G1, 0.3, 0.5, color=c, lw=2.2)
    ax[2].annotate(f"{nm} {G1:.3f}", (0.5, G1), xytext=(0.62, YG[nm]), fontsize=6.5, va="center", arrowprops=dict(arrowstyle="-", color=c, lw=0.7), bbox=dict(fc="white", ec="none", pad=0.5))
ax[0].plot([0, 1], [0, 1], "k:", lw=0.8, label="chance $k/(m-1)$")
ax[0].set(xlabel="$k/m$", ylabel=r"$\mathrm{DLP}_k$", ylim=(0, 1.02))
ax[0].set_title("(a) Step 1: the DLP curve\n" + r"$\mathrm{DLP}_k=\frac{1}{nk}\sum_i |N^X_k(i)\cap N^Y_k(i)|$", fontsize=7.5)
ax[0].legend(frameon=False, fontsize=6.5, loc="lower right")
ax[1].set_xscale("log"); ax[1].axvline(10 / n, color="0.6", ls=":", lw=0.8)
ax[1].set(xlabel="$k/n$ (log scale)", ylabel=r"$\nu_k$", xlim=(8e-5, 1))
ax[1].set_title("(b) Step 2: chance correction, $L$\n" + r"$\nu_k=\frac{(n-1)\,\mathrm{DLP}_k-k}{n-1-k},\quad L=\nu_{10}$", fontsize=7.5)
ax[1].text(6e-4, 0.26, "$L$ (dots), exact over\nall 70,000 points:\n" + "\n".join(Ltxt), fontsize=6, va="top", bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.5)); ax[1].set_ylim(-0.03, 0.48); ax[2].set_ylim(-0.03, 0.48)
ax[2].axvspan(0.3, 0.5, color="0.9", zorder=0)
ax[2].set(xlabel="$k/m$", ylabel=r"$\nu_k$", xlim=(0, 1))
ax[2].set_title("(c) Step 3: band average, $G$\n" + r"$G=\frac{1}{|\mathcal{K}|}\sum_{k\in\mathcal{K}}\nu_k,\quad \mathcal{K}:\ 0.3\leq k/n\leq 0.5$", fontsize=7.5)
fig.tight_layout(w_pad=1.2)
for e in ("pdf", "png"): fig.savefig(f"{F}/fig_LGcalc.{e}", dpi=200)
with open(CK, "a") as fh:
    fh.write("== fig_LGcalc (MNIST 70,000; profile = mean over ten subsamples of 2,000; L exact)\n")
    for c in check: fh.write("  " + " ".join(map(str, c)) + "\n")
print(*check, sep="\n")
