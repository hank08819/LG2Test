"""Parameter paths on MNIST-60k (the 60,000 training digits): t-SNE perplexity, UMAP n_neighbors, t-SNE exaggeration (openTSNE
FFT) and the number of PCA components before t-SNE; three runs each (t-SNE paths: seed 0 PCA initialization, seeds 1-2 random).
(a) paths in the (L,G) plane, mean +- sd over runs, PCA and NPCA marked; (b) G against energy; (c) seed standard deviation of G
against that of L for every setting. Writes tex/tab_sweep_rows.tex.
usage: python3 fig_tradeoff_big.py <big dir> <fig dir> <tex dir> <checks file>"""
import sys, json, glob, collections, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
B, F, T, CK = sys.argv[1:5]
plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False})
R = collections.defaultdict(list)
for f in glob.glob(f"{B}/sweep_*.json"): r = json.load(open(f)); R[(r["path"], r["value"])].append(r)
MP = json.load(open(f"{B}/maps.json"))
PATHS = [("perplexity", "t-SNE, perplexity", "#c0392b", "o"), ("umap", "UMAP, \\texttt{n\\_neighbors}", "#2c7fb8", "D"),
         ("exaggeration", "t-SNE (FFT), exaggeration", "#d95f0e", "s"), ("pcak", "PCA($k$) + t-SNE, $k$", "0.35", "^")]
LEG = {"perplexity": "t-SNE perplexity", "umap": "UMAP n_neighbors", "exaggeration": "t-SNE exaggeration", "pcak": "PCA($k$) + t-SNE"}
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4), gridspec_kw=dict(wspace=0.45)); check = []; rows = []; S = {}
for p, tl, c, mk in PATHS:
    vs = sorted(v for q, v in R if q == p)
    L = np.array([[x["L"] for x in R[(p, v)]] for v in vs]); G = np.array([[x["G"] for x in R[(p, v)]] for v in vs]); J = np.array([[x["J"] for x in R[(p, v)]] for v in vs])
    S[p] = dict(v=vs, L=L.mean(1), G=G.mean(1), J=J.mean(1), sL=L.std(1, ddof=1), sG=G.std(1, ddof=1))
    a = ax[0]; a.errorbar(S[p]["L"], S[p]["G"], xerr=S[p]["sL"], yerr=S[p]["sG"], fmt="-" + mk, ms=3, lw=1, color=c, capsize=1, elinewidth=0.5, label=LEG[p])
    for i in (0, len(vs) - 1): a.annotate(f"{vs[i]:g}", (S[p]["L"][i], S[p]["G"][i]), fontsize=5, xytext=(3, 2), textcoords="offset points", color=c)
    if p != "pcak": ax[1].plot(S[p]["J"] / 1000, S[p]["G"], "-" + mk, ms=3, lw=1, color=c)
    ax[2].scatter(np.maximum(S[p]["sL"], 1.2e-4), S[p]["sG"], s=12, marker=mk, color=c, zorder=3)
    for i, v in enumerate(vs):
        rows.append(f"{tl if i == 0 else ''} & {v:g} & {S[p]['L'][i]:.3f} & {S[p]['G'][i]:.3f} & {S[p]['sG'][i]:.3f} & {S[p]['J'][i]:,.0f} \\\\")
        check.append((p, v, "L", round(S[p]["L"][i], 4), "G", round(S[p]["G"][i], 4), "sdL", round(S[p]["sL"][i], 4), "sdG", round(S[p]["sG"][i], 4), "J", round(S[p]["J"][i])))
    rows.append("\\addlinespace")
rows = rows[:-1]
for m, mk in (("PCA", "s"), ("NPCA", "s")):
    ax[0].scatter(MP[m]["L"], MP[m]["G"], marker=mk, s=22, color="0.55" if m == "PCA" else "black", zorder=4); ax[0].annotate(m, (MP[m]["L"], MP[m]["G"]), fontsize=5.5, xytext=(4, 2 if m == "PCA" else -7), textcoords="offset points")
ax[0].set(xlabel="$L$", ylabel="$G$"); ax[0].set_title("(a) Parameter paths, $n$ = 60,000", loc="left", fontsize=7)
ax[0].legend(frameon=False, fontsize=5.2, loc="upper center", bbox_to_anchor=(1.75, -0.22), ncol=4, handletextpad=0.3)
ax[1].set(xlabel="energy (kJ, one core at 15 W)", ylabel="$G$"); ax[1].set_title("(b) $G$ against energy", loc="left", fontsize=7)
mx = max(max(S[p]["sL"].max(), S[p]["sG"].max()) for p in S) * 1.1
lo = 1e-4; ax[2].plot([lo, mx], [lo, mx], "k:", lw=0.7); ax[2].plot([lo, mx / 10], [10 * lo, mx], color="0.7", ls="--", lw=0.6); ax[2].text(1.3e-4, 2.2e-3, "×10", fontsize=5.5, color="0.4")
ax[2].set_xscale("log"); ax[2].set_yscale("log"); ax[2].set(xlabel="seed s.d. of $L$", ylabel="seed s.d. of $G$", xlim=(lo, mx), ylim=(lo, mx))
ax[2].set_title("(c) Seeds move $G$, not $L$", loc="left", fontsize=7)
for e in ("pdf", "png"): fig.savefig(f"{F}/fig_tradeoff.{e}", bbox_inches="tight", dpi=220)
open(f"{T}/tab_sweep_rows.tex", "w").write("\n".join(rows) + "\n")
# summary numbers
P = S["perplexity"]; i30, i500, i200 = P["v"].index(30), P["v"].index(500), P["v"].index(200)
for j, nm in ((i500, "500"), (i200, "200")):
    dL, dG = P["L"][j] - P["L"][i30], P["G"][j] - P["G"][i30]
    check.append((f"perplexity 30->{nm}: dL", round(dL, 4), "dG", round(dG, 4), "eta", round(-dG / dL, 3), "energy ratio", round(P["J"][j] / P["J"][i30], 2)))
U = S["umap"]; dom = []
for i, v in enumerate(U["v"]):
    d = [P["v"][j] for j in range(len(P["v"])) if P["L"][j] >= U["L"][i] and P["G"][j] >= U["G"][i]]; dom.append((v, d))
check.append(("UMAP settings dominated by perplexities (L and G both >=)", dom))
E = S["exaggeration"]; check.append(("exaggeration 1->32 dL, dG", round(E["L"][-1] - E["L"][0], 4), round(E["G"][-1] - E["G"][0], 4)))
K = S["pcak"]; check.append(("PCA-k 50->2 L, G", (round(K["L"][-1], 4), round(K["G"][-1], 4)), (round(K["L"][0], 4), round(K["G"][0], 4)), "sdG at k=2", round(K["sG"][0], 4)))
rat = [(p, v, round(S[p]["sG"][i] / max(S[p]["sL"][i], 1e-6), 1)) for p in S for i, v in enumerate(S[p]["v"])]
check.append(("sdG/sdL per setting: n settings with sdG>sdL of", sum(1 for p in S for i in range(len(S[p]["v"])) if S[p]["sG"][i] > S[p]["sL"][i]), sum(len(S[p]["v"]) for p in S)))
check.append(("sdG/sdL at perplexity 30", [r for r in rat if r[0] == "perplexity" and r[1] == 30]))
check.append(("PCA, NPCA (60k) L/G", round(MP["PCA"]["L"], 4), round(MP["PCA"]["G"], 4), round(MP["NPCA"]["L"], 4), round(MP["NPCA"]["G"], 4)))
check.append(("max G over all path settings", max((round(S[p]["G"][i], 4), p, S[p]["v"][i]) for p in S for i in range(len(S[p]["v"])))))
with open(CK, "a") as fh:
    fh.write("== fig_tradeoff + tab_sweep_rows (MNIST-60k paths, 3 runs each)\n")
    for c in check: fh.write("  " + " ".join(map(str, c)) + "\n")
print(*check[-9:], sep="\n")
