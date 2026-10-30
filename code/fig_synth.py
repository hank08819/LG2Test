"""The three synthetic manifolds (Swiss roll, S-curve, Tree), whose reference is the geodesic distance.
Row 1: profiles nu_k of eleven methods (seed 0). Row 2: G against L (means over seeds). Row 3: maps of the Swiss roll.
usage: python3 fig_synth.py <synth dir> <fig dir> <tex dir>"""
import sys, json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D, F, T = sys.argv[1:4]
DS = [("Swiss_roll", "Swiss roll"), ("S-curve", "S-curve"), ("Tree", "Tree")]
ORD = ["PCA", "NPCA", "KPCA", "LLE", "Isomap", "t-SNE (PCA init)", "SR-tSNE2", "UMAP", "PHATE", "PaCMAP", "TriMAP"]
LAB = {"t-SNE (PCA init)": "t-SNE"}
COL = {"PCA": "0.45", "NPCA": "black", "KPCA": "#b07aa1", "LLE": "#9c755f", "Isomap": "#59a14f", "t-SNE (PCA init)": "#e15759",
       "SR-tSNE2": "#8b0000", "UMAP": "#2c7fb8", "PHATE": "#7b4fa0", "PaCMAP": "#17becf", "TriMAP": "#f28e2b"}
def load(d):
    P = json.load(open(f"{D}/paper_{d}.json")); O = json.load(open(f"{D}/own_{d}.json")); n = P["n"]; M = {}
    for m, runs in list(P["methods"].items()) + [(k, v) for k, v in O["methods"].items() if k in ("NPCA", "SR-tSNE2")]:
        if m not in ORD: continue
        if "profile_K" in runs[0]: K, nu = np.array(runs[0]["profile_K"]), np.array(runs[0]["profile_nu"])
        else: nu = np.load(f"{D}/nu_{d}_{m}.npy"); K = np.arange(1, len(nu) + 1)
        M[m] = dict(L=np.mean([r["L"] for r in runs]), G=np.mean([r["G"] for r in runs]), K=K, nu=nu)
    return n, M
BLUE, RED, PGREY = "#1f6fd1", "#c0392b", "0.25"
fig = plt.figure(figsize=(7.2, 6.6)); gs = fig.add_gridspec(3, 15, hspace=0.75, wspace=1.4); rows = []; L = "abcdefghijk"
def clean(a):
    for sd_ in ("top", "right"): a.spines[sd_].set_visible(False)
    a.tick_params(labelsize=6)
for j, (d, nm) in enumerate(DS):
    n, M = load(d); rows.append((nm, M)); bG = max(M, key=lambda m: M[m]["G"]); bL = max(M, key=lambda m: M[m]["L"])
    a = fig.add_subplot(gs[0, 5 * j:5 * j + 5]); clean(a)
    for m in M:
        if m not in (bG, bL, "PCA"): a.plot(M[m]["K"] / n, M[m]["nu"], color="0.8", lw=0.7, zorder=1)
    if "PCA" not in (bG, bL): a.plot(M["PCA"]["K"] / n, M["PCA"]["nu"], color=PGREY, lw=1.3, ls="--", zorder=2)
    a.plot(M[bL]["K"] / n, M[bL]["nu"], color=RED, lw=1.8, zorder=3); a.plot(M[bG]["K"] / n, M[bG]["nu"], color=BLUE, lw=1.8, zorder=4)
    a.axvspan(0.3, 0.5, color="#dbe8f7", zorder=0); a.set_xscale("log"); a.set_xlabel("$k/n$", fontsize=7)
    a.set_ylim(top=a.get_ylim()[1] * 1.3)
    a.text(0.03, 0.98, f"best $G$: {LAB.get(bG, bG)} {M[bG]['G']:.2f}", transform=a.transAxes, color=BLUE, fontsize=5.8, va="top", fontweight="bold")
    a.text(0.03, 0.88, f"best $L$: {LAB.get(bL, bL)} {M[bL]['L']:.2f}", transform=a.transAxes, color=RED, fontsize=5.8, va="top", fontweight="bold")
    a.text(0.03, 0.78, f"PCA: $G$ {M['PCA']['G']:.2f}", transform=a.transAxes, color=PGREY, fontsize=5.8, va="top")
    if j == 0: a.set_ylabel(r"$\nu_k$", fontsize=7)
    a.set_title(f"({L[j]}) {nm}", loc="left", fontsize=8)
    b = fig.add_subplot(gs[1, 5 * j:5 * j + 5]); clean(b)
    for m in M:
        c = BLUE if m == bG else RED if m == bL else PGREY if m == "PCA" else "0.7"
        b.scatter(M[m]["L"], M[m]["G"], s=30 if c != "0.7" else 14, color=c, zorder=3, lw=0)
        if c != "0.7":
            off, ha = ((-5, -9), "right") if c == RED else ((5, 3), "left") if c == PGREY else ((-5, 4), "right")
            b.annotate(LAB.get(m, m), (M[m]["L"], M[m]["G"]), xytext=off, textcoords="offset points", fontsize=6, color=c, ha=ha, fontweight="bold")
    b.set_xlabel("$L$", fontsize=7); b.set_title(f"({L[3 + j]}) {nm}: $G$ against $L$", loc="left", fontsize=8)
    if j == 0: b.set_ylabel("$G$", fontsize=7)
from matplotlib.lines import Line2D
fig.legend([Line2D([], [], color=BLUE, lw=1.8), Line2D([], [], color=RED, lw=1.8), Line2D([], [], color=PGREY, lw=1.3, ls="--"), Line2D([], [], color="0.8", lw=0.9)],
           ["highest $G$", "highest $L$", "PCA", "other methods"], loc="upper center", ncol=4, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, 0.995))
z = np.load(f"{D}/maps_Swiss_roll.npz"); y = z["y"]; MP = [("PCA", "PCA"), ("NPCA", "NPCA"), ("t_SNE", "t-SNE"), ("SR_tSNE2", "SR-tSNE2"), ("UMAP", "UMAP")]
SR = dict(rows[0][1])
for i, (k, nm) in enumerate(MP):
    a = fig.add_subplot(gs[2, 3 * i:3 * i + 3]); Y = z[k]; a.scatter(Y[:, 0], Y[:, 1], c=y, cmap="viridis", s=1.2, lw=0)
    key = {"t-SNE": "t-SNE (PCA init)"}.get(nm, nm); a.set_xticks([]); a.set_yticks([])
    a.set_title(f"({L[6 + i]}) {nm}\n$L$={SR[key]['L']:.2f}, $G$={SR[key]['G']:.2f}", loc="left", fontsize=6.5)
    [a.spines[q].set_color("0.6") for q in a.spines]
fig.savefig(f"{F}/fig_synth.pdf", bbox_inches="tight"); fig.savefig(f"{F}/fig_synth.png", dpi=200, bbox_inches="tight")
with open(f"{T}/synth_checks.txt", "w") as fh:
    for nm, M in rows:
        for m in ORD:
            if m in M: fh.write(f"{nm} {m} L {M[m]['L']:.3f} G {M[m]['G']:.3f}\n")
print(open(f"{T}/synth_checks.txt").read())
