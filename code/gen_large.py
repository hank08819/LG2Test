"""V7 large-data tables, figures and number checks, all from results/large. Missing entries render as ---.
usage: python gen_large.py <results/large> <figs dir> <tex dir>"""
import sys, os, json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R, F, T = sys.argv[1:4]
ST = [("path100k", "Pathology-100K", "Pathology", "Phikon"), ("seaad1m", "SEA-AD (single cell)", "Single cell", "50 PCs"),
      ("instacart", "Instacart (supply chain)", "Supply chain", "purchases"), ("marco1m", "MS MARCO (agent memory)", "Agent memory", "BGE-base"),
      ("mnist", "MNIST", "Vision", "pixels"), ("cifar10", "CIFAR-10", "Vision", "ResNet-18"), ("fmnist", "Fashion-MNIST", "Vision", "pixels"),
      ("speech", "Speech Commands", "Audio", "wav2vec 2.0"), ("renal", "Renal-Cortex", "Single cell", "50 PCs"), ("brain", "Brain-Cortex", "Single cell", "50 PCs"),
      ("covertype", "Covertype", "Remote sensing", "attributes"), ("agnews", "AG News", "Text", "BERT")]
NS = {"mnist": (70000, 784), "cifar10": (60000, 512), "fmnist": (70000, 784), "speech": (59387, 768), "renal": (50027, 50), "brain": (50212, 50),
      "covertype": (59999, 54), "agnews": (60000, 768), "path100k": (100000, 768), "seaad1m": (1378211, 50), "instacart": (195827, 2000), "marco1m": (1000000, 768)}
MS = ["PCA", "NPCA", "t-SNE", "t-SNE-200", "SR-tSNE2", "UMAP", "PaCMAP"]
TA = [("t-SNE", "t-SNE, perplexity 30"), ("t-SNE-200", "t-SNE, perplexity 200"), ("SR-tSNE2", "SR-tSNE2"), ("UMAP", "UMAP")]
def J(p): return json.load(open(p)) if os.path.exists(p) else {}
S = {s: J(f"{R}/{s}/scores.json") for s, *_ in ST}; TT = {s: J(f"{R}/{s}/lgtest.json") for s, *_ in ST}
chk = []; f3 = lambda v: "---" if v is None else f"{v:.3f}"

# dataset table and L/G table
rows = []
for s, nm, dom, rep in ST:
    n, d = NS[s]; rows.append(f"{nm} & {dom} & {n:,} & {d:,} & {rep} \\\\" + ("\n\\midrule" if s == "marco1m" else ""))
open(f"{T}/tab_large_data_rows.tex", "w").write("\n".join(rows) + "\n")
rows = []
for s, nm, *_ in ST:
    r = S[s]; L = {m: r.get(m, {}).get("L") for m in MS}; G = {m: r.get(m, {}).get("G") for m in MS}
    bl = max((v, m) for m, v in L.items() if v is not None)[1]; bg = max((v, m) for m, v in G.items() if v is not None)[1]
    cell = lambda m, X, b: "---" if X[m] is None else (r"\textbf{%s}" % f3(X[m]) if m == b else f3(X[m]))
    rows.append(nm + " & " + " & ".join(cell(m, L, bl) for m in MS) + " & " + " & ".join(cell(m, G, bg) for m in MS) + r" \\" + ("\n\\midrule" if s == "marco1m" else ""))
    chk.append(f"{s}: best L {bl} {L[bl]:.3f}; best G {bg} {G[bg]:.3f}; PCA G {f3(G['PCA'])}")
open(f"{T}/tab_large_LG_rows.tex", "w").write("\n".join(rows) + "\n")

# figure: reversal test at scale
names = [nm for _, nm, *_ in ST]; y = np.arange(len(ST))[::-1]
fig, ax = plt.subplots(1, 4, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1.25, 1, 1, 1]})
P = np.full((len(ST), len(TA)), np.nan)
for i, (s, *_) in enumerate(ST):
    for j, (a, _) in enumerate(TA):
        if a in TT[s]: P[i, j] = TT[s][a]["p_reversal"]
im = ax[0].imshow(-np.log10(np.clip(P, 1e-12, 1)), cmap="Greens", vmin=0, vmax=6, aspect="auto")
for i in range(len(ST)):
    for j in range(len(TA)):
        p = P[i, j]; txt = "---" if np.isnan(p) else (("✓ " if p < 0.05 else "") + ("<.001" if p < 0.001 else f"{p:.3f}"))
        ax[0].text(j, i, txt, ha="center", va="center", fontsize=7, color="white" if (not np.isnan(p) and p < 1e-3) else "black")
ax[0].set_xticks(range(len(TA))); ax[0].set_xticklabels(["t-SNE", "t-SNE$_{200}$", "SR-tSNE2", "UMAP"], fontsize=8)
ax[0].set_yticks(range(len(ST))); ax[0].set_yticklabels(names, fontsize=8); ax[0].set_title("(a) p of the declaration (✓: reversal at 0.05)", fontsize=9)
cols = {"t-SNE": "#d62728", "t-SNE-200": "#ff7f0e", "SR-tSNE2": "#8c1c13", "UMAP": "#1f77b4"}; off = dict(zip(cols, (0.27, 0.09, -0.09, -0.27)))
tcrit = 2.093
for k, (key, se, lab) in enumerate((("dL", "seL", r"(b) $L^A-L^{\rm PCA}$"), ("dG", "seG", r"(c) $G^{\rm PCA}-G^A$"))):
    a = ax[k + 1]
    for i, (s, *_) in enumerate(ST):
        for m, c in cols.items():
            t = TT[s].get(m)
            if t: a.errorbar(t[key], y[i] + off[m], xerr=tcrit * t[se], fmt="o", ms=3, color=c, lw=1, label=m if i == 0 else None)
    a.axvline(0, color="k", lw=0.6); a.set_yticks(y); a.set_yticklabels(names if k == 0 else [], fontsize=8); a.set_title(lab + " (95% interval)", fontsize=9)
from matplotlib.lines import Line2D
fig.legend(handles=[Line2D([], [], marker="o", ls="", ms=4, color=c) for c in cols.values()], labels=["t-SNE", "t-SNE$_{200}$", "SR-tSNE2", "UMAP"], fontsize=8, frameon=False, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04))
a = ax[3]
for i, (s, *_) in enumerate(ST):
    for m, c in cols.items():
        t = TT[s].get(m)
        if t and t.get("crossover") is not None and t["crossover"] < 0.99:
            ci = t.get("crossover_ci"); lo, hi = (ci if ci else (t["crossover"], t["crossover"]))
            a.errorbar(t["crossover"], y[i] + off[m], xerr=[[t["crossover"] - max(lo, 0)], [min(hi, 1) - t["crossover"]]], fmt="o", ms=3, color=c, lw=1)
a.set_xlim(0, 1); a.set_yticks(y); a.set_yticklabels([]); a.set_title(r"(d) crossover $s^\times$ (95% interval)", fontsize=9); a.set_xlabel("k/n")
for a in ax[1:]: a.set_ylim(-0.6, len(ST) - 0.4)
APPC = {"path100k": "#ad1457", "seaad1m": "#2e7d32", "instacart": "#ef6c00", "marco1m": "#5e35b1"}
ax[0].axhline(3.5, color="k", lw=1.2)
for a in ax[1:]: a.axhline((y[3] + y[4]) / 2, color="0.5", lw=0.8, ls="--")
for a in (ax[0], ax[1]):
    for tl, (s_, *_) in zip(a.get_yticklabels(), ST):
        if s_ in APPC: tl.set_color(APPC[s_]); tl.set_fontweight("bold")
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(f"{F}/fig_large.pdf", bbox_inches="tight"); fig.savefig(f"{F}/fig_large.png", dpi=150, bbox_inches="tight"); plt.close(fig)
for a_, lab in TA:
    dec = [nm for s, nm, *_ in ST if TT[s].get(a_, {}).get("reversal")]; tot = [nm for s, nm, *_ in ST if a_ in TT[s]]
    chk.append(f"reversal {a_}: {len(dec)} of {len(tot)}: {dec}")
sm = [nm for s, nm, *_ in ST if "t-SNE" in TT[s] and "t-SNE-200" in TT[s] and TT[s]["t-SNE-200"]["dG"] < TT[s]["t-SNE"]["dG"]]
chk.append(f"pp200 smaller dG than pp30 on {len(sm)} of {sum(1 for s,*_ in ST if 't-SNE-200' in TT[s])}: {sm}")

# figure: SR-tSNE2 against t-SNE
done = [(s, nm) for s, nm, *_ in ST if "SR-tSNE2" in S[s] and "t-SNE" in S[s]]
g = [100 * (S[s]["SR-tSNE2"]["odlp"] / S[s]["t-SNE"]["odlp"] - 1) for s, _ in done]
dL = [S[s]["SR-tSNE2"]["L"] - S[s]["t-SNE"]["L"] for s, _ in done]; dGs = [S[s]["SR-tSNE2"]["G"] - S[s]["t-SNE"]["G"] for s, _ in done]
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6)); x = np.arange(len(done)); lab = [nm for _, nm in done]
ax[0].bar(x, g, color="#8c1c13"); ax[0].set_ylabel("gain in O-DLP@10 (%)"); ax[0].set_title("(a) exact-rank neighbors returned, SR-tSNE2 over t-SNE", fontsize=9)
ax[1].bar(x, dL, color="#d62728"); ax[1].set_ylabel(r"$L^{\rm SR}-L^{\rm t\text{-}SNE}$"); ax[1].set_title("(b) change in L", fontsize=9)
ax[2].bar(x, dGs, color="#7f7f7f"); ax[2].set_ylabel(r"$G_{\rm SR}-G_{\rm t\text{-}SNE}$"); ax[2].set_title("(c) change in G", fontsize=9); ax[2].set_ylim(-0.02, 0.02)
for a in ax: a.set_xticks(x); a.set_xticklabels(lab, rotation=45, ha="right", fontsize=8); a.axhline(0, color="k", lw=0.6)
fig.tight_layout(); fig.savefig(f"{F}/fig_srlarge.pdf"); fig.savefig(f"{F}/fig_srlarge.png", dpi=150); plt.close(fig)
chk.append(f"SR-tSNE2 on {len(done)} stores: O-DLP gain {min(g):.0f}% to {max(g):.0f}%; >35% on {sum(v>35 for v in g)}; dL {min(dL):+.3f} to {max(dL):+.3f}; max |dG| {max(abs(v) for v in dGs):.4f}")
for (s, nm), v in zip(done, g): chk.append(f"  {nm}: O-DLP {S[s]['t-SNE']['odlp']:.4f} -> {S[s]['SR-tSNE2']['odlp']:.4f} ({v:+.0f}%), L {S[s]['t-SNE']['L']:.3f} -> {S[s]['SR-tSNE2']['L']:.3f}, G {S[s]['t-SNE']['G']:.3f} -> {S[s]['SR-tSNE2']['G']:.3f}")

# figure: MNIST size ladder, three subsets, two perplexities
lad = {(pp, sd): J(f"{R}/scale_mnist{'' if pp == 30 and sd == 0 else ('_pp200' if sd == 0 else f'_pp{pp}_s{sd}')}.json") for pp in (30, 200) for sd in (0, 1, 2)}
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for k, pp in enumerate((30, 200)):
    for sd, mk in zip((0, 1, 2), ("o", "s", "^")):
        d = lad[(pp, sd)]; ns = sorted(int(v) for v in d)
        if not ns: continue
        dg = [d[str(n)]["test"]["dG"] for n in ns]; rv = [d[str(n)]["test"]["reversal"] for n in ns]
        ax[k].plot(ns, dg, "-", color="#d62728" if pp == 30 else "#ff7f0e", lw=1, alpha=0.6)
        ax[k].scatter(ns, dg, marker=mk, s=36, facecolors=["#d62728" if r and pp == 30 else "#ff7f0e" if r else "white" for r in rv], edgecolors="#d62728" if pp == 30 else "#ff7f0e", label=f"subset {sd}")
    ax[k].axhline(0, color="k", lw=0.6); ax[k].set_xscale("log"); ax[k].set_xlabel("n"); ax[k].set_ylabel(r"$G^{\rm PCA}-G^{\rm t\text{-}SNE}$")
    ax[k].set_title(f"({'ab'[k]}) t-SNE, perplexity {pp}; filled: reversal at 0.05", fontsize=9); ax[k].legend(fontsize=7, frameon=False)
for sd, mk in zip((0, 1, 2), ("o", "s", "^")):
    d = lad[(30, sd)]; ns = sorted(int(v) for v in d)
    if ns: ax[2].plot(ns, [d[str(n)]["PCA"]["G"] for n in ns], marker=mk, color="#1f77b4", lw=1, label=f"subset {sd}")
ax[2].set_xscale("log"); ax[2].set_xlabel("n"); ax[2].set_ylabel("G of PCA fitted on the subset"); ax[2].set_title("(c) PCA on the same subsets", fontsize=9); ax[2].legend(fontsize=7, frameon=False)
for a in ax:
    a.set_xticks([2500, 5000, 10000, 20000, 40000, 70000]); a.set_xticklabels(["2.5k", "5k", "10k", "20k", "40k", "70k"]); a.minorticks_off()
fig.tight_layout(); fig.savefig(f"{F}/fig_scale.pdf"); fig.savefig(f"{F}/fig_scale.png", dpi=150); plt.close(fig)
for pp in (30, 200):
    for n in (2500, 5000, 10000, 20000, 40000, 70000):
        rv = [lad[(pp, sd)].get(str(n), {}).get("test", {}).get("reversal") for sd in (0, 1, 2)]
        pg = [lad[(pp, sd)].get(str(n), {}).get("PCA", {}).get("G") for sd in (0, 1, 2)]
        chk.append(f"ladder pp{pp} n={n}: reversal {sum(bool(v) for v in rv)} of {sum(v is not None for v in rv)}; PCA G {pg}")
open(f"{T}/large_checks.txt", "w").write("\n".join(chk) + "\n"); print("\n".join(chk))
