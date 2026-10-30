"""V8 big-data figures for validation, robustness and findings, all from results/large and results/big.
usage: python fig_bigval.py <results dir> <figs dir> <tex dir>
Writes fig_profiles, fig_validation, fig_robust, fig_LGplane (pdf+png), tab_validation_rows.tex, tab_sens_rows.tex, bigval_checks.txt."""
import sys, os, re, json, glob, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau
R, F, T = sys.argv[1:4]
ST = [("path100k", "Pathology-100K"), ("seaad1m", "SEA-AD (single cell)"), ("instacart", "Instacart (supply chain)"), ("marco1m", "MS MARCO (agent memory)"),
      ("mnist", "MNIST"), ("cifar10", "CIFAR-10"), ("fmnist", "Fashion-MNIST"), ("speech", "Speech Commands"), ("renal", "Renal-Cortex"),
      ("brain", "Brain-Cortex"), ("covertype", "Covertype"), ("agnews", "AG News")]
APP = {"path100k": "#ad1457", "seaad1m": "#2e7d32", "instacart": "#ef6c00", "marco1m": "#5e35b1"}   # the four applications, same colors in every figure
NAPP = 4
MS = ["PCA", "NPCA", "t-SNE", "t-SNE-200", "SR-tSNE2", "UMAP", "PaCMAP"]
LAB = {"t-SNE-200": "t-SNE$_{200}$"}
COL = {"PCA": "#7f7f7f", "NPCA": "k", "t-SNE": "#c0392b", "t-SNE-200": "#f1948a", "SR-tSNE2": "#6e0f0f", "UMAP": "#1f4e79", "PaCMAP": "#17becf"}
LS = {"NPCA": "--"}
SC = plt.get_cmap("tab20")(np.linspace(0, 1, 20))[[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 5]]
plt.rcParams.update({"font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6})
def tag(ax, s): ax.set_title(s, fontsize=8, loc="left")
def save(fig, name):
    for e in ("pdf", "png"): fig.savefig(os.path.join(F, f"{name}.{e}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
chk = []
S = {s: json.load(open(f"{R}/large/{s}/scores.json")) for s, _ in ST}
NU = {s: {m: np.load(f"{R}/large/{s}/nu_{m}.npy").astype(float) for m in MS} for s, _ in ST}
m_sub = 2000; K = np.arange(1, m_sub - 1); f = K / m_sub
def band(nu, a, b): return float(nu[(f >= a) & (f <= b)].mean())
def auc_ln(nu): return float((nu / K).sum() / (1 / K).sum())
SCORES = [("$L=\\nu_{10}$ (exact)", None), ("Middle band, $0.1\\le k/n\\le0.2$", lambda v: band(v, .1, .2)), ("$\\mathrm{AUC}_{\\ln K}$", auc_ln),
          ("$G$, $0.2\\le k/n\\le0.5$", lambda v: band(v, .2, .5)), ("$G$, $0.3\\le k/n\\le0.5$", lambda v: band(v, .3, .5)),
          ("$G$, $0.4\\le k/n\\le0.5$", lambda v: band(v, .4, .5)), ("Midpoint $\\nu_{k^\\star}$", lambda v: float(v[K == (m_sub - 1) // 2][0]))]
# store embeddings
E = [dict(ds=s, m=m, L=S[s][m]["L"], G=S[s][m]["G"], rho=S[s][m]["rho"], tri=S[s][m]["triplet"], nu=NU[s][m]) for s, _ in ST for m in MS]
SW = [json.load(open(p)) for p in sorted(glob.glob(f"{R}/big/sweep_*.json"))]
P = [dict(ds="mnist60k", m=f"{r['path']}_{r['value']}", L=r["L"], G=r["G"], rho=r["rho"], tri=r["triplet"]) for r in SW]
ALL = E + P; N, NE, NP = len(ALL), len(E), len(P)
sp = lambda a, b: float(spearmanr(a, b)[0])
g, l, rh, tr = (np.array([e[k] for e in ALL]) for k in ("G", "L", "rho", "tri"))
pool = dict(G_rho=sp(g, rh), G_tri=sp(g, tr), L_rho=sp(l, rh), L_tri=sp(l, tr), rho_tri=sp(rh, tr))
chk.append(f"pooled N={N} ({NE} store embeddings: 12 datasets x 7 methods; {NP} MNIST-60k parameter-path runs)")
chk.append("pooled Spearman: G~rho {G_rho:.3f}, G~triplet {G_tri:.3f}, L~rho {L_rho:.3f}, L~triplet {L_tri:.3f}, rho~triplet {rho_tri:.3f}".format(**pool))
# scale ladder over the 84 store embeddings
erh, etr = np.array([e["rho"] for e in E]), np.array([e["tri"] for e in E])
lad = []
for nm, fn in SCORES:
    v = np.array([e["L"] for e in E]) if fn is None else np.array([fn(e["nu"]) for e in E])
    lad.append((nm, sp(v, erh), sp(v, etr), v))
Gnu = lad[4][3]; Gex = np.array([e["G"] for e in E])
chk.append(f"G from saved profile (one 2000-pt subsample) vs G in scores (mean of 10): Spearman {sp(Gnu, Gex):.3f}, max |diff| {np.abs(Gnu - Gex).max():.3f}")
chk.append("scale ladder (84 store embeddings), Spearman with rho / triplet:")
for nm, a, b, _ in lad: chk.append(f"  {nm}: {a:.3f} / {b:.3f}")
with open(f"{T}/tab_validation_rows.tex", "w") as fh:
    best = max(range(len(lad)), key=lambda i: lad[i][1] if i == 4 else -9)
    for i, (nm, a, b, _) in enumerate(lad):
        A_, B_ = (f"\\textbf{{{a:.2f}}}", f"\\textbf{{{b:.2f}}}") if i == 4 else (f"{a:.2f}", f"{b:.2f}")
        fh.write(f"{nm} & {A_} & {B_} \\\\\n")
# within-dataset
W = []
for s, nm in ST:
    es = [e for e in E if e["ds"] == s]; rr = [e["rho"] for e in es]
    W.append((nm, sp([e["G"] for e in es], rr), sp([e["L"] for e in es], rr), sp([e["G"] for e in es], [e["tri"] for e in es])))
W.append(("MNIST parameter paths", sp([e["G"] for e in P], [e["rho"] for e in P]), sp([e["L"] for e in P], [e["rho"] for e in P]), sp([e["G"] for e in P], [e["tri"] for e in P])))
chk.append("within-dataset Spearman (G~rho, L~rho, G~triplet):")
for w in W: chk.append(f"  {w[0]}: {w[1]:.2f}, {w[2]:.2f}, {w[3]:.2f}")
wg = np.array([w[1] for w in W]); wl = np.array([w[2] for w in W])
chk.append(f"within G~rho: mean {wg.mean():.3f}, median {np.median(wg):.3f}, min {wg.min():.2f}; >=0.7 on {(wg >= .7).sum()} of {len(wg)}; within L~rho: mean {wl.mean():.3f}, min {wl.min():.2f}, max {wl.max():.2f}")
chk.append(f"within: G~rho > L~rho on {(wg > wl).sum()} of {len(wg)}")

# ---------------- fig_profiles ----------------
BLUE, RED = "#1f6fd1", "#c0392b"
fig, axs = plt.subplots(3, 4, figsize=(7.2, 5.6), gridspec_kw=dict(wspace=0.32, hspace=0.55))
for i, (s, nm) in enumerate(ST):
    ax = axs.flat[i]; bG = max(MS, key=lambda m: S[s][m]["G"]); bL = max(MS, key=lambda m: S[s][m]["L"])
    for m in MS:
        if m not in (bG, bL): ax.plot(f, NU[s][m], color="0.78", lw=0.7, zorder=1)
    ax.plot(f, NU[s][bL], color=RED, lw=1.8, zorder=3); ax.plot(f, NU[s][bG], color=BLUE, lw=1.8, zorder=4)
    ax.axvspan(.3, .5, color="#dbe8f7", zorder=0); ax.set_xscale("log"); ax.set_xlim(f[0], 1)
    for sd_ in ("top", "right"): ax.spines[sd_].set_visible(False)
    ax.text(0.03, 0.97, f"best $G$: {LAB.get(bG, bG)} {S[s][bG]['G']:.2f}", transform=ax.transAxes, color=BLUE, fontsize=5.6, va="top", fontweight="bold")
    ax.text(0.03, 0.86, f"best $L$: {LAB.get(bL, bL)} {S[s][bL]['L']:.2f}", transform=ax.transAxes, color=RED, fontsize=5.6, va="top", fontweight="bold")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.32); tag(ax, f"({chr(97 + i)}) {nm}")
    if s in APP: ax._left_title.set_color(APP[s]); ax._left_title.set_fontweight("bold"); ax._left_title.set_fontsize(6.2)
    if i % 4 == 0: ax.set_ylabel(r"$\nu_k$")
    if i >= 8: ax.set_xlabel("$k/n$")
from matplotlib.lines import Line2D
fig.legend([Line2D([], [], color=BLUE, lw=1.8), Line2D([], [], color=RED, lw=1.8), Line2D([], [], color="0.78", lw=0.9), plt.Rectangle((0, 0), 1, 1, color="#dbe8f7")],
           ["method with the highest $G$", "method with the highest $L$", "other methods", "band 0.3–0.5 of $k/n$ giving $G$"],
           loc="lower center", ncol=4, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.02))
save(fig, "fig_profiles")
dsr = [(nm, band(NU[s]["SR-tSNE2"], .3, .5) - band(NU[s]["t-SNE"], .3, .5)) for s, nm in ST]
dnp = [(nm, band(NU[s]["NPCA"], .3, .5) - band(NU[s]["PCA"], .3, .5), S[s]["NPCA"]["G"] - S[s]["PCA"]["G"]) for s, nm in ST]
chk.append("profile G-band diff SR-tSNE2 - t-SNE (subsample nu): " + ", ".join(f"{a} {b:+.4f}" for a, b in dsr))
sm = [(nm, float((NU[s]["SR-tSNE2"] - NU[s]["t-SNE"])[f <= .01].min())) for s, nm in ST]
chk.append("SR-tSNE2 - t-SNE on subsample profile, min over k/m<=0.01 (scales above full-data k=10): " + ", ".join(f"{a} {b:+.3f}" for a, b in sm))
chk.append("NPCA - PCA G (scores.json): " + ", ".join(f"{a} {c:+.3f}" for a, b, c in dnp))
# profile shapes: PCA rising? compare nu at k=10 vs band
rise = [nm for s, nm in ST if band(NU[s]["PCA"], .3, .5) > NU[s]["PCA"][9]]
fall = [nm for s, nm in ST if band(NU[s]["t-SNE"], .3, .5) < NU[s]["t-SNE"][9]]
chk.append(f"PCA profile higher in G band than at k=10 (subsample): {len(rise)} of 12 {rise}")
chk.append(f"t-SNE profile lower in G band than at k=10 (subsample): {len(fall)} of 12 {fall}")

# ---------------- fig_validation ----------------
MC = dict(COL); PCOL = "#e67e22"
fig = plt.figure(figsize=(7.2, 5.2)); gs = fig.add_gridspec(2, 6, wspace=1.4, hspace=0.55)
def scat(ax, xk, yk, ttl):
    ax.scatter([e[xk] for e in P], [e[yk] for e in P], s=7, color=PCOL, marker="^", alpha=.7, lw=0, label="MNIST parameter paths")
    for m in MS: ax.scatter([e[xk] for e in E if e["m"] == m], [e[yk] for e in E if e["m"] == m], s=9, color=MC[m], lw=0, label=LAB.get(m, m))
    tag(ax, ttl)
ax = fig.add_subplot(gs[0, 0:2]); scat(ax, "rho", "G", f"(a) $G$, $\\rho_S$={pool['G_rho']:.2f}"); ax.set_xlabel(r"distance $\rho$"); ax.set_ylabel("$G$")
ax = fig.add_subplot(gs[0, 2:4]); scat(ax, "tri", "G", f"(b) $G$, $\\rho_S$={pool['G_tri']:.2f}"); ax.set_xlabel("triplet accuracy"); ax.set_ylabel("$G$")
ax = fig.add_subplot(gs[0, 4:6]); scat(ax, "rho", "L", f"(c) $L$, $\\rho_S$={pool['L_rho']:.2f}"); ax.set_xlabel(r"distance $\rho$"); ax.set_ylabel("$L$")
ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.02), frameon=False, fontsize=5.8, handletextpad=0.1, markerscale=1.3)
gb = gs[1, :].subgridspec(1, 2, wspace=0.75, width_ratios=[1, 1.05])
ax = fig.add_subplot(gb[0]); x = np.arange(len(lad)); w = 0.38
ax.axvspan(2.5, 6.5, color="#dbe9f8", zorder=0, lw=0); ax.text(4.5, 1.02, "global band", ha="center", fontsize=6, color="#1f6fd1")
ax.bar(x - w / 2, [a for _, a, _, _ in lad], w, color="#1f4e79", label=r"distance $\rho$"); ax.bar(x + w / 2, [b for _, _, b, _ in lad], w, color="#e67e22", label="triplet accuracy")
ax.set_xticks(x); ax.set_xticklabels(["$L$", "middle", r"AUC$_{\ln K}$", "$G_{.2-.5}$", "$G_{.3-.5}$", "$G_{.4-.5}$", "mid"], fontsize=6, rotation=30)
ax.set_xlim(-.6, 6.6); ax.set_ylim(-.3, 1.3); ax.axhline(0, color="k", lw=.5); ax.set_ylabel("Spearman, 84 maps"); ax.legend(frameon=False, fontsize=6, loc="upper left", ncol=2); tag(ax, "(d) agreement grows with scale")
ax = fig.add_subplot(gb[1]); y = np.arange(len(W))[::-1]
for yy, g_, l_ in zip(y, wg, wl): ax.plot([l_, g_], [yy, yy], color="0.85", lw=1, zorder=1)
ax.scatter(wl, y, s=14, color="0.55", zorder=3, label="$L$"); ax.scatter(wg, y, s=18, color="#1f6fd1", zorder=4, label="$G$")
ax.set_yticks(y); ax.set_yticklabels([w[0] for w in W], fontsize=5.8); ax.axvline(0, color="k", lw=.5); ax.set_xlim(-1.05, 1.05)
ax.axhline(y[NAPP - 1] - .5, color="0.6", lw=.6, ls="--")
for tl, w_ in zip(ax.get_yticklabels(), W):
    k_ = [s_ for s_, n_ in ST if n_ == w_[0]]
    if k_ and k_[0] in APP: tl.set_color(APP[k_[0]]); tl.set_fontweight("bold")
ax.set_xlabel(r"Spearman with distance $\rho$ in a dataset"); ax.legend(frameon=False, fontsize=6, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, handletextpad=0.1, borderaxespad=0.2); tag(ax, "(e) within each dataset")
save(fig, "fig_validation")

# ---------------- fig_robust ----------------
fig, axs = plt.subplots(1, 4, figsize=(7.4, 2.3), gridspec_kw=dict(wspace=0.62, width_ratios=[1, 1, 1.15, 1]))
alts = [(r"$.2$–$.5$", lad[3][3]), (r"$.4$–$.5$", lad[5][3]), ("mid", lad[6][3]), ("meso", lad[1][3])]
ax = axs[0]; cor, tm, tmin = [], [], []
for nm, v in alts:
    cor.append(sp(v, Gnu)); taus = []
    for s, _ in ST:
        idx = [i for i, e in enumerate(E) if e["ds"] == s]; taus.append(kendalltau(v[idx], Gnu[idx])[0])
    tm.append(np.mean(taus)); tmin.append(np.min(taus))
yl = [r"$0.2$–$0.5$", r"$0.4$–$0.5$", "midpoint", r"middle $0.1$–$0.2$"]; y = np.arange(len(alts))[::-1]
for yy, c, t, mn in zip(y, cor, tm, tmin):
    ax.plot([mn, t], [yy - .14, yy - .14], color="0.6", lw=1.2, solid_capstyle="round"); ax.plot(t, yy - .14, "o", color="0.45", ms=4)
    ax.plot(c, yy + .14, "o", color="#1f6fd1", ms=5); ax.text(c + .06, yy + .14, f"{c:.2f}", ha="left", va="center", fontsize=5.5, color="#1f6fd1")
ax.set_yticks(y); ax.set_yticklabels(yl, fontsize=6); ax.set_xlim(-0.9, 1.35); ax.set_xticks([-0.5, 0, 0.5, 1]); ax.axvline(0, color="0.85", lw=.6, zorder=0); ax.set_ylim(-.6, len(alts) - .4); ax.set_xlabel("agreement with $G$")
ax.plot([], [], "o", color="#1f6fd1", ms=4, label="Spearman, 84 maps"); ax.plot([], [], "o-", color="0.45", ms=3, lw=1.2, label=r"Kendall $\tau$ in a dataset (mean, min)")
ax.legend(frameon=False, fontsize=5, loc="upper center", bbox_to_anchor=(0.45, -0.3), handlelength=1.2); tag(ax, "(a) other bands")
chk.append("band alternatives vs G .3-.5 (84 embeddings, subsample profile): " + "; ".join(f"{a}: rho {c:.3f}, mean tau {t:.3f}, min tau {mn:.3f}" for (a, _), c, t, mn in zip(alts, cor, tm, tmin)))
with open(f"{T}/tab_sens_rows.tex", "w") as fh:
    rows = [("$G$, $0.3\\le k/n\\le0.5$ (default)", Gnu, 1.0, 1.0, 1.0)] + [(SCORES[j][0], lad[j][3], c, t, mn) for j, c, t, mn in zip((3, 5, 6, 1), cor, tm, tmin)]
    for nm, v, c, t, mn in rows: fh.write(f"{nm} & {c:.3f} & {t:.2f} & {mn:.2f} & {sp(v, erh):.2f} & {sp(v, etr):.2f} \\\\\n")
# (b) seeds
grp = {}
for r in SW: grp.setdefault((r["path"], r["value"]), []).append(r)
sdL, sdG, keys = [], [], []
for k_, rs in sorted(grp.items()):
    if len(rs) >= 2: sdL.append(np.std([r["L"] for r in rs], ddof=1)); sdG.append(np.std([r["G"] for r in rs], ddof=1)); keys.append(k_)
sdL, sdG = np.array(sdL), np.array(sdG); PC = {"perplexity": "#c0392b", "umap": "#1f4e79", "exaggeration": "#e67e22", "pcak": "#2e8b57"}
ax = axs[1]; lo, hi = 3e-6, 0.3
ax.fill_between([lo, hi], [lo, hi], [hi, hi], color="#dbe9f8", zorder=0, lw=0)
ax.plot([lo, hi], [lo, hi], "k:", lw=.7, label="$y=x$"); ax.plot([lo, hi / 10], [10 * lo, hi], color="0.5", ls="--", lw=.6, label="$y=10x$")
for p_, c_ in PC.items():
    ii = [i for i, k_ in enumerate(keys) if k_[0] == p_]; ax.scatter(sdL[ii], sdG[ii], s=12, color=c_, lw=0.3, edgecolor="w", label={"umap": "UMAP neighbors", "pcak": "PCA components"}.get(p_, p_), zorder=3)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_xlabel("run s.d. of $L$"); ax.set_ylabel("run s.d. of $G$")
ax.legend(frameon=False, fontsize=5, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, handletextpad=0.1, columnspacing=0.6); tag(ax, "(b) runs, MNIST-60k")
rat = sdG / sdL
chk.append(f"seed sd (3 runs, {len(keys)} settings, MNIST-60k): sd(G)>sd(L) on {(rat > 1).sum()} of {len(rat)}; ratio median {np.median(rat):.1f}, max {rat.max():.0f} at {keys[int(rat.argmax())]}")
for p_ in PC:
    ii = [i for i, k_ in enumerate(keys) if k_[0] == p_]
    chk.append(f"  {p_}: ratio range {rat[ii].min():.1f}-{rat[ii].max():.0f}; max sd L {sdL[ii].max():.4f}, max sd G {sdG[ii].max():.4f}")
pp30 = keys.index(("perplexity", 30)); chk.append(f"  perplexity 30: sd G {sdG[pp30]:.4f}, sd L {sdL[pp30]:.4f}, ratio {rat[pp30]:.0f}")
# (c), (d) subsampling
CV = {s: json.load(open(f"{R}/large/{s}/conv.json")) for s, _ in ST}
ax = axs[2]; devs = {}
ax.axhspan(0, 0.01, color="#dbe9f8", zorder=0, lw=0)
for i, (s, nm) in enumerate(ST):
    for mth, ls_ in (("PCA", "-"), ("t-SNE", "--")):
        d = CV[s][mth]; ms = sorted(int(k_) for k_ in d); top = np.mean(d[str(ms[-1])])
        dv = [abs(np.mean(d[str(m_)]) - top) for m_ in ms]
        if s in APP: ax.plot(ms, dv, color=APP[s], ls=ls_, lw=1.3, marker="o", ms=2.5, zorder=3)
        else: ax.plot(ms, dv, color="0.78", ls=ls_, lw=.6, zorder=2)
        for m_, v_ in zip(ms, dv): devs.setdefault(m_, []).append((v_, nm, mth, float(np.std(d[str(m_)], ddof=1))))
ax.set_xscale("log"); ax.set_xticks([1000, 2000, 4000, 8000]); ax.set_xticklabels(["1k", "2k", "4k", "8k"]); ax.minorticks_off(); ax.set_xlabel("subsample size $m$"); ax.set_ylabel(r"$|\bar G_m-\bar G_{8000}|$")
ax.set_ylim(0, 0.0135); ax.text(7600, 0.0102, "0.01", fontsize=5, ha="right", color="0.35"); tag(ax, "(c) subsample error")
hs = [plt.Line2D([], [], color=APP[s], lw=1.3) for s, _ in ST[:NAPP]] + [plt.Line2D([], [], color="0.78", lw=.8), plt.Line2D([], [], color="k", lw=.8), plt.Line2D([], [], color="k", ls="--", lw=.8)]
ax.legend(hs, [nm for _, nm in ST[:NAPP]] + ["8 benchmarks", "PCA", "t-SNE"], frameon=False, fontsize=5, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, handlelength=1.4, columnspacing=0.7)
for m_ in sorted(devs):
    v = devs[m_]; mx = max(v); chk.append(f"subsample m={m_}: max |mean - mean at largest m| {mx[0]:.4f} ({mx[1]} {mx[2]}); max sd over 5 draws {max(a[3] for a in v):.4f}")
ax = axs[3]; s = "seaad1m"
for mth, ls_ in (("PCA", "-"), ("t-SNE", "--")):
    d = CV[s][mth]; ms = sorted(int(k_) for k_ in d)
    ax.errorbar(ms, [np.mean(d[str(m_)]) for m_ in ms], [np.std(d[str(m_)], ddof=1) for m_ in ms], color=APP[s], ls=ls_, marker="o", ms=2.5, lw=1.1, capsize=1.5)
    ax.axhline(S[s][mth]["G"], color="0.6", ls=":", lw=.7); ax.text(1050, S[s][mth]["G"] + 0.003, mth, fontsize=6, color=APP[s])
ax.set_xscale("log"); ax.set_xticks([1000, 2000, 4000, 8000]); ax.set_xticklabels(["1k", "2k", "4k", "8k"]); ax.minorticks_off(); ax.set_xlabel("subsample size $m$"); ax.set_ylabel("$G$"); tag(ax, "(d) SEA-AD, 1.38M")
chk.append(f"SEA-AD conv: " + "; ".join(f"{mth} " + ", ".join(f"m={m_}: {np.mean(CV[s][mth][str(m_)]):.3f} ({np.std(CV[s][mth][str(m_)], ddof=1):.3f})" for m_ in sorted(int(k_) for k_ in CV[s][mth])) for mth in ("PCA", "t-SNE")))
save(fig, "fig_robust")

# ---------------- fig_LGplane ----------------
Lm = np.array([[S[s][m]["L"] for m in MS] for s, _ in ST]); Gm = np.array([[S[s][m]["G"] for m in MS] for s, _ in ST])
fig = plt.figure(figsize=(7.4, 3.6)); gs = fig.add_gridspec(1, 4, wspace=0.12, width_ratios=[1, 1, 1, 1.25])
def heat(ax, M, ttl, cmap, vmin, vmax, bold=True, ylab=True):
    ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(M.shape[0]):
        b = int(M[i].argmax())
        for j in range(M.shape[1]):
            v = M[i, j]; dark = abs((v - vmin) / (vmax - vmin) - .5) > .3
            ax.text(j, i, f"{v:.2f}".replace("0.", ".").replace("-.", "−."), ha="center", va="center", fontsize=4.6,
                    fontweight="bold" if (bold and j == b) else "normal", color="w" if (cmap != "RdBu" and (v - vmin) / (vmax - vmin) > .6) else "k")
    ax.set_xticks(range(len(MS))); ax.set_xticklabels([LAB.get(m, m) for m in MS], rotation=60, fontsize=5.5)
    ax.set_yticks(range(len(ST))); ax.set_yticklabels([nm for _, nm in ST] if ylab else [], fontsize=6); tag(ax, ttl)
    ax.axhline(NAPP - .5, color="k", lw=1.2)
    for tl, (s_, _) in zip(ax.get_yticklabels(), ST):
        if s_ in APP: tl.set_color(APP[s_]); tl.set_fontweight("bold")
heat(fig.add_subplot(gs[0]), Lm, "(a) $L$", "Reds", 0, Lm.max())
heat(fig.add_subplot(gs[1]), Gm, "(b) $G$", "Blues", 0, Gm.max(), ylab=False)
D = Gm - Lm; v = np.abs(D).max(); heat(fig.add_subplot(gs[2]), D, "(c) $G-L$", "RdBu", -v, v, bold=False, ylab=False)
ax = fig.add_subplot(gs[3]); NE_ = ["t-SNE", "t-SNE-200", "SR-tSNE2", "UMAP", "PaCMAP"]; hgt = .16; CX = {}
for j, m in enumerate(NE_):
    for i, (s, nm) in enumerate(ST):
        a, b = NU[s][m], NU[s]["PCA"]; hit = np.where(b >= a)[0]; sx = float(f[hit[0]]) if len(hit) else None; CX[(nm, m)] = sx
        yy = (len(ST) - 1 - i) + (j - 2) * hgt
        if sx is None: ax.barh(yy, 1, hgt, color="white", edgecolor=COL[m], hatch="////", lw=.4)
        else: ax.barh(yy, sx, hgt, color=COL[m])
ax.set_yticks(range(len(ST))); ax.set_yticklabels([nm for _, nm in ST][::-1], fontsize=6); ax.yaxis.tick_right(); ax.set_xlim(0, 1)
ax.axhline(len(ST) - NAPP - .5, color="k", lw=1.2)
for tl, (s_, _) in zip(ax.get_yticklabels(), ST[::-1]):
    if s_ in APP: tl.set_color(APP[s_]); tl.set_fontweight("bold")
ax.set_xlabel(r"crossover $s^\times$ against PCA"); tag(ax, "(d) where PCA overtakes")
from matplotlib.patches import Patch
ax.legend([Patch(color=COL[m]) for m in NE_] + [Patch(facecolor="white", edgecolor="0.3", hatch="////")], [LAB.get(m, m) for m in NE_] + ["never"],
          frameon=False, fontsize=5.3, loc="lower center", bbox_to_anchor=(0.5, -0.33), ncol=3)
save(fig, "fig_LGplane")
for (s, nm) in ST:
    bl = MS[int(np.argmax([S[s][m]["L"] for m in MS]))]; bg = MS[int(np.argmax([S[s][m]["G"] for m in MS]))]
    chk.append(f"{nm}: best L {bl} {max(S[s][m]['L'] for m in MS):.3f}; best G {bg} {max(S[s][m]['G'] for m in MS):.3f}")
chk.append("crossover from subsample profile (first k/m with nu_PCA >= nu_A; None = never): ")
for m in NE_: chk.append(f"  {m}: " + ", ".join(f"{nm} {('none' if CX[(nm, m)] is None else f'{CX[(nm, m)]:.3f}')}" for _, nm in ST))
TT = {s: json.load(open(f"{R}/large/{s}/lgtest.json")) for s, _ in ST}
chk.append("  t-SNE crossover from lgtest (5k subsample): " + ", ".join(f"{nm} {TT[s]['t-SNE'].get('crossover')}" for s, nm in ST))
for fn_ in ("tab_validation_rows.tex", "tab_sens_rows.tex"):
    t_ = open(f"{T}/{fn_}").read(); open(f"{T}/{fn_}", "w").write(re.sub(r"(?<![\\\w])-(\d)", r"$-$\1", t_))
open(f"{T}/bigval_checks.txt", "w").write("\n".join(chk) + "\n"); print("\n".join(chk))
