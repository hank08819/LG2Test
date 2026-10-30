"""Our two methods on the scale, large data. (a) SR-tSNE2 on MNIST-60k: O-DLP@10 relative to its t-SNE start against the rank
weight lambda, for the default schedule and for learning rate 50 with 2,000 iterations (tune_sr_big.json); (b) the same for L;
(c) O-DLP@10 gain of SR-tSNE2 over t-SNE on the twelve large datasets and the three small stores; (d) G of NPCA against G of PCA.
Also writes tex/tab_own_rows.tex (twelve large datasets).
usage: python3 fig_own_big.py <large dir> <big dir> <v6 dir> <fig dir> <tex dir> <checks file>"""
import sys, json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
LD, BD, V6, F, T, CK = sys.argv[1:7]
ST = [("path100k", "Pathology-100K"), ("seaad1m", "SEA-AD"), ("instacart", "Instacart"), ("marco1m", "MS MARCO"),
      ("mnist", "MNIST"), ("cifar10", "CIFAR-10"), ("fmnist", "Fashion-MNIST"), ("speech", "Speech Commands"), ("renal", "Renal-Cortex"), ("brain", "Brain-Cortex"),
      ("covertype", "Covertype"), ("agnews", "AG News")]
APP = {"path100k": "#ad1457", "seaad1m": "#2e7d32", "instacart": "#ef6c00", "marco1m": "#5e35b1"}
SM = [("path", f"{V6}/path/path.json", "Pathology-7K"), ("retail", f"{V6}/apps/retail.json", "Online Retail"), ("8k", f"{V6}/fin8k/8k.json", "8-K")]
plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False})
tu = json.load(open(f"{BD}/tune_sr_big.json")); o0 = tu["start"]["odlp"]; L0 = tu["lam0_lr50_it2000"]["L"]
check = []; fig = plt.figure(figsize=(7.2, 5.0)); gs = fig.add_gridspec(2, 6, wspace=1.6, hspace=0.6, height_ratios=[1, 1.15])
ax = [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]), fig.add_subplot(gs[1, 0:3]), fig.add_subplot(gs[0, 4:6])]; AXE = fig.add_subplot(gs[1, 3:6])
lam = [0, 5, 20, 50]
s1o = [o0] + [tu[f"lam{l}"]["odlp"] for l in lam[1:]]; l2 = [l for l in lam if f"lam{l}_lr50_it2000" in tu]; s2o = [tu[f"lam{l}_lr50_it2000"]["odlp"] for l in l2]
s1L = [L0] + [tu[f"lam{l}"]["L"] for l in lam[1:]]; s2L = [tu[f"lam{l}_lr50_it2000"]["L"] for l in l2]
xs = [1, 5, 20, 50]; xs2 = [max(l, 1) for l in l2]
for a, (y1, y2), base, lab, nm in ((ax[0], (s1o, s2o), o0, "O-DLP@10 / start", "odlp"), (ax[1], (s1L, s2L), L0, "$L$ − $L$ at $\\lambda=0$", "L")):
    f = (lambda v: np.array(v) / base) if nm == "odlp" else (lambda v: np.array(v) - base)
    a.plot(xs, f(y1), "o-", color="0.5", ms=3, lw=1, label="default schedule"); a.plot(xs2, f(y2), "s-", color="#8b0000", ms=3, lw=1.2, label="lr 50, 2,000 it.")
    a.set_xscale("log"); a.set_xticks(xs); a.set_xticklabels(["0", "5", "20", "50"]); a.minorticks_off(); a.set_xlabel("rank weight $\\lambda$"); a.set_ylabel(lab)
    a.axhline(1 if nm == "odlp" else 0, color="0.7", lw=0.6, ls=":")
    check += [(f"tune {nm} default lambda {lam} (lambda 0 = start; L at lambda 0 taken from lr50 lambda 0 run)", [round(v, 4) for v in y1]), (f"tune {nm} lr50 lambda {l2}", [round(v, 4) for v in y2])]
ax[0].legend(frameon=False, fontsize=5.5, loc="upper left"); ax[0].set_title("(a) O-DLP, MNIST-60k", loc="left", fontsize=7); ax[1].set_title("(b) $L$, MNIST-60k", loc="left", fontsize=7)
g_lab, g_val, g_col, npc = [], [], [], []
rows = []
for st, nm in ST:
    r = json.load(open(f"{LD}/{st}/scores.json")); t, s, p, q = r["t-SNE"], r["SR-tSNE2"], r["PCA"], r["NPCA"]
    g_lab.append(nm); g_val.append(100 * (s["odlp"] / t["odlp"] - 1)); g_col.append(APP.get(st, "0.6")); npc.append((nm, p["G"], q["G"], "o", APP.get(st, "black")))
    rows.append(f"{nm} & {p['L']:.3f}/{p['G']:.3f} & {q['L']:.3f}/{q['G']:.3f} & {t['L']:.3f}/{t['G']:.3f} & {s['L']:.3f}/{s['G']:.3f} & ${s['L'] - t['L']:+.3f}$ & ${s['G'] - t['G']:+.4f}$ \\\\" + ("\n\\midrule" if st == "marco1m" else ""))
    check.append((nm, "PCA", round(p["L"], 4), round(p["G"], 4), "NPCA", round(q["L"], 4), round(q["G"], 4), "NPCA-PCA G", round(q["G"] - p["G"], 4), "t-SNE", round(t["L"], 4), round(t["G"], 4), "SR", round(s["L"], 4), round(s["G"], 4), "odlp gain %", round(g_val[-1], 1)))
for st, f, nm in SM:
    r = json.load(open(f)); t, s, p, q = r["t-SNE"], r["SR-tSNE2"], r["PCA"], r["NPCA"]
    npc.append((nm, p["G"], q["G"], "s", "white"))
    check.append((nm, "(small store) PCA G", round(p["G"], 4), "NPCA G", round(q["G"], 4), "NPCA-PCA G", round(q["G"] - p["G"], 4), "odlp gain %", round(100 * (s["odlp"] / t["odlp"] - 1), 1)))
a = ax[2]; o = np.arange(len(g_val))
a.barh(o, g_val, color=g_col, height=0.7); a.set_yticks(o); a.set_yticklabels(g_lab, fontsize=6); a.invert_yaxis(); a.axvline(0, color="k", lw=0.5); a.axhline(3.5, color="0.7", lw=0.6, ls="--")
for tl, (st_, _) in zip(a.get_yticklabels(), ST):
    if st_ in APP: tl.set_color(APP[st_]); tl.set_fontweight("bold")
for i, v in enumerate(g_val): a.text(max(v, 0) + 1, i, f"{v:+.0f}%", va="center", ha="left", fontsize=4.8)
a.set_xlim(0, max(g_val) + 14); a.set_xlabel("exact-rank neighbors, gain over t-SNE (%)"); a.set_title("(d) SR-tSNE2: more neighbors at their exact rank", loc="left", fontsize=7)
a = ax[3]
for nm, p, q, mk, c in npc:
    big = c not in ("black", "white"); a.scatter(p, q, s=26 if big else 14, marker=mk, color=c if mk == "o" else "white", edgecolor=c if big else ("0.4" if mk == "o" else "black"), lw=0.6, zorder=3 + big)
    if big: a.annotate(nm, (p, q), xytext={"Pathology-100K": (4, -9), "SEA-AD": (-4, 8), "Instacart": (-5, 6), "MS MARCO": (5, -6)}[nm], textcoords="offset points", fontsize=5.5, color=c, fontweight="bold", ha="right" if nm in ("SEA-AD", "Instacart") else "left")
lo = min(min(e[1] for e in npc), min(e[2] for e in npc)) - 0.03; hi = max(max(e[1] for e in npc), max(e[2] for e in npc)) + 0.03
a.plot([lo, hi], [lo, hi], "k:", lw=0.7); a.set_xlim(lo, hi); a.set_ylim(lo, hi); a.set_xlabel("$G$ of PCA"); a.set_ylabel("$G$ of NPCA")
a.set_title("(c) NPCA vs. PCA, $G$", loc="left", fontsize=7); a.text(0.03, 0.97, "colored: four applications\nblack: benchmarks\nsquares: smaller datasets", transform=a.transAxes, ha="left", va="top", fontsize=5)
dL, dG = [], []
for st, nm in ST:
    r = json.load(open(f"{LD}/{st}/scores.json")); dL.append(r["SR-tSNE2"]["L"] - r["t-SNE"]["L"]); dG.append(r["SR-tSNE2"]["G"] - r["t-SNE"]["G"])
a = AXE; o = np.arange(len(ST)); h = 0.38
a.barh(o - h / 2, dL, h, color="#c0392b", label="change in $L$"); a.barh(o + h / 2, dG, h, color="#1f6fd1", label="change in $G$")
for i, (x, y) in enumerate(zip(dL, dG)):
    a.text(max(x, 0) + 0.001, i - h / 2, f"{x:+.3f}", va="center", fontsize=4.8)
a.set_yticks(o); a.set_yticklabels([nm for _, nm in ST], fontsize=6); a.invert_yaxis(); a.axvline(0, color="k", lw=0.5); a.axhline(3.5, color="0.7", lw=0.6, ls="--")
for tl, (st_, _) in zip(a.get_yticklabels(), ST):
    if st_ in APP: tl.set_color(APP[st_]); tl.set_fontweight("bold")
a.set_xlim(-0.004, max(dL) + 0.016); a.set_xlabel("SR-tSNE2 $-$ t-SNE"); a.set_title("(e) SR-tSNE2: $L$ rises, $|\\Delta G|\\leq 0.0005$", loc="left", fontsize=7)
a.legend(frameon=False, fontsize=5.5, loc="lower right")
for e in ("pdf", "png"): fig.savefig(f"{F}/fig_own.{e}", bbox_inches="tight", dpi=220)
open(f"{T}/tab_own_rows.tex", "w").write("\n".join(rows) + "\n")
d = [e[2] - e[1] for e in npc if e[3] == "o"]
check.append(("NPCA-PCA G over 12 large: min, max, n within 0.005", round(min(d), 4), round(max(d), 4), sum(abs(x) <= 0.005 for x in d)))
gl = g_val[:12]; check.append(("SR odlp gain over 12 large: min, max, n>35", round(min(gl), 1), round(max(gl), 1), sum(x > 35 for x in gl)))
with open(CK, "a") as fh:
    fh.write("== fig_own + tab_own_rows (12 large datasets; small stores from v6 jsons, SR-tSNE2 lambda=5 setting as in fig_apps)\n")
    for c in check: fh.write("  " + " ".join(map(str, c)) + "\n")
print(*check, sep="\n")
