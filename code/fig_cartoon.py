"""Cartoon of the LG reversal test. (a) Two maps of the same data: map A keeps each point's neighbors and scrambles the
layout of the groups, map B keeps the layout and blurs the neighbors. (b) Two one-sided tests give one declaration.
(c) The crossover tells the user which map answers a question of a given size. Schematic; no data.
usage: python3 fig_cartoon.py <fig dir>"""
import sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
F = sys.argv[1]; RED, BLUE, PUR, INK = "#b2182b", "#2166ac", "#5e35b1", "0.2"
GC = ["#e08214", "#1b7837", "#762a83"]                       # three groups of points
plt.rcParams.update({"font.size": 7})
fig = plt.figure(figsize=(7.2, 2.75)); gs = fig.add_gridspec(1, 3, wspace=0.12, width_ratios=[1.25, 1.0, 1.05])
rng = np.random.default_rng(3)

def blob(c, s, m=14): return np.asarray(c) + s * rng.standard_normal((m, 2))
def box(a, x, y, w, h, fc, ec, lw=0.8): a.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.03", fc=fc, ec=ec, lw=lw, transform=a.transAxes))
def arrow(a, p, q, c=INK, cs="arc3,rad=0"): a.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, color=c, lw=0.9, connectionstyle=cs, transform=a.transAxes))

# (a) the data and two maps
a = fig.add_subplot(gs[0]); a.axis("off"); a.set_xlim(0, 1); a.set_ylim(0, 1)
a.set_title("(a) two maps of the same data", loc="left", fontsize=7.5, fontweight="bold")
def panel(x0, y0, cents, s, title, tc, note):
    box(a, x0, y0, 0.29, 0.36, "white", "0.6")
    for c, col in zip(cents, GC):
        P = blob(c, s); a.scatter(x0 + 0.145 + P[:, 0], y0 + 0.17 + P[:, 1], s=4, color=col, lw=0)
    a.text(x0 + 0.145, y0 + 0.385, title, ha="center", fontsize=6.6, color=tc, fontweight="bold")
    a.text(x0 + 0.145, y0 - 0.055, note, ha="center", va="top", fontsize=5.8, color=INK, linespacing=1.25)
line = [(-0.09, -0.07), (0.0, 0.0), (0.09, 0.07)]           # true layout: orange - green - purple on a line
panel(0.35, 0.56, line, 0.013, "data", INK, "")
panel(0.02, 0.10, [(0.08, -0.07), (-0.09, 0.07), (0.0, 0.0)], 0.012, "map A", RED, "neighbors kept ✓\nlayout of groups lost ✗")
panel(0.68, 0.10, line, 0.034, "map B", BLUE, "neighbors blurred ✗\nlayout of groups kept ✓")
arrow(a, (0.40, 0.55), (0.25, 0.47), RED); arrow(a, (0.60, 0.55), (0.75, 0.47), BLUE)

# (b) two one-sided tests, one declaration
b = fig.add_subplot(gs[1]); b.axis("off"); b.set_xlim(0, 1); b.set_ylim(0, 1)
b.set_title("(b) two one-sided tests", loc="left", fontsize=7.5, fontweight="bold")
box(b, 0.02, 0.70, 0.44, 0.19, "#fbe9eb", RED); b.text(0.24, 0.795, "local test\nis A better for\nneighbors? $p_L$", ha="center", va="center", fontsize=6, linespacing=1.15)
box(b, 0.54, 0.70, 0.44, 0.19, "#e6eef7", BLUE); b.text(0.76, 0.795, "global test\nis B better for\nthe layout? $p_G$", ha="center", va="center", fontsize=6, linespacing=1.15)
arrow(b, (0.24, 0.68), (0.42, 0.52)); arrow(b, (0.76, 0.68), (0.58, 0.52))
b.text(0.5, 0.465, "both yes?", ha="center", va="center", fontsize=6.4, fontweight="bold")
box(b, 0.0, 0.08, 0.52, 0.26, "#efeaf8", PUR); b.text(0.26, 0.21, "yes: reversal\nA local, B global\n$p=\\max(p_L,p_G)$", ha="center", va="center", fontsize=6, linespacing=1.2)
box(b, 0.60, 0.08, 0.40, 0.26, "0.95", "0.55"); b.text(0.80, 0.21, "no: one map\nis enough", ha="center", va="center", fontsize=6, linespacing=1.2)
arrow(b, (0.44, 0.43), (0.30, 0.36), PUR); arrow(b, (0.56, 0.43), (0.74, 0.36), "0.45")
b.text(0.5, -0.04, "nearby points share neighbors:\n20 $k$-means blocks, jackknife", ha="center", va="top", fontsize=5.6, color="0.35", linespacing=1.2)

# (c) which map for which question
c = fig.add_subplot(gs[2]); c.set_title("(c) which map for which question", loc="left", fontsize=7.5, fontweight="bold")
s = np.linspace(0.005, 0.5, 300); u = 1 - np.exp(-s / 0.16); fa, fb = 0.36 - 0.22 * u, 0.08 + 0.24 * u
c.plot(s, fa, color=RED, lw=1.8); c.plot(s, fb, color=BLUE, lw=1.8); sx = s[np.argmin(np.abs(fa - fb))]
c.axvspan(sx - 0.04, sx + 0.04, color=PUR, alpha=0.15, lw=0); c.axvline(sx, color=PUR, lw=1)
c.text(sx + 0.048, 0.445, "crossover $s^\\times n$ points,\nwith its interval", fontsize=5.8, color=PUR, va="top", linespacing=1.15)
c.text(0.012, 0.375, "map A", color=RED, fontsize=6.4, fontweight="bold"); c.text(0.42, 0.33, "map B", color=BLUE, fontsize=6.4, fontweight="bold")
P = 0.012 * rng.standard_normal((5, 2)); c.scatter(0.055 + P[:, 0], 0.03 + 0.6 * P[:, 1], s=4, color=INK, lw=0)
c.scatter([0.055], [0.03], s=260, facecolors="none", edgecolors=INK, linestyles="--", lw=0.7)
P = rng.uniform(-1, 1, (60, 2)); P = P[(P ** 2).sum(1) < 1][:40]; c.scatter(0.36 + 0.05 * P[:, 0], 0.03 + 0.034 * P[:, 1], s=4, color=INK, lw=0)
c.scatter([0.36], [0.03], s=1500, facecolors="none", edgecolors=INK, linestyles="--", lw=0.7)
c.text(0.055, -0.045, "small question:\nuse map A", ha="center", va="top", fontsize=5.8, color=RED, linespacing=1.15)
c.text(0.36, -0.045, "large question:\nuse map B", ha="center", va="top", fontsize=5.8, color=BLUE, linespacing=1.15)
c.set_xlim(0, 0.5); c.set_ylim(-0.1, 0.45); c.set_xticks([]); c.set_yticks([])
c.set_xlabel("size of the question (points)  →", fontsize=6.4); c.set_ylabel("fidelity $\\nu_k$", fontsize=6.4)
for k in ("top", "right"): c.spines[k].set_visible(False)
for e in ("pdf", "png"): fig.savefig(f"{F}/fig_cartoon.{e}", bbox_inches="tight", dpi=220)
