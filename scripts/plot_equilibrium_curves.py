"""What the equilibrium curve at N = 160 looks like once the chains no longer freeze. Usage:

    pip install -e ".[plots]"          # matplotlib is needed by the plotting scripts only
    python scripts/plot_equilibrium_curves.py . docs/figures/equilibrium_curves.png

First argument: the repository root. Reads the two parallel-tempering results
(cqg_n160_capped_tempering.csv, cqg_n160_lam1_nocap_tempering.csv), the two plain
sweeps at the same size, and the digitised published points in docs/published/.
Every number plotted comes from a committed file. Used in the design brief and
ASSUMPTIONS section D.
"""
import csv
import statistics as st
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

repo, out = sys.argv[1], sys.argv[2]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def tempered(name):
    d = defaultdict(list)
    for r in csv.DictReader(open(f"{repo}/results/{name}.csv", newline="")):
        d[float(r["g"])].append(float(r["phi"]))
    gs = sorted(d)
    return gs, [st.mean(d[g]) for g in gs]


def plain(name, g_min):
    """The plain sweep, kept only where a single chain reaches equilibrium."""
    d = defaultdict(list)
    for r in csv.DictReader(open(f"{repo}/results/{name}.csv", newline="")):
        if float(r["g"]) >= g_min:
            d[float(r["g"])].append(float(r["phi"]))
    gs = sorted(d)
    return gs, [st.mean(d[g]) for g in gs]


def published(path, prefix):
    rows = [r for r in csv.DictReader(open(f"{repo}/docs/published/{path}", newline=""))
            if r["series"].startswith(prefix)]
    return [float(r["g"]) for r in rows], [float(r["S_over_N"]) for r in rows]


plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, ax = plt.subplots(figsize=(13.0, 7.0), dpi=160, facecolor=SURFACE)
ax.set_facecolor(SURFACE)

# the two regions we cannot speak for, marked so the reader sees the whole published range
ax.axvspan(0.08, 2.0, color="#f2f1ed", zorder=0, lw=0)
ax.text(0.095, 0.06, "neither method reaches\nequilibrium in here",
        color=INK2, fontsize=9.5, va="bottom", ha="left")
ax.axvspan(2.0, 5.0, color="#f8f7f3", zorder=0, lw=0)
ax.text(2.1, 0.06, "a single chain freezes here;\nthese points come from running\nall couplings at once",
        color=INK2, fontsize=9.5, va="bottom", ha="left")

for level, label in ((1.0, "flat sheet"), (1.5, "closed knots")):
    ax.plot([0.08, 220], [level, level], color=INK2, lw=1, ls=(0, (2, 3)), zorder=1)
    ax.text(225, level, label, ha="left", va="center", fontsize=9.5, color=INK2)

gs, phi = tempered("cqg_n160_lam1_nocap_tempering")
ax.plot(gs, phi, color=INK, lw=2.6, marker="o", ms=6, mec=SURFACE, mew=1.2, zorder=5,
        label="Ours, full energy (λ = 1)")
gs2, phi2 = tempered("cqg_n160_capped_tempering")
ax.plot(gs2, phi2, color=INK2, lw=2.2, ls=(0, (5, 3)), marker="s", ms=5, mec=SURFACE, mew=1.2, zorder=4,
        label="Ours, folding forbidden (the cap)")
gs3, phi3 = plain("cqg_n160_lam1_nocap_vs_t25fig3", 5.0)
ax.plot(gs3, phi3, color=INK, lw=2.6, marker="o", ms=6, mfc="none", mew=1.6, zorder=6,
        label="Ours, full energy, one chain at a time")

x8, y8 = published("KTB19_fig8a_N160_digitised.csv", "cool")
ax.scatter(x8, y8, s=85, color=BLUE, edgecolors=SURFACE, linewidths=1.6, zorder=7,
           label="Published: Kelly et al. 2019, Fig. 8a")
x3, y3 = published("T25_fig3_digitised.csv", "cool")
ax.scatter(x3, y3, s=85, color=ORANGE, edgecolors=SURFACE, linewidths=1.6, zorder=7,
           label="Published: Trugenberger 2025, Fig. 3 (cooling)")
x3h, y3h = published("T25_fig3_digitised.csv", "heat")
ax.scatter(x3h, y3h, s=75, facecolors=SURFACE, edgecolors=ORANGE, linewidths=2.2, zorder=7,
           label="Published: Trugenberger 2025, Fig. 3 (heating)")

ax.annotate("Here the 2025 figure climbs to 0.99 while we are\nstill at 0.62. This narrow window is the whole\nof the disagreement, and it is unexplained.",
            xy=(5.06, 0.992), xytext=(0.115, 1.45), color=INK, fontsize=10.5, va="top",
            arrowprops=dict(arrowstyle="-", color=INK2, lw=1))
ax.annotate("Everywhere out here we sit on the published points,\nboth papers', to between 0.001 and 0.04",
            xy=(30, 0.191), xytext=(11, 0.74), color=INK, fontsize=10.5,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=1))

ax.set_xscale("log")
ax.set_xlim(0.08, 220)
ax.set_ylim(0, 1.62)
ax.set_xticks([0.1, 0.3, 1, 2, 3, 5, 10, 30, 100])
ax.set_xticklabels(["0.1", "0.3", "1", "2", "3", "5", "10", "30", "100"])
ax.minorticks_off()
ax.grid(True, which="major", color=GRID, lw=1)
ax.tick_params(colors=INK2, length=0)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.set_xlabel("coupling g  (acts like temperature: cold on the left, hot on the right)", color=INK2)
ax.set_ylabel("blocks per crossing, φ", color=INK2)
fig.suptitle("Over the whole published range we sit on their points, except in one narrow window",
             x=0.04, y=0.975, ha="left", fontsize=14, color=INK, fontweight="bold")
ax.set_title("N = 160 points, four independent runs each, agreeing to 0.002 down to g = 2.7. Where both our "
             "methods reach, they agree to 0.0005. Exploratory.",
             loc="left", fontsize=10.5, color=INK2, pad=10)
leg = ax.legend(loc="upper right", frameon=True, fontsize=9.5, labelcolor=INK, borderpad=0.7,
                bbox_to_anchor=(1.0, 0.99))
leg.get_frame().set_facecolor(SURFACE)
leg.get_frame().set_edgecolor(GRID)
fig.tight_layout(rect=(0, 0, 1, 0.945))
fig.savefig(out, facecolor=SURFACE)
print("saved", out)
