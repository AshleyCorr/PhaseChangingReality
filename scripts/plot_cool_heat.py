"""Cooling against heating at N = 160: lambda = 0 (a loop) beside lambda = 1 (no loop). Usage:

    pip install -e ".[plots]"          # matplotlib is needed by the plotting scripts only
    python scripts/plot_cool_heat.py . docs/figures/cool_heat_lam0_vs_lam1.png

First argument: the repository root. Reads committed result files only
(cqg_n160_lam1_nocap_vs_t25fig3.csv and cqg_lam0_first_look.csv), one line per
replica, nothing averaged across replicas. Both runs are exploratory and the figure
says so. Was used in paper/introduction_plain_language.md (removed 2026-09-24, superseded).
"""
import csv
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

repo, out = sys.argv[1], sys.argv[2]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE = "#2a78d6", "#eb6834"          # validated earlier: categorical slots 1 and 2


def replicas(path, n, g_lo, g_hi):
    """{(leg, replica): [(g, phi), ...]} for one size, couplings inside the window, sorted by g."""
    d = defaultdict(list)
    for r in csv.DictReader(open(path, newline="")):
        g = float(r["g"])
        if int(r["N"]) == n and g_lo <= g <= g_hi:
            d[(r["leg"], r["replica"])].append((g, float(r["phi"])))
    return {k: sorted(v) for k, v in d.items()}


panels = [
    ("λ = 1: the full energy", "cooling and heating give the same curve, in one piece",
     replicas(f"{repo}/results/cqg_n160_lam1_nocap_vs_t25fig3.csv", 160, 4.4, 32)),
    ("λ = 0: square count only", "cooling and heating disagree; the cold state is shattered",
     replicas(f"{repo}/results/cqg_lam0_first_look.csv", 160, 4.4, 32)),
]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.2), dpi=160, facecolor=SURFACE, sharey=True)
for ax, (title, sub, data) in zip(axes, panels):
    ax.set_facecolor(SURFACE)
    for (leg, rep), pts in sorted(data.items()):
        colour = BLUE if leg == "cool" else ORANGE
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=colour, lw=3.4 if leg == "cool" else 1.6, alpha=0.9, solid_capstyle="round",
                zorder=3 if leg == "heat" else 2, marker="o", ms=6 if leg == "cool" else 3.5, mec=SURFACE, mew=0.8,
                label={"cool": "cooling (four runs)", "heat": "heating (four runs)"}[leg] if rep == "0" else None)
    ax.set_xscale("log")
    ax.set_xlim(4.2, 34)
    ax.set_ylim(0, 1.62)
    ax.set_xticks([4.5, 5, 6, 7, 8, 10, 15, 20, 30])
    ax.set_xticklabels(["4.5", "5", "6", "7", "8", "10", "15", "20", "30"])
    ax.minorticks_off()
    ax.grid(True, which="major", color=GRID, lw=1)
    ax.tick_params(colors=INK2, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_title(title, loc="left", fontsize=13, color=INK, fontweight="bold", pad=24)
    ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=10.5, color=INK2, va="bottom")
    ax.set_xlabel("coupling g  (acts like temperature: cold on the left, hot on the right)", color=INK2)

axes[0].set_ylabel("squares per vertex, φ", color=INK2)
for ax in axes:                                   # reference heights, labelled once
    ax.axhline(1.0, color=INK2, lw=1, ls=(0, (2, 3)), zorder=1)
    ax.axhline(1.5, color=INK2, lw=1, ls=(0, (2, 3)), zorder=1)
axes[0].text(33, 1.015, "flat sheet = 1", color=INK2, fontsize=9.5, ha="right", va="bottom")
axes[0].text(33, 1.515, "4-cubes = 1.5", color=INK2, fontsize=9.5, ha="right", va="bottom")

axes[1].annotate("cooling: stays low,\nthen jumps", xy=(5.05, 0.80), xytext=(7.7, 0.80), color=INK, fontsize=10.5,
                 arrowprops=dict(arrowstyle="-", color=INK2, lw=1))
axes[1].annotate("heating: the shattered state\nhangs on to a higher g", xy=(6.5, 1.47), xytext=(9.2, 1.30),
                 color=INK, fontsize=10.5, arrowprops=dict(arrowstyle="-", color=INK2, lw=1))
axes[1].text(11.5, 1.07, "Between g = 5 and g = 7 a run is in\none state or the other, never between.",
             color=INK2, fontsize=10, ha="left", va="center")
axes[0].annotate("both directions,\nall eight runs", xy=(6.1, 0.545), xytext=(8.5, 1.18), color=INK, fontsize=10.5,
                 arrowprops=dict(arrowstyle="-", color=INK2, lw=1))
leg = axes[0].legend(loc="lower left", frameon=True, fontsize=10, labelcolor=INK, borderpad=0.8)
leg.get_frame().set_facecolor(SURFACE)
leg.get_frame().set_edgecolor(GRID)

fig.suptitle("Same code, same size (N = 160): one ingredient of the energy decides whether there are two states",
             x=0.045, y=0.985, ha="left", fontsize=13.5, color=INK, fontweight="bold")
fig.text(0.045, 0.012, "Exploratory runs of this project, four replicas each; not yet a finding. "
         "Each line is one run; every dot is an average over 4000 sweeps at that coupling.\n"
         "Below g = 4.5 the runs are frozen and are not shown.",
         fontsize=9.5, color=INK2, ha="left", va="bottom")
fig.tight_layout(rect=(0, 0.06, 1, 0.965))
fig.savefig(out, facecolor=SURFACE)
print("saved", out)
