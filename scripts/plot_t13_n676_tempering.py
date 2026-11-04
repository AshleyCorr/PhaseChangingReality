"""T13 at N = 676: the two replica-exchange series only, on a log10 coupling axis, with no title. Usage:

    python scripts/plot_t13_n676_tempering.py docs/figures/t13_n676_replica_exchange

Writes <stem>.png and <stem>.pdf. Reads committed result files only (results/t13_temper_n676_{melt,torus}.csv); one
marker per replica and coupling, nothing averaged. Drawn on 25 September 2026 at the model's author's request, for his
use; the words of the key are those of t13_three_sizes.png. Colors: the validated categorical slots 1 and 2 (blue,
orange), checked with the dataviz validator on a light surface (all checks pass); the two series also differ in
marker shape (squares, triangles), so identity never rests on color alone.
"""
import csv
import math
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

stem = sys.argv[1] if len(sys.argv) > 1 else "docs/figures/t13_n676_replica_exchange"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#ffffff"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def by_replica(path):
    d = defaultdict(list)
    for r in csv.DictReader(open(path, newline="")):
        d[r["replica"]].append((math.log10(float(r["g"])), float(r["phi"])))
    return {k: sorted(v) for k, v in d.items()}


series = [
    ("replica exchange,\nevery copy started random", by_replica("results/t13_temper_n676_melt.csv"),
     dict(color=BLUE, marker="s", ms=6.5, mfc="none", mew=1.5)),
    ("replica exchange,\nevery copy started as the lattice torus", by_replica("results/t13_temper_n676_torus.csv"),
     dict(color=ORANGE, marker="^", ms=7, mfc="none", mew=1.5)),
]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "mathtext.fontset": "dejavusans"})
fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=200, facecolor=SURFACE)
ax.set_facecolor(SURFACE)
xs_all = []
for label, reps, style in series:
    for i, (rep, pts) in enumerate(sorted(reps.items())):
        xs, ys = zip(*pts)
        xs_all += xs
        ax.plot(xs, ys, lw=0, mec=style["color"], zorder=3, label=label if i == 0 else None,
                **{k: v for k, v in style.items() if k != "color"})
lo, hi = min(xs_all), max(xs_all)
pad = 0.03 * (hi - lo)
ax.set_xlim(lo - pad, hi + pad)
ax.set_ylim(0.1, 1.08)
ax.grid(True, color=GRID, lw=1)
ax.tick_params(colors=INK2, length=0)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color(GRID)
ax.set_xlabel(r"$\log_{10}(g)$", color=INK)
ax.set_ylabel(r"$S/N$", color=INK)
leg = ax.legend(loc="upper right", fontsize=10.5, frameon=True, framealpha=1.0, edgecolor=GRID, labelcolor=INK,
                handletextpad=0.4, borderpad=0.7, labelspacing=0.9)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig("%s.%s" % (stem, ext), facecolor=SURFACE)
print("wrote %s.png and %s.pdf; x from %.3f to %.3f" % (stem, stem, lo, hi))
