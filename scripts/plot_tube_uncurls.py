"""A tube uncurling into the flat sheet above lambda = 1: energy per vertex against sweeps, one line per run. Usage:

    pip install -e ".[plots]"          # matplotlib is needed by the plotting scripts only
    python scripts/plot_tube_uncurls.py . docs/figures/tube_uncurls.png

First argument: the repository root. Reads results/cqg_tube_uncurls_lam125.csv
(exploratory; N = 64; g = 1.0 and 1.5). The dashed levels are exact
(test_sheet_tube_cube_ladder): 4 (lambda - 1) = 1 per vertex for the tube, 0 for the
flat sheet. Used in docs/design/model_x_brief.md.
"""
import csv
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

repo, out = sys.argv[1], sys.argv[2]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE = "#2a78d6"
N, LAM = 64, 1.25
TUBE = 4 * (LAM - 1)

runs = defaultdict(list)
for r in csv.DictReader(open(f"{repo}/results/cqg_tube_uncurls_lam125.csv", newline="")):
    if int(r["N"]) == N:
        runs[(float(r["g"]), r["replica"])].append((int(r["sweep"]), float(r["energy"])))

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.0), dpi=160, facecolor=SURFACE, sharey=True)
for ax, (g, note) in zip(axes, ((1.0, "colder: two of the four runs are still tubes at the end"),
                                (1.5, "a little warmer: all four uncurl, three of them into a perfect flat sheet"))):
    ax.set_facecolor(SURFACE)
    for level, label in ((TUBE, "tube:\n1 large\ndimension"), (0.0, "flat sheet:\n2 large\ndimensions")):
        ax.plot([0, 30000], [level, level], color=INK2, lw=1, ls=(0, (2, 3)), zorder=1)
        ax.text(30700, level, label, ha="left", va="center", fontsize=9.5, color=INK2)
    for (gg, rep), pts in sorted(runs.items()):
        if gg == g:
            xs, ys = zip(*pts)
            ax.plot((0,) + xs, (TUBE,) + ys, color=BLUE, lw=1.8, alpha=0.85, zorder=3, drawstyle="steps-post")
    ax.set_xlim(0, 38000)
    ax.set_xticks([0, 10000, 20000, 30000])
    ax.set_ylim(-0.3, 1.6)
    ax.set_xlabel("sweeps since the start (every run starts as a perfect tube)", color=INK2)
    ax.grid(True, which="major", color=GRID, lw=1)
    ax.tick_params(colors=INK2, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_title(f"g = {g:g}", loc="left", fontsize=13, color=INK, fontweight="bold", pad=22)
    ax.text(0, 1.02, note, transform=ax.transAxes, fontsize=10.5, color=INK2, va="bottom")
axes[0].set_ylabel("energy per vertex", color=INK2)
fig.suptitle("The right way round, in miniature: a curled-up dimension opens out into a flat sheet and gives off energy",
             x=0.045, y=0.985, ha="left", fontsize=13.5, color=INK, fontweight="bold")
fig.text(0.045, 0.012, f"Exploratory runs, λ = {LAM}, N = {N}, four runs a panel, each line one run. The tube lies exactly "
         f"{TUBE:g} per vertex above the flat sheet.\n"
         "Fixed temperature, so the energy given off is carried away. Spikes are the energy paid to get over the wall.\n"
         "Sixty-four points are not a space; this shows the mechanism, not the universe.",
         fontsize=9.5, color=INK2, ha="left", va="bottom")
fig.tight_layout(rect=(0, 0.095, 1, 0.95))
fig.savefig(out, facecolor=SURFACE)
print("saved", out)
