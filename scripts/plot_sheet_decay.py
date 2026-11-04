"""The flat sheet giving way below lambda = 1: energy per vertex against sweeps, one line per run. Usage:

    pip install -e ".[plots]"          # matplotlib is needed by the plotting scripts only
    python scripts/plot_sheet_decay.py . docs/figures/sheet_decay.png

First argument: the repository root. Reads results/cqg_sheet_decay_lam025.csv and
results/cqg_sheet_decay_lam050.csv (exploratory runs; N = 64, g = 3). The dashed
levels are exact (test_sheet_tube_cube_ladder): 0 for the flat sheet, -4 (1 - lambda)
for a tube, -8 (1 - lambda) for 4-cubes. Used in docs/design/model_x_brief.md.
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
N, G = 64, 3.0


def runs(path):
    d = defaultdict(list)
    for r in csv.DictReader(open(path, newline="")):
        if int(r["N"]) == N and float(r["g"]) == G:
            d[r["replica"]].append((int(r["sweep"]), float(r["energy"])))
    return d


plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.0), dpi=160, facecolor=SURFACE)
for ax, (lam, name) in zip(axes, ((0.25, "cqg_sheet_decay_lam025"), (0.5, "cqg_sheet_decay_lam050"))):
    ax.set_facecolor(SURFACE)
    tube, cubes = -4 * (1 - lam), -8 * (1 - lam)
    for level, label in ((0.0, "flat sheet:\n2 large\ndimensions"), (tube, "tube:\n1 large\ndimension"),
                         (cubes, "closed knots:\nno large\ndimension")):
        ax.plot([0, 20000], [level, level], color=INK2, lw=1, ls=(0, (2, 3)), zorder=1)
        ax.text(20500, level, label, ha="left", va="center", fontsize=9.5, color=INK2)
    for rep, pts in sorted(runs(f"{repo}/results/{name}.csv").items()):
        xs, ys = zip(*pts)
        ax.plot((0,) + xs, (0.0,) + ys, color=BLUE, lw=1.8, alpha=0.85, zorder=3, drawstyle="steps-post")
    ax.set_xlim(0, 25500)
    ax.set_xticks([0, 5000, 10000, 15000, 20000])
    ax.set_ylim(cubes * 1.12, 1.0)
    ax.set_xlabel("sweeps since the start (every run starts as a perfect flat sheet)", color=INK2)
    ax.grid(True, which="major", color=GRID, lw=1)
    ax.tick_params(colors=INK2, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_title(f"λ = {lam}", loc="left", fontsize=13, color=INK, fontweight="bold", pad=22)
    ax.text(0, 1.02, f"exact levels: tube {tube:+.0f}, knots {cubes:+.0f} per vertex; way out of the sheet costs {16 * lam:.0f}",
            transform=ax.transAxes, fontsize=10.5, color=INK2, va="bottom")
axes[0].set_ylabel("energy per vertex", color=INK2)
fig.suptitle("A flat sheet that is stable for now: it waits, then drops in steps that land on the exact levels",
             x=0.045, y=0.985, ha="left", fontsize=13.5, color=INK, fontweight="bold")
fig.text(0.045, 0.012, f"Exploratory runs, N = {N}, g = {G:g}, four runs a panel, each line one run. Fixed temperature, so "
         "the energy given off is carried away.\nThis is the mechanism of claim 4 running the wrong way round: here "
         "space is the state that gives way.", fontsize=9.5, color=INK2, ha="left", va="bottom")
fig.tight_layout(rect=(0, 0.07, 1, 0.95))
fig.savefig(out, facecolor=SURFACE)
print("saved", out)
