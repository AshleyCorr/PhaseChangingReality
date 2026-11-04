"""Is the leftover disorder fixed by the energy released? Usage:

    pip install -e ".[plots]"          # matplotlib is needed by the plotting scripts only
    python scripts/plot_leftover_determined.py . docs/figures/leftover_determined.png

First argument: the repository root. The line is the equilibrium curve at lambda =
1.25 measured by parallel tempering (results/cqg_n64_lam125_tempering.csv): what a
network at a given energy per vertex looks like when it has settled. The points are
sealed runs (results/cqg_spark_wide_lam125.csv), each of which starts as a perfect
tube plus a fixed lump of spare energy and never exchanges energy with anything.

The prediction, written into the sealed run's config before it ran, is that a sealed
run must land on the line: its total energy is fixed at the start, so where it ends
is not free. Used in ASSUMPTIONS section D.
"""
import csv
import math
import statistics as st
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

repo, out = sys.argv[1], sys.argv[2]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE = "#2a78d6", "#eb6834"
LAM, N = 1.25, 64

eq = defaultdict(list)
for r in csv.DictReader(open(f"{repo}/results/cqg_n64_lam125_tempering.csv", newline="")):
    energy = 16 * (1 - float(r["phi"])) + 4 * LAM * float(r["surplus"])
    eq[float(r["g"])].append((energy, float(r["phi"])))
curve = sorted((st.mean(e for e, _ in v), st.mean(p for _, p in v)) for v in eq.values())

sealed = defaultdict(list)
for r in csv.DictReader(open(f"{repo}/results/cqg_spark_wide_lam125.csv", newline="")):
    if r["left"] == "True":
        sealed[float(r["spark"])].append(float(r["phi_end"]))

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, ax = plt.subplots(figsize=(11.5, 6.8), dpi=160, facecolor=SURFACE)
ax.set_facecolor(SURFACE)

ax.plot([c[0] for c in curve], [c[1] for c in curve], color=INK, lw=2.4, marker="o", ms=5, mec=SURFACE, mew=1,
        zorder=3, label="Equilibrium: where a settled network sits at this energy\n(measured separately, by tempering)")
for spark, vals in sorted(sealed.items()):
    total = 1.0 + spark / N
    m = st.mean(vals)
    e = st.stdev(vals) / math.sqrt(len(vals))
    ax.errorbar([total], [m], yerr=[e], color=ORANGE, marker="o", ms=11, mec=SURFACE, mew=1.5, capsize=6,
                lw=0, elinewidth=2, zorder=5,
                label="Sealed run: started as a tube plus a lump of spare energy" if spark == 12 else None)
    ax.annotate(f"spark {spark:g}", xy=(total, m), xytext=(0, -22 if spark >= 96 else 16),
                textcoords="offset points", ha="center", fontsize=9.5, color=INK2)

ax.set_xlim(0, 6.6)
ax.set_ylim(0.58, 1.04)
ax.grid(True, which="major", color=GRID, lw=1)
ax.tick_params(colors=INK2, length=0)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.set_xlabel("total energy per point, which a sealed run cannot change", color=INK2)
ax.set_ylabel("blocks per crossing, φ  (1.00 = a perfect flat sheet)", color=INK2)
ax.annotate("Every sealed run lands on the line.\nWhere it ends is not free: it is fixed by\nthe energy that was in it at the start.",
            xy=(3.55, 0.805), xytext=(4.25, 0.90), color=INK, fontsize=11, va="center",
            arrowprops=dict(arrowstyle="-", color=INK2, lw=1))
fig.suptitle("The leftover disorder is fixed by the energy released, with no freedom",
             x=0.035, y=0.975, ha="left", fontsize=14, color=INK, fontweight="bold")
ax.set_title("λ = 1.25, N = 64. The curve was measured first and the points predicted from it before they were run. "
             "Exploratory.", loc="left", fontsize=10.5, color=INK2, pad=10)
leg = ax.legend(loc="lower left", frameon=True, fontsize=10, labelcolor=INK, borderpad=0.8)
leg.get_frame().set_facecolor(SURFACE)
leg.get_frame().set_edgecolor(GRID)
fig.text(0.035, 0.012, "Ten runs a point; error bars are the spread between them. The three largest sparks agree with "
         "the prediction to 0.003. The two smallest sit 0.008 to 0.014 high,\nwhich is most likely incomplete settling: "
         "with little spare energy a sealed run explores slowly.", fontsize=9.5, color=INK2, ha="left", va="bottom")
fig.tight_layout(rect=(0, 0.055, 1, 0.94))
fig.savefig(out, facecolor=SURFACE)
print("saved", out)
