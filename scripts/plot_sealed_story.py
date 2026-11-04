"""The three sealed-run measurements side by side. Usage:

    pip install -e ".[plots]"          # matplotlib is needed by the plotting scripts only
    python scripts/plot_sealed_story.py . docs/figures/sealed_story.png

First argument: the repository root. Reads results/cqg_spark_threshold_lam125.csv
and results/cqg_tube_waiting_lam125.csv. All at lambda = 1.25, starting from a
perfect tube, which lies exactly 1 per vertex above the flat sheet.

Panel 1: how big a spark a SEALED tube needs before its (S,X) counts can leave their initial values.
Panel 2: how long an OPEN tube waits before converting, against the prediction we
         wrote down beforehand and which the runs do not support.
Panel 3: where a sealed tube ends up, against how much energy was put in.
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
TUBE = 1.0                                          # energy per vertex of the tube at lambda = 1.25

spark_rows = list(csv.DictReader(open(f"{repo}/results/cqg_spark_threshold_lam125.csv", newline="")))
wait_rows = list(csv.DictReader(open(f"{repo}/results/cqg_tube_waiting_lam125.csv", newline="")))
sizes = sorted({int(r["N"]) for r in spark_rows})

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.4), dpi=160, facecolor=SURFACE)
for ax in axes:
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, lw=1)
    ax.tick_params(colors=INK2, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)

# ---- 1. the spark a sealed tube needs -------------------------------------------------------
ax = axes[0]
fraction = defaultdict(dict)
for r in spark_rows:
    fraction[int(r["N"])].setdefault(float(r["spark"]), []).append(r["left"] == "True")
pattern = {n: sorted((s, sum(v) == len(v)) for s, v in fraction[n].items()) for n in sizes}
assert all(pattern[n] == pattern[sizes[0]] for n in sizes), \
    "the sizes no longer agree, so one curve would misrepresent them"
xs = sorted(fraction[sizes[0]])
ys = [100 * sum(fraction[sizes[0]][x]) / len(fraction[sizes[0]][x]) for x in xs]
ax.plot(xs, ys, color=BLUE, lw=3, marker="o", ms=9, mec=SURFACE, mew=1.4, drawstyle="steps-post", zorder=4)
ax.axvline(12, color=ORANGE, lw=2, ls=(0, (4, 3)), zorder=1)
ax.text(12.3, 55, "12", color=ORANGE, fontsize=15, fontweight="bold", va="center")
ax.set_xlim(3, 17)
ax.set_ylim(-8, 112)
ax.set_yticks([0, 50, 100])
ax.set_yticklabels(["0%", "50%", "100%"])
ax.set_xlabel("energy given to the sealed system at the start", color=INK2)
ax.set_title("1  How big a spark does it need?", loc="left", fontsize=12.5, fontweight="bold", color=INK, pad=22)
ax.text(0, 1.02, "Sealed: nothing can get in or out.", transform=ax.transAxes, fontsize=10.5, color=INK2, va="bottom")
ax.text(3.35, 90, "Below 12: 0/8 escapes;\nat 12 and above: 8/8.\n\n"
        "All five sizes gave exactly this,\nfrom 48 to 192 points, 320 runs.\nThe five curves lie on top of one\n"
        "another, so one is drawn.\n\n"
        "12,000 sweeps; global seed energy.\nEscape does not certify conversion\nor localized energy delivery.",
        fontsize=10.5, color=INK, va="top")

# ---- 2. how long an open tube waits ---------------------------------------------------------
ax = axes[1]
runs = defaultdict(list)
for r in wait_rows:
    runs[(int(r["N"]), r["replica"])].append(r)
xs, means, errs = [], [], []
for n in sorted({int(r["N"]) for r in wait_rows}):
    waits = []
    for (nn, rep), rs in runs.items():
        if nn == n:
            w = next((int(rr["sweep"]) for rr in rs
                      if float(rr["phi"]) != 1.25 or float(rr["surplus"]) != 1.0), None)
            if w:
                waits.append(w)
    xs.append(n)
    means.append(st.mean(waits))
    errs.append(st.stdev(waits) / math.sqrt(len(waits)))
pred = [means[0] * xs[0] / n for n in xs]
ax.errorbar(xs, means, yerr=errs, color=INK, lw=2.4, marker="o", ms=8, mec=SURFACE, mew=1.2, capsize=5,
            zorder=4, label="what the runs did (12 runs a point)")
ax.plot(xs, pred, color=ORANGE, lw=2.2, ls=(0, (5, 3)), marker="s", ms=7, mec=SURFACE, mew=1.2, zorder=3,
        label="what we predicted beforehand")
ax.set_xlim(48, 215)
ax.set_ylim(0, 2150)
ax.set_xticks(xs)
ax.set_xlabel("size of the system (points)", color=INK2)
ax.set_ylabel("sweeps to first recorded change in (S,X)", color=INK2)
ax.set_title("2  A prediction of ours, not supported", loc="left", fontsize=12.5, fontweight="bold", color=INK, pad=22)
ax.text(0, 1.02, "Open: the energy given off is carried away.", transform=ax.transAxes, fontsize=10.5, color=INK2,
        va="bottom")
ax.annotate("we said a bigger system should convert\nsooner: more places for it to start",
            xy=(192, pred[-1]), xytext=(103, 250), color=ORANGE, fontsize=10, va="bottom",
            arrowprops=dict(arrowstyle="-", color=ORANGE, lw=1))
ax.text(52, 2090, "It does not. The waiting time is flat.\nAt the biggest size the prediction is out by three\n"
        "standard errors, which is suggestive rather than\nsettled with only 12 runs at each size.",
        fontsize=10.5, color=INK, va="top")
leg = ax.legend(loc="upper right", frameon=True, fontsize=9.5, labelcolor=INK, borderpad=0.7,
                bbox_to_anchor=(1.0, 0.70))
leg.get_frame().set_facecolor(SURFACE)
leg.get_frame().set_edgecolor(GRID)

# ---- 3. where a sealed tube ends up ---------------------------------------------------------
ax = axes[2]
ends = defaultdict(list)
for r in spark_rows:
    if int(r["N"]) == 64:
        ends[float(r["spark"])].append(float(r["energy_end"]))
xs = sorted(ends)
for level, label in ((TUBE, "the tube it started as"), (0.0, "a perfect flat sheet")):
    ax.plot([3, 17], [level, level], color=INK2, lw=1, ls=(0, (2, 3)), zorder=1)
    ax.text(16.8, level + 0.05, label, ha="right", fontsize=9.5, color=INK2)
ax.plot([3, 17], [TUBE + 3 / 64, TUBE + 17 / 64], color=ORANGE, lw=1.8, zorder=2,
        label="everything the system was given")
ax.text(16.8, TUBE + 17 / 64 + 0.06, "all the energy it was given", ha="right", fontsize=9.5, color=ORANGE)
for x in xs:
    ax.scatter([x] * len(ends[x]), ends[x], s=55, color=BLUE, edgecolors=SURFACE, linewidths=1.2, zorder=4)
ax.plot(xs, [st.mean(ends[x]) for x in xs], color=INK, lw=2, zorder=3)
ax.axvline(12, color=ORANGE, lw=2, ls=(0, (4, 3)), zorder=1)
ax.set_xlim(3, 17)
ax.set_ylim(-0.3, 2.75)
ax.set_xlabel("energy given to the sealed system at the start", color=INK2)
ax.set_ylabel("energy per point left in the network at the end", color=INK2)
ax.set_title("3  Graph energy at the recording limit", loc="left", fontsize=12.5, fontweight="bold", color=INK, pad=22)
ax.text(0, 1.02, "64 points, eight runs at each spark.", transform=ax.transAxes, fontsize=10.5, color=INK2, va="bottom")
ax.annotate("under 12: unchanged\n(S,X), neutral moves allowed", xy=(8, TUBE), xytext=(3.4, 0.45), color=INK, fontsize=10.5,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=1))
ax.text(3.35, 2.72, "At 12 and above: 8/8 escape the initial counts.\n"
        "Graph plus store energy is conserved.\n"
        "Final energy alone does not certify geometry,\n"
        "equilibration or a unique final state.",
        color=INK, fontsize=10.5, va="top", ha="left")

fig.suptitle("Three things the sealed runs say about a curled-up dimension opening out",
             x=0.028, y=0.985, ha="left", fontsize=15, color=INK, fontweight="bold")
fig.text(0.028, 0.012, "All at λ = 1.25, starting from a perfect tube (one large dimension), which sits exactly 1 unit of "
         "energy per point above the flat sheet (two large dimensions).\nExploratory runs of this project; not yet "
         "findings. Panels 1 and 3 share their runs.", fontsize=9.5, color=INK2, ha="left", va="bottom")
fig.tight_layout(rect=(0, 0.045, 1, 0.935))
fig.savefig(out, facecolor=SURFACE)
print("saved", out)
