"""T13 at one size: the author's protocol beside replica exchange. Usage:

    pip install -e ".[plots]"          # matplotlib is needed by the plotting scripts only
    python scripts/plot_t13_two_protocols.py 196 docs/figures/t13_n196_two_protocols.png

Reads committed result files only: results/t13_seq_n<N>.csv (protocol P, his protocol as he
described it: single chains, cold descent from a melt then cold ascent from the lattice torus, each
coupling starting from the last) and results/t13_temper_n<N>_melt.csv (protocol E, parallel
tempering from a melt). One line per replica; nothing is averaged across replicas. Made for the
reply to the model's author (docs/outreach/reply_draft_2026-09-22.md), so the words are his:
lattice torus, replica exchange.
"""
import csv
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

n, out = int(sys.argv[1]), sys.argv[2]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, BLACK = "#2a78d6", "#eb6834", "#1a1a1a"


def by_replica(path, key, want=None):
    d = defaultdict(list)
    for r in csv.DictReader(open(path, newline="")):
        if int(r["N"]) == n and (want is None or r[key] == want):
            d[r["replica"]].append((float(r["g"]), float(r["phi"])))
    return {k: sorted(v) for k, v in d.items()}


p_cool = by_replica("results/t13_seq_n%d.csv" % n, "leg", "cool")
p_heat = by_replica("results/t13_seq_n%d.csv" % n, "leg", "heat")
e_melt = by_replica("results/t13_temper_n%d_melt.csv" % n, "replica")
trips = sorted({int(r["round_trips"]) for r in csv.DictReader(open("results/t13_temper_n%d_melt.csv" % n, newline=""))})

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, ax = plt.subplots(figsize=(10.5, 6.4), dpi=160, facecolor=SURFACE)
ax.set_facecolor(SURFACE)
for rep, pts in sorted(p_cool.items()):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=BLUE, lw=2.2, alpha=0.85, marker="o", ms=3.5, mec=SURFACE, mew=0.6, zorder=2,
            label="continuation protocol, cooling from a random graph (%d chains)" % len(p_cool) if rep == "0" else None)
for rep, pts in sorted(p_heat.items()):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=ORANGE, lw=1.8, alpha=0.9, marker="o", ms=3.5, mec=SURFACE, mew=0.6, zorder=3,
            label="continuation protocol, heating from the lattice torus (%d chains)" % len(p_heat) if rep == "0" else None)
for rep, pts in sorted(e_melt.items()):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=BLACK, lw=0, marker="s", ms=6.5, mfc="none", mec=BLACK, mew=1.3, zorder=4,
            label="replica exchange from a random graph (%d replicas; %d to %d round trips per run)"
                  % (len(e_melt), trips[0], trips[-1]) if rep == "0" else None)

ax.set_xscale("log")
ax.set_xlim(1.4, 13)
ax.set_ylim(0.2, 1.08)
ax.set_xticks([1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12])
ax.set_xticklabels(["1.5", "2", "2.5", "3", "4", "5", "6", "8", "10", "12"])
ax.minorticks_off()
ax.grid(True, which="major", color=GRID, lw=1)
ax.tick_params(colors=INK2, length=0)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.axhline(1.0, color=INK2, lw=1, ls=(0, (2, 3)), zorder=1)
ax.text(12.8, 1.005, "lattice torus = 1", color=INK2, fontsize=9.5, ha="right", va="bottom")
ax.set_xlabel("coupling g  (cold on the left, hot on the right; log scale)", color=INK2)
ax.set_ylabel("squares per vertex, S/N", color=INK2)
ax.annotate("heating: the lattice survives past the transition,\nthen collapses between g ≈ 3.5 and 2.7",
            xy=(3.1, 0.93), xytext=(4.6, 0.86), color=INK, fontsize=10.5,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=1))
ax.annotate("cooling, and replica exchange:\none smooth curve, no jump",
            xy=(2.35, 0.905), xytext=(1.5, 0.62), color=INK, fontsize=10.5,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=1))
leg = ax.legend(loc="lower left", frameon=True, fontsize=9.5, labelcolor=INK, borderpad=0.8)
leg.get_frame().set_facecolor(SURFACE)
leg.get_frame().set_edgecolor(GRID)
ax.set_title("N = %d, full Hamiltonian, hard-core rule only: continuation beside replica exchange" % n,
             loc="left", fontsize=13, color=INK, fontweight="bold", pad=14)
fig.text(0.04, 0.012,
         "Continuation protocol: 40 couplings from 12 to 1.5, each starting from the previous coupling's final graph,\n"
         "240 sweeps of warm-up and 10,000 measured sweeps per coupling. Replica exchange: 20 couplings from 9 to 2.2,\n"
         "20,000 equilibration + 100,000 measurement sweeps, swap rate 0.20 to 0.58.\n"
         "Files: results/t13_seq_n%d.csv, results/t13_temper_n%d_melt.csv; "
         "seeds in configs/." % (n, n),
         fontsize=9, color=INK2, ha="left", va="bottom")
fig.tight_layout(rect=(0, 0.09, 1, 1))
fig.savefig(out, facecolor=SURFACE)
print("saved", out)
