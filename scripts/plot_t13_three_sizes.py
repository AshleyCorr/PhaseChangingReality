"""T13 at N = 196, 484 and 676: the author's protocol beside replica exchange from both starts. Usage:

    python scripts/plot_t13_three_sizes.py docs/figures/t13_three_sizes.png

Reads committed result files only (results/t13_seq_n<N>.csv, results/t13_temper_n<N>_{melt,torus}.csv).
One line per replica; nothing is averaged. Unlike t13_n196_two_protocols.png it also draws replica
exchange started from the lattice torus, which never converged, so that the window where the two
starts disagree is visible rather than left out. Made for the second reply to the model's author,
so the words are his: lattice torus, replica exchange.
"""
import csv
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = sys.argv[1]
SIZES = (196, 484, 676)
SURFACE, INK, INK2, GRID, SHADE = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#f1f0ec"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"   # validated, categorical slots 1-4


def by_replica(path, leg=None):
    d, trips = defaultdict(list), set()
    for r in csv.DictReader(open(path, newline="")):
        if leg is None or r["leg"] == leg:
            d[r["replica"]].append((float(r["g"]), float(r["phi"])))
            if r.get("round_trips", "") != "":
                trips.add(int(r["round_trips"]))
    return {k: sorted(v) for k, v in d.items()}, trips


plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5})
fig, axes = plt.subplots(1, 3, figsize=(15, 5.6), dpi=160, facecolor=SURFACE, sharey=True)
for ax, n in zip(axes, SIZES):
    ax.set_facecolor(SURFACE)
    ax.axvspan(1.4, 3.2, color=SHADE, zorder=0)
    ax.text(1.47, 0.40, "below g ≈ 3.2: the two\nreplica-exchange starts\nnever meet", color=INK2, fontsize=9,
            va="bottom")
    series = [
        ("his protocol, cooling from a random graph", by_replica("results/t13_seq_n%d.csv" % n, "cool")[0],
         dict(color=BLUE, lw=2, marker="o", ms=3.2)),
        ("his protocol, heating from the lattice torus", by_replica("results/t13_seq_n%d.csv" % n, "heat")[0],
         dict(color=ORANGE, lw=1.8, marker="o", ms=3.2)),
    ]
    melt, t_melt = by_replica("results/t13_temper_n%d_melt.csv" % n)
    tor, t_tor = by_replica("results/t13_temper_n%d_torus.csv" % n)
    series.append(("replica exchange, every copy started random", melt,
                   dict(color=AQUA, lw=0, marker="s", ms=6, mfc="none", mew=1.4)))
    series.append(("replica exchange, every copy started as the lattice torus", tor,
                   dict(color=YELLOW, lw=0, marker="^", ms=6.5, mfc="none", mew=1.4)))
    for label, reps, style in series:
        for i, (rep, pts) in enumerate(sorted(reps.items())):
            xs, ys = zip(*pts)
            ax.plot(xs, ys, alpha=0.9, mec=style["color"], zorder=3, label=label if i == 0 else None,
                    **{k: v for k, v in style.items()})
    ax.set_xscale("log")
    ax.set_xlim(1.4, 13)
    ax.set_ylim(0.1, 1.08)
    ax.set_xticks([1.5, 2, 3, 4, 6, 8, 12])
    ax.set_xticklabels(["1.5", "2", "3", "4", "6", "8", "12"])
    ax.minorticks_off()
    ax.grid(True, color=GRID, lw=1)
    ax.tick_params(colors=INK2, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_title("N = %d  (p = %d)\nround trips: random start %s, lattice start %s"
                 % (n, int(round((n / 4) ** 0.5)), "–".join(str(t) for t in sorted(t_melt)[::max(1, len(t_melt) - 1)]),
                    "/".join(str(t) for t in sorted(t_tor))), color=INK, fontsize=11)
    ax.set_xlabel("coupling g  (log scale)", color=INK2)
axes[0].set_ylabel("squares per vertex, S/N", color=INK2)
fig.suptitle("Full Hamiltonian, hard-core rule only. Above g ≈ 3.2 all four agree to 0.002; "
             "below it, heating from the lattice holds on to a coupling that does not move with N",
             color=INK, fontsize=12, x=0.01, ha="left")
handles, labels = axes[0].get_legend_handles_labels()
leg = fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=10, frameon=False, labelcolor=INK)
fig.tight_layout(rect=(0, 0.1, 1, 0.94))
fig.savefig(out, facecolor=SURFACE)
print("wrote", out)
