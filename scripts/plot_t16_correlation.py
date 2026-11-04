"""T16 for the model's author: correlations, susceptibility, and the Fig. 9a correlation length. Usage:

    python scripts/plot_t16_correlation.py docs/figures/t16_correlation.png

Reads committed result files only (results/t16_*). Three panels:
  left    C(r) against graph distance r at several couplings (his procedure, cooling, N = 676, and
          replica exchange at N = 196), replica-averaged; the curves are what xi is computed from.
  middle  the susceptibility N var(S/N) against g, his procedure at N = 196, 484, 676 and replica
          exchange at N = 196, at the couplings where each is read (PREREGISTRATION T16).
  right   xi / diameter by Eq. (4.15) of the 2019 paper, same curves, same couplings, with errors.
Words are his: replica exchange, coupling g.
"""
import csv
import glob
import math
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "scripts")
from analyse_t16 import pooled, read_couplings_p          # noqa: E402

out = sys.argv[1]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"   # validated, categorical slots 1-4


def load(pattern):
    rows = []
    for f in sorted(glob.glob(pattern)):
        rows += list(csv.DictReader(open(f, newline="")))
    return rows


def c_of_r(tag, n, leg, k):
    cs = [np.load(f)["c_mean"] for f in sorted(glob.glob("results/t16_%s_rep*_curves/N%d_%sk%d.npz" % (tag, n, leg, k)))]
    g = float(np.load(glob.glob("results/t16_%s_rep0_curves/N%d_%sk%d.npz" % (tag, n, leg, k))[0])["g"])
    return g, np.nanmean(np.array(cs), axis=0)


series = []
for n, color, marker in ((196, BLUE, "o"), (484, ORANGE, "s"), (676, AQUA, "D")):
    rows = load("results/t16_seq_n%d_rep*.csv" % n)
    gs = read_couplings_p(rows)
    every = sorted({float(r["g"]) for r in rows}, reverse=True)
    series.append(("his procedure, N = %d" % n, pooled([r for r in rows if float(r["g"]) in gs]), gs, color, marker,
                   every))
rows = load("results/t16_temper_n196_rep*.csv")
curve = pooled(rows)
series.append(("replica exchange, N = 196", curve, sorted(curve, reverse=True), YELLOW, "^", sorted(curve, reverse=True)))


def gapped(cv, gs, every, key):
    """Values at every coupling, NaN where the curve is not read, so the line breaks at the gap."""
    return every, [cv[g][key] if g in gs else math.nan for g in every]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5})
fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), dpi=160, facecolor=SURFACE)
for ax in axes:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=1)
    ax.tick_params(colors=INK2, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)

ax = axes[0]
shades = ["#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]       # one-hue ramp, hot to cold
for (k, shade) in zip((0, 10, 18, 22, 26), shades):
    g, c = c_of_r("seq_n676", 676, "cool_", k)
    ax.plot(range(1, 9), c[1:9], color=shade, lw=2, marker="o", ms=4, label="g = %.1f" % g)
ax.axhline(0, color=INK2, lw=1)
ax.set_xlabel("graph distance r", color=INK2)
ax.set_ylabel("C(r), Eq. (4.14) of the 2019 paper", color=INK2)
ax.set_title("Correlations reach one or two steps,\nat every coupling (his procedure, N = 676)", color=INK, fontsize=11)
leg = ax.legend(fontsize=9, frameon=False, labelcolor=INK)

ax = axes[1]
for label, cv, gs, color, marker, every in series:
    xs, ys = gapped(cv, gs, every, "chi")
    ax.plot(xs, ys, color=color, lw=1.8, marker=marker, ms=4.5, mec=color, label=label)
ax.set_xscale("log")
ax.set_xticks([1.5, 2, 3, 4, 6, 8, 12])
ax.set_xticklabels(["1.5", "2", "3", "4", "6", "8", "12"])
ax.minorticks_off()
ax.set_ylim(0, 0.22)
ax.set_xlabel("coupling g  (log scale)", color=INK2)
ax.set_ylabel("susceptibility  N var(S/N)", color=INK2)
ax.set_title("The susceptibility peaks at about 0.18\nat every size: it does not grow with N", color=INK, fontsize=11)

ax = axes[2]
for label, cv, gs, color, marker, every in series:
    xs, ys = gapped(cv, gs, every, "xi")
    _, es = gapped(cv, gs, every, "xi_err")
    ax.errorbar(xs, ys, yerr=es, color=color, lw=1.4, marker=marker, ms=4.5, mec=color, capsize=2, label=label)
ax.set_xscale("log")
ax.set_xticks([1.5, 2, 3, 4, 6, 8, 12])
ax.set_xticklabels(["1.5", "2", "3", "4", "6", "8", "12"])
ax.minorticks_off()
ax.set_ylim(0, 2.0)
ax.text(1.62, 1.93, "N = 676 at g = 1.58: 1.17 ± 0.84,\nhalf its snapshots skipped (lattice)", color=INK2,
        fontsize=8.5, va="top")
ax.set_xlabel("coupling g  (log scale)", color=INK2)
ax.set_ylabel("ξ / diameter, Eq. (4.15) of the 2019 paper", color=INK2)
ax.set_title("ξ / diameter as defined: spikes from single couplings,\nno trend with N", color=INK, fontsize=11)

handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=10, frameon=False, labelcolor=INK)
fig.suptitle("Full Hamiltonian, hard-core rule only. Shown only at couplings where cooling and heating agree "
             "(his procedure) or replica exchange passes its checks", color=INK, fontsize=12, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0.07, 1, 0.94))
fig.savefig(out, facecolor=SURFACE)
print("wrote", out)
