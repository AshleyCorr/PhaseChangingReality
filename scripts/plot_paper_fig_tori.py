"""Schematic figure of the curled-torus paper (docs/papers/curled_torus/). Usage:

    python scripts/plot_paper_fig_tori.py docs/papers/curled_torus/fig_tori.pdf

Drawn, not simulated. (a) the flat lattice torus, every vertex at local dimension d = 2; (b) the 4 x L
torus, whose short direction is a single square, drawn as a square tube, every vertex at d = 1; (c) a
schematic of the change under way: the curled order on one side of a front, the flat order on the
other. Panel (c) illustrates a possible interface, not a demonstrated unique front;
it is not a snapshot, and its shape is not read from any run.
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = sys.argv[1]
INK, INK2 = "#1c1e22", "#5d636b"
SHEET, TUBE = "#eb6834", "#2a78d6"          # validated categorical slots 2 and 1 (dataviz palette)
EDGE = "#9aa0a6"


def wrap_marks(ax, x0, x1, y0, y1):
    for x in (x0, x1):
        ax.annotate("", xy=(x, y1 + 0.35), xytext=(x, y1 + 0.05), arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8, ls=":"))


def flat(ax, n=6):
    for i in range(n):
        ax.plot([0, n - 1], [i, i], color=EDGE, lw=1, zorder=1)
        ax.plot([i, i], [0, n - 1], color=EDGE, lw=1, zorder=1)
        # wrap-around stubs: the torus closes both ways
        ax.plot([-0.45, 0], [i, i], color=EDGE, lw=1, ls=":")
        ax.plot([n - 1, n - 0.55], [i, i], color=EDGE, lw=1, ls=":")
        ax.plot([i, i], [-0.45, 0], color=EDGE, lw=1, ls=":")
        ax.plot([i, i], [n - 1, n - 0.55], color=EDGE, lw=1, ls=":")
    xs = [x for x in range(n) for _ in range(n)]
    ys = [y for _ in range(n) for y in range(n)]
    ax.scatter(xs, ys, s=22, color=SHEET, edgecolor="white", linewidth=0.6, zorder=3)
    ax.set_xlim(-0.8, n - 0.2)
    ax.set_ylim(-0.8, n - 0.2)


RING = [(0.0, 0.0), (0.0, 1.0), (0.42, 1.32), (0.42, 0.32)]   # one square cross-section, oblique view


def ring_at(x):
    return [(x + dx, dy) for dx, dy in RING]


def tube(ax, x_from, x_to, step=1.0, color=TUBE):
    cols = []
    x = x_from
    while x <= x_to + 1e-9:
        cols.append(ring_at(x))
        x += step
    for r in cols:                                   # each column: a 4-cycle, itself a square
        xs = [p[0] for p in r] + [r[0][0]]
        ys = [p[1] for p in r] + [r[0][1]]
        ax.plot(xs, ys, color=EDGE, lw=1, zorder=1)
    for a, b in zip(cols, cols[1:]):                 # along the tube
        for p, q in zip(a, b):
            ax.plot([p[0], q[0]], [p[1], q[1]], color=EDGE, lw=1, zorder=1)
    for r in cols:
        ax.scatter([p[0] for p in r], [p[1] for p in r], s=22, color=color, edgecolor="white", linewidth=0.6, zorder=3)
    return cols


def panel(ax, title):
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=9, loc="left", color=INK)


plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5})
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.25), gridspec_kw=dict(width_ratios=[1.0, 1.25, 1.6]))

ax = axes[0]
panel(ax, "(a) flat torus: d = 2, H = 0")
flat(ax)

ax = axes[1]
panel(ax, "(b) curled 4 × L torus: d = 1")
cols = tube(ax, 0.0, 5.0)
ax.plot([-0.5, 0.0], [0.5, 0.5], color=EDGE, lw=1, ls=":")
ax.plot([5.42, 5.95], [0.8, 0.8], color=EDGE, lw=1, ls=":")
ax.text(2.7, -0.75, "each column is one square;\nH = 4(λ − 1) per vertex", ha="center", va="top", fontsize=7.5, color=INK2)
ax.set_xlim(-0.8, 6.2)
ax.set_ylim(-1.9, 2.1)

ax = axes[2]
panel(ax, "(c) the change under way (schematic)")
tube(ax, 0.0, 3.0)
# the flat order beyond the front, six dots around (the width the clean runs reach), five columns shown
x0, y0, dy, ROWS, COLS = 4.4, -1.3, 0.72, 6, 5
for i in range(ROWS):
    y = y0 + dy * i
    ax.plot([x0, x0 + 0.8 * (COLS - 1)], [y, y], color=EDGE, lw=1, zorder=1)
for j in range(COLS):
    x = x0 + 0.8 * j
    ax.plot([x, x], [y0, y0 + dy * (ROWS - 1)], color=EDGE, lw=1, zorder=1)
ax.scatter([x0 + 0.8 * j for j in range(COLS) for _ in range(ROWS)], [y0 + dy * i for _ in range(COLS) for i in range(ROWS)],
           s=22, color=SHEET, edgecolor="white", linewidth=0.6, zorder=3)
for p in ring_at(3.0):                               # the front: last ring joined to the sheet's edge
    ax.plot([p[0], x0], [p[1], y0 + dy * round((p[1] - y0) / dy)], color=EDGE, lw=1, ls="--", zorder=1)
ax.plot([3.9, 3.9], [-1.9, 2.35], color=INK2, lw=0.8, ls=(0, (3, 3)))
ax.annotate("", xy=(2.7, 2.6), xytext=(3.9, 2.6), arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8))
ax.text(4.0, 2.6, "possible advancing boundary", ha="left", va="center", fontsize=7.5, color=INK2)
ax.text(1.6, -1.55, "curled (d = 1)", ha="center", fontsize=7.5, color=TUBE)
ax.text(6.0, -1.95, "flat (d = 2)", ha="center", fontsize=7.5, color=SHEET)
ax.set_xlim(-0.6, 7.9)
ax.set_ylim(-2.3, 2.9)

fig.tight_layout(w_pad=0.6)
fig.savefig(out, dpi=220 if out.endswith(".png") else None)
print("wrote", out)
