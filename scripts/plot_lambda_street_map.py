"""Labelled diagram: the street-map picture of the energy and of the knob lambda. Usage:

    pip install -e ".[plots]"          # matplotlib is needed by the plotting scripts only
    python scripts/plot_lambda_street_map.py docs/figures/lambda_street_map.png

A drawing, not a plot of data. The statements in panel 4 come from the exhaustive
results at N = 18: results/dip_census_fine.csv (the window 1 < lambda < 1.6) and
scripts/exact_small_averages.py (folds fading out above it). If those change, change
the text here. Used in docs/design/ (and formerly in paper/introduction_plain_language.md, removed 2026-09-24).
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

out = sys.argv[1]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#c9c8c3"
BLUE, ORANGE = "#2a78d6", "#eb6834"
BLUE_TINT, ORANGE_TINT = "#d5e4f7", "#fbdccd"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig = plt.figure(figsize=(13.5, 8.6), dpi=160, facecolor=SURFACE)
fig.suptitle("The energy as a street map, and what the knob λ does", x=0.04, y=0.975, ha="left",
             fontsize=15, fontweight="bold", color=INK)


def panel(rect, title, number):
    ax = fig.add_axes(rect)
    ax.set_facecolor(SURFACE)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.text(rect[0], 0.895, f"{number}  {title}", fontsize=12.5, fontweight="bold", color=INK, ha="left", va="bottom")
    return ax


def street(ax, p, q, **kw):
    style = dict(color=INK2, lw=2.2, solid_capstyle="round", zorder=3)
    style.update(kw)
    ax.plot([p[0], q[0]], [p[1], q[1]], **style)


def corners(ax, pts, **kw):
    style = dict(s=42, color=INK, zorder=5, edgecolors=SURFACE, linewidths=1.2)
    style.update(kw)
    ax.scatter([p[0] for p in pts], [p[1] for p in pts], **style)


def note(ax, text, xy, xytext, ha="left", va="center"):
    ax.annotate(text, xy=xy, xytext=xytext, ha=ha, va=va, fontsize=10.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=1, shrinkA=2, shrinkB=3))


# ---- 1. one block -------------------------------------------------------------------------
ax = panel([0.04, 0.45, 0.24, 0.43], "A block", "1")
sq = [(0, 0), (1, 0), (1, 1), (0, 1)]
ax.add_patch(Polygon(sq, closed=True, facecolor=BLUE_TINT, edgecolor="none", zorder=1))
for i in range(4):
    street(ax, sq[i], sq[(i + 1) % 4])
corners(ax, sq)
ax.text(0.5, 0.5, "block\n(a square)\nearns 16", ha="center", va="center", fontsize=11, color=INK)
note(ax, "street\n(an edge)", (1.0, 0.5), (1.32, 0.5))
note(ax, "crossing\n(a vertex)", (0, 1), (-0.12, 1.28), ha="center", va="bottom")
ax.text(0.5, -0.16, "four streets closing a loop", ha="center", va="top", fontsize=10.5, color=INK2)
ax.set_xlim(-0.6, 1.95)
ax.set_ylim(-0.42, 1.62)

# ---- 2. flat grid -------------------------------------------------------------------------
ax = panel([0.30, 0.45, 0.28, 0.43], "Flat map: two blocks per street", "2")
for bx in (1, 2):
    ax.add_patch(Polygon([(bx, 1), (bx + 1, 1), (bx + 1, 2), (bx, 2)], closed=True, facecolor=BLUE_TINT,
                         edgecolor="none", zorder=1))
for x in range(5):
    street(ax, (x, 0), (x, 3), color=GRID, lw=1.6, zorder=2)
for y in range(4):
    street(ax, (0, y), (4, y), color=GRID, lw=1.6, zorder=2)
for bx in (1, 2):
    for p, q in (((bx, 1), (bx + 1, 1)), ((bx, 2), (bx + 1, 2)), ((bx, 1), (bx, 2)), ((bx + 1, 1), (bx + 1, 2))):
        street(ax, p, q)
street(ax, (2, 1), (2, 2), color=INK, lw=5)
corners(ax, [(x, y) for x in (1, 2, 3) for y in (1, 2)])
ax.text(1.5, 1.5, "block 1", ha="center", va="center", fontsize=11, color=INK)
ax.text(2.5, 1.5, "block 2", ha="center", va="center", fontsize=11, color=INK)
note(ax, "this stretch of street has\none block on each side:\n2 blocks, no fine", (2, 2.0), (2.0, 3.45), ha="center", va="bottom")
ax.set_xlim(-0.3, 4.3)
ax.set_ylim(-0.4, 4.6)

# ---- 3. the third block --------------------------------------------------------------------
ax = panel([0.60, 0.45, 0.39, 0.43], "A third block on one street: the map folds", "3")


def flat(x, y, z=0.0):
    """Oblique view: x to the right, y into the page, z up."""
    return (x + 0.62 * y, z + 0.5 * y)


for gx in range(-2, 3):
    street(ax, flat(gx, -0.5), flat(gx, 1.5), color=GRID, lw=1.4, zorder=1)
for gy in (0, 1):
    street(ax, flat(-2, gy), flat(2, gy), color=GRID, lw=1.4, zorder=1)
left = [flat(-1, 0), flat(0, 0), flat(0, 1), flat(-1, 1)]
right = [flat(0, 0), flat(1, 0), flat(1, 1), flat(0, 1)]
up = [flat(0, 0), flat(0, 1), flat(0, 1, 1), flat(0, 0, 1)]
for poly in (left, right):
    ax.add_patch(Polygon(poly, closed=True, facecolor=BLUE_TINT, edgecolor="none", zorder=2))
    for i in range(4):
        street(ax, poly[i], poly[(i + 1) % 4])
ax.add_patch(Polygon(up, closed=True, facecolor=ORANGE_TINT, edgecolor="none", zorder=4, alpha=0.95))
for i in range(4):
    street(ax, up[i], up[(i + 1) % 4], color=ORANGE, lw=2.4, zorder=5)
street(ax, flat(0, 0), flat(0, 1), color=INK, lw=5, zorder=6)
corners(ax, left + right + up, zorder=7)
ax.text(*flat(-0.7, 0.45), "1", ha="center", va="center", fontsize=13, fontweight="bold", color=INK, zorder=8)
ax.text(*flat(0.72, 0.4), "2", ha="center", va="center", fontsize=13, fontweight="bold", color=INK, zorder=8)
ax.text(*flat(0, 0.5, 0.5), "3", ha="center", va="center", fontsize=13, fontweight="bold", color=INK, zorder=8)
note(ax, "block 3 cannot lie flat, so it\nstands up out of the map.\nIt still earns 16, but is\nfined 4λ for this street", flat(0, 0.8, 0.85), (1.45, 1.5), ha="left")
note(ax, "the same stretch of street,\nnow with 3 blocks on it", flat(0, 0.1, 0.0), (-0.9, -0.42), ha="center", va="top")
ax.text(*flat(1.15, 0.5), "blocks 1 and 2\nlie flat, as before", ha="left", va="center", fontsize=10.5, color=INK, zorder=8)
ax.set_xlim(-1.95, 3.55)
ax.set_ylim(-1.0, 1.85)

# ---- the knob ------------------------------------------------------------------------------
ax = fig.add_axes([0.04, 0.05, 0.94, 0.34])
ax.set_facecolor(SURFACE)
ax.axis("off")
ax.set_xlim(0, 100)
ax.set_ylim(0, 40)
ax.text(0, 40, "4  The knob λ sets the fine.  What the cold network becomes:", fontsize=12.5, fontweight="bold",
        color=INK, va="top")
ticks = {"0": 6, "1": 34, "about 1.6": 50, "2": 58, "5 to 10": 78, "very large": 95}
y0 = 17
bands = [(6, 34, ORANGE_TINT, "λ = 0 is no fine at all. Below 1, folding pays:\ntight closed knots (4-cubes) are lowest,\nand the network falls into pieces"),
         (34, 50, BLUE_TINT, "flat sheet lowest, and\nsome folded arrangements\nare dips above it\n(exact, at N = 18)"),
         (50, 95, BLUE_TINT, "flat sheet lowest; no dips found above it (N = 18).\nFolds just get rarer: by λ ≈ 5 to 10 the model\nbehaves as if folding were forbidden")]
for a, b, colour, text in bands:
    ax.add_patch(FancyBboxPatch((a + 0.35, y0 - 7.5), b - a - 0.7, 15, boxstyle="round,pad=0,rounding_size=1.2",
                                facecolor=colour, edgecolor="none", zorder=1))
    ax.text((a + b) / 2, y0, text, ha="center", va="center", fontsize=10, color=INK, zorder=3)
ax.add_patch(FancyBboxPatch((34.35, y0 - 7.5), 15.3, 15, boxstyle="round,pad=0,rounding_size=1.2", facecolor="none",
                            edgecolor=ORANGE, lw=2, zorder=2))
ax.plot([6, 95], [y0 - 10.5, y0 - 10.5], color=INK2, lw=1.5, zorder=2)
for label, x in ticks.items():
    ax.plot([x, x], [y0 - 11.3, y0 - 9.7], color=INK2, lw=1.5)
    ax.text(x, y0 - 12.3, ("λ = " if label == "0" else "") + label, ha="center", va="top", fontsize=10.5, color=INK)
ax.annotate("at λ = 1 the fine equals the reward: knots, tubes and\nthe flat sheet all tie. This is the true curvature formula", xy=(34, y0 + 7.5),
            xytext=(33.4, y0 + 10), ha="right", va="bottom", fontsize=10, color=INK,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=1))
ax.annotate("folding forbidden:\nthe \"cap\" of the 2019 paper", xy=(95, y0 + 7.5), xytext=(95, y0 + 10), ha="right",
            va="bottom", fontsize=10, color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=1))
ax.text(42, y0 + 8.6, "candidates for X", ha="center", va="bottom", fontsize=10.5, color=INK, fontweight="bold")
fig.text(0.98, 0.012, "Scale not even. Statements about dips are exact for 18 vertices only; nothing is known yet for large networks.",
         ha="right", va="bottom", fontsize=9.5, color=INK2)
fig.savefig(out, facecolor=SURFACE)
print("saved", out)
