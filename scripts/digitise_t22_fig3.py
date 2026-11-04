"""Read the data points off [T22] Fig. 3 (D = 3, N = 500), the Gate C target (ASSUMPTIONS Q21). Usage:

    pip install pillow scipy          # needed by this script only
    curl -L -o squares3.png https://arxiv.org/html/2112.03778v2/squares3.png
    python scripts/digitise_t22_fig3.py squares3.png docs/published/T22_fig3_digitised.csv

The image is the authors' and is not kept in this repository; the numbers read from it are. Method, as
digitise_t25_fig3.py: find the plot frame, find the major tick marks inside it, fit pixel -> value through the
tick labels and the frame edges (x: -1 at the left edge, -0.5 ... 1.0 inside, 1.5 at the right edge; y: 0 at the
bottom, 2 ... 10 inside, 12 at the top), then take the centre of every blue marker.

The axis label is "log(hbar g)" with no base. It is the natural log: the markers sit at ln 0.40, ln 0.45, ...,
ln 4.00, the couplings 0.40 to 4.00 in steps of 0.05 (their spacing shrinks exactly as ln of evenly spaced values
does; checked when this script runs). The y axis is squares per vertex counted by incidence, 4S/N, whose maximum
on the cubic lattice is 12, as the caption says. So S/N = y/4, and the order parameter phi = S/(3N) = y/12.
"""
import sys

import numpy as np
from PIL import Image
from scipy import ndimage, signal


def frame(dark):
    rows, cols = dark.sum(axis=1), dark.sum(axis=0)
    fr = np.flatnonzero(rows > 0.6 * rows.max())
    fc = np.flatnonzero(cols > 0.6 * cols.max())
    return fr.min(), fr.max(), fc.min(), fc.max()


def interior_ticks(dark, top, bottom, left, right, axis):
    """Centres of the tick marks pointing inward from the bottom (x) or the left (y) frame line."""
    h, w = dark.shape
    n_along = w if axis == "x" else h
    length = np.zeros(n_along, dtype=int)
    for p in range(n_along):
        n = 0
        while True:
            y, x = (bottom - 1 - n, p) if axis == "x" else (p, left + 1 + n)
            if not (0 <= y < h and 0 <= x < w) or not dark[y, x]:
                break
            n += 1
        length[p] = n
    lo, hi = (left + 8, right - 8) if axis == "x" else (top + 8, bottom - 8)
    inside = np.zeros(n_along, dtype=bool)
    inside[lo:hi] = True
    labelled, k = ndimage.label(inside & (length >= 4))
    return [float(np.mean(np.flatnonzero(labelled == i + 1))) for i in range(k)]


def main(image_path, out_path):
    im = np.asarray(Image.open(image_path).convert("RGB")).astype(int)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    dark = (r < 170) & (g < 170) & (b < 170)          # the frame is black on the left, grey on the right
    top, bottom, left, right = frame(dark)
    xt = interior_ticks(dark, top, bottom, left, right, "x")
    yt = interior_ticks(dark, top, bottom, left, right, "y")
    if len(xt) != 4 or len(yt) != 5:
        raise RuntimeError(f"expected 4 and 5 interior ticks, found {len(xt)} and {len(yt)}")
    cx = np.polyfit([left] + xt + [right], [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5], 1)
    cy = np.polyfit([bottom] + sorted(yt, reverse=True) + [top], [0, 2, 4, 6, 8, 10, 12], 1)
    resid_x = np.polyval(cx, [left] + xt + [right]) - np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
    row_col_inside = np.zeros_like(b, dtype=bool)
    row_col_inside[top + 2:bottom - 2, left + 2:right - 2] = True
    blue = ((b > 150) & (b - r > 60) & row_col_inside).astype(float)
    # Where the curve is steep the markers touch, so a component can hold several. Use the first marker, which
    # stands alone, as a template, and take every place it fits: peaks of the template match, at least 5 px apart.
    labelled, k = ndimage.label(blue > 0)
    sizes = ndimage.sum(blue, labelled, range(1, k + 1))
    first = min((i for i in range(1, k + 1) if sizes[i - 1] >= 15),
                key=lambda i: np.nonzero(labelled == i)[1].mean())
    ys, xs = np.nonzero(labelled == first)
    tmpl = blue[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    match = signal.fftconvolve(blue, tmpl[::-1, ::-1], mode="same")
    cand = np.argwhere((match == ndimage.maximum_filter(match, size=3)) & (match >= 0.6 * tmpl.sum()))
    cand = sorted(cand.tolist(), key=lambda c: -match[c[0], c[1]])
    kept = []                                    # non-maximum suppression: markers are about 12 px across
    for py, px in cand:
        if all((py - qy) ** 2 + (px - qx) ** 2 > 7 ** 2 for qy, qx in kept):
            kept.append((py, px))
    pts = sorted((float(np.polyval(cx, px)), float(np.polyval(cy, py))) for py, px in kept)
    with open(out_path, "w", newline="\n") as fh:
        fh.write("ln_hbar_g,hbar_g,squares_per_vertex,S_over_N,phi\n")
        for x, y in pts:
            fh.write(f"{x:.4f},{np.exp(x):.4f},{y:.3f},{y / 4:.4f},{y / 12:.4f}\n")
    print(f"{len(pts)} points -> {out_path}; x fit residual {np.max(np.abs(resid_x)):.4f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
