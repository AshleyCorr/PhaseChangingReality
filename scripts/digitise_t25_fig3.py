"""Read the data points off [T25] Fig. 3. Usage:

    pip install pillow scipy          # needed by this script only
    curl -L -o Order_parameter.png https://arxiv.org/html/2512.17676v2/Order_parameter.png
    python scripts/digitise_t25_fig3.py Order_parameter.png docs/published/T25_fig3_digitised.csv

The image is the authors' and is not kept in this repository; the numbers read
from it are. Method: find the grey plot frame, find the major tick marks on both
axes, fit pixel -> value through the tick labels (-1.0 ... 2.0 and 0.0 ... 1.0),
then take the centre of every red and every blue dot. The fit residual is about
0.001 in both directions; a dot is about 0.02 wide in S/N, so trust +/- 0.005.

The axis label is "log[g]". It is base 10: the curve reaches the random-phase
floor at 2.25, i.e. g = 178. With a natural log that would be g = 9.5, where the
square density is still about 0.3.
"""
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

X_LABELS = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
Y_LABELS = np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])              # top to bottom


def main(image_path, out_path):
    im = np.asarray(Image.open(image_path).convert("RGB")).astype(int)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    h, w = r.shape
    grey = (np.abs(r - g) < 25) & (np.abs(g - b) < 25) & (r < 215)   # axes and ticks are grey ink
    rows, cols = grey.sum(axis=1), grey.sum(axis=0)
    frame_rows = np.flatnonzero(rows > 0.6 * rows.max())
    frame_cols = np.flatnonzero(cols > 0.6 * cols.max())
    top, bottom, left, right = frame_rows.min(), frame_rows.max(), frame_cols.min(), frame_cols.max()

    def major_ticks(axis):
        n_along = w if axis == "x" else h
        length = np.zeros(n_along, dtype=int)
        for p in range(n_along):                                 # walk inward from the frame line
            n = 0
            while True:
                y, x = (bottom - 4 - n, p) if axis == "x" else (p, left + 4 + n)
                if not (0 <= y < h and 0 <= x < w) or not grey[y, x]:
                    break
                n += 1
            length[p] = n
        lo, hi = (left + 6, right - 6) if axis == "x" else (top + 6, bottom - 6)
        inside = np.zeros(n_along, dtype=bool)
        inside[lo:hi] = True
        labelled, k = ndimage.label(inside & (length >= 0.75 * length[lo:hi].max()))
        return [float(np.mean(np.flatnonzero(labelled == i + 1))) for i in range(k)]

    xt, yt = major_ticks("x"), major_ticks("y")
    if len(xt) != len(X_LABELS) or len(yt) != len(Y_LABELS):
        raise RuntimeError(f"expected 7 and 6 major ticks, found {len(xt)} and {len(yt)}")
    cx, cy = np.polyfit(xt, X_LABELS, 1), np.polyfit(yt, Y_LABELS, 1)

    points = []
    for name, mask in [("heat_from_torus_red", (r > 180) & (g < 120) & (b < 120)),
                       ("cool_from_random_blue", (b > 180) & (r < 120) & (g < 120))]:
        labelled, k = ndimage.label(mask)
        for i in range(1, k + 1):
            ys, xs = np.nonzero(labelled == i)
            if len(ys) >= 30:                                    # a dot, not a stray pixel
                points.append((name, float(np.polyval(cx, xs.mean())), float(np.polyval(cy, ys.mean()))))
    points.sort(key=lambda p: (p[0], -p[1]))
    with open(out_path, "w", newline="\n") as fh:
        fh.write("series,log10_g,g,S_over_N\n")
        for name, log_g, phi in points:
            fh.write(f"{name},{log_g:.3f},{10**log_g:.3f},{phi:.3f}\n")
    print(f"{len(points)} points -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
