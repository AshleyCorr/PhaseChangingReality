"""Read the N = 160 series off [KTB19] Fig. 8(a). Usage:

    pip install pillow scipy          # needed by this script only
    curl -L -o figure8a.png https://arxiv.org/html/1901.09870v2/figure8a.png
    python scripts/digitise_ktb19_fig8a.py figure8a.png docs/published/KTB19_fig8a_N160_digitised.csv

The image is the authors' and is not kept in this repository; the numbers read
from it are. Method: the plot frame is found from its long unbroken lines; the
x axis is calibrated on its six major ticks (-2 ... 3); the y axis on the frame
itself, which runs from phi = 0 at the bottom to phi = 1 at the top. N = 160 is
the fourth series, drawn in purple. Dots partly hidden behind other series are
dropped (fewer than 40 clean pixels), so a few couplings are missing.

The axis is log(G) with a NATURAL log: the recovered couplings come out as the
integers 1, 2, 4, 5, ... 19, which is the check that the calibration is right.
[T25] Fig. 3 uses base 10. Trust phi to about +/- 0.005.
"""
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

X_LABELS = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
PURPLE = (126, 47, 142)                                              # fourth default colour of the plotting program


def longest_run(flags):
    best = current = 0
    for f in flags:
        current = current + 1 if f else 0
        best = max(best, current)
    return best


def main(image_path, out_path):
    im = np.asarray(Image.open(image_path).convert("RGB")).astype(int)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    h, w = r.shape
    ink = (r < 140) & (g < 140) & (b < 140) & (np.abs(r - b) < 30)      # black and grey only
    col_run = np.array([longest_run(ink[:, x]) for x in range(w)])
    row_run = np.array([longest_run(ink[y, :]) for y in range(h)])
    verticals = np.flatnonzero(col_run > 0.6 * h)
    horizontals = np.flatnonzero(row_run > 0.6 * w)
    left, right, top, bottom = verticals.min(), verticals.max(), horizontals.min(), horizontals.max()
    if bottom - top < 0.5 * h:
        raise RuntimeError("could not find both the top and the bottom of the plot frame")

    length = np.zeros(w, dtype=int)                                      # tick marks rising from the bottom line
    for x in range(left + 8, right - 8):
        n = 0
        while n <= 40 and ink[bottom - 3 - n, x]:
            n += 1
        length[x] = n
    labelled, k = ndimage.label(length >= max(3, 0.6 * length.max()))
    ticks = [float(np.mean(np.flatnonzero(labelled == i + 1))) for i in range(k)]
    if len(ticks) != len(X_LABELS):
        raise RuntimeError(f"expected 6 major x ticks, found {len(ticks)}")
    cx = np.polyfit(ticks, X_LABELS, 1)

    mask = (np.abs(r - PURPLE[0]) < 35) & (np.abs(g - PURPLE[1]) < 35) & (np.abs(b - PURPLE[2]) < 35)
    mask[: top + int(0.45 * (bottom - top)), int(left + 0.8 * (right - left)):] = False   # the legend box
    labelled, k = ndimage.label(mask)
    points = []
    for i in range(1, k + 1):
        ys, xs = np.nonzero(labelled == i)
        if len(ys) >= 40:
            points.append((float(np.polyval(cx, xs.mean())), (bottom - ys.mean()) / (bottom - top)))
    points.sort()
    with open(out_path, "w", newline="\n") as fh:
        fh.write("series,ln_G,g,S_over_N\n")
        for ln_g, phi in points:
            fh.write(f"cool_N160,{ln_g:.3f},{np.exp(ln_g):.3f},{phi:.3f}\n")
    print(f"{len(points)} points -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
