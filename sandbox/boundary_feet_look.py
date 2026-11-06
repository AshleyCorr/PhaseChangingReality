"""Sandbox look (5 November 2026, night, her words "One seed on the boundary with a foot on each side is strong if the
math works"): the loose-band tube of `loose_band_look.py` (4 x L, T37's cell, a band of L/8 columns loosened by k
switches), followed past the escape. At the escape and at fixed sweeps after it, the new sheet points (d = 2, not
made so by the loosening) are split into the foot inside the band and the foot outside it, and the connected pieces
the sheet points form (the loosened points included, since the feet grow from them) are counted, and whether the
two feet share a piece. Asked: does the seed have a foot on each side; are the feet joined across the boundary;
does the outside foot grow. No lengths, nothing but the wiring's density. A look, not a test."""
import sys

import numpy as np

from graphity.cqg import NO_CAP, run_chain, torus, total_squares
from graphity.dimension import local_dimension, piece_labels
from loose_band_look import loosen


def one(L, k, rep, lam=1.25, g=1.5, block=10, max_sweeps=20000, min_size=8, after=(0, 50, 200, 500), width=None):
    width = width or L // 8
    adj, part = torus(L, 4, NO_CAP)
    n = adj.shape[0]
    x_of = np.arange(n) // 4
    centre = L // 2
    in_band = np.abs(((x_of - centre + L // 2) % L) - L // 2) < width / 2
    rng = np.random.default_rng([20261105, L, k, rep])
    loosen(adj, part, rng, in_band, k)
    side_u = np.flatnonzero(part == 0)
    seed = int(np.random.SeedSequence([20261105, L, k, rep]).generate_state(1)[0])
    init = local_dimension(adj) == 2
    first, escape, rows = True, None, []
    for sweeps in range(0, max_sweeps, block):
        run_chain(adj, side_u, 1.0 / g, 0, block, seed if first else -1, lam=lam, cap=NO_CAP)
        first = False
        t = sweeps + block
        new_pts = (local_dimension(adj) == 2) & ~init
        if escape is None:
            if new_pts.sum() < min_size:
                continue
            escape = t
        dt = t - escape
        if dt in after:
            sheet = local_dimension(adj) == 2            # the seed with the loosened points it grew from
            labels = piece_labels(adj, sheet)
            lab_in = set(labels[new_pts & in_band]) - {-1}
            lab_out = set(labels[new_pts & ~in_band]) - {-1}
            rows.append(dict(dt=dt, inside=int((new_pts & in_band).sum()), outside=int((new_pts & ~in_band).sum()),
                             pieces=int(labels.max() + 1), joined=bool(lab_in & lab_out),
                             total_sheet=int(sheet.sum())))
        if dt >= max(after):
            break
    return escape, rows


if __name__ == "__main__":
    L = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    by_dt = {}
    for rep in range(reps):
        escape, rows = one(L, k, rep)
        print(f"L {L} k {k} rep {rep:2d}: escape {escape} | " + " | ".join(
            f"+{r['dt']}: in {r['inside']:3d} out {r['outside']:3d} pieces {r['pieces']} joined {r['joined']} sheet {r['total_sheet']:3d}"
            for r in rows), flush=True)
        for r in rows:
            by_dt.setdefault(r["dt"], []).append(r)
    band_cols = L // 8
    for dt, rs in sorted(by_dt.items()):
        both = np.mean([r["inside"] > 0 and r["outside"] > 0 for r in rs])
        joined = np.mean([r["joined"] for r in rs])
        print(f"== L {L} k {k} +{dt} sweeps after the escape ({len(rs)} replicas): a foot on each side in {both:.2f}; "
              f"the feet joined through one piece in {joined:.2f}; median inside {np.median([r['inside'] for r in rs]):.0f} of {4 * band_cols} band points, "
              f"median outside {np.median([r['outside'] for r in rs]):.0f} of {4 * (L - band_cols)}; "
              f"median sheet pieces {np.median([r['pieces'] for r in rs]):.0f}", flush=True)
