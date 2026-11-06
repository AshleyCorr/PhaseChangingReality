"""Sandbox look (5 November 2026, her words "This cycle leaves lower density shell inside the higher density layers
of shell. How could that affect 3d seeds?"): a 4 x L tube packed everywhere except one loose band. Where does the
escape start? A look, not a test.

The tube is the exact 4 x L torus at T37's cell (lambda 1.25, g 1.5, no cap). The band is the L/8 columns centred
on x = L/2; it is loosened by k random valid switches whose four points all lie in the band (any valid switch
accepted, no energy asked), which removes squares there and nothing elsewhere. k = 0 is the control. The chain then
runs in blocks of `block` sweeps, reading the local dimension between blocks (no random numbers drawn). The
loosening itself makes about a dozen points in the band read d = 2 (seen 5 November, first version of this look), so
the escape is read as growth: the first block at which at least `min_size` points that did not read d = 2 right
after the loosening do so; the seed is those new points; its position is the circular mean of their x along the
tube. Reported per replica: the band's square deficit after loosening, the escape sweep, the seed's distance from
the band's centre in links (0 to L/2), and the share of the new points inside the band. If the new sheet points sit
in the band more often than the band's share of the tube (one in eight), the loose band seeds the escape."""
import sys

import numpy as np

from graphity.cqg import NO_CAP, run_chain, torus, total_squares
from graphity.dimension import local_dimension
from graphity.loose_band import loosen  # moved to the package on 7 November 2026 for T107; same code  # noqa: F401


def circ_mean_x(xs, L):
    ang = 2 * np.pi * xs / L
    return (np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()) % (2 * np.pi)) * L / (2 * np.pi)


def circ_dist(a, b, L):
    d = abs(a - b) % L
    return min(d, L - d)


def one(L, k, rep, lam=1.25, g=1.5, block=10, max_sweeps=20000, min_size=8, width=None):
    width = width or L // 8
    adj, part = torus(L, 4, NO_CAP)
    n = adj.shape[0]
    x_of = np.arange(n) // 4
    centre = L // 2
    in_band = np.abs(((x_of - centre + L // 2) % L) - L // 2) < width / 2
    rng = np.random.default_rng([20261105, L, k, rep])
    s0 = total_squares(adj)
    made = loosen(adj, part, rng, in_band, k)
    deficit = s0 - total_squares(adj)
    side_u = np.flatnonzero(part == 0)
    seed = int(np.random.SeedSequence([20261105, L, k, rep]).generate_state(1)[0])
    init = local_dimension(adj) == 2          # what the loosening itself made read as sheet
    first = True
    for sweeps in range(0, max_sweeps, block):
        run_chain(adj, side_u, 1.0 / g, 0, block, seed if first else -1, lam=lam, cap=NO_CAP)
        first = False
        new_pts = (local_dimension(adj) == 2) & ~init
        if new_pts.sum() < min_size:
            continue
        pos = circ_mean_x(x_of[new_pts], L)
        return dict(k=k, rep=rep, made=made, deficit=int(deficit), init=int(init.sum()), sweep=sweeps + block,
                    size=int(new_pts.sum()), dist=circ_dist(pos, centre, L),
                    in_band=float(in_band[new_pts].mean()))
    return dict(k=k, rep=rep, made=made, deficit=int(deficit), init=int(init.sum()), sweep=None, size=0, dist=None,
                in_band=None)


if __name__ == "__main__":
    L = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    for k in (0, 2, 4, 8):
        rows = [one(L, k, rep) for rep in range(reps)]
        esc = [r for r in rows if r["sweep"] is not None]
        for r in rows:
            print(f"L {L} k {k} rep {r['rep']:2d}: deficit {r['deficit']:2d} sheet points made by loosening {r['init']:2d} "
                  f"escape sweep {r['sweep']} new sheet points {r['size']:3d} "
                  f"dist {r['dist'] if r['dist'] is None else round(r['dist'], 1)} "
                  f"share in band {r['in_band'] if r['in_band'] is None else round(r['in_band'], 2)}", flush=True)
        if esc:
            share = np.mean([r["in_band"] for r in esc])
            print(f"== L {L} k {k}: escaped {len(esc)}/{reps}; median escape sweep {np.median([r['sweep'] for r in esc]):.0f}; "
                  f"new sheet points in band {share:.2f} (band share of tube {L // 8 / L:.3f}); "
                  f"median dist from band centre {np.median([r['dist'] for r in esc]):.1f} links "
                  f"(uniform would be {L / 4:.1f})", flush=True)
