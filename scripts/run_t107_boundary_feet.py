"""T107: the seed on the boundary of the loose remnant, and the kind on each side. Usage:

    python scripts/run_t107_boundary_feet.py configs/t107_boundary_feet.json

PREREGISTRATION.md section T107 (registered 6 November 2026, before this ran). For each length L and replica: the
4 x L tube at T37's cell (lambda, g, no cap) with a band of L/8 columns loosened by k switches
(graphity.loose_band); the chain run in blocks of `block` sweeps; the escape read as growth (the first block at
which at least `min_size` points not made to read d = 2 by the loosening do so); at each `after` time past the
escape the new sheet points are split into the inside foot (in the band) and the outside foot, the connected
pieces of sheet counted with the loosened points included, and whether the two feet share one piece recorded.
Then, on the wiring at the last `after` time, her curl state on the squares and her clock (price from gravity,
room judged once on the made state) under her kind rule (the first pair across the band's edge sets the outward
kind, every pair after follows it) and, as the control, the coin; read at the freeze: net twist inside and
outside, stores by sign on each side, pairs made, photons, the energy left on squares and held in stores.
One row per (L, replica, after) for the feet and one per (L, replica, rule) for the kinds, in two CSVs. Seeds:
the loosening from SeedSequence([seed, L, k, replica]); the chain from the same sequence's first state; the pair
moves from SeedSequence([seed + 1, L, k, replica]). Writes results/<name>_feet.csv and results/<name>_kind.csv with
.meta.json beside each and refuses to overwrite.
"""
import json
import platform
import sys
from pathlib import Path

import numpy as np

from graphity import __version__
from graphity.cqg import NO_CAP, run_chain, total_squares
from graphity.curl_pair_moves import PairMoves
from graphity.curl_pairs import CurlState
from graphity.dimension import local_dimension, piece_labels
from graphity.loose_band import loose_band_tube
from graphity.results import ResultWriter, provenance


def square_distances(state, L, centre):
    """Each square's circular distance in columns from the centre column (the circular mean of its corners' x)."""
    ang = 2 * np.pi * (np.arange(len(state.adj)) // 4) / L
    d = np.empty(len(state.sq))
    for q, (black, white) in enumerate(state.sq):
        c = list(black) + list(white)
        m = np.arctan2(np.sin(ang[c]).mean(), np.cos(ang[c]).mean()) * L / (2 * np.pi)
        d[q] = abs(((m - centre + L / 2) % L) - L / 2)
    return d


def one(L, rep, cfg, feet_out, kind_out):
    k, lam, g, block = cfg["k"], cfg["lambda"], cfg["g"], cfg["block"]
    after, min_size, max_sweeps = cfg["after"], cfg["min_size"], cfg["max_sweeps"]
    ss = np.random.SeedSequence([cfg["seed"], L, k, rep])
    adj, part, in_band, made = loose_band_tube(L, k, np.random.default_rng(ss))
    chain_seed = int(ss.generate_state(1)[0])
    side_u = np.flatnonzero(part == 0)
    init = local_dimension(adj) == 2
    s_packed = 4 * L
    first, escape = True, None
    for sweeps in range(0, max_sweeps, block):
        run_chain(adj, side_u, 1.0 / g, 0, block, chain_seed if first else -1, lam=lam, cap=NO_CAP)
        first = False
        t = sweeps + block
        new_pts = (local_dimension(adj) == 2) & ~init
        if escape is None:
            if new_pts.sum() < min_size:
                continue
            escape = t
        dt = t - escape
        if dt in after:
            sheet = local_dimension(adj) == 2
            labels = piece_labels(adj, sheet)
            lab_in = set(labels[new_pts & in_band]) - {-1}
            lab_out = set(labels[new_pts & ~in_band]) - {-1}
            feet_out.write(dict(L=L, n_points=4 * L, replica=rep, k=k, made=made, deficit=s_packed - total_squares(adj),
                                escape=escape, dt=dt, inside=int((new_pts & in_band).sum()),
                                outside=int((new_pts & ~in_band).sum()), band_points=int(in_band.sum()),
                                pieces=int(labels.max() + 1), joined=bool(lab_in & lab_out),
                                total_sheet=int(sheet.sum())))
        if dt >= max(after):
            break
    if escape is None:
        feet_out.write(dict(L=L, n_points=4 * L, replica=rep, k=k, made=made, deficit=s_packed - total_squares(adj),
                            escape=None, dt=None, inside=0, outside=0, band_points=int(in_band.sum()), pieces=0,
                            joined=False, total_sheet=int((local_dimension(adj) == 2).sum())))
        return
    # her curl state and clock on the escaped wiring, under her kind rule and under the coin
    base = CurlState(adj, part)
    dist = square_distances(base, L, L // 2)
    edge = (L // 8) / 2
    inside_sq = dist < edge
    pair_seed = int(np.random.SeedSequence([cfg["seed"] + 1, L, k, rep]).generate_state(1)[0])
    for rule in ("boundary", "coin"):
        st = CurlState(adj, part)
        pm = PairMoves(st, lam=lam, barrier=float(cfg["barrier"]), seed=pair_seed, price_from="gravity",
                       clock="simultaneous", kind_rule=rule, dist=dist, edge=edge)
        pm.run(max_sweeps=cfg["max_pair_sweeps"])
        c, s = st.c, pm.store
        free = s == 0
        twist = 0.5 * np.sign(c) * free + s
        cen = pm.census()
        kind_out.write(dict(L=L, n_points=4 * L, replica=rep, k=k, escape=escape, rule=rule, squares=len(st.sq),
                            inside_squares=int(inside_sq.sum()), outward_kind=pm.outward_kind,
                            twist_in=float(twist[inside_sq].sum()), twist_out=float(twist[~inside_sq].sum()),
                            stores_in_plus=int((s[inside_sq] > 0).sum()), stores_in_minus=int((s[inside_sq] < 0).sum()),
                            stores_out_plus=int((s[~inside_sq] > 0).sum()), stores_out_minus=int((s[~inside_sq] < 0).sum()),
                            pairs_made=cen["pairs_made"], photons=cen["photons"], light=cen["light"],
                            stored_on_squares=cen["stored_on_squares"], held_in_stores=cen["held_in_stores"],
                            gravity=cen["gravity"], net_twist=cen["net_twist"], sweeps=cen["sweeps"],
                            books_ok=abs(pm.books() - pm.release) < 1e-9))
        print(f"L={L} rep={rep} {rule}: escape {escape} outward {pm.outward_kind:+d} twist in {twist[inside_sq].sum():+.1f} "
              f"out {twist[~inside_sq].sum():+.1f} pairs {cen['pairs_made']} photons {cen['photons']}", flush=True)


def main(path):
    cfg = json.loads(Path(path).read_text())
    meta = dict(config=cfg, config_path=str(path), graphity_version=__version__, numpy=np.__version__,
                python=platform.python_version(), **provenance())
    with ResultWriter(cfg["name"] + "_feet", meta) as feet_out, ResultWriter(cfg["name"] + "_kind", meta) as kind_out:
        for L in cfg["lengths"]:
            for rep in range(cfg["replicas"]):
                one(L, rep, cfg, feet_out, kind_out)


if __name__ == "__main__":
    main(sys.argv[1])
