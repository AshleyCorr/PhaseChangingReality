"""How long does an arrangement sit before it gives way? Usage:

    python scripts/run_waiting_time.py configs/<name>.json

Starts from the exact lx x ly torus at a fixed coupling and records only the sweep
at which the graph first leaves it (S or X changes). Sweeping the coupling gives an
Arrhenius plot: if the way out is a single move costing B, then the waiting time
should be 1 / (w exp(-B/g)) sweeps, where w is how many such moves the sampler
offers per sweep, so ln(wait) against 1/g is a straight line of slope B.

Both B and w can be counted exactly beforehand for a given starting arrangement, so
this is a prediction with no free parameters, not a fit. For the tube at lambda =
1.25 they are B = 12 and w = 3 (ASSUMPTIONS Q13).

Config keys: name, sides, replicas, seed, "couplings", "n_sweeps", plus "cap" and
"lambda" as in run_cqg_sweep.py. Metropolis only: the Arrhenius form above assumes
it. One row per (size, coupling, replica): left says whether it gave way inside
n_sweeps, and wait is the sweep at which it did.
"""
import json
import platform
import sys
from pathlib import Path

import numba
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_cqg_sweep import model_of, seeder, shape_of              # noqa: E402  (the path line must come first)

from graphity import __version__                                   # noqa: E402
from graphity.cqg import NO_CAP, is_valid, run_chain, surplus, torus, total_squares   # noqa: E402
from graphity.results import ResultWriter                          # noqa: E402


def main(path, out_dir="results"):
    cfg = json.loads(Path(path).read_text())
    lam, cap, glauber = model_of(cfg)
    if glauber:
        raise ValueError('the Arrhenius reading assumes Metropolis; use "acceptance": "metropolis"')
    couplings = [float(g) for g in cfg["couplings"]]
    for entry in cfg["sides"]:                                     # fail before any work is done
        seeder(cfg, entry, 0)
    meta = dict(config=cfg, package=__version__, python=platform.python_version(),
                numpy=np.__version__, numba=numba.__version__)
    with ResultWriter(cfg["name"], meta, out_dir) as out:
        for entry in cfg["sides"]:
            lx, ly = shape_of(entry)
            n = lx * ly
            for g in couplings:
                gave_way = 0
                for rep in range(cfg["replicas"]):
                    adj, part = torus(lx, ly, cap)
                    side_u = np.flatnonzero(part == 0)
                    start_s, start_x = total_squares(adj), surplus(adj)     # before a single sweep has run
                    s, x, acc = run_chain(adj, side_u, 1.0 / g, 0, cfg["n_sweeps"], seeder(cfg, entry, rep)(1),
                                          lam, cap, glauber)
                    assert is_valid(adj, cap)
                    moved = np.flatnonzero((s != start_s) | (x != start_x))
                    left = moved.size > 0
                    gave_way += left
                    out.write(dict(N=n, lx=lx, ly=ly, g=g, replica=rep, left=left,
                                   wait=(int(moved[0]) + 1 if left else ""), n_sweeps=cfg["n_sweeps"],
                                   phi_start=float(start_s / n), acceptance=float(acc), lam=lam,
                                   cap=("none" if cap == NO_CAP else cap)))
                print(f"N = {n}, g = {g}: {gave_way} of {cfg['replicas']} gave way", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
