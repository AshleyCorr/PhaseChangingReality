"""How big a spark does a sealed arrangement need before it will convert? Usage:

    python scripts/run_spark_threshold.py configs/<name>.json

Sealed runs (graphity/sealed.py, ASSUMPTION Q12): the demon starts with a fixed
lump of energy, the "spark", nothing leaks, and there is no other source of energy
anywhere. The run either leaves its starting arrangement or it does not. Sweeping
the spark gives the height of the wall round that arrangement, in absolute units;
sweeping the size as well says whether the wall is local (the same however big the
system) or grows with it.

Config keys: name, sides, replicas, seed, "sparks" (a list), "n_sweeps", and
"start" ("torus", the default: the exact lx x ly torus, which for ly = 4 is a tube),
plus "cap", "lambda" as in run_cqg_sweep.py. Metropolis only: a sealed run has no
acceptance rule to choose.

One row per (size, spark, replica): left says whether the graph ever left its
starting arrangement (S or X changed at all), first_change is the sweep it did so,
and the rest describe where it ended. The *_start columns are the starting arrangement
itself, before any sweep has run.
"""
import json
import platform
import sys
from pathlib import Path

import numba
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_cqg_sweep import model_of, seeder, shape_of                # noqa: E402  (the path line must come first)

from graphity import __version__                                     # noqa: E402
from graphity.cqg import NO_CAP, hamiltonian, is_valid, surplus, total_squares, torus   # noqa: E402
from graphity.results import ResultWriter                            # noqa: E402
from graphity.sealed import run_sealed                               # noqa: E402


def main(path, out_dir="results"):
    cfg = json.loads(Path(path).read_text())
    lam, cap, glauber = model_of(cfg)
    if glauber:
        raise ValueError("a sealed run has no acceptance rule to choose; use \"acceptance\": \"metropolis\"")
    if cfg.get("start", "torus") != "torus":
        raise ValueError('only "start": "torus" is implemented')
    for entry in cfg["sides"]:                                       # fail before any work is done
        seeder(cfg, entry, 0)
    meta = dict(config=cfg, package=__version__, python=platform.python_version(),
                numpy=np.__version__, numba=numba.__version__)
    with ResultWriter(cfg["name"], meta, out_dir) as out:
        for entry in cfg["sides"]:
            lx, ly = shape_of(entry)
            n = lx * ly
            for spark in cfg["sparks"]:
                left_count = 0
                for rep in range(cfg["replicas"]):
                    adj, part = torus(lx, ly, cap)
                    side_u = np.flatnonzero(part == 0)
                    phi_start = total_squares(adj) / n              # before a single sweep has run
                    energy_start = hamiltonian(adj, lam) / n
                    surplus_start = surplus(adj) / n
                    s, x, demon, lost, acc = run_sealed(adj, side_u, float(spark), cfg["n_sweeps"],
                                                        seeder(cfg, entry, rep)(1), lam, cap, 0.0)
                    assert is_valid(adj, cap)
                    moved = np.flatnonzero((s != phi_start * n) | (x != surplus_start * n))
                    left = moved.size > 0
                    left_count += left
                    energy = 16.0 * (n - s) + 4.0 * lam * x
                    out.write(dict(N=n, lx=lx, ly=ly, spark=float(spark), replica=rep, left=left,
                                   first_change=(int(moved[0]) + 1 if left else ""),
                                   phi_start=float(phi_start), phi_end=float(s[-1] / n),
                                   energy_start=float(energy_start), energy_end=float(energy[-1] / n),
                                   lowest_energy=float(energy.min() / n), demon_end=float(demon[-1]),
                                   acceptance=float(acc), lam=lam, cap=("none" if cap == NO_CAP else cap)))
                print(f"N = {n}, spark {spark}: {left_count} of {cfg['replicas']} left the starting arrangement",
                      flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
