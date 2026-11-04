"""Which arrangements are "dips" of the energy at the smallest sizes? Usage:

    python scripts/dip_census.py configs/dip_census_small.json

A DIP is a state from which every valid single edge switch raises H. A dip that is
not the lowest state is what VISION claim 4 calls "stable for the time being but
not the most stable arrangement available": a candidate, in miniature, for the
starting point of a reaction-like change. Exhaustive and exact, no random numbers;
it uses the complete class lists of task T4 (src/graphity/small_graphs.py), so it
can only be run where those exist (N <= 18).

Config keys: "sizes" (vertices per side, n; N = 2n) and "lambdas". No cap.

One row per (size, lambda, class). way_out is the smallest change of H over all
valid single switches; is_dip says it is positive; above_ground is the height of
the class above the lowest class at that lambda. A size with no states is skipped.
"""
import json
import platform
import sys
from pathlib import Path

import networkx
import numba
import numpy as np

from graphity import __version__
from graphity.connectivity import connectivity
from graphity.cqg import NO_CAP, hamiltonian, surplus, total_squares
from graphity.results import ResultWriter
from graphity.small_graphs import count_labelled_states, explore, one_switch_away


def main(path, out_dir="results"):
    cfg = json.loads(Path(path).read_text())
    meta = dict(config=cfg, package=__version__, python=platform.python_version(), numpy=np.__version__,
                numba=numba.__version__, networkx=networkx.__version__)
    with ResultWriter(cfg["name"], meta, out_dir) as out:
        for n in cfg["sizes"]:
            _, start = count_labelled_states(n, NO_CAP)
            if start is None:
                continue
            classes, _ = explore(start, NO_CAP)
            around = [one_switch_away(adj, NO_CAP) for adj, _ in classes.reps]
            for lam in cfg["lambdas"]:
                energy = [float(hamiltonian(adj, lam)) for adj, _ in classes.reps]
                ground = min(energy)
                dips = 0
                for k, (adj, _) in enumerate(classes.reps):
                    way_out = min((float(hamiltonian(b, lam)) - energy[k] for b in around[k]), default=float("inf"))
                    pieces, largest, in_babies, cubes = connectivity(adj)
                    dips += way_out > 1e-9
                    out.write(dict(N=2 * n, lam=lam, class_id=k, squares=int(total_squares(adj)),
                                   surplus=int(surplus(adj)), pieces=int(pieces), in_babies=int(in_babies),
                                   H=energy[k], above_ground=energy[k] - ground, way_out=way_out,
                                   is_dip=way_out > 1e-9))
                print(f"N = {2 * n}, lambda = {lam}: {dips} dip(s) among {len(classes.reps)} classes", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
