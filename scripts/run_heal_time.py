"""T66: does flat three-direction space heal damage in fewer sweeps than flat two-direction space? Usage:

    python scripts/run_heal_time.py configs/t66_d3_n216.json

Implements PREREGISTRATION.md section T66 (written 2026-10-16, before any registered run; the owner's prediction of
the same day). A flat torus (two directions, Lx x Ly, or three, L x L x L), untied, sealed with a bath of C empty
stores, is given E = rho * N units packed into one store (T9's protocol, as T65 and the sandbox look of 16 October).
Every `record_every` sweeps the run writes the damage census: damaged = points whose local dimension d exceeds the
number of directions D; flat = points at d = D; curled = points at d < D; and the stores' mean and total. The
analysis (`scripts/analyse_t66.py`) reads the peak of the damage and the sweeps from the peak to the scar level.

The chain is `graphity.sealed_tie_d.run_sealed_bath_table_d` with a table of zeros (no tie; the tie was withdrawn,
VISION Update 58), which is the untied sealed chain of any D with the shared bath. Energy is conserved to the last
bit and the drift is written in every row. Seeds: SeedSequence([seed, N, D, round(100 rho), replica]).
"""
import json
import platform
import sys
from pathlib import Path

import numba
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphity import __version__                                          # noqa: E402
from graphity.cqg_d import torus                                           # noqa: E402
from graphity.dimension import local_dimension_d                           # noqa: E402
from graphity.results import ResultWriter, provenance                      # noqa: E402
from graphity.sealed_tie_d import energy_table_d, run_sealed_bath_table_d  # noqa: E402


def census(adj, dim):
    d = local_dimension_d(adj)
    return dict(damaged=int((d > dim).sum()), flat=int((d == dim).sum()), curled=int((d < dim).sum()))


def run_cell(dims, lam, rho, c_bath, n_sweeps, every, seed):
    """One replica: yields (sweep, census, store_mean, store_total, drift) at sweep 0 and every `every` sweeps."""
    adj, part = torus(dims)
    dim = len(dims)
    n = adj.shape[0]
    side_u = np.flatnonzero(part == 0)
    ftab = np.zeros(dim + 1)
    stores = np.zeros(c_bath)
    stores[0] = float(rho) * n
    e0 = energy_table_d(adj, lam, ftab) + stores.sum()
    yield 0, census(adj, dim), float(stores.mean()), float(stores.sum()), 0.0
    for b in range(n_sweeps // every):
        run_sealed_bath_table_d(adj, side_u, stores, every, seed if b == 0 else -1, lam, ftab, by_vertex=False)
        drift = abs(energy_table_d(adj, lam, ftab) + stores.sum() - e0)
        yield (b + 1) * every, census(adj, dim), float(stores.mean()), float(stores.sum()), float(drift)


def main(path, out_dir="results"):
    cfg = json.loads(Path(path).read_text())
    dims = [int(x) for x in cfg["dims"]]
    dim = len(dims)
    n = int(np.prod(dims))
    lam = float(cfg["lambda"])
    c_bath = int(round(eval(str(cfg["capacity"]), {"N": n})))
    every, n_sweeps = int(cfg["record_every"]), int(cfg["n_sweeps"])
    label = "x".join(str(x) for x in dims)
    meta = dict(config=cfg, config_path=str(path), package=__version__, python=platform.python_version(),
                numpy=np.__version__, numba=numba.__version__, **provenance(),
                preregistration="PREREGISTRATION.md section %s, written 2026-10-16" % cfg.get("section", "T66"))
    with ResultWriter(cfg["name"], meta, out_dir) as out:
        for rho in cfg["rho_per_point"]:
            for rep in range(int(cfg["replicas"])):
                seed = int(np.random.SeedSequence([int(cfg["seed"]), n, dim, int(round(100 * float(rho))), rep]).generate_state(1)[0])
                for sweep, c, mean, total, drift in run_cell(dims, lam, float(rho), c_bath, n_sweeps, every, seed):
                    out.write(dict(N=n, D=dim, dims=label, lam=lam, rho=float(rho), E=float(rho) * n, C=c_bath,
                                   replica=rep, sweep=sweep, **c, store_mean=mean, store_total=total, drift=drift))
    print("wrote", Path(out_dir) / (cfg["name"] + ".csv"))


if __name__ == "__main__":
    main(sys.argv[1], *(sys.argv[2:3]))
