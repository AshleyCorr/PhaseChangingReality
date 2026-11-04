"""T36: search an equilibrated λ = 1 graph for allotropes, regions whose points touch only 2 squares in a background of 3.
Usage:

    python scripts/run_allotrope_search.py configs/t36_allotropes_n196.json

Implements PREREGISTRATION.md section T36 (piece 12). The method was suggested by the model's author (private
communication, 25 September 2026): equilibrate a finite graph at a coupling where points touch about three squares on
average, and look for points, or groups of points, that touch only two. Each replica starts from the lattice torus
(sides given) or from a melt of it (`start`: "melt" or "lattice"), runs `n_equil` sweeps at coupling g with the full
Hamiltonian (λ = 1, no cap; graphity.cqg.run_chain, the random stream carried on), then `n_snap` more blocks of
`snap_every` sweeps; after every block the graph is saved and one row is written: S, X, the histogram of each point's
square count c(v) (the squares that contain v; 4 on the flat torus, mean 4S/N), and the set of points with c(v) <= 2.
"""
import json
import platform
import sys
from pathlib import Path

import numba
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphity import __version__                                          # noqa: E402
from graphity.cqg import NO_CAP, run_chain, squares_on_edge, surplus, torus, total_squares   # noqa: E402
from graphity.results import ResultWriter                                 # noqa: E402

C_BINS = 9


def square_counts(adj):
    """c(v): the number of squares containing v = (sum of squares on v's edges) / 2."""
    n = adj.shape[0]
    c = np.zeros(n, dtype=np.int64)
    for v in range(n):
        c[v] = sum(int(squares_on_edge(adj, v, int(w))) for w in adj[v]) // 2
    return c


def main(path, out_dir="results"):
    cfg = json.loads(Path(path).read_text())
    lx, ly = cfg["side"]
    n = lx * ly
    lam = float(cfg.get("lambda", 1.0))
    adj_dir = Path(out_dir) / (cfg["name"] + "_adj")
    meta = dict(config=cfg, config_path=str(path), package=__version__, python=platform.python_version(),
                numpy=np.__version__, numba=numba.__version__,
                preregistration="PREREGISTRATION.md section T36, written 2026-09-25")
    with ResultWriter(cfg["name"], meta, out_dir) as out:
        for g in cfg["couplings"]:
            for start in cfg["starts"]:
                for rep in range(int(cfg["replicas"])):
                    adj, part = torus(lx, ly, NO_CAP)
                    side_u = np.flatnonzero(part == 0)
                    seed = int(np.random.SeedSequence([int(cfg["seed"]), n, int(round(g * 1000)), 0 if start == "melt" else 1, rep]).generate_state(1)[0])
                    first = True
                    if start == "melt":
                        run_chain(adj, side_u, 0.0, int(cfg.get("n_melt", 200)), 1, seed, lam, NO_CAP, False)
                        first = False
                    run_chain(adj, side_u, 1.0 / g, int(cfg["n_equil"]), 1, seed if first else -1, lam, NO_CAP, False)
                    for b in range(int(cfg["n_snap"]) + 1):
                        if b:
                            run_chain(adj, side_u, 1.0 / g, 0, int(cfg["snap_every"]), -1, lam, NO_CAP, False)
                        c = square_counts(adj)
                        low = np.flatnonzero(c <= 2)
                        hist = np.bincount(np.minimum(c, C_BINS - 1), minlength=C_BINS)[:C_BINS]
                        out.write(dict(N=n, g=g, start=start, replica=rep, snap=b, sweep=int(cfg["n_equil"]) + b * int(cfg["snap_every"]),
                                       S=int(total_squares(adj)), X=int(surplus(adj)), mean_c=float(c.mean()),
                                       **{"c%d" % k: int(hist[k]) for k in range(C_BINS)},
                                       n_low=len(low), low=" ".join(map(str, low))))
                        if cfg.get("save_adjacency") and b % int(cfg.get("save_every_snap", 10)) == 0:
                            adj_dir.mkdir(parents=True, exist_ok=True)
                            np.savez(adj_dir / ("g%d_%s_rep%d_snap%d.npz" % (round(g * 1000), start, rep, b)), adj=adj, part=part)
                    print("N=%d g=%.3f %s rep %d: mean c %.3f, points with c <= 2: %d" % (n, g, start, rep, c.mean(), len(low)), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "results")
