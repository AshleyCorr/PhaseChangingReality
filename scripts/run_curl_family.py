"""T79: replay T78's chains from the same seeds and count the ring, its uncurled twin and a family of flat grids on
every snapshot. Usage:

    python scripts/run_curl_family.py configs/t79_curl_family_n196_g3433.json

Implements PREREGISTRATION.md section T79. The chain is `graphity.cqg.run_chain` with the seed scheme of
scripts/run_curl_vs_flat.py, so the snapshots are the same graphs as T78's (checked by the analyzer: the replayed
ring count must equal T78's n_ring on every snapshot). Objects: ring3 (four links at three squares), ring2 (four links
at exactly two), flat a x b grids with no grid link above two squares for the registered sizes, and the 4 x 4 grid
without that condition (T78's patch) as the second identity check. Nothing is read here.
"""
import json
import platform
import sys
from pathlib import Path

import numba
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphity import __version__                                          # noqa: E402
from graphity.cqg import NO_CAP, run_chain, surplus, torus, total_squares  # noqa: E402
from graphity.objects import count_grid, count_ring2, count_rings         # noqa: E402
from graphity.results import ResultWriter, provenance                     # noqa: E402

GRIDS = [(2, 2), (2, 3), (2, 4), (3, 3), (3, 4), (3, 5), (4, 4), (4, 5)]


def counts(adj):
    row = dict(n_ring3=int(count_rings(adj)), n_ring2=int(count_ring2(adj)), n_patch=int(count_grid(adj, 4, 4, False)))
    for a, b in GRIDS:
        row["n_flat_%dx%d" % (a, b)] = int(count_grid(adj, a, b, True))
    return row


def main(path, out_dir="results"):
    cfg = json.loads(Path(path).read_text())
    lx, ly = cfg["side"]
    n = lx * ly
    lam = float(cfg.get("lambda", 1.0))
    meta = dict(config=cfg, config_path=str(path), package=__version__, python=platform.python_version(),
                numpy=np.__version__, numba=numba.__version__,
                preregistration="PREREGISTRATION.md section T79, written 2026-10-22", **provenance())
    with ResultWriter(cfg["name"], meta, out_dir) as out:
        for g in cfg["couplings"]:
            for rep in range(int(cfg["replicas"])):
                adj, part = torus(lx, ly, NO_CAP)
                side_u = np.flatnonzero(part == 0)
                seed = int(np.random.SeedSequence([int(cfg["seed"]), n, int(round(g * 1000)), rep]).generate_state(1)[0])
                run_chain(adj, side_u, 0.0, int(cfg.get("n_melt", 200)), 1, seed, lam, NO_CAP, False)
                run_chain(adj, side_u, 1.0 / g, int(cfg["n_equil"]), 1, -1, lam, NO_CAP, False)
                for b in range(int(cfg["n_snap"]) + 1):
                    if b:
                        run_chain(adj, side_u, 1.0 / g, 0, int(cfg["snap_every"]), -1, lam, NO_CAP, False)
                    s, x = int(total_squares(adj)), int(surplus(adj))
                    out.write(dict(N=n, g=g, replica=rep, snap=b, sweep=int(cfg["n_equil"]) + b * int(cfg["snap_every"]),
                                   S=s, X=x, D=4 * (n - s) + x, H=16 * (n - s) + 4 * lam * x, **counts(adj)))
                print("N=%d g=%.3f rep %d done" % (n, g, rep), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "results")
