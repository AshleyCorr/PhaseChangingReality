"""T78: in the published model's warm background, count the curled ring and the flat patch snapshot by snapshot.
Usage:

    python scripts/run_curl_vs_flat.py configs/t78_curl_vs_flat_n196.json

Implements PREREGISTRATION.md section T78. The published model exactly (λ = 1, no cap, the hard-core rule, Metropolis;
graphity.cqg.run_chain with the random stream carried on). Each replica starts from a melt of the lattice torus, runs
`n_equil` sweeps at coupling g, then `n_snap` blocks of `snap_every` sweeps; after every block one row is written:
S, X, D = 4 (N − S) + X, H = 16 (N − S) + 4 X, and the two object counts of graphity.objects (rings: squares whose four
links carry three squares; patches: 4 x 4 grids with their 24 links). The graph is saved every `save_every_snap`
blocks. Nothing is read here; scripts/analyse_t78.py reads the CSV under the registered rules.
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
from graphity.objects import count_patches, count_rings                   # noqa: E402
from graphity.results import ResultWriter, provenance                     # noqa: E402


def main(path, out_dir="results"):
    cfg = json.loads(Path(path).read_text())
    lx, ly = cfg["side"]
    n = lx * ly
    lam = float(cfg.get("lambda", 1.0))
    adj_dir = Path(out_dir) / (cfg["name"] + "_adj")
    meta = dict(config=cfg, config_path=str(path), package=__version__, python=platform.python_version(),
                numpy=np.__version__, numba=numba.__version__,
                preregistration="PREREGISTRATION.md section T78, written 2026-10-22", **provenance())
    with ResultWriter(cfg["name"], meta, out_dir) as out:
        for g in cfg["couplings"]:
            for rep in range(int(cfg["replicas"])):
                adj, part = torus(lx, ly, NO_CAP)
                side_u = np.flatnonzero(part == 0)
                seed = int(np.random.SeedSequence([int(cfg["seed"]), n, int(round(g * 1000)), rep]).generate_state(1)[0])
                run_chain(adj, side_u, 0.0, int(cfg.get("n_melt", 200)), 1, seed, lam, NO_CAP, False)
                run_chain(adj, side_u, 1.0 / g, int(cfg["n_equil"]), 1, -1, lam, NO_CAP, False)
                tot_r = tot_p = 0
                for b in range(int(cfg["n_snap"]) + 1):
                    if b:
                        run_chain(adj, side_u, 1.0 / g, 0, int(cfg["snap_every"]), -1, lam, NO_CAP, False)
                    s, x = int(total_squares(adj)), int(surplus(adj))
                    r, p = int(count_rings(adj)), int(count_patches(adj))
                    tot_r += r
                    tot_p += p
                    out.write(dict(N=n, g=g, replica=rep, snap=b, sweep=int(cfg["n_equil"]) + b * int(cfg["snap_every"]),
                                   S=s, X=x, D=4 * (n - s) + x, H=16 * (n - s) + 4 * lam * x, n_ring=r, n_patch=p))
                    if cfg.get("save_adjacency") and b % int(cfg.get("save_every_snap", 100)) == 0:
                        adj_dir.mkdir(parents=True, exist_ok=True)
                        np.savez(adj_dir / ("g%d_rep%d_snap%d.npz" % (round(g * 1000), rep, b)), adj=adj, part=part)
                print("N=%d g=%.3f rep %d: rings %d, patches %d over %d snapshots" % (n, g, rep, tot_r, tot_p, int(cfg["n_snap"]) + 1), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "results")
