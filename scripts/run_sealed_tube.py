"""T9: a tube in a sealed box with a bath of C demons. Usage:

    python scripts/run_sealed_tube.py configs/t9_n64.json

Implements PREREGISTRATION.md section T9. One demon starts with the spark (12 units), the rest
empty. Every run goes the full length; recorded every `record_every` sweeps are phi, X, the
mean demon energy (legacy CSV name bath_T, not a calibrated temperature) and the conservation check, and at the end the local-
dimension histogram and the pieces of the vertices at each d. Nothing is stopped early: what
the product is after the lump has had nowhere to go is the whole question.
"""
import json
import platform
import sys
from pathlib import Path

import numba
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphity import __version__                                        # noqa: E402
from graphity.cqg import NO_CAP, hamiltonian, torus                     # noqa: E402
from graphity.dimension import local_dimension, pieces_of               # noqa: E402
from graphity.results import ResultWriter                               # noqa: E402
from graphity.sealed import run_sealed_bath                             # noqa: E402

PHI_TUBE, PHI_SHEET = 1.25, 1.0
D_BINS = 7


def main(path, out_dir="results"):
    cfg = json.loads(Path(path).read_text())
    lam, spark = float(cfg["lambda"]), float(cfg["spark"])
    n_sweeps, every = int(cfg["n_sweeps"]), int(cfg.get("record_every", 100))
    meta = dict(config=cfg, config_path=str(path), package=__version__,
                python=platform.python_version(), numpy=np.__version__, numba=numba.__version__,
                preregistration="PREREGISTRATION.md section T9, written 2026-09-22")
    lx, ly = cfg["side"]
    n = lx * ly
    caps = [max(1, int(round(eval(str(c), {"N": n})))) for c in cfg["capacities"]]
    with ResultWriter(cfg["name"], meta, out_dir) as out:
        for c in caps:
            for rep in range(int(cfg["replicas"])):
                adj, part = torus(lx, ly, NO_CAP)
                side_u = np.flatnonzero(part == 0)
                demons = np.zeros(c)
                demons[0] = spark
                e0 = hamiltonian(adj, lam) + demons.sum()
                seed = int(np.random.SeedSequence([int(cfg["seed"]), n, c, rep]).generate_state(1)[0])
                s, x, mean, tot, acc = run_sealed_bath(adj, side_u, demons, n_sweeps, seed, lam, NO_CAP)
                h = 16.0 * (n - s) + 4.0 * lam * x
                drift = float(np.abs(h + tot - e0).max())
                d = local_dimension(adj)
                hist = np.bincount(d, minlength=D_BINS)[:D_BINS]
                p1 = pieces_of(adj, d == 1)
                p2 = pieces_of(adj, d == 2)
                f_final = (PHI_TUBE - s[-1] / n) / (PHI_TUBE - PHI_SHEET)
                for k in range(every - 1, n_sweeps, every):
                    out.write(dict(N=n, C=c, replica=rep, sweep=k + 1, phi=s[k] / n, surplus=x[k] / n,
                                   bath_T=mean[k], bath_total=tot[k],
                                   f=(PHI_TUBE - s[k] / n) / (PHI_TUBE - PHI_SHEET),
                                   final=int(k + 1 == n_sweeps),
                                   drift=drift, acceptance=acc, f_final=f_final,
                                   **{"d%d" % i: int(hist[i]) for i in range(D_BINS)},
                                   pieces_d1=len(p1), largest_d1=(p1[0] if p1 else 0),
                                   pieces_d2=len(p2), largest_d2=(p2[0] if p2 else 0)))
                print("N=%-4d C=%-4d rep=%-2d  f_final=%.2f  mean store energy %.2f  d-hist %s  d1 pieces %d  drift %.1e"
                      % (n, c, rep, f_final, mean[-1], hist.tolist(), len(p1), drift), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "results")
