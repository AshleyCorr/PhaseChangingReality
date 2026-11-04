"""T47 part A: does the front of the tube's opening move at a fixed speed? Usage:

    python scripts/run_front_speed.py configs/t47_front_bath_l96.json

Implements PREREGISTRATION.md section T47 (TASKS T13 rung 2, the speed limit; VISION Update 36, the owner's idea that
time curls behind a moving present). A 4 x L tube at lambda = 1.25 is given one seed before the run, move A at column 0
(scripts/run_seeded_tube.plant_seeds, unchanged), and is then sealed: energy is conserved from the planted state on.
Two baths, as the config says: "bath": the shared bath of C = 2N empty stores (T17's box), where released energy is
shared at once; "local": one empty store per vertex (graphity.sealed.run_sealed_bath with by_vertex, T26), where released
energy stays where it is released. Recorded every `record_every` sweeps: the converted fraction
f = (1.25 - S/N) / 0.25 (0 for the tube, 1 for the flat sheet), the bath temperature and the conservation check.
The front's progress is read from f: with one seed two fronts move apart, so f L / 2 is how far each has gone, in
columns. Config keys: name, section, side ([L, 4]), bath ("bath" or "local"), replicas, n_sweeps, record_every, seed,
lambda.
"""
import json
import platform
import sys
from pathlib import Path

import numba
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from graphity import __version__                                        # noqa: E402
from graphity.cqg import NO_CAP, hamiltonian, torus                     # noqa: E402
from graphity.results import ResultWriter                               # noqa: E402
from graphity.sealed import run_sealed_bath                             # noqa: E402
from run_seeded_tube import plant_seeds                                 # noqa: E402

PHI_TUBE, PHI_SHEET = 1.25, 1.0


def main(path, out_dir="results"):
    cfg = json.loads(Path(path).read_text())
    lam = float(cfg["lambda"])
    n_sweeps, every = int(cfg["n_sweeps"]), int(cfg.get("record_every", 10))
    lx, ly = cfg["side"]
    n = lx * ly
    local = cfg["bath"] == "local"
    meta = dict(config=cfg, config_path=str(path), package=__version__, python=platform.python_version(),
                numpy=np.__version__, numba=numba.__version__,
                preregistration="PREREGISTRATION.md section %s" % cfg.get("section", "T47"))
    with ResultWriter(cfg["name"], meta, out_dir) as out:
        for rep in range(int(cfg["replicas"])):
            adj, part = torus(lx, ly, NO_CAP)
            side_u = np.flatnonzero(part == 0)
            h_tube = hamiltonian(adj, lam)
            cols = plant_seeds(adj, part, lx, ly, 1)
            e0 = hamiltonian(adj, lam)
            stores = np.zeros(n if local else 2 * n)
            seed = int(np.random.SeedSequence([int(cfg["seed"]), n, int(local), rep]).generate_state(1)[0])
            base = dict(N=n, L=lx, bath=cfg["bath"], lam=lam, replica=rep, seed_column=cols[0],
                        planted_cost=e0 - h_tube)
            out.write(dict(**base, sweep=0, f=0.0, bath_T=0.0, drift=0.0))
            for b in range(1, n_sweeps // every + 1):
                s, x, mean, tot, _ = run_sealed_bath(adj, side_u, stores, every, seed if b == 1 else -1, lam, NO_CAP,
                                                     by_vertex=local)
                h = 16.0 * (n - s[-1]) + 4.0 * lam * x[-1]
                f = (PHI_TUBE - s[-1] / n) / (PHI_TUBE - PHI_SHEET)
                out.write(dict(**base, sweep=b * every, f=float(f), bath_T=float(mean[-1]),
                               drift=float(abs(h + tot[-1] - e0))))
            print("L=%d bath=%s rep=%-2d f_end=%.3f T=%.3f" % (lx, cfg["bath"], rep, f, mean[-1]), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "results")
