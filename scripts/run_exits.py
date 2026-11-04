"""T22: count every exit from the curled torus and every fall-back into it. Usage:

    python scripts/run_exits.py configs/t22_exits_n64.json

Implements PREREGISTRATION.md section T22 with graphity.exits.run_until_through (move by move).
"""
import json
import platform
import sys
from pathlib import Path

import numba
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphity import __version__                                          # noqa: E402
from graphity.cqg import NO_CAP, torus                                    # noqa: E402
from graphity.exits import run_until_through                              # noqa: E402
from graphity.results import ResultWriter                                 # noqa: E402


def main(path, out_dir="results"):
    cfg = json.loads(Path(path).read_text())
    lx, ly = cfg["side"]
    n = lx * ly
    g = float(cfg["g"])
    meta = dict(config=cfg, config_path=str(path), package=__version__,
                python=platform.python_version(), numpy=np.__version__, numba=numba.__version__,
                preregistration="PREREGISTRATION.md section T22, written 2026-09-24")
    with ResultWriter(cfg["name"], meta, out_dir) as out:
        for lam in cfg["lambdas"]:
            for rep in range(int(cfg["replicas"])):
                adj, part = torus(lx, ly, NO_CAP)
                seed = int(np.random.SeedSequence([int(cfg["seed"]), n, int(round(lam * 100)), rep]).generate_state(1)[0])
                e, f, first, end, through = run_until_through(adj, np.flatnonzero(part == 0), g, float(lam), NO_CAP,
                                                              int(cfg["max_sweeps"]), seed, float(cfg["stop_fraction"]))
                out.write(dict(N=n, lam=lam, g=g, replica=rep, exits=e, fallbacks=f,
                               first_exit_sweeps=(first / (2.0 * n) if first > 0 else ""),
                               end_sweeps=end / (2.0 * n), went_through=bool(through)))
                print("N=%d lam=%.2f rep=%d done" % (n, lam, rep), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "results")
