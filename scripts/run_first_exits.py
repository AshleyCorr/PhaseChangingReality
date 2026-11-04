"""T59: the first exit from the curled torus, timed by the move, and what a long wait was offered. Usage:

    python scripts/run_first_exits.py configs/t59_lam130_n64_00.json

Implements PREREGISTRATION.md section T59 with graphity.exits. A config has one of two modes.

  "clocks"  fresh tubes. Each row is one tube: its first exit counted move by move, and what a look every sweep and
            every `look` sweeps would have reported for the same history (first_exit_clocks).
  "offers"  T22's own runs replayed from their seeds and stopped at the first exit. Each row is one stretch of one
            wait: how many exits of each kind the chain was offered, and the smallest acceptance draw made against
            them (offers_until_exit).

Seeds follow T22's rule (scripts/run_exits.py): the config's seed, N, lambda in hundredths and the replica.
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
from graphity.exits import first_exit_clocks, offers_until_exit           # noqa: E402
from graphity.results import ResultWriter, provenance                     # noqa: E402


def seed_of(base, n, lam, rep):
    return int(np.random.SeedSequence([int(base), n, int(round(lam * 100)), int(rep)]).generate_state(1)[0])


def clocks(cfg, out):
    lx, ly = cfg["side"]
    n = lx * ly
    g, lam, look = float(cfg["g"]), float(cfg["lambda"]), int(cfg["look"])
    start = int(cfg["replica_start"])
    for rep in range(start, start + int(cfg["replicas"])):
        adj, part = torus(lx, ly, NO_CAP)
        seed = seed_of(cfg["seed"], n, lam, rep)
        first, seen_1, seen_k, exits_1, exits_k, fallbacks = first_exit_clocks(
            adj, np.flatnonzero(part == 0), g, lam, NO_CAP, int(cfg["max_sweeps"]), seed, look)
        out.write(dict(N=n, lam=lam, g=g, replica=rep, seed=seed,
                       first_exit_attempts=(first if first > 0 else ""),
                       first_exit_sweeps=(first / (2.0 * n) if first > 0 else ""),
                       seen_every_sweep=(seen_1 if seen_1 > 0 else ""),
                       seen_every_look=(seen_k if seen_k > 0 else ""),
                       look=look, exits_before_sweep_look=exits_1, exits_before_look=exits_k, fallbacks=fallbacks))
        if (rep - start + 1) % 25 == 0:
            print("%s: %d of %d tubes done" % (cfg["name"], rep - start + 1, int(cfg["replicas"])), flush=True)


def offers(cfg, out):
    g, lam, stretch = float(cfg["g"]), float(cfg["lambda"]), int(cfg["stretch"])
    for target in cfg["targets"]:
        lx, ly = target["side"]
        n = lx * ly
        rep = int(target["replica"])
        adj, part = torus(lx, ly, NO_CAP)
        seed = seed_of(target["seed"], n, lam, rep)
        first, offered, smallest = offers_until_exit(adj, np.flatnonzero(part == 0), g, lam, NO_CAP,
                                                     int(cfg["max_sweeps"]), seed, stretch)
        last = (first - 1) // (stretch * 2 * n) if first > 0 else offered.shape[0] - 1
        for i in range(last + 1):
            whole = first < 0 or i < last
            sweeps = stretch if whole else first / (2.0 * n) - last * stretch
            out.write(dict(N=n, lam=lam, g=g, replica=rep, role=target["role"], seed=seed,
                           first_exit_attempts=(first if first > 0 else ""),
                           first_exit_sweeps=(first / (2.0 * n) if first > 0 else ""),
                           stretch=i, sweeps_in_stretch=sweeps, whole_stretch=int(whole),
                           offered_a=int(offered[i, 0]), offered_b=int(offered[i, 1]),
                           offered_other=int(offered[i, 2]), neutral=int(offered[i, 3]),
                           refused=int(offered[i, 4]),
                           smallest_draw_a=float(smallest[i, 0]), smallest_draw_b=float(smallest[i, 1])))
        print("%s: N=%d replica %d done" % (cfg["name"], n, rep), flush=True)


def main(path, out_dir="results"):
    cfg = json.loads(Path(path).read_text())
    meta = dict(config=cfg, config_path=str(path), package=__version__,
                python=platform.python_version(), numpy=np.__version__, numba=numba.__version__,
                preregistration="PREREGISTRATION.md section T59, written 2026-10-10", **provenance())
    with ResultWriter(cfg["name"], meta, out_dir) as out:
        {"clocks": clocks, "offers": offers}[cfg["mode"]](cfg, out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "results")
