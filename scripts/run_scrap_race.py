"""T25: does the scrap freeze in before it heals, when the box cools? Usage:

    python scripts/run_scrap_race.py configs/t25_race_tc1000.json

Implements PREREGISTRATION.md section T25 (series paper 2; programme piece 6). Each replica first makes a
flat sheet with one leftover exactly as T19 does (a 24 x 4 tube at lambda = 1.25, one seed planted as move A,
a cold sealed box of 2N empty stores, `make_sweeps` sweeps); a replica whose product is not exactly one
leftover piece at d = 1 is recorded and not followed. The sheet is then cooled at fixed coupling in blocks of
`block` sweeps: over `t_cool` sweeps the coupling falls from `g_hot` to `g_cold` by the same factor each
block (g at the end of block b of nb is g_hot (g_cold / g_hot)^(b / nb)), and it is then held at `g_cold` for
`hold` sweeps. The random stream is carried on between blocks (seed < 0). Every block the vertices at d = 1
are read: how many, in how many pieces, and how many vertices are not flat at all.
"""
import json
import platform
import sys
from pathlib import Path

import numba
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from graphity import __version__                                          # noqa: E402
from graphity.cqg import NO_CAP, run_chain, torus                         # noqa: E402
from graphity.dimension import local_dimension, pieces_of                 # noqa: E402
from graphity.results import ResultWriter                                 # noqa: E402
from graphity.sealed import run_sealed_bath                               # noqa: E402
from graphity.spark import energy_of                                      # noqa: E402
from run_seeded_tube import plant_seeds                                   # noqa: E402


def schedule(g_hot, g_cold, t_cool, hold, block):
    """The coupling at the end of each block: cooling blocks, then holding blocks."""
    nb = max(1, int(round(t_cool / block)))
    cool = [g_hot * (g_cold / g_hot) ** ((b + 1) / nb) for b in range(nb)]
    return cool + [g_cold] * int(round(hold / block))


def main(path, out_dir="results"):
    cfg = json.loads(Path(path).read_text())
    lam = float(cfg["lambda"])
    lx, ly = cfg["side"]
    n = lx * ly
    block = int(cfg["block"])
    g_hot, g_cold = float(cfg["g_hot"]), float(cfg["g_cold"])
    meta = dict(config=cfg, config_path=str(path), package=__version__,
                python=platform.python_version(), numpy=np.__version__, numba=numba.__version__,
                preregistration="PREREGISTRATION.md section T25, written 2026-09-24")
    with ResultWriter(cfg["name"], meta, out_dir) as out:
        for t_cool in cfg["t_cool"]:
            gs = schedule(g_hot, g_cold, int(t_cool), int(cfg["hold"]), block)
            for rep in range(int(cfg["replicas"])):
                adj, part = torus(lx, ly, NO_CAP)
                side_u = np.flatnonzero(part == 0)
                plant_seeds(adj, part, lx, ly, 1)
                seed = int(np.random.SeedSequence([int(cfg["seed"]), n, int(t_cool), rep]).generate_state(1)[0])
                run_sealed_bath(adj, side_u, np.zeros(2 * n), int(cfg["make_sweeps"]), seed, lam, NO_CAP)
                d = local_dimension(adj)
                pieces = pieces_of(adj, d == 1)
                base = dict(N=n, t_cool=int(t_cool), g_hot=g_hot, g_cold=g_cold, replica=rep)
                if len(pieces) != 1:
                    out.write(dict(**base, followed=False, block=0, sweep=0, g=g_hot, n_d1=int((d == 1).sum()),
                                   pieces_d1=len(pieces), n_damage=int((d != 2).sum()), h=float(energy_of(adj, lam)),
                                   final=True))
                    print("t_cool=%d rep %d: %d leftover pieces, not followed" % (t_cool, rep, len(pieces)), flush=True)
                    continue
                for b, g in enumerate([g_hot] + gs):
                    if b:
                        run_chain(adj, side_u, 1.0 / g, 0, block, seed + 1 if b == 1 else -1, lam, NO_CAP, False)
                    d = local_dimension(adj)
                    out.write(dict(**base, followed=True, block=b, sweep=b * block, g=g, n_d1=int((d == 1).sum()),
                                   pieces_d1=len(pieces_of(adj, d == 1)), n_damage=int((d != 2).sum()),
                                   h=float(energy_of(adj, lam)), final=(b == len(gs))))
                print("t_cool=%d rep %d: followed; final d=1 vertices %d, not flat %d"
                      % (t_cool, rep, int((d == 1).sum()), int((d != 2).sum())), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "results")
