"""T19: does a leftover move through the sheet? Usage:

    python scripts/run_leftover_mobility.py configs/t19_mobility.json

Implements PREREGISTRATION.md section T19 (series papers 2 and 5). Each replica first makes a flat sheet with
leftovers exactly as T17 does at k = 1 (a 24 x 4 tube, one seed planted as move A, a cold sealed box of 2N
empty demons, 30,000 sweeps). A replica whose product is not exactly one leftover piece at d = 1 is recorded
and not followed. The sheet with one leftover is then run at fixed coupling g (cqg.run_chain, random stream
carried on) for `follow_sweeps`, and every `block` sweeps the vertices at d = 1 are read. The leftover's
**displacement** is the shortest graph distance, in the current graph, from its current vertices to the
vertices it started on; it is 0 while it sits where it began.
"""
import json
import platform
import sys
from collections import deque
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
from run_seeded_tube import plant_seeds                                   # noqa: E402


def distance_to(adj, sources, targets):
    """Shortest path length in adj from any vertex of `sources` to any of `targets` (-1 if none)."""
    targets = set(int(t) for t in targets)
    seen = {int(s): 0 for s in sources}
    queue = deque(seen)
    while queue:
        v = queue.popleft()
        if v in targets:
            return seen[v]
        for w in adj[v]:
            w = int(w)
            if w not in seen:
                seen[w] = seen[v] + 1
                queue.append(w)
    return -1


def main(path, out_dir="results"):
    cfg = json.loads(Path(path).read_text())
    lam = float(cfg["lambda"])
    lx, ly = cfg["side"]
    n = lx * ly
    block, follow = int(cfg["block"]), int(cfg["follow_sweeps"])
    meta = dict(config=cfg, config_path=str(path), package=__version__,
                python=platform.python_version(), numpy=np.__version__, numba=numba.__version__,
                preregistration="PREREGISTRATION.md section T19, written 2026-09-24")
    with ResultWriter(cfg["name"], meta, out_dir) as out:
        for g in cfg["couplings"]:
            for rep in range(int(cfg["replicas"])):
                adj, part = torus(lx, ly, NO_CAP)
                side_u = np.flatnonzero(part == 0)
                plant_seeds(adj, part, lx, ly, 1)
                seed = int(np.random.SeedSequence([int(cfg["seed"]), n, int(g * 100), rep]).generate_state(1)[0])
                run_sealed_bath(adj, side_u, np.zeros(2 * n), int(cfg["make_sweeps"]), seed, lam, NO_CAP)
                d = local_dimension(adj)
                start = np.flatnonzero(d == 1)
                pieces = pieces_of(adj, d == 1)
                if len(pieces) != 1:
                    out.write(dict(N=n, g=g, replica=rep, sweep=0, followed=False, pieces_d1=len(pieces),
                                   n_d1=len(start), displacement="", d1_vertices=" ".join(map(str, start))))
                    print("g=%.1f rep %d: %d leftover pieces, not followed" % (g, rep, len(pieces)), flush=True)
                    continue
                for b in range(follow // block + 1):
                    if b:
                        run_chain(adj, side_u, 1.0 / g, 0, block, seed + 1 if b == 1 else -1, lam, NO_CAP, False)
                    d = local_dimension(adj)
                    now = np.flatnonzero(d == 1)
                    out.write(dict(N=n, g=g, replica=rep, sweep=b * block, followed=True,
                                   pieces_d1=len(pieces_of(adj, d == 1)), n_d1=len(now),
                                   displacement=(distance_to(adj, now, start) if len(now) else ""),
                                   d1_vertices=" ".join(map(str, now))))
                print("g=%.1f rep %d: followed; final d=1 vertices %d" % (g, rep, len(now)), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "results")
