"""T43: how long does a planted allotrope last at λ = 1? Usage:

    python scripts/run_planted_allotrope.py configs/t43_allotrope_lifetime_g3433.json [out_dir]

Implements PREREGISTRATION.md section T43 (the test proposed in docs/design/planted_allotrope.md, section 6). The
objects are built exactly by scripts/build_planted_allotrope.py (build_all(240, 14)), and each replica starts from one
of them with no warm-up:
  hyperbolic_fold (234), the scored object; hyperbolic_handle (228); hyperbolic_background (240, the Fig. 7 tiling,
  control); flat_handle (188); flat_torus (196, the 14 x 14 lattice torus, control).
The chain is graphity.cqg.run_chain at λ = 1 (the published model), no cap, Metropolis, run in blocks with the random
stream carried on (seed on the first block, -1 afterwards; ASSUMPTIONS Q14). A snapshot is read every 10 sweeps for the
first 2,000 sweeps and every 100 after, to n_sweeps, plus one at sweep 0. Reading uses no random numbers, so the chain
is the same as if it had run in one call.

WHAT A ROW HOLDS (section 6). The planted set R is fixed at the start, by label: the points whose square count c(v)
differs from the background's (3 on the hyperbolic objects, 4 on the flat ones); it is empty for the two controls.
The far points are those at least 3 steps from R in the starting graph (all points when R is empty). At each snapshot:
  f_R  the share of R touching at most (background - 1) squares;
  f_B  the same share among the far points (the chance level);
  q_R  the share of R with the planted link pattern: squares on its four links (1,1,1,1), Fig. 9's, on the hyperbolic
       objects, and (1,1,2,2) on the flat handle;
  q_B  the share of far points with the background's link pattern: (1,1,2,2) hyperbolic, (2,2,2,2) flat;
  H    = 16 (N - S) + 4 X.
f_R and q_R are left empty when R is empty. For T36's persistence excess (analyse_t43.py) each row also holds n_low,
the points of the whole graph touching at most (background - 1) squares, and low_R, the labels of the points of R that
do.

Seeds: np.random.SeedSequence([seed, object index in the config, round(1000 g), replica]).
"""
import json
import platform
import sys
from pathlib import Path

import numba
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))
import build_planted_allotrope as bpa                                      # noqa: E402
from graphity import __version__                                          # noqa: E402
from graphity.cqg import (ENERGY_PER_SQUARE, ENERGY_PER_SURPLUS, NO_CAP,   # noqa: E402
                          run_chain, squares_on_edge, surplus, total_squares)
from graphity.results import ResultWriter                                 # noqa: E402

LAM = 1.0                                   # the published model; this is CQG (VISION Update 22)
FINE_EVERY, FINE_UNTIL, COARSE_EVERY = 10, 2000, 100
FAR = 3                                     # far points: at least this many steps from R in the starting graph
# squares on the four links, sorted: (planted pattern, background pattern), by the background's square count
PATTERNS = {3: ((1, 1, 1, 1), (1, 1, 2, 2)), 4: ((1, 1, 2, 2), (2, 2, 2, 2))}


def snapshot_sweeps(n_sweeps):
    """0, then every 10 sweeps to 2,000, then every 100 to n_sweeps."""
    fine = list(range(FINE_EVERY, min(FINE_UNTIL, n_sweeps) + 1, FINE_EVERY))
    coarse = list(range(FINE_UNTIL + COARSE_EVERY, n_sweeps + 1, COARSE_EVERY))
    return [0] + fine + coarse


def link_squares(adj):
    """(N, 4): the squares on each of every point's four links."""
    n = adj.shape[0]
    return np.array([[squares_on_edge(adj, v, int(adj[v, k])) for k in range(4)] for v in range(n)], dtype=np.int64)


def planted_and_far(adj, bg_c):
    """R (the points whose square count differs from the background's) and the far points, in the starting graph."""
    R, _ = bpa.region(adj, bg_c)
    if len(R) == 0:
        return R, np.arange(adj.shape[0])
    return R, np.flatnonzero(bpa.distances(adj, R) >= FAR)


def measure(adj, R, far, bg_c):
    """The section-6 numbers at one snapshot."""
    e = link_squares(adj)
    low = e.sum(axis=1) // 2 <= bg_c - 1
    links = np.sort(e, axis=1)
    pat_r, pat_b = (np.array(p) for p in PATTERNS[bg_c])
    has_r, has_b = (links == pat_r).all(axis=1), (links == pat_b).all(axis=1)
    return dict(f_R=float(low[R].mean()) if len(R) else "", f_B=float(low[far].mean()),
                q_R=float(has_r[R].mean()) if len(R) else "", q_B=float(has_b[far].mean()),
                n_R=len(R), n_low=int(low.sum()), low_R=" ".join(str(int(v)) for v in R[low[R]]))


def energy(n, s, x):
    return int(ENERGY_PER_SQUARE * (n - s) + ENERGY_PER_SURPLUS * LAM * x)


def main(path, out_dir="results"):
    cfg = json.loads(Path(path).read_text())
    objects = bpa.build_all(240, 14)
    n_sweeps = int(cfg["n_sweeps"])
    sweeps = snapshot_sweeps(n_sweeps)
    sizes = {}
    for name in cfg["objects"]:
        adj, _, bg_c = objects[name]
        R, far = planted_and_far(adj, bg_c)
        sizes[name] = dict(N=int(adj.shape[0]), background_squares=bg_c, planted=len(R), far=len(far))
    meta = dict(config=cfg, config_path=str(path), package=__version__, python=platform.python_version(),
                numpy=np.__version__, numba=numba.__version__, objects=sizes, snapshot_sweeps=[sweeps[1], sweeps[-1]],
                preregistration="PREREGISTRATION.md section T43, written 2026-09-25")
    with ResultWriter(cfg["name"], meta, out_dir) as out:
        for k, name in enumerate(cfg["objects"]):
            start, part, bg_c = objects[name]
            n = start.shape[0]
            side_u = np.flatnonzero(part == 0)
            R, far = planted_and_far(start, bg_c)
            for g in cfg["couplings"]:
                for rep in range(int(cfg["replicas"])):
                    adj = start.copy()
                    seed = int(np.random.SeedSequence([int(cfg["seed"]), k, int(round(g * 1000)), rep]).generate_state(1)[0])
                    h = energy(n, int(total_squares(adj)), int(surplus(adj)))
                    done = 0
                    for sw in sweeps:
                        if sw > done:
                            s, x, _ = run_chain(adj, side_u, 1.0 / g, 0, sw - done, seed if done == 0 else -1,
                                                LAM, NO_CAP, False)
                            h, done = energy(n, int(s[-1]), int(x[-1])), sw
                        row = dict(object=name, N=n, g=g, replica=rep, sweep=sw, H=h, **measure(adj, R, far, bg_c))
                        out.write(row)
                    print("%s g=%.3f rep %d: at sweep %d H %d, f_R %s, f_B %.3f, q_B %.3f"
                          % (name, g, rep, done, h, row["f_R"], row["f_B"], row["q_B"]), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "results")
