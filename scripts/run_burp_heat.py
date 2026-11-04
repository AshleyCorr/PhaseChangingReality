"""T61: if the burp's heat stays, does the tube come out wider? Usage:

    python scripts/run_burp_heat.py configs/t61_lam125_n96.json [out_dir]

Implements PREREGISTRATION.md section T61. Each replica is a sealed tube (a 4 x L torus, N = 4L points) with C
stores of energy beside it and a spark of the wall's size in one store. Energy is conserved: what the change
releases can only go into the stores, so fewer stores means a hotter new arrangement. Three spells:

  sealed   `hold` sweeps of graphity.sealed.run_sealed_bath (H + stores constant to the last unit);
  drained  every 100 sweeps each store is multiplied by 1 - leak and what was taken is counted, until less than
           `drain_stop` units are left in the stores (H + stores + drained stays constant);
  cold     `cold_sweeps` of the thermal chain at g = `cold_g`, which takes no uphill move.

The end state is read from its wiring, not from its counts alone: exactly flat or not (every edge on two squares),
its width if flat (scripts/read_endpoint_width.py), the energy left, the points at each d, and for every point the
number of points within 3, 4 and 5 steps. An endless flat sheet has 25, 41 and 61; the tube with its seam shifted
by two has 22, 30 and 38. `sees3` is the share of points with exactly 25 within three steps: 0 on any flat torus
six around, 1 on any flat torus eight or more around. The end graph is saved.

Seeds: numpy SeedSequence([seed, N, round(100 lambda), C, replica]), so a replica's stream does not depend on
which other cells are in its config.
"""
import json
import platform
import sys
from pathlib import Path

import numba
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from graphity import __version__                                           # noqa: E402
from graphity.cqg import NO_CAP, hamiltonian, is_valid, run_chain, torus   # noqa: E402
from graphity.dimension import local_dimension                             # noqa: E402
from graphity.results import ResultWriter, provenance                      # noqa: E402
from graphity.sealed import run_sealed_bath                                # noqa: E402
from read_endpoint_width import is_flat, width                             # noqa: E402
from read_open_shells import shells                                        # noqa: E402

DRAIN_BLOCK = 100
MAX_DRAIN_BLOCKS = 20000


def wall(lam):
    """The cheapest single move out of the tube (paper 1): 32 - 16 lambda below lambda = 4/3, 64 - 40 lambda above."""
    return min(32.0 - 16.0 * lam, 64.0 - 40.0 * lam)


def room(adj):
    """(mean points within 3, within 4, within 5 steps; share of points with exactly 25 within 3 steps)."""
    n = adj.shape[0]
    balls = np.zeros((n, 6), dtype=np.int64)
    for v in range(n):
        s = shells(adj, v)[:6]
        balls[v, :len(s)] = s
    balls = np.cumsum(balls, axis=1)
    return float(balls[:, 3].mean()), float(balls[:, 4].mean()), float(balls[:, 5].mean()), float((balls[:, 3] == 25).mean())


def replica(length, stores, lam, seed, hold, leak, drain_stop, cold_sweeps, cold_g):
    """One sealed, drained, cooled tube. Returns (row fields, end graph, side labels)."""
    adj, part = torus(length, 4, NO_CAP)
    n = adj.shape[0]
    side = np.flatnonzero(part == 0).astype(np.int64)
    demons = np.zeros(stores, dtype=np.float64)
    spark = wall(lam)
    demons[0] = spark
    start = float(hamiltonian(adj, lam)) + spark
    s, _x, mean, _sum, _acc = run_sealed_bath(adj, side, demons, hold, seed, lam, NO_CAP, None, False)
    drift = abs(float(hamiltonian(adj, lam)) + float(demons.sum()) - start)
    hold_fields = dict(phi_hold_end=s[-1] / n, phi_hold_min=float(s.min()) / n, store_mean_hold=float(mean[-1]),
                       h_hold_end=float(hamiltonian(adj, lam)))
    drained, blocks = 0.0, 0
    while demons.sum() > drain_stop and blocks < MAX_DRAIN_BLOCKS:
        taken = float(demons.sum()) * leak
        demons *= (1.0 - leak)
        drained += taken
        run_sealed_bath(adj, side, demons, DRAIN_BLOCK, -1, lam, NO_CAP, None, False)
        blocks += 1
    drift = max(drift, abs(float(hamiltonian(adj, lam)) + float(demons.sum()) + drained - start))
    h_drained = float(hamiltonian(adj, lam))
    if cold_sweeps > 0:
        run_chain(adj, side, 1.0 / cold_g, cold_sweeps - 1, 1, -1, lam, NO_CAP)
    flat = bool(is_flat(adj))
    d = np.bincount(local_dimension(adj), minlength=7)[:7]
    near3, near4, near5, sees3 = room(adj)
    fields = dict(spark=spark, **hold_fields, drained=drained, drain_sweeps=blocks * DRAIN_BLOCK,
                  left_in_stores=float(demons.sum()), h_drained=h_drained, h_end=float(hamiltonian(adj, lam)),
                  valid=bool(is_valid(adj, NO_CAP)), flat=flat, width=(width(adj) if flat else ""),
                  near3=round(near3, 4), near4=round(near4, 4), near5=round(near5, 4), sees3=round(sees3, 4),
                  **{"d%d" % j: int(d[j]) for j in range(7)}, drift=drift)
    return fields, adj, part


def seed_of(seed, n, lam, stores, rep):
    return int(np.random.SeedSequence([int(seed), int(n), int(round(100 * lam)), int(stores), int(rep)]).generate_state(1)[0])


def main(path, out_dir="results"):
    cfg = json.loads(Path(path).read_text())
    lam = float(cfg["lambda"])
    meta = dict(config=cfg, config_path=str(path), package=__version__, python=platform.python_version(),
                numpy=np.__version__, numba=numba.__version__,
                preregistration="PREREGISTRATION.md section T61, written 2026-10-11", **provenance())
    adj_dir = Path(out_dir) / (cfg["name"] + "_adj")
    if cfg.get("save_adjacency") and adj_dir.exists():
        raise SystemExit("refusing to overwrite %s" % adj_dir)
    with ResultWriter(cfg["name"], meta, out_dir) as out:
        for length in cfg["lengths"]:
            n = 4 * int(length)
            hold = int(cfg["hold_sweeps_at_192"] * max(1.0, n / 192.0))
            for ratio in cfg["stores_per_point"]:
                stores = int(round(float(ratio) * n))
                for rep in range(int(cfg["replicas"])):
                    seed = seed_of(cfg["seed"], n, lam, stores, rep)
                    fields, adj, part = replica(int(length), stores, lam, seed, hold, float(cfg["leak"]),
                                                float(cfg["drain_stop"]), int(cfg["cold_sweeps"]), float(cfg["cold_g"]))
                    out.write(dict(N=n, length=int(length), lam=lam, stores=stores, stores_per_point=float(ratio),
                                   replica=rep, seed=seed, hold_sweeps=hold, **fields))
                    if cfg.get("save_adjacency"):
                        adj_dir.mkdir(parents=True, exist_ok=True)
                        np.savez(adj_dir / ("N%d_C%d_rep%d.npz" % (n, stores, rep)), adj=adj, part=part, lam=lam)
                print("%s N=%d stores=%d done" % (cfg["name"], n, stores), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
