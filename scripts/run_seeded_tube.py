"""T17: does each seed leave its own leftover? Usage:

    python scripts/run_seeded_tube.py configs/t17_seeds.json

Implements PREREGISTRATION.md section T17 (TASKS T14). A long 4 x L tube at lambda = 1.25 in a cold
sealed box (C = 2N demons, all empty) is given k seeds before the run: move A, the tube's cheapest way
out (Delta S = -2, Delta X = -4; the step a 12-unit spark pays for), applied at k columns spaced L/k
apart. Energy is conserved from that state on. The run is T10's in every other respect, and the
leftover is counted as T10 counts it: connected pieces of vertices at local dimension d = 1 at the end.

How a seed is chosen (`plant_seeds`): for each target column c, the first switch in a fixed order
(u1 in side 0 at column c, then u2, then the neighbour slots) that the chain could propose, that is
valid, that has (Delta S, Delta X) = (-2, -4), and whose four vertices all lie within one column of c.
Deterministic, so a seed's position is part of the configuration, not of the random stream.
"""
import json
import platform
import sys
from pathlib import Path

import numba
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphity import __version__                                        # noqa: E402
from graphity.cqg import NO_CAP, _switch, hamiltonian, is_valid, surplus, torus, total_squares  # noqa: E402
from graphity.dimension import local_dimension, pieces_of               # noqa: E402
from graphity.results import ResultWriter                               # noqa: E402
from graphity.sealed import run_sealed_bath                             # noqa: E402

PHI_TUBE, PHI_SHEET = 1.25, 1.0
D_BINS = 7


def column(v, ly):
    return v // ly


def near(c, x, lx):
    return min((x - c) % lx, (c - x) % lx) <= 1


def plant_seeds(adj, part, lx, ly, k):
    """Apply move A at k columns spaced lx/k apart. Returns the columns used. Modifies adj."""
    if lx % k:
        raise ValueError("k must divide the tube length")
    side0 = np.flatnonzero(part == 0)
    cols = []
    for j in range(k):
        c = j * lx // k
        s0, x0 = total_squares(adj), surplus(adj)
        done = False
        for u1 in [u for u in side0 if column(u, ly) == c]:
            for u2 in side0:
                if u2 == u1:
                    continue
                for v1 in adj[u1]:
                    for v2 in adj[u2]:
                        u1_, v1_, u2_, v2_ = int(u1), int(v1), int(u2), int(v2)
                        if v1_ == v2_ or not all(near(c, column(w, ly), lx) for w in (u1_, v1_, u2_, v2_)):
                            continue
                        trial = adj.copy()
                        _switch(trial, u1_, v1_, u2_, v2_)
                        if not is_valid(trial, NO_CAP):
                            continue
                        if total_squares(trial) - s0 == -2 and surplus(trial) - x0 == -4:
                            adj[:] = trial
                            done = True
                            break
                    if done:
                        break
                if done:
                    break
            if done:
                break
        if not done:
            raise RuntimeError("no move A found at column %d" % c)
        cols.append(c)
    return cols


def main(path, out_dir="results"):
    cfg = json.loads(Path(path).read_text())
    lam = float(cfg["lambda"])
    n_sweeps, every = int(cfg["n_sweeps"]), int(cfg.get("record_every", 100))
    lx, ly = cfg["side"]
    n = lx * ly
    # T18 (2026-09-23): a list of bath sizes. With "capacities" the seed derivation includes C; with a
    # single "capacity" (T17) it is unchanged, so T17's runs reproduce exactly.
    many = "capacities" in cfg
    baths = [int(round(eval(str(c), {"N": n}))) for c in (cfg["capacities"] if many else [cfg["capacity"]])]
    meta = dict(config=cfg, config_path=str(path), package=__version__,
                python=platform.python_version(), numpy=np.__version__, numba=numba.__version__,
                preregistration="PREREGISTRATION.md section %s" % cfg.get("section", "T17"))
    runs = [(c, k, rep) for c in baths for k in cfg["seeds"] for rep in range(int(cfg["replicas"]))]
    with ResultWriter(cfg["name"], meta, out_dir) as out:
        for c_bath, k, rep in runs:
            adj, part = torus(lx, ly, NO_CAP)
            side_u = np.flatnonzero(part == 0)
            h_tube = hamiltonian(adj, lam)
            cols = plant_seeds(adj, part, lx, ly, k)
            h_planted = hamiltonian(adj, lam)
            demons = np.zeros(c_bath)
            e0 = h_planted
            parts = [int(cfg["seed"]), n, k, rep] + ([c_bath] if many else [])
            seed = int(np.random.SeedSequence(parts).generate_state(1)[0])
            s, x, mean, tot, acc = run_sealed_bath(adj, side_u, demons, n_sweeps, seed, lam, NO_CAP)
            h = 16.0 * (n - s) + 4.0 * lam * x
            drift = float(np.abs(h + tot - e0).max())
            d = local_dimension(adj)
            hist = np.bincount(d, minlength=D_BINS)[:D_BINS]
            p1 = pieces_of(adj, d == 1)
            f_final = (PHI_TUBE - s[-1] / n) / (PHI_TUBE - PHI_SHEET)
            out.write(dict(N=n, C=c_bath, lam=lam, k=k, replica=rep, seed_columns=" ".join(map(str, cols)),
                           planted_cost=h_planted - h_tube, f_final=f_final, phi_final=s[-1] / n,
                           bath_T=mean[-1], drift=drift, acceptance=acc,
                           **{"d%d" % i: int(hist[i]) for i in range(D_BINS)},
                           pieces_d1=len(p1), sizes_d1=" ".join(map(str, p1)),
                           h_final=float(h[-1])))
            print("N=%d C=%d k=%d rep=%-2d cost %.1f f_final=%.2f d1 pieces %d sizes %s drift %.1e"
                  % (n, c_bath, k, rep, h_planted - h_tube, f_final, len(p1), p1[:6], drift), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "results")
