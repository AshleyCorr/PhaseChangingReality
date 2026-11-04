"""Quench of the 2D CQG model: melt, then run at ONE coupling and record every sweep. Usage:

    python scripts/run_cqg_quench.py configs/<name>.json

run_cqg_sweep.py reports averages at each of many couplings; this reports the
history at one. Published curves of this kind, both for N = 100 ... 200:
    [KTB19] Fig. 6  "mean field" action, our lambda = 0: phi passes 1 within a
                    few hundred sweeps and climbs on towards 1.5, the graph
                    breaking into baby universes (its Fig. 5).
    [KTB19] Fig. 7  exact action, our lambda = 1: phi is held near 1.
The paper defines neither its sweep nor the temperature of its quench, so only
shapes and heights can be compared with it, not the clock.

Config keys: name, sides, n_melt, n_sweeps, replicas, seed, and
    "g"             the coupling quenched to, or a list of them (each gets its own
                    fresh start). 0 means zero temperature: a switch is accepted
                    only if it does not raise H (Metropolis only).
    "start"         "melt" (default): from a melted torus, as in the published
                    quenches. "torus": from the flat torus itself, to watch a
                    stable-for-now sheet give way (design brief, experiment 2).
    "record_every"  write every k-th sweep (default 1).
    "cap", "lambda", "acceptance", "seed_scheme": as in run_cqg_sweep.py.
One sweep = 2N attempted switches. Seed step 0 is the melt, step 1 + i the quench
to the i-th coupling.

Columns: phi = S/N, surplus = X/N, energy = H/N = 16 (1 - phi) + 4 lambda surplus,
and the connectivity numbers of ASSUMPTIONS
Q8 as they stand after that sweep (not averages): pieces, largest_frac,
baby_frac, cube_frac.
"""
import json
import platform
import sys
from pathlib import Path

import numba
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_cqg_sweep import model_of, seeder, shape_of     # noqa: E402  (the path line must come first)

from graphity import __version__                          # noqa: E402
from graphity.cqg import NO_CAP, is_valid, run_chain, torus   # noqa: E402
from graphity.results import ResultWriter                 # noqa: E402

ZERO_TEMPERATURE = 1e300       # 1/g for g = 0: exp(-dH / g) underflows to exactly 0 for every dH > 0


def main(path, out_dir="results"):
    cfg = json.loads(Path(path).read_text())
    lam, cap, glauber = model_of(cfg)
    couplings = [float(g) for g in (cfg["g"] if isinstance(cfg["g"], list) else [cfg["g"]])]
    if any(g < 0 or (g == 0 and glauber) for g in couplings):
        raise ValueError("g must be >= 0, and g = 0 (zero temperature) needs the Metropolis rule")
    start = cfg.get("start", "melt")
    if start not in ("melt", "torus"):
        raise ValueError(f"start must be 'melt' or 'torus', not {start!r}")
    every = int(cfg.get("record_every", 1))
    for entry in cfg["sides"]:                                               # fail before any work is done
        seeder(cfg, entry, 0)
    meta = dict(config=cfg, package=__version__, python=platform.python_version(),
                numpy=np.__version__, numba=numba.__version__)
    with ResultWriter(cfg["name"], meta, out_dir) as out:
        for entry in cfg["sides"]:
            lx, ly = shape_of(entry)
            n = lx * ly
            for rep in range(cfg["replicas"]):
                seed_of = seeder(cfg, entry, rep)
                for i, g in enumerate(couplings):
                    inv_g = ZERO_TEMPERATURE if g == 0 else 1.0 / g
                    adj, part = torus(lx, ly, cap)
                    side_u = np.flatnonzero(part == 0)
                    if start == "melt":
                        run_chain(adj, side_u, 0.0, cfg["n_melt"], 1, seed_of(0), lam, cap, glauber)   # hot start (Q5)
                    conn = np.zeros((cfg["n_sweeps"], 4), dtype=np.int64)
                    s, x, acc = run_chain(adj, side_u, inv_g, 0, cfg["n_sweeps"], seed_of(1 + i), lam, cap, glauber,
                                          conn)
                    assert is_valid(adj, cap)
                    for j in range(every - 1, cfg["n_sweeps"], every):
                        out.write(dict(N=n, replica=rep, sweep=j + 1, phi=float(s[j] / n), surplus=float(x[j] / n),
                                       pieces=int(conn[j, 0]), largest_frac=float(conn[j, 1] / n),
                                       baby_frac=float(conn[j, 2] / n), cube_frac=float(16 * conn[j, 3] / n),
                                       lx=lx, ly=ly, lam=lam, cap=("none" if cap == NO_CAP else cap), g=g,
                                       energy=float(16 * (1 - s[j] / n) + 4 * lam * x[j] / n)))
                    print(f"N = {n}, replica {rep}, g = {g}: phi {s[0] / n:.3f} -> {s[-1] / n:.3f}, "
                          f"pieces {conn[-1, 0]}, acceptance {acc:.4f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
