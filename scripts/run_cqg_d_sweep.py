"""Cooling-then-heating sweep at any dimension D (graphity/cqg_d.py). Usage:

    python scripts/run_cqg_d_sweep.py configs/gatec_t22_fig3.json

Written for Gate C (TASKS, six links per point): reproduce [T22] Fig. 3, squares per vertex against the coupling
at D = 3, N = 500, full curvature. Writes results/<name>.csv and .meta.json and refuses to overwrite either.

Config keys:
    "start"       {"circulant": m, "offsets": [...]}: 2m points, no squares at all (cqg_d.circulant), or
                  {"torus": [L1, ..., LD]}.
    "lambda"      default 1.0 (the published model).
    "acceptance"  "metropolis" (default) or "glauber".
    "couplings"   the cooling leg, in order; the heating leg is the same list reversed, carrying on from the cold
                  end (its first entry is not repeated).
    "n_melt", "n_equil", "n_meas", "replicas", "seed": as run_cqg_sweep.py; every (replica, step) gets its own
                  statistically independent seed.
                  With "n_melt": 0 the start state is used exactly as built (no sweep at infinite temperature).
    "heat_equil", "heat_meas"  optional sweep counts for the heating leg only (default: n_equil, n_meas), for a slow
                  cool followed by a fast heat. Configs without them run exactly as before.

Columns: squares_per_vertex = 4S/N (the published axis; 12 on the cubic lattice at D = 3), phi = S / (D(D-1)/2 N)
(1 on the flat D-torus), their error from a block bootstrap, tau_int in sweeps (ASSUMPTION Q6), acceptance, and
surplus = X/N.
"""
import json
import platform
import sys
from pathlib import Path

import numba
import numpy as np

from graphity import __version__
from graphity.analysis import autocorr_time, block_bootstrap_mean_var
from graphity.cqg_d import circulant, is_valid, run_chain, torus
from graphity.results import ResultWriter


def start_of(cfg):
    st = cfg["start"]
    if "circulant" in st:
        return circulant(int(st["circulant"]), [int(s) for s in st["offsets"]])
    if "torus" in st:
        return torus([int(x) for x in st["torus"]])
    raise ValueError("start must name 'circulant' or 'torus'")


def main(path, out_dir="results"):
    cfg = json.loads(Path(path).read_text())
    lam = float(cfg.get("lambda", 1.0))
    acceptance = cfg.get("acceptance", "metropolis")
    if acceptance not in ("metropolis", "glauber"):
        raise ValueError(f"acceptance must be 'metropolis' or 'glauber', not {acceptance!r}")
    glauber = acceptance == "glauber"
    adj0, part = start_of(cfg)
    if not is_valid(adj0):
        raise ValueError("the start graph breaks the hard-core rule")
    n, deg = adj0.shape
    dim = deg // 2
    flat_s = dim * (dim - 1) / 2 * n
    meta = dict(config=cfg, package=__version__, python=platform.python_version(),
                numpy=np.__version__, numba=numba.__version__)
    with ResultWriter(cfg["name"], meta, out_dir) as out:
        for rep in range(cfg["replicas"]):
            def seed_of(step):
                return int(np.random.SeedSequence([cfg["seed"], n, deg, rep, step]).generate_state(1)[0])
            adj = adj0.copy()
            side_u = np.flatnonzero(part == 0)
            if cfg["n_melt"] > 0:                                                        # hot start; with n_melt 0 the
                run_chain(adj, side_u, 0.0, cfg["n_melt"], 1, seed_of(0), lam, glauber)   # start state is used as it is
                                                                                          # (2026-09-25; before this, one
                                                                                          # infinite-temperature sweep
                                                                                          # still ran: explore_3d_window_*)
            gs = [float(g) for g in cfg["couplings"]]
            k = 0
            sweeps = {"cool": (cfg["n_equil"], cfg["n_meas"]),
                      "heat": (cfg.get("heat_equil", cfg["n_equil"]), cfg.get("heat_meas", cfg["n_meas"]))}
            for leg, values in [("cool", gs), ("heat", gs[::-1][1:])]:
                n_equil, n_meas = sweeps[leg]
                for g in values:
                    k += 1
                    s, x, acc = run_chain(adj, side_u, 1.0 / g, n_equil, n_meas, seed_of(k), lam, glauber)
                    spv = 4.0 * s / n
                    _, spv_err, _, _ = block_bootstrap_mean_var(spv, seed=seed_of(k))
                    row = dict(N=n, D=dim, replica=rep, leg=leg, g=g, ln_g=float(np.log(g)),
                               squares_per_vertex=float(spv.mean()), spv_err=spv_err,
                               phi=float(s.mean() / flat_s), acceptance=float(acc),
                               tau_int=autocorr_time(spv), surplus=float(x.mean() / n), lam=lam)
                    out.write(row)
                    print(row, flush=True)
            assert is_valid(adj)


if __name__ == "__main__":
    main(sys.argv[1])
