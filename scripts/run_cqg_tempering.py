"""Parallel tempering for the 2D CQG model: all couplings at once, graphs swapping between them. Usage:

    python scripts/run_cqg_tempering.py configs/<name>.json

Use it where a single chain freezes (run_cqg_sweep.py below g of about 4). The
method and why it is exact: src/graphity/tempering.py. One row per (size,
replica, coupling), with the columns of run_cqg_sweep.py (leg is "temper") and:
    swap_rate    how often a swap with the next COLDER coupling was accepted
                 (empty for the coldest). Aim for 0.2 to 0.4; outside that, change
                 the spacing of the couplings, do not just run longer.
    round_trips  how many times a graph went from the hottest coupling to the
                 coldest and back during the whole run. Zero means the cold end was
                 never refreshed from the hot end and its numbers are not to be trusted.

Config keys: name, sides, replicas, seed, and
    "couplings"         hottest first, strictly decreasing.
    "start"             "melt" (default): every copy starts from its own melted torus.
                        "torus": every copy starts from the flat torus.
    "n_melt"            sweeps at infinite temperature for "melt".
    "sweeps_per_round"  ordinary sweeps between two sets of swap offers.
    "rounds_equil", "rounds_meas"   rounds thrown away, then rounds recorded.
    "cap", "lambda", "acceptance"   as in run_cqg_sweep.py.
Seeds all come from numpy's SeedSequence([seed, lx, ly, replica]) (ASSUMPTION Q11).
"""
import json
import platform
import sys
from pathlib import Path

import numba
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_cqg_sweep import model_of, shape_of                     # noqa: E402  (the path line must come first)

from graphity import __version__                                  # noqa: E402
from graphity.analysis import autocorr_time, block_bootstrap_mean_var   # noqa: E402
from graphity.cqg import NO_CAP, is_valid, run_chain, torus      # noqa: E402
from graphity.results import ResultWriter                         # noqa: E402
from graphity.tempering import temper                             # noqa: E402


def main(path, out_dir="results"):
    cfg = json.loads(Path(path).read_text())
    lam, cap, glauber = model_of(cfg)
    couplings = [float(g) for g in cfg["couplings"]]
    start = cfg.get("start", "melt")
    if start not in ("melt", "torus"):
        raise ValueError(f"start must be 'melt' or 'torus', not {start!r}")
    if any(a <= b for a, b in zip(couplings, couplings[1:])):
        raise ValueError("couplings must be strictly decreasing (hottest first)")
    meta = dict(config=cfg, package=__version__, python=platform.python_version(),
                numpy=np.__version__, numba=numba.__version__)
    with ResultWriter(cfg["name"], meta, out_dir) as out:
        for entry in cfg["sides"]:
            lx, ly = shape_of(entry)
            n = lx * ly
            for rep in range(cfg["replicas"]):
                melt_seeds, temper_seeds = np.random.SeedSequence([cfg["seed"], lx, ly, rep]).spawn(2)
                graphs = []
                for melt_seed in melt_seeds.generate_state(len(couplings)):
                    adj, part = torus(lx, ly, cap)
                    side_u = np.flatnonzero(part == 0)
                    if start == "melt":
                        run_chain(adj, side_u, 0.0, cfg["n_melt"], 1, int(melt_seed), lam, cap, glauber)
                    graphs.append(adj)
                result = temper(graphs, side_u, couplings, cfg["rounds_equil"] + cfg["rounds_meas"],
                                cfg["sweeps_per_round"], temper_seeds, lam, cap, glauber,
                                measure_from=cfg["rounds_equil"])
                assert all(is_valid(g, cap) for g in graphs)
                for k, g in enumerate(couplings):
                    phi = result["squares"][k] / n
                    conn = result["connectivity"][k]
                    _, phi_err, _, var_err = block_bootstrap_mean_var(phi, seed=cfg["seed"] + k)
                    row = dict(N=n, replica=rep, leg="temper", g=g, phi=float(phi.mean()), phi_sd=float(phi.std()),
                               chi=float(n * phi.var()), acceptance=float(result["acceptance"][k]),
                               phi_err=phi_err, chi_err=n * var_err, tau_int=autocorr_time(phi), lx=lx, ly=ly,
                               lam=lam, cap=("none" if cap == NO_CAP else cap),
                               surplus=float(result["surplus"][k].mean() / n),
                               pieces=float(conn[:, 0].mean()), largest_frac=float(conn[:, 1].mean() / n),
                               baby_frac=float(conn[:, 2].mean() / n), cube_frac=float(16 * conn[:, 3].mean() / n),
                               swap_rate=(float(result["swap_rate"][k]) if k < len(couplings) - 1 else ""),
                               round_trips=int(result["round_trips"]))
                    out.write(row)
                print(f"N = {n}, replica {rep}: round trips {result['round_trips']}, swap rates "
                      + " ".join(f"{r:.2f}" for r in result["swap_rate"])
                      + f"; coldest phi {result['squares'][-1].mean() / n:.3f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
