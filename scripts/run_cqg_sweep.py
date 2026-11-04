"""Cooling-then-heating sweep for the 2D CQG model. Usage:

    python scripts/run_cqg_sweep.py configs/cqg_first_look.json

Writes results/<name>.csv and .meta.json, and refuses to overwrite either (see
graphity/results.py). phi = S/N is the order parameter (1 on the flat torus, ~0
on a large random graph) [KTB19 Eq. (32), D = 2].

Config key "sides": each entry is either L (an L x L torus) or [lx, ly].

Optional config key "heat_start":
    "continue" (default)  the heating leg carries on from the cold end of the
                          cooling leg, whatever state that froze into.
    "torus"               the heating leg starts from a fresh flat torus, as in
                          the published protocol [T25 Fig. 3].

Optional config key "seed_scheme" (ASSUMPTION Q7):
    "legacy" (default)    seed + 100000 * replica + L + step. Kept so that old
                          configs reproduce bit for bit. It reuses seeds across
                          sizes and has no meaning for a rectangle, so [lx, ly]
                          entries are refused under it.
    "independent"         every (shape, replica, step) gets its own statistically
                          independent seed. Use this for every new config.

Optional config keys for the model (ASSUMPTIONS Q1, Q3, Q4). The defaults are
the model of the first look, so old configs reproduce bit for bit:
    "cap"         2 (default): no edge may carry more than two squares, the model
                  of [KTB19 Sec. 4]. null: no cap, the space of [T25].
    "lambda"      strength of the local term, default 1.0. H = 16 (N - S) + 4
                  lambda X. It has no effect under the cap, where X is zero.
    "acceptance"  "metropolis" (default) or "glauber" [T25 Eq. (28)].

Columns phi_err and chi_err are block-bootstrap error bars and tau_int is the
autocorrelation time in sweeps (ASSUMPTION Q6). Do not trust the error bars of a
row whose tau_int is not far below n_meas / 20, or is NaN (frozen chain).
Column surplus is X/N, the surplus squares on over-full edges per vertex: zero
under the cap and on a flat torus, 2 on a 4-cube. Without the cap phi can exceed 1.

Connectivity columns (task T3; graphity/connectivity.py), each an average over the
measurement sweeps: pieces is the number of connected pieces, largest_frac the
share of vertices in the largest piece, baby_frac the share of vertices in "baby
universes" (pieces with three squares on every edge, the lowest-energy pieces at
lambda = 0 [KTB19 Sec. 3.3.1]) and cube_frac the share in those baby universes
that are 4-cubes. A connected space has 1, 1, 0, 0; a graph shattered into 4-cubes
has N/16, 16/N, 1, 1. Looking uses no random numbers, so it leaves the chain, and
therefore every other column, exactly as it was.
"""
import json
import platform
import sys
from pathlib import Path

import numba
import numpy as np

from graphity import __version__
from graphity.analysis import autocorr_time, block_bootstrap_mean_var
from graphity.cqg import CAP, NO_CAP, is_valid, run_chain, torus
from graphity.results import ResultWriter


def shape_of(entry):
    """A "sides" entry is either L, meaning L x L, or a pair [lx, ly]."""
    if isinstance(entry, int):
        return entry, entry
    lx, ly = entry
    return int(lx), int(ly)


def seeder(cfg, entry, rep):
    """Return f(step) -> seed. Step 0 is the melt, then one step per coupling visited."""
    scheme = cfg.get("seed_scheme", "legacy")
    lx, ly = shape_of(entry)
    if scheme == "legacy":
        if not isinstance(entry, int):
            raise ValueError('[lx, ly] entries need "seed_scheme": "independent"')
        return lambda step: cfg["seed"] + 100000 * rep + entry + step
    if scheme == "independent":
        return lambda step: int(np.random.SeedSequence(
            [cfg["seed"], lx, ly, rep, step]).generate_state(1)[0])
    raise ValueError(f"seed_scheme must be 'legacy' or 'independent', not {scheme!r}")


def model_of(cfg):
    """Return (lam, cap, glauber) from the optional model keys; the defaults are the first-look model."""
    cap = cfg.get("cap", CAP)
    if cap not in (CAP, None):
        raise ValueError(f"cap must be {CAP} or null, not {cap!r}")
    acceptance = cfg.get("acceptance", "metropolis")
    if acceptance not in ("metropolis", "glauber"):
        raise ValueError(f"acceptance must be 'metropolis' or 'glauber', not {acceptance!r}")
    return float(cfg.get("lambda", 1.0)), (NO_CAP if cap is None else CAP), acceptance == "glauber"


def main(path, out_dir="results"):
    cfg = json.loads(Path(path).read_text())
    heat_start = cfg.get("heat_start", "continue")
    if heat_start not in ("continue", "torus"):
        raise ValueError(f"heat_start must be 'continue' or 'torus', not {heat_start!r}")
    lam, cap, glauber = model_of(cfg)
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
                adj, part = torus(lx, ly, cap)
                side_u = np.flatnonzero(part == 0)
                run_chain(adj, side_u, 0.0, cfg["n_melt"], 1, seed_of(0), lam, cap, glauber)   # hot start (Q5)
                gs = cfg["couplings"]
                if heat_start == "torus":
                    legs = [("cool", gs), ("heat", gs[::-1])]
                else:
                    legs = [("cool", gs), ("heat", gs[::-1][1:])]
                k = 0
                for leg, values in legs:
                    if leg == "heat" and heat_start == "torus":
                        assert is_valid(adj, cap)                            # check the cooled state before dropping it
                        adj, _ = torus(lx, ly, cap)
                    for g in values:
                        k += 1
                        conn = np.zeros((cfg["n_meas"], 4), dtype=np.int64)
                        s, x, acc = run_chain(adj, side_u, 1.0 / g, cfg["n_equil"], cfg["n_meas"],
                                              seed_of(k), lam, cap, glauber, conn)
                        phi = s / n
                        _, phi_err, _, var_err = block_bootstrap_mean_var(phi, seed=seed_of(k))
                        row = dict(N=n, replica=rep, leg=leg, g=g, phi=float(phi.mean()),
                                   phi_sd=float(phi.std()), chi=float(n * phi.var()),
                                   acceptance=float(acc), phi_err=phi_err,
                                   chi_err=n * var_err, tau_int=autocorr_time(phi),
                                   lx=lx, ly=ly, lam=lam, cap=("none" if cap == NO_CAP else cap),
                                   surplus=float(x.mean() / n),
                                   pieces=float(conn[:, 0].mean()),
                                   largest_frac=float(conn[:, 1].mean() / n),
                                   baby_frac=float(conn[:, 2].mean() / n),
                                   cube_frac=float(16 * conn[:, 3].mean() / n))
                        out.write(row)
                        print(row, flush=True)
                assert is_valid(adj, cap)


if __name__ == "__main__":
    main(sys.argv[1])
