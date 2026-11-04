"""T83: is curl on a warm two-dimensional sheet felt at a distance? Usage:

    python scripts/run_t83_curl_correlation.py configs/t83_<name>.json

Writes results/<name>.csv, .meta.json and results/<name>_curves/*.npz (one C(r) per cell and replica).

Each cell (lambda, g, size, replica) is its own chain: a perfect flat torus, no cap, Metropolis, `n_equil` sweeps
of warm-up, then `n_meas` measured sweeps in blocks of `snapshot_every`, one snapshot after each block, the random
stream seeded once per cell and carried on (ASSUMPTIONS Q14). From each snapshot: the per-edge field S_e / 2 and its
correlation by graph distance, C(r), exactly as T16 measured it (`graphity.correlation`, ASSUMPTIONS Q19); the share
of edges that are flat (S_e = 2), which says whether the sheet has survived the coupling; and phi. The row holds the
snapshot-averaged C(r) at r = 1 .. 6, the flat share, phi, the acceptance and tau_int.
"""
import json
import platform
import sys
import warnings
from pathlib import Path

import numba
import numpy as np

from graphity import __version__
from graphity.analysis import autocorr_time
from graphity.correlation import correlation_by_distance, correlation_curve, edge_squares
from graphity.cqg import NO_CAP, is_valid, run_chain, torus
from graphity.results import ResultWriter, provenance

MAX_R = 16
warnings.filterwarnings("ignore", "Mean of empty slice")


def main(path, out_dir="results"):
    cfg = json.loads(Path(path).read_text())
    if cfg.get("cap", None) is not None or cfg.get("acceptance", "metropolis") != "metropolis":
        raise ValueError("T83 runs without the cap, Metropolis only")
    name = cfg["name"]
    meta = dict(config=cfg, config_path=str(path), package=__version__, python=platform.python_version(),
                numpy=np.__version__, numba=numba.__version__, preregistration="PREREGISTRATION.md section T83",
                provenance=provenance())
    store = Path(out_dir) / (name + "_curves")
    block = int(cfg["snapshot_every"])
    with ResultWriter(name, meta, out_dir) as out:
        store.mkdir(parents=True, exist_ok=True)
        for lam in cfg["lambdas"]:
            for side in cfg["sides"]:
                lx, ly = (side, side) if isinstance(side, int) else (int(side[0]), int(side[1]))
                n = lx * ly
                for g in cfg["couplings"]:
                    for rep in range(int(cfg["replicas"])):
                        seed = int(np.random.SeedSequence([cfg["seed"], int(round(lam * 1000)), lx, ly,
                                                           int(round(g * 1000)), rep]).generate_state(1)[0])
                        adj, part = torus(lx, ly, NO_CAP)
                        side_u = np.flatnonzero(part == 0)
                        run_chain(adj, side_u, 1.0 / g, int(cfg["n_equil"]), 0, seed, lam, NO_CAP, False)
                        curves, flat, s_all, acc = [], [], [], 0.0
                        for _ in range(int(cfg["n_meas"]) // block):
                            s, _, a = run_chain(adj, side_u, 1.0 / g, 0, block, -1, lam, NO_CAP, False)
                            s_all.append(s)
                            acc += a
                            num, count, edge_var, _, diam = correlation_by_distance(adj, MAX_R)
                            curves.append(correlation_curve(num, count, edge_var))
                            flat.append(float(np.mean(edge_squares(adj) == 2)))
                        assert is_valid(adj, NO_CAP)
                        s = np.concatenate(s_all).astype(np.int64)
                        c = np.nanmean(np.array(curves), axis=0)
                        c_err = np.nanstd(np.array(curves), axis=0, ddof=1) / np.sqrt(len(curves))
                        np.savez_compressed(store / ("lam%d_L%dx%d_g%d_rep%d.npz" % (round(lam * 100), lx, ly,
                                                                                   round(g * 100), rep)),
                                            c_mean=c, c_err=c_err, curves=np.array(curves), flat=np.array(flat))
                        row = dict(lam=lam, lx=lx, ly=ly, N=n, g=g, replica=rep, phi=float(s.mean() / n),
                                   chi=float(s.var() / n), tau_int=autocorr_time(s / n),
                                   acceptance=acc / len(s_all), flat_share=float(np.mean(flat)),
                                   flat_share_min=float(np.min(flat)), snapshots=len(curves))
                        for r in range(1, 7):
                            row["c%d" % r] = float(c[r]) if np.isfinite(c[r]) else ""
                            row["c%d_err" % r] = float(c_err[r]) if np.isfinite(c_err[r]) else ""
                        out.write(row)
                        print("lam=%.2f L=%d g=%.2f rep=%d flat=%.3f c1=%.3f c2=%.3f c3=%.3f" % (
                            lam, lx, g, rep, row["flat_share"], c[1], c[2], c[3]), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
