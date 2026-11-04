"""T6 by parallel tempering and energy histograms. Usage:

    python scripts/run_t6_tempering.py configs/t6_lam0_control.json

Writes results/<name>.csv, .meta.json and results/<name>_hist/*.npz, and refuses to overwrite
(rule 5). Implements PREREGISTRATION.md section T6 and adds nothing to it.

WHY TEMPERING AND NOT FLAT-HISTOGRAM SAMPLING. The flat-histogram instrument was built and
validated and then failed on this model at production size: zero round trips across its own
window, so its weights ran away and its second stage froze (ASSUMPTIONS Q17). [RdF15] asks our
exact question in a neighbouring model -- is the transition really first order -- and answers it
with parallel tempering plus energy histograms, which is also the machinery this project already
has and has validated against exact averages (Q11). The pre-registered observables, sizes, knob
settings, seeds, gates and criteria are all unchanged; only the method is.

WHAT IS WRITTEN. For each (size, replica, coupling), the moments of the energy and the whole
energy histogram. The histogram is the object the analysis needs: two humps, the gap between
them and the valley between them are all read off it.

REWEIGHTING, and why it is needed. The two humps carry equal weight at one particular coupling,
which will not be one of the couplings on the ladder. From a histogram measured at g, the
histogram at a nearby g' follows by multiplying each energy's count by exp(-H(1/g' - 1/g)) --
the standard single-histogram reweighting. That locates the transition coupling without running
there, and it is what makes "the coupling where the two humps balance" a usable definition.
Reweighting is only trustworthy where the two histograms overlap, so the analysis reports how
far it had to shift and refuses a shift that is too large.
"""
import json
import platform
import sys
from pathlib import Path

import numba
import numpy as np

from graphity import __version__
from graphity.analysis import autocorr_time
from graphity.cqg import ENERGY_PER_SQUARE, ENERGY_PER_SURPLUS, CAP, NO_CAP, is_valid, run_chain, torus
from graphity.results import ResultWriter
from graphity.tempering import temper


def shape_of(entry):
    return (entry, entry) if isinstance(entry, int) else (int(entry[0]), int(entry[1]))


def main(path, out_dir="results"):
    cfg = json.loads(Path(path).read_text())
    lam = float(cfg.get("lambda", 1.0))
    cap = NO_CAP if cfg.get("cap", None) is None else int(cfg["cap"])
    glauber = cfg.get("acceptance", "metropolis") == "glauber"
    couplings = [float(g) for g in cfg["couplings"]]
    if any(a <= b for a, b in zip(couplings, couplings[1:])):
        raise ValueError("couplings must be strictly decreasing (hottest first)")

    meta = dict(config=cfg, config_path=str(path), package=__version__,
                python=platform.python_version(), numpy=np.__version__, numba=numba.__version__,
                preregistration="PREREGISTRATION.md section T6, revised 2026-09-21",
                method="parallel tempering with energy histograms; see ASSUMPTIONS Q17 for why "
                       "the flat-histogram instrument was set aside")
    store = Path(out_dir) / (cfg["name"] + "_hist")
    store.mkdir(parents=True, exist_ok=True)

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
                    if cfg.get("start", "melt") == "melt":
                        run_chain(adj, side_u, 0.0, cfg["n_melt"], 1, int(melt_seed), lam, cap, glauber)
                    graphs.append(adj)
                res = temper(graphs, side_u, couplings,
                             cfg["rounds_equil"] + cfg["rounds_meas"], cfg["sweeps_per_round"],
                             temper_seeds, lam, cap, glauber, measure_from=cfg["rounds_equil"])
                assert all(is_valid(g, cap) for g in graphs)

                for k, g in enumerate(couplings):
                    s = res["squares"][k].astype(np.int64)
                    x = res["surplus"][k].astype(np.int64)
                    h = ENERGY_PER_SQUARE * (n - s) + ENERGY_PER_SURPLUS * lam * x
                    levels, counts = np.unique(h, return_counts=True)
                    # The joint (S, X) histogram, which is what the analysis now works from.
                    # S is an integer with unit spacing at every lambda, so a histogram in S has
                    # no comb in it; H does, because which energies are reachable is arithmetic
                    # between 16 and 4*lambda (PREREGISTRATION.md, T6 amendment 1). Storing the
                    # pair also fixes H exactly for any lambda, rather than only the one run.
                    pairs, pair_counts = np.unique(np.stack([s, x], axis=1), axis=0,
                                                   return_counts=True)
                    np.savez_compressed(store / ("N%d_rep%d_k%d.npz" % (n, rep, k)),
                                        levels=levels, counts=counts, g=g, lam=lam, N=n,
                                        s_bin=pairs[:, 0].astype(np.int32),
                                        x_bin=pairs[:, 1].astype(np.int32),
                                        sx_counts=pair_counts.astype(np.int64))
                    conn = res["connectivity"][k]
                    out.write(dict(
                        N=n, lx=lx, ly=ly, replica=rep, k=k, g=g, lam=lam,
                        cap=("none" if cap == NO_CAP else cap),
                        phi=float(s.mean() / n), surplus=float(x.mean() / n),
                        e_mean=float(h.mean() / n), e_var=float(h.var()),
                        c_per_point=float(h.var() / (g * g * n)),
                        m2=float((h ** 2).mean()), m4=float((h ** 4).mean()),
                        binder=float(1.0 - (h ** 4).mean() / (3.0 * (h ** 2).mean() ** 2)),
                        levels_seen=int(levels.size), sweeps=int(h.size),
                        tau_int=autocorr_time(s / n),
                        acceptance=float(res["acceptance"][k]),
                        swap_rate=(float(res["swap_rate"][k]) if k < len(couplings) - 1 else ""),
                        round_trips=int(res["round_trips"]),
                        pieces=float(conn[:, 0].mean()), largest_frac=float(conn[:, 1].mean() / n),
                        baby_frac=float(conn[:, 2].mean() / n),
                        cube_frac=float(16 * conn[:, 3].mean() / n)))
                print("N=%-5d rep=%-2d trips=%-4d swaps %s  phi cold %.3f  levels %d..%d"
                      % (n, rep, res["round_trips"],
                         " ".join("%.2f" % r for r in res["swap_rate"][:6]),
                         res["squares"][-1].mean() / n,
                         min(np.unique(res["squares"][k]).size for k in range(len(couplings))),
                         max(np.unique(res["squares"][k]).size for k in range(len(couplings)))),
                      flush=True)


if __name__ == "__main__":
    # The second argument is the output directory. It was silently ignored before, so a smoke
    # test aimed at a scratch directory wrote into results/ instead.
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "results")
