"""T15 rung 1: does the time spent in each arrangement equal the count's prediction? Usage:

    python scripts/run_t15_rung1.py configs/t15_rung1.json

Implements PREREGISTRATION.md section T15 rung 1 (docs/design/quantum_loop_design.md, rung 1). At N = 16 and
18 every arrangement (class up to renaming within sides) is listed by exploration (small_graphs.explore, Q9),
with its squares S, surplus X and symmetry count A. With interchangeable points each class has probability
proportional to exp(-H/g); with named points, (n!)^2/A exp(-H/g). Two routes are checked against the
interchangeable probabilities:
  A. the interchangeable chain (graphity.interchangeable, Q20): fraction of sweeps spent in each class;
  B. the named chain (cqg.run_chain, one sweep at a time, random stream carried on): fraction of sweeps in
     each class, reweighted by A and renormalised.
The named chain's raw fractions are also compared with the named probabilities, as a control.
Classes are identified by igraph's canonical labelling (graphity.symmetry.canonical_key).
"""
import json
import platform
import sys
from math import exp, factorial
from pathlib import Path

import numba
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphity import __version__, interchangeable, symmetry               # noqa: E402
from graphity.cqg import NO_CAP, run_chain, surplus, total_squares         # noqa: E402
from graphity.results import ResultWriter                                 # noqa: E402
from graphity.small_graphs import circulant, explore                      # noqa: E402


def class_table(n_half):
    start = circulant(n_half)
    part = np.array([0] * n_half + [1] * n_half)
    classes, _ = explore(start, NO_CAP)
    table = []
    for adj, _ in classes.reps:
        table.append(dict(S=int(total_squares(adj)), X=int(surplus(adj)), A=symmetry.count(adj, part),
                          key=symmetry.canonical_key(adj, part)))
    return start, part, table


def exact(table, n_half, lam, g):
    h = np.array([16 * (2 * n_half - c["S"]) + 4 * lam * c["X"] for c in table])
    w_int = np.exp(-(h - h.min()) / g)
    w_nam = w_int * np.array([factorial(n_half) ** 2 / c["A"] for c in table])
    return w_int / w_int.sum(), w_nam / w_nam.sum()


def main(path, out_dir="results"):
    cfg = json.loads(Path(path).read_text())
    meta = dict(config=cfg, config_path=str(path), package=__version__,
                python=platform.python_version(), numpy=np.__version__, numba=numba.__version__,
                preregistration="PREREGISTRATION.md section T15 rung 1, written 2026-09-24")
    burn, meas = int(cfg["burn_in"]), int(cfg["sweeps"])
    with ResultWriter(cfg["name"], meta, out_dir) as out:
        for n_half in cfg["n_half"]:
            start, part, table = class_table(n_half)
            index = {c["key"]: k for k, c in enumerate(table)}
            side_u = np.flatnonzero(part == 0)
            print("N=%d: %d classes, symmetries %s" % (2 * n_half, len(table), sorted(c["A"] for c in table)), flush=True)
            for lam, g in cfg["settings"]:
                p_int, p_nam = exact(table, n_half, lam, g)
                for rep in range(int(cfg["replicas"])):
                    seed = int(np.random.SeedSequence([int(cfg["seed"]), n_half, int(lam * 100), int(g * 100), rep]).generate_state(1)[0])
                    counts_i = np.zeros(len(table))

                    def tally(sweep, adj, counts=counts_i):
                        if sweep >= burn:
                            counts[index[symmetry.canonical_key(adj, part)]] += 1

                    adj = start.copy()
                    interchangeable.run(adj, part, burn + meas, seed, lam, NO_CAP, g=g, on_sweep=tally)
                    counts_n = np.zeros(len(table))
                    adj = start.copy()
                    for sweep in range(burn + meas):
                        run_chain(adj, side_u, 1.0 / g, 0, 1, seed if sweep == 0 else -1, lam, NO_CAP, False)
                        if sweep >= burn:
                            counts_n[index[symmetry.canonical_key(adj, part)]] += 1
                    f_i = counts_i / counts_i.sum()
                    f_n = counts_n / counts_n.sum()
                    reweighted = f_n * np.array([c["A"] for c in table])
                    reweighted /= reweighted.sum()
                    for k, c in enumerate(table):
                        out.write(dict(N=2 * n_half, lam=lam, g=g, replica=rep, class_id=k, S=c["S"], X=c["X"], A=c["A"],
                                       p_interchangeable=p_int[k], p_named=p_nam[k],
                                       f_interchangeable_chain=f_i[k], f_named_chain=f_n[k],
                                       f_named_reweighted=reweighted[k]))
                    print("  N=%d lam %.1f g %.1f rep %d: TV(A route) %.3f  TV(B route) %.3f  TV(named control) %.3f"
                          % (2 * n_half, lam, g, rep, 0.5 * np.abs(f_i - p_int).sum(), 0.5 * np.abs(reweighted - p_int).sum(),
                             0.5 * np.abs(f_n - p_nam).sum()), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "results")
