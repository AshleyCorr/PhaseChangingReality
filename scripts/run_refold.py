"""T21: does a sealed sheet given energy fold rather than melt when the points are interchangeable? Usage:

    python scripts/run_refold.py configs/t21_refold_n64_interchangeable.json

Implements PREREGISTRATION.md section T21 (series paper 4, the black-hole piece). O20's measurement
(`scripts/run_sealed_sheet_budget.py`) repeated at N = 36 and 64 with named and with interchangeable points
side by side, under two sealed protocols:
  "single"  O20's own: one demon holding the whole budget (samples the graph nearly uniformly below the
            total energy, which favours disorder; kept so the comparison with O20 is like for like);
  "bath"    C = 2N demons, the budget starting in one of them (T9's bath; a proper temperature).
Named runs use graphity.sealed (run_sealed, run_sealed_bath); interchangeable runs use
graphity.interchangeable.run with the same demons (ASSUMPTIONS Q20). The product is read as O20 reads it:
vertices whose local dimension fell below 2 are folded, above 2 melted; the folded share of the damage is
folded / (folded + melted).
"""
import json
import platform
import sys
from pathlib import Path

import numba
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphity import __version__, interchangeable, symmetry               # noqa: E402
from graphity.cqg import NO_CAP, hamiltonian, torus                       # noqa: E402
from graphity.dimension import local_dimension                            # noqa: E402
from graphity.results import ResultWriter                                 # noqa: E402
from graphity.sealed import run_sealed, run_sealed_bath                   # noqa: E402


def one_run(lx, ly, budget, protocol, points, n_sweeps, seed, lam):
    adj, part = torus(lx, ly, NO_CAP)
    n = lx * ly
    side_u = np.flatnonzero(part == 0)
    e0 = hamiltonian(adj, lam) + budget * n
    demons = np.zeros(1 if protocol == "single" else 2 * n)
    demons[0] = budget * n
    if points == "interchangeable":
        s, x, _, total = interchangeable.run(adj, part, n_sweeps, seed, lam, NO_CAP, demons=demons)
        end_total = total[-1]
    elif protocol == "single":
        s, x, demon, lost, _ = run_sealed(adj, side_u, float(budget * n), n_sweeps, seed, lam, NO_CAP)
        end_total = demon[-1]
    else:
        s, x, _, total, _ = run_sealed_bath(adj, side_u, demons, n_sweeps, seed, lam, NO_CAP)
        end_total = total[-1]
    h_end = 16.0 * (n - s[-1]) + 4.0 * lam * x[-1]
    hist = np.bincount(local_dimension(adj), minlength=7)[:7]
    folded, melted = int(hist[0] + hist[1]), int(hist[3:].sum())
    return dict(folded=folded, flat=int(hist[2]), melted=melted,
                folded_share=(folded / (folded + melted)) if folded + melted else "",
                phi_end=s[-1] / n, h_end_per_point=h_end / n, drift=abs(h_end + end_total - e0),
                symmetries_end=symmetry.count(adj, part), **{"d%d" % i: int(hist[i]) for i in range(7)})


def main(path, out_dir="results"):
    cfg = json.loads(Path(path).read_text())
    lam = float(cfg["lambda"])
    lx, ly = cfg["side"]
    meta = dict(config=cfg, config_path=str(path), package=__version__,
                python=platform.python_version(), numpy=np.__version__, numba=numba.__version__,
                preregistration="PREREGISTRATION.md section T21, written 2026-09-24")
    with ResultWriter(cfg["name"], meta, out_dir) as out:
        for protocol in cfg["protocols"]:
            budgets = (cfg["budgets_by_protocol"][protocol] if "budgets_by_protocol" in cfg
                       else cfg["budgets_per_point"])
            for budget in budgets:
                for rep in range(int(cfg["replicas"])):
                    seed = int(np.random.SeedSequence([int(cfg["seed"]), lx * ly, int(budget * 100), rep,
                                                       0 if protocol == "single" else 1]).generate_state(1)[0])
                    row = one_run(lx, ly, float(budget), protocol, cfg["points"], int(cfg["n_sweeps"]), seed, lam)
                    out.write(dict(N=lx * ly, points=cfg["points"], protocol=protocol, budget_per_point=budget,
                                   replica=rep, **row))
                    print("N=%d %s %s budget %.1f rep %d: folded %d flat %d melted %d share %s drift %.1e A %d"
                          % (lx * ly, cfg["points"], protocol, budget, rep, row["folded"], row["flat"],
                             row["melted"], row["folded_share"], row["drift"], row["symmetries_end"]), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "results")
