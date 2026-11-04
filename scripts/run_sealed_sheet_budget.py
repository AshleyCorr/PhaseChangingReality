"""What does energy do to a sheet of space when it cannot leave? Usage:

    python scripts/run_sealed_sheet_budget.py configs/sealed_sheet_budget_lam125.json

EXPLORATORY (design track; TASKS T13 rung 3; the author's picture of a black hole). A black hole
is a sealed system: nothing leaves it, which is the fixed-total-energy setting of ASSUMPTIONS Q12.
The question it makes concrete is what concentrated energy does to flat space in this model:

  re-curl it   -- vertices whose local dimension falls to 1 or 0, a region folded back up, which
                  is the author's black hole and this model's X;
  melt it      -- vertices whose local dimension rises to 3 or more, a bubble of the random phase,
                  which is [T25]'s black hole.

The two cost different amounts and the difference is exact. At lambda > 1 a square added to an
edge that already carries two costs 4(lambda - 1) per added square; a square destroyed costs 16.
So curling is the cheaper damage per square, by four times at lambda = 1.25, and a budget that
cannot afford to melt may still afford to curl. The run measures which one the energy buys.

Each row is one replica at one budget: the budget per point, the demon (the energy still loose)
at the end, and the census of local dimension over the vertices. The sheet starts perfect, so
every vertex begins at d = 2 and any row that is not all-d=2 is damage the energy paid for.

These are square-count classes, not geometry certificates. The legacy temperature
column is NaN unless the config supplies a positive thermometer_step, justified
separately as an accessible level spacing. Even then it is a canonical proxy, not
a calibrated temperature. The energy coefficient 4*lambda is NOT that spacing.
The measured second-half mean is always recorded as demon_mean. Existing results
are retained; their temperature column used the old unjustified 4*lambda spacing.
"""
import json
import platform
import sys
from pathlib import Path

import numba
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphity import __version__                                          # noqa: E402
from graphity.cqg import ENERGY_PER_SQUARE, ENERGY_PER_SURPLUS, NO_CAP, torus, total_squares, surplus  # noqa: E402
from graphity.dimension import local_dimension                            # noqa: E402
from graphity.results import ResultWriter                                 # noqa: E402
from graphity.sealed import demon_temperature, run_sealed                 # noqa: E402

D_BINS = 7


def main(path, out_dir="results"):
    cfg = json.loads(Path(path).read_text())
    lam = float(cfg["lambda"])
    thermometer_step = cfg.get("thermometer_step")
    if thermometer_step is not None:
        thermometer_step = float(thermometer_step)
        if not np.isfinite(thermometer_step) or thermometer_step <= 0:
            raise ValueError("thermometer_step must be a finite positive level spacing")
    meta = dict(config=cfg, config_path=str(path), package=__version__,
                python=platform.python_version(), numpy=np.__version__,
                numba=numba.__version__, purpose=cfg.get("_purpose", ""),
                thermometer=dict(step=thermometer_step, status="conditional canonical proxy"))
    with ResultWriter(cfg["name"], meta, out_dir) as out:
        for lx, ly in cfg["sides"]:
            n = lx * ly
            for budget in cfg["budgets_per_point"]:
                for rep in range(int(cfg["replicas"])):
                    adj, part = torus(lx, ly, NO_CAP)
                    side_u = np.flatnonzero(part == 0)
                    seed = int(np.random.SeedSequence(
                        [int(cfg["seed"]), n, int(float(budget) * 100), rep]).generate_state(1)[0])
                    s, x, demon, lost, acceptance = run_sealed(
                        adj, side_u, float(budget) * n, int(cfg["n_sweeps"]), seed, lam, NO_CAP)
                    d = local_dimension(adj)
                    hist = np.bincount(d, minlength=D_BINS)[:D_BINS]
                    h = ENERGY_PER_SQUARE * (n - total_squares(adj)) + ENERGY_PER_SURPLUS * lam * surplus(adj)
                    settled = demon[int(cfg["n_sweeps"]) // 2:]
                    row = dict(n=n, budget_per_point=budget, replica=rep, acceptance=round(acceptance, 4),
                               h_per_point=h / n, demon_end=demon[-1],
                               demon_mean=float(np.mean(settled)),
                               temperature=(demon_temperature(settled, thermometer_step)
                                            if thermometer_step is not None else float("nan")),
                               squares=int(s[-1]), surplus=int(x[-1]),
                               curled=int(hist[0] + hist[1]), flat=int(hist[2]),
                               melted=int(hist[3:].sum()))
                    row.update({f"d{k}": int(hist[k]) for k in range(D_BINS)})
                    out.write(row)
                    print("N=%d budget %.2f rep %d: curled %d, flat %d, melted %d, demon %.0f" % (
                        n, budget, rep, row["curled"], row["flat"], row["melted"], demon[-1]), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
