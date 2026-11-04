"""Does a two-number coarse law predict the microscopic runs? PREREGISTRATION T12. Usage:

    python scripts/run_front_law.py configs/front_law_lam125.json

A tube part-converted into a sheet is supposed to be describable by the fraction converted and the
two fronts, and nothing else:

    H(f) = H(tube) - Delta * (converted points) + 2 sigma,   Delta = 4(lambda - 1), known exactly.

Each replica starts from a perfect tube (a torus with one side of length 4) and is followed in short
blocks. After every block the chain is paused -- not copied -- and three things are read: the energy,
the number of points that now read as sheet (local dimension 2), and the number of separate sheet
patches, which says whether the run has one converted region as the coarse model assumes or several.
The chain is advanced with seed < 0 after the first block so the random stream carries on (Q14), and
reading the graph uses no random numbers, so the run is the same as if it had never been paused.

Both acceptance rules are run, because prediction 4 is that the coarse law survives a change of the
microscopic rule.
"""
import json
import platform
import sys
from pathlib import Path

import numba
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphity import __version__                                          # noqa: E402
from graphity.cqg import (ENERGY_PER_SQUARE, ENERGY_PER_SURPLUS, NO_CAP,   # noqa: E402
                          run_chain, surplus, torus, total_squares)
from graphity.dimension import local_dimension, pieces_of                 # noqa: E402
from graphity.results import ResultWriter                                 # noqa: E402


def read_state(adj, lam, n):
    d = local_dimension(adj)
    sheet = d == 2
    converted = int(sheet.sum())
    patches = pieces_of(adj, sheet)
    h = ENERGY_PER_SQUARE * (n - total_squares(adj)) + ENERGY_PER_SURPLUS * lam * surplus(adj)
    return h, converted, len(patches), (patches[0] if patches else 0)


def main(path, out_dir="results"):
    cfg = json.loads(Path(path).read_text())
    lam = float(cfg["lambda"])
    block = int(cfg.get("block", 25))
    cap_sweeps = int(cfg["n_sweeps"])
    stop_at = float(cfg.get("stop_at", 0.98))
    meta = dict(config=cfg, config_path=str(path), package=__version__,
                python=platform.python_version(), numpy=np.__version__, numba=numba.__version__,
                preregistration="PREREGISTRATION.md section T12, written 2026-09-22")
    with ResultWriter(cfg["name"], meta, out_dir) as out:
        for glauber in cfg["acceptance"]:
            rule = "glauber" if glauber else "metropolis"
            for n in cfg["sizes"]:
                lx = n // 4
                for g in cfg["couplings"]:
                    for rep in range(int(cfg["replicas"])):
                        adj, part = torus(lx, 4, NO_CAP)
                        side_u = np.flatnonzero(part == 0)
                        seed = int(np.random.SeedSequence(
                            [int(cfg["seed"]), n, int(g * 100), rep, int(glauber)]).generate_state(1)[0])
                        done, first = 0, True
                        while done < cap_sweeps:
                            run_chain(adj, side_u, 1.0 / g, 0, block, seed if first else -1,
                                      lam, NO_CAP, bool(glauber))
                            first = False
                            done += block
                            h, converted, patches, largest = read_state(adj, lam, n)
                            out.write(dict(rule=rule, n=n, g=g, replica=rep, sweep=done,
                                           h=h, h_per_point=h / n, converted=converted,
                                           fraction=converted / n, patches=patches,
                                           largest_patch=largest))
                            if converted >= stop_at * n:
                                break
                        print("%-10s N=%3d g=%.1f rep %d: %.2f converted by sweep %d"
                              % (rule, n, g, rep, converted / n, done), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
