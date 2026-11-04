"""Where can a long-range force live in this energy? Exact, no sampling. Usage:

    python scripts/run_force_census.py configs/force_census.json

EXPLORATORY (TASKS T13 rung 3; follows O22). O22 proved that two defects whose damage shares no
square cost exactly twice one, at every separation, so there is no force between point defects.
That proof has three conditions -- fixed wiring, zero temperature, damage confined to a patch --
and this script tests the third, which is the one a defect can escape.

The test is the standard one for telling a local defect from a topological one: **price it in
systems of different size.** A defect whose damage sits in a patch costs the same in a big system
as in a small one. A defect whose damage cannot be confined costs more as the system grows, and
such defects interact over long distances even though the energy is local -- that is how vortices
in two dimensions come to have a logarithmic interaction (Kosterlitz-Thouless, from general
knowledge, not read by us).

Two objects are priced here:

  a point defect  -- the cheapest single switch out of a flat sheet, the same local pattern at
                     every size;
  a whole order   -- the gap per point between the flat sheet and the tube, which is what a
                     boundary between the two orders has to pay for every point it sweeps.

The second is the interesting one. If the gap per point is fixed, then a boundary between two
orders feels a force that does not fall off with distance at all: moving it one step converts a
fixed number of points at a fixed price each. That is a long-range force, from a local energy,
and this project has already watched it act -- it is what drives the front in T7.

PREDICTION, written before running (ours): the point defect costs the same at every size, so it
is local and O22's proof applies to it; the gap per point is exactly 4(lambda - 1) at every size,
so the boundary force is constant and does not weaken with separation.
"""
import json
import platform
import sys
from pathlib import Path

import numba
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphity import __version__                                          # noqa: E402
from graphity.cqg import NO_CAP, _switch, hamiltonian, is_valid, torus     # noqa: E402
from graphity.results import ResultWriter                                  # noqa: E402


def point_defect_cost(lx, ly, lam):
    """The same local switch pattern at any size: cross two neighbouring rungs of the lattice."""
    adj, _ = torus(lx, ly, NO_CAP)
    before = hamiltonian(adj, lam)
    u1, v1 = 0, ly            # vertex 0 and its neighbour one step along x
    u2, v2 = 1, ly + 1        # the same rung, one step along y
    _switch(adj, u1, v1, u2, v2)
    if not is_valid(adj, NO_CAP):
        return None, None
    return hamiltonian(adj, lam) - before, adj


def main(path, out_dir="results"):
    cfg = json.loads(Path(path).read_text())
    meta = dict(config=cfg, config_path=str(path), package=__version__,
                python=platform.python_version(), numpy=np.__version__,
                numba=numba.__version__, purpose=cfg.get("_purpose", ""))
    with ResultWriter(cfg["name"], meta, out_dir) as out:
        for lam in cfg["lambdas"]:
            lam = float(lam)
            for lx, ly in cfg["sides"]:
                n = lx * ly
                cost, _ = point_defect_cost(lx, ly, lam)
                # the two orders at this size: a flat sheet, and a tube (one side curled to 4)
                sheet, _ = torus(lx, ly, NO_CAP)
                h_sheet = hamiltonian(sheet, lam)
                long_side = n // 4                 # the tube is this size with one side curled to 4
                if long_side * 4 != n or long_side % 2 or long_side < 4:
                    h_tube = None                      # no tube of exactly this many points
                else:
                    tube, _ = torus(long_side, 4, NO_CAP)
                    h_tube = hamiltonian(tube, lam)
                out.write(dict(lam=lam, lx=lx, ly=ly, n=n,
                               point_defect=cost,
                               point_defect_per_point=None if cost is None else cost / n,
                               sheet=h_sheet, tube=h_tube,
                               gap=None if h_tube is None else h_tube - h_sheet,
                               gap_per_point=None if h_tube is None else (h_tube - h_sheet) / n,
                               expected_gap_per_point=4 * (lam - 1)))
                gpp = None if h_tube is None else (h_tube - h_sheet) / n
                print("lambda %.2f  N=%4d: point defect %s, gap per point %s (expected %+.3f)"
                      % (lam, n, cost, "none at this size" if gpp is None else "%+.3f" % gpp,
                         4 * (lam - 1)), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
