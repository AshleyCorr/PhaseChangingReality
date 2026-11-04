"""T50: does the tube's opening front slow where energy sits? Usage:

    python scripts/run_front_matter.py configs/t50_front_matter_e3.json

Implements PREREGISTRATION.md section T50 (the owner's question of 26 September, "does the front slow where matter
sits?", the model's counterpart of time running slower near mass; programme piece 10). T47 part A's setting, unchanged:
a 4 x L tube at lambda = 1.25, one seed (move A at column 0, scripts/run_seeded_tube.plant_seeds), sealed with one store
per point (graphity.sealed.run_sealed_bath with by_vertex: only moves made from a point spend its store, and released
energy stays where it is released). One seed makes two fronts that move apart, one to the right and one to the left.

"Matter" here is energy placed before the run in the stores of every point of a band of `band_width` columns on the
right, centred `band_centre` columns from the seed, `energy_per_point` each (below the 12 that starts an opening, so the
band cannot open by itself). The mirror band on the left holds nothing, so each replica carries its own control: the
right front crosses the band, the left front crosses the same distance through bare tube. A config with energy 0 is the
control of the control.

Recorded every `record_every` sweeps: how many columns on each side of the seed have opened (a column is open when at
least 3 of its 4 points are at local dimension 2, the sheet's; the tube's is 1), the converted fraction f as in T47, the
energy left in the band's stores, and the conservation check. Config keys: name, section, side ([L, 4]), lambda,
band_centre, band_width, energy_per_point, replicas, n_sweeps, record_every, seed.
"""
import json
import platform
import sys
from pathlib import Path

import numba
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from graphity import __version__                                        # noqa: E402
from graphity.cqg import NO_CAP, hamiltonian, torus                     # noqa: E402
from graphity.dimension import local_dimension                          # noqa: E402
from graphity.results import ResultWriter                               # noqa: E402
from graphity.sealed import run_sealed_bath                             # noqa: E402
from run_seeded_tube import plant_seeds                                 # noqa: E402

PHI_TUBE, PHI_SHEET = 1.25, 1.0
LY = 4


def band_columns(lx, centre, width):
    """The band's columns on the right of the seed (column 0), and their mirror on the left."""
    right = [(centre - width // 2 + k) % lx for k in range(width)]
    return right, [(-c) % lx for c in right]


def opened_columns(adj, lx):
    """Per column, whether at least 3 of its 4 points are at d = 2 (vertex v sits in column v // 4, cqg.torus)."""
    d = np.asarray(local_dimension(adj)).reshape(lx, LY)
    return (d == 2).sum(axis=1) >= 3


def sides(opened, lx):
    """Opened columns on the right of the seed (1 .. L/2 - 1) and on the left (L/2 + 1 .. L - 1)."""
    return int(opened[1:lx // 2].sum()), int(opened[lx // 2 + 1:].sum())


def main(path, out_dir="results"):
    cfg = json.loads(Path(path).read_text())
    lam = float(cfg["lambda"])
    n_sweeps, every = int(cfg["n_sweeps"]), int(cfg["record_every"])
    lx, ly = cfg["side"]
    if ly != LY:
        raise ValueError("the tube is 4 points round")
    n = lx * ly
    right, mirror = band_columns(lx, int(cfg["band_centre"]), int(cfg["band_width"]))
    band_points = np.array([c * LY + y for c in right for y in range(LY)], dtype=np.int64)
    e_point = float(cfg["energy_per_point"])
    meta = dict(config=cfg, config_path=str(path), package=__version__, python=platform.python_version(),
                numpy=np.__version__, numba=numba.__version__,
                preregistration="PREREGISTRATION.md section %s" % cfg.get("section", "T50"))
    with ResultWriter(cfg["name"], meta, out_dir) as out:
        for rep in range(int(cfg["replicas"])):
            adj, part = torus(lx, ly, NO_CAP)
            side_u = np.flatnonzero(part == 0)
            cols = plant_seeds(adj, part, lx, ly, 1)
            stores = np.zeros(n)
            stores[band_points] = e_point
            e0 = hamiltonian(adj, lam) + stores.sum()
            seed = int(np.random.SeedSequence([int(cfg["seed"]), n, int(round(e_point * 100)), rep]).generate_state(1)[0])
            base = dict(N=n, L=lx, lam=lam, replica=rep, seed_column=cols[0], energy_per_point=e_point,
                        band_centre=int(cfg["band_centre"]), band_width=int(cfg["band_width"]))
            r, l_ = sides(opened_columns(adj, lx), lx)
            out.write(dict(**base, sweep=0, right=r, left=l_, f=0.0, band_energy=float(stores[band_points].sum()), drift=0.0))
            for b in range(1, n_sweeps // every + 1):
                s, x, mean, tot, _ = run_sealed_bath(adj, side_u, stores, every, seed if b == 1 else -1, lam, NO_CAP,
                                                     by_vertex=True)
                h = 16.0 * (n - s[-1]) + 4.0 * lam * x[-1]
                r, l_ = sides(opened_columns(adj, lx), lx)
                out.write(dict(**base, sweep=b * every, right=r, left=l_,
                               f=float((PHI_TUBE - s[-1] / n) / (PHI_TUBE - PHI_SHEET)),
                               band_energy=float(stores[band_points].sum()), drift=float(abs(h + tot[-1] - e0))))
            print("L=%d e=%g rep=%-2d right %d left %d (of %d) drift %.2g" % (lx, e_point, rep, r, l_, lx // 2 - 1,
                                                                             abs(h + tot[-1] - e0)), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "results")
