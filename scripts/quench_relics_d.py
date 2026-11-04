"""Where do damaged six-link flat tori come to rest? A search for the six-link relic by quenching (gravity, step 1). Usage:

    python scripts/quench_relics_d.py configs/quench_relics_d.json

EXPLORATORY (the question and the reading were written in the config before the run). The four-link relic was found as
the resting state of real decays (O13, O16), not by construction; the six-link search by construction found no relic
among one- and two-switch objects (O58). Here each saved final graph of a six-link run that ended damaged (T34: flat
6 x 6 x 6 given energy) is quenched: `cqg_d.run_chain` at a coupling so cold that only moves that lower the energy or
leave it unchanged are accepted, for `n_sweeps` sweeps. What remains is read exactly: the energy above flat, the connected
pieces of points not at d = 3 (flat), and whether the state is a dip, meaning every single switch whose first side-0 point
lies within distance 3 of a defect point (with its partner within distance 4) costs energy (`exact_walls_d.walls`; moves
farther away touch only flat space and cost at least 64). A dip with one small defect piece is a relic.
"""
import glob
import json
import platform
import sys
from pathlib import Path

import numba
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from graphity import __version__, symmetry                    # noqa: E402
from graphity.cqg_d import run_chain                          # noqa: E402
from graphity.dimension import local_dimension_d, piece_labels  # noqa: E402
from graphity.results import ResultWriter                     # noqa: E402
from graphity.sealed_d import energy_d                        # noqa: E402
from exact_walls_d import walls, within                       # noqa: E402

COLD = 1.0e6          # inverse coupling: exp(-COLD * dH) is zero for any uphill move


def dip_test(adj, part, lam, defect_points):
    near = set()
    for v in defect_points:
        near |= set(int(w) for w in within(adj, int(v), 3))
    cheapest = None
    for u in sorted(near):
        if part[u] != 0:
            continue
        w = walls(adj, part, lam, u1=u, top=1, near=4)
        if w and (cheapest is None or w[0][0] < cheapest[0]):
            cheapest = w[0]
    return cheapest


def main(path, out_dir="results"):
    cfg = json.loads(Path(path).read_text())
    lam = float(cfg["lambda"])
    meta = dict(config=cfg, config_path=str(path), package=__version__, python=platform.python_version(),
                numpy=np.__version__, numba=numba.__version__, purpose=cfg.get("_purpose", ""))
    with ResultWriter(cfg["name"], meta, out_dir) as out:
        files = []
        for pat in cfg["inputs"]:
            files += sorted(glob.glob(str(Path(pat))))
        for i, f in enumerate(files):
            z = np.load(f)
            adj, part = z["adj"].copy(), z["part"]
            n = adj.shape[0]
            side_u = np.flatnonzero(part == 0)
            h_before = float(energy_d(adj, lam))
            run_chain(adj, side_u, COLD, 0, int(cfg["n_sweeps"]), int(cfg["seed"]) + i, lam, False)
            h_after = float(energy_d(adj, lam))
            d = local_dimension_d(adj)
            labels = piece_labels(adj, d != 3)
            sizes = sorted(np.bincount(labels[labels >= 0]).tolist(), reverse=True) if labels.max() >= 0 else []
            defect = np.flatnonzero(d != 3)
            cheapest = dip_test(adj, part, lam, defect) if 0 < len(defect) <= cfg.get("max_defect_points", 64) else None
            is_dip = cheapest is not None and cheapest[0] > 0
            row = dict(source=Path(f).parent.name + "/" + Path(f).name, N=n, lam=lam, h_before=h_before, h_after=h_after,
                       pieces=len(sizes), sizes=" ".join(map(str, sizes[:12])), defect_points=len(defect),
                       d_hist=" ".join(str(int(v)) for v in np.bincount(d, minlength=8)[:8]),
                       cheapest_exit=(cheapest[0] if cheapest else ""), dip=is_dip,
                       symmetries=(symmetry.count(adj, part) if is_dip and len(sizes) == 1 else ""))
            out.write(row)
            print("%-40s H %8.1f -> %7.1f | defect pieces %d %s | cheapest exit %s -> %s"
                  % (row["source"], h_before, h_after, len(sizes), sizes[:6], row["cheapest_exit"], "DIP" if is_dip else "-"),
                  flush=True)
            if is_dip and cfg.get("save_dips"):
                d_dir = Path(out_dir) / (cfg["name"] + "_dips")
                d_dir.mkdir(parents=True, exist_ok=True)
                np.savez(d_dir / ("%s_%s" % (Path(f).parent.name, Path(f).name)), adj=adj, part=part, lam=lam, energy=h_after)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "results")
