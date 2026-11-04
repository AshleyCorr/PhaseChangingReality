"""How would a field on the points (gravity brief, option A) weigh the rungs of the ladder? Usage:

    python scripts/exact_tree_count_ladder.py configs/exact_tree_count_ladder.json

EXACT, exploratory. For each arrangement: ln det' L / N, the field's free energy per point in units of g / 2 (det' L, the
product of the nonzero Laplacian eigenvalues, is N times the number of spanning trees; for a gas of k separate pieces the
zero modes are k and the product runs over the rest). Differences between rungs at equal N are what the field would add
to the curling cost per point.
"""
import json
import platform
import sys
from pathlib import Path

import numba
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from graphity import __version__                          # noqa: E402
from graphity.cqg_d import torus                          # noqa: E402
from graphity.results import ResultWriter                 # noqa: E402
from exact_walls_d import gas                             # noqa: E402
from exact_tree_count_pull import laplacian               # noqa: E402


def log_det_prime(adj):
    ev = np.linalg.eigvalsh(laplacian(adj))
    return float(np.sum(np.log(ev[ev > 1e-9]))), int(np.sum(ev <= 1e-9))


def main(path, out_dir="results"):
    cfg = json.loads(Path(path).read_text())
    meta = dict(config=cfg, config_path=str(path), package=__version__, python=platform.python_version(),
                numpy=np.__version__, numba=numba.__version__, purpose=cfg.get("_purpose", ""))
    with ResultWriter(cfg["name"], meta, out_dir) as out:
        for r in cfg["rungs"]:
            adj = gas(r["gas"]["dims"], r["gas"]["copies"])[0] if "gas" in r else torus(r["dims"])[0]
            n = adj.shape[0]
            ld, zeros = log_det_prime(adj)
            out.write(dict(links=r["links"], label=r["label"], N=n, zero_modes=zeros, log_det_prime=ld, per_point=ld / n))
            print("%d links  %-22s N=%-4d zero modes %d  ln det' L / N = %.5f" % (r["links"], r["label"], n, zeros, ld / n))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "results")
