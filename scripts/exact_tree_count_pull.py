"""Would a massless field on the graph, or a count of hidden spanning trees, pull two identical defects together?
(gravity brief, section 6; series paper 5). Usage:

    python scripts/exact_tree_count_pull.py configs/exact_tree_count_pull.json

EXACT, exploratory: a calculation on fixed wirings of two candidate terms, neither of which is in the model. The
literature read on 25 September (docs/reading/notes/gravity_mechanisms_2026-09-25.md) says a pull at a distance needs
a medium with no energy gap, and the model's flat space has one (every move out of it costs 32, 64 or 128). The
simplest gapless medium a graph carries for free is a field on its points whose energy is the sum over links of the
squared difference across the link. Integrating that field out exactly leaves two terms in the free energy of a wiring
(ours, the standard Gaussian integral):

  (1) (g / 2) ln det' L, with L the graph Laplacian; by Kirchhoff's theorem det' L = N tau, tau the number of spanning
      trees. Two defects that change the wiring locally then interact through ln tau: a Casimir-type force.
  (2) if a defect is also a source for the field (a charge q on its points), -(1 / 2) q^T L^+ q, with L^+ the graph's
      own Green's function. Two sources then interact through q1^T L^+ q2: a Newton-type force, carried by the
      emergent geometry.

The opposite reading of (1), spanning trees counted as hidden structure so that a wiring's weight grows with tau,
flips its sign; both are reported. The defect is O56's: the switch that curls a line of four points into a 4-cycle
along x (four links: the 16 x 16 and 24 x 24 sheets; six links: the 8 x 8 x 8 torus). The second copy is the first
translated along y by an even step (and, in 3D, also diagonally), so both are identical and the sides are kept.
Printed and written per separation r: ln tau and the cross term q1^T L^+ q2 (charge 1 on each of the four points of
each defect), each relative to the farthest placement.
"""
import json
import platform
import sys
from pathlib import Path

import numba
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from graphity import __version__                                 # noqa: E402
from graphity.cqg_d import is_valid, torus                       # noqa: E402
from graphity.results import ResultWriter                        # noqa: E402
from exact_relic_d import index, make_relic                      # noqa: E402


def laplacian(adj):
    n = adj.shape[0]
    lap = np.zeros((n, n))
    for u in range(n):
        for v in adj[u]:
            lap[u, v] -= 1.0
        lap[u, u] += adj.shape[1]
    return lap


def log_trees(lap):
    """ln tau: the log-determinant of the Laplacian with one row and column removed (Kirchhoff)."""
    sign, val = np.linalg.slogdet(lap[1:, 1:])
    assert sign > 0
    return float(val)


def main(path, out_dir="results"):
    cfg = json.loads(Path(path).read_text())
    meta = dict(config=cfg, config_path=str(path), package=__version__, python=platform.python_version(),
                numpy=np.__version__, numba=numba.__version__, purpose=cfg.get("_purpose", ""))
    with ResultWriter(cfg["name"], meta, out_dir) as out:
        for dims in cfg["tori"]:
            coords, look = index(dims)
            flat, part = torus(dims)
            n = flat.shape[0]
            rows = []
            steps = [s for s in range(2, dims[1] // 2 + 1, 2)]
            placements = [("y", tuple([0, s] + [0] * (len(dims) - 2))) for s in steps]
            if len(dims) == 3:
                placements += [("yz", (0, s, s)) for s in steps]
            lt_flat = log_trees(laplacian(flat))
            one = flat.copy()
            make_relic(one, dims, look, tuple([0] * len(dims)))
            lt_one = log_trees(laplacian(one))
            for kind, shift in placements:
                g2 = flat.copy()
                a = make_relic(g2, dims, look, tuple([0] * len(dims)))
                b = make_relic(g2, dims, look, shift)
                if not is_valid(g2) or set(a) & set(b):
                    continue
                lap = laplacian(g2)
                q1, q2 = np.zeros(n), np.zeros(n)
                q1[a], q2[b] = 1.0, 1.0
                # q1 L+ q2 up to a constant: (L + J/N)^-1 = L+ + J/N on this connected graph, and the extra term
                # (sum q1)(sum q2)/N does not depend on the placement (a direct solve; the SVD behind pinv failed at 16^3)
                cross = float(q1 @ np.linalg.solve(lap + 1.0 / n, q2))
                r = int(np.abs(np.array(shift)).sum())
                rows.append(dict(dims="x".join(map(str, dims)), kind=kind, shift=" ".join(map(str, shift)), r=r,
                                 log_trees=log_trees(lap), cross=cross))
            # each kind of placement is referred to its own farthest placement
            far = {row["kind"]: row for row in rows}
            print("%s: ln tau flat %.4f, one defect %+.4f vs flat" % ("x".join(map(str, dims)), lt_flat, lt_one - lt_flat))
            for row in rows:
                row["d_log_trees"] = row["log_trees"] - far[row["kind"]]["log_trees"]
                row["d_cross"] = row["cross"] - far[row["kind"]]["cross"]
                row["ln_tau_one_minus_flat"] = lt_one - lt_flat
                out.write(row)
                print("  %-3s shift %-8s r=%-3d  ln tau - farthest %+.6f   q1 L+ q2 - farthest %+.5f"
                      % (row["kind"], row["shift"], row["r"], row["d_log_trees"], row["d_cross"]), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "results")
