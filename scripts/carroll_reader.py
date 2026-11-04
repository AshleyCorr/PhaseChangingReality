"""T91: the toy's states read through Cao, Carroll and Michalakis's mutual-information geometry, on windows. Exact.

Usage:
    python scripts/carroll_reader.py <stage> [radius ...]     stage is "stage1" (R = 2, 3, 4, 6) or "heldout" (R = 10)

THE READER ([CCM16], Secs. III.B and III.C; docs/reading/notes/2026-11-02_carroll_space_from_entanglement.md). Each
point is a factor of the Hilbert space; the mutual information between two points defines a distance, d = Φ(I/I0) with
Φ = −ln (their Eq. 13). With a constant mutual information on every link and the shortest-path completion (their Eq. 14),
that distance is the graph distance times a constant, so the reader on a classical wiring is: the graph distances
between the points of a window, then classical multidimensional scaling (their Eqs. 17 to 26). Constants drop out.

WINDOWS. The reader is applied to a ball of graph radius R round a center point, using full-graph shortest-path
distances between the ball's points (whole tori read four for every shape because each wrapped cycle takes two large
eigenvalues; the note's section 7 as corrected). For each window: the eigenvalues of B = −J D² J / 2 in decreasing order,
λ1 ≥ λ2 ≥ ..., their reading rule ε_D = 1 − Σ_{i≤D} |λ_i| / Σ |λ_i| (unnumbered in [CCM16]; their Eq. 26 is the stress), and the registered count

    D_MDS = the number of eigenvalues at least 0.1 λ1.

Also the toy's own local dimension at the center (graphity.dimension.local_dimension: squares-free pairs at a point).

STATES. Built (four links): flat 32 × 32 torus; the curled tube 4 × 64; the 4-cube (the 4 × 4 torus). Saved: T51's
opened 1,024-point sheets after the slowest cooling, the first wiring of each of the first four runs
(results/t51_l256_300k_*_adj; their knots, O107; centers every eighth point) and T78's warm
melts at N = 196, g = 3.3 (results/t78_curl_vs_flat_n196_g3300_adj, snapshot 800). Centers: one point of each built
shape (every point is equivalent by symmetry), every eighth point of each T51 wiring, every point of each melt.
Pure functions, tested in tests/test_carroll_reader.py.
"""
import csv
import glob
import json
import platform
import sys
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphity import __version__                     # noqa: E402
from graphity.cqg import NO_CAP, torus                 # noqa: E402
from graphity.dimension import local_dimension         # noqa: E402

THRESHOLD = 0.1
STAGES = {"stage1": (2, 3, 4, 6), "heldout": (10,)}


def graph_of(adj):
    g = nx.Graph()
    for u in range(len(adj)):
        for v in adj[u]:
            if v >= 0:
                g.add_edge(int(u), int(v))
    return g


def mds_spectrum(dist):
    """Eigenvalues of B = −J D² J / 2, decreasing (classical multidimensional scaling, [CCM16] Eqs. 22 to 25)."""
    n = len(dist)
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ (np.asarray(dist, dtype=float) ** 2) @ j
    return np.sort(np.linalg.eigvalsh(b))[::-1]


def d_mds(w, threshold=THRESHOLD):
    """The registered count: eigenvalues at least `threshold` times the largest."""
    return int(np.sum(w >= threshold * w[0])) if w[0] > 0 else 0


def eps(w, d):
    """Their reading rule ε_D ([CCM16] Sec. III.C, unnumbered; Eq. 26 is the stress function)."""
    a = np.abs(w)
    return float(1 - a[:d].sum() / a.sum()) if a.sum() > 0 else float("nan")


def window(g, center, radius, lengths=None):
    """The ball of graph radius `radius` round `center` and the full-graph distances between its points."""
    if lengths is None:
        lengths = dict(nx.single_source_shortest_path_length(g, center, cutoff=radius))
    pts = sorted(lengths)
    sub = {p: nx.single_source_shortest_path_length(g, p) for p in pts}
    return pts, [[sub[p][q] for q in pts] for p in pts]


def read_state(adj, name, radii, centers=None):
    g = graph_of(adj)
    dims = local_dimension(adj)
    rows = []
    for c in (centers if centers is not None else range(len(adj))):
        for r in radii:
            pts, dist = window(g, c, r)
            w = mds_spectrum(dist)
            rows.append(dict(state=name, center=int(c), R=r, points=len(pts), d_local=int(dims[c]), d_mds=d_mds(w),
                             l1=float(w[0]), l2=float(w[1]) if len(w) > 1 else 0.0,
                             l3=float(w[2]) if len(w) > 2 else 0.0, l4=float(w[3]) if len(w) > 3 else 0.0,
                             l5=float(w[4]) if len(w) > 4 else 0.0,
                             eps1=eps(w, 1), eps2=eps(w, 2), eps3=eps(w, 3), eps4=eps(w, 4)))
    return rows


def states():
    out = [("flat_32x32", torus(32, 32, NO_CAP)[0], [0]),
           ("tube_4x64", torus(4, 64, NO_CAP)[0], [0]),
           ("cube_4x4", torus(4, 4, NO_CAP)[0], [0])]
    for f in [sorted(glob.glob(d + "/*.npz"))[0] for d in sorted(glob.glob("results/t51_l256_300k_*_adj"))[:4]]:
        adj = np.load(f)["adj"]
        out.append(("t51_" + Path(f).parent.name.replace("_adj", "") + "_" + Path(f).stem, adj, list(range(0, len(adj), 8))))
    for f in sorted(glob.glob("results/t78_curl_vs_flat_n196_g3300_adj/*snap800.npz"))[:4]:
        adj = np.load(f)["adj"]
        out.append(("t78_melt_" + Path(f).stem, adj, None))
    return out


def main(stage, out_dir="results"):
    radii = STAGES[stage]
    name = "t91_carroll_reader_" + stage
    path = Path(out_dir) / (name + ".csv")
    if path.exists():
        raise SystemExit("refusing to overwrite " + str(path))
    rows = []
    for sname, adj, centers in states():
        rows += read_state(adj, sname, radii, centers)
        print(sname, "done", flush=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    meta = dict(script="scripts/carroll_reader.py", stage=stage, radii=radii, threshold=THRESHOLD, package=__version__,
                python=platform.python_version(), numpy=np.__version__, networkx=nx.__version__,
                preregistration="PREREGISTRATION.md section T91")
    (Path(out_dir) / (name + ".meta.json")).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
