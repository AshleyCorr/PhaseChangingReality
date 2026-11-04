"""T86: which of the toy's objects carries a wave under the adopted rule? Exact, no chain. Usage:

    python scripts/t86_wave_carrier.py <N> <npz> [<npz> ...]        (writes results/t86_wave_carrier_N<N>.csv and .meta.json)

Implements PREREGISTRATION.md section T86 (written 2026-10-30 before any registered computation). The rule (VISION
addendum of 2026-10-30; docs/design/amplitude_rule_brief.md section 3): a complex amplitude on every point of the
arrangement, i dpsi/dt = L psi with L the graph Laplacian of the WHOLE arrangement, probability |psi|^2.

For every saved final graph and every connected piece of points off the sheet (local dimension != 2;
graphity.dimension.piece_labels, as T55 read them), the object is the piece plus its collar (the sheet points one link
away), and its kind is the piece's size: 2 (a quarter of the eight-point knot of two kinds, O107), 4 (a column, the
curled loop of four), 8 (a double column), else "other". Two measures, both exact from one eigendecomposition per graph:

  1. MODE: the largest localisation of any eigenspace of L on the object (the squared norm of the projection of the
     eigenspace's orthonormal basis onto the object, divided by the eigenspace's dimension, as
     scripts/analyse_t15_resonator.py computes it; 1 for a mode living entirely on the object, |object|/N for one
     spread evenly).
  2. STAY: the probability still on the object at t = pi/2 after starting on one point of the piece (averaged over the
     piece's points), and the same at t = pi; the isolated loop of four keeps 1 at every t.

Controls per graph: three random connected sets of sheet points of the same sizes (seeded), with the same measures.

Words, per kind at a size, over all instances at that size: CARRIES if the majority of instances have MODE >= 0.5 and
STAY(pi/2) >= 0.5; LEAKS if the majority have STAY(pi/2) < 0.5; else MIXED. The verdict is read at the held-out size only
(rule 14). One row per object or control per graph. Pure functions, tested in tests/test_t86_t87.py.
"""
import json
import platform
import sys
from pathlib import Path

import numba
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from graphity import __version__                                   # noqa: E402
from graphity.dimension import local_dimension, piece_labels       # noqa: E402
from graphity.results import ResultWriter, provenance              # noqa: E402

SHEET_D = 2
KINDS = {2: "quarter_knot", 4: "column", 8: "double_column"}
T_HALF, T_PI = np.pi / 2, np.pi


def dense(adj):
    n = adj.shape[0]
    a = np.zeros((n, n))
    for v in range(n):
        for w in adj[v]:
            if w >= 0:
                a[v, w] = 1.0
    return a


def laplacian(a):
    return np.diag(a.sum(axis=1)) - a


def pieces_of(adj):
    d = local_dimension(adj)
    labels = piece_labels(adj, d != SHEET_D)
    if labels.max() < 0:
        return [], d
    return [tuple(int(v) for v in np.flatnonzero(labels == k)) for k in range(int(labels.max()) + 1)], d


def collar(adj, points):
    keep = set(points)
    return sorted(keep | {int(w) for v in points for w in adj[v] if w >= 0})


def eigenspaces(w, v, tol=1e-9):
    out, i = [], 0
    while i < len(w):
        j = i
        while j + 1 < len(w) and abs(w[j + 1] - w[i]) < tol:
            j += 1
        out.append((float(w[i]), v[:, i:j + 1]))
        i = j + 1
    return out


def best_mode(spaces, points):
    """Largest localisation over eigenspaces: the share of the eigenspace's weight on S (sum over an orthonormal basis
    of the eigenspace of the squared entries on S, divided by the eigenspace's dimension): 1 for a mode living entirely
    on S, |S|/N for one spread evenly."""
    best = 0.0
    for _, basis in spaces:
        sub = basis[points, :]                       # (|S|, dim)
        loc = float(np.sum(sub * sub)) / basis.shape[1]  # share of the eigenspace's weight on S, per dimension
        best = max(best, loc)
    return best


def stay(w, v, start_points, object_points, t):
    """Mean over start points of the probability on the object at time t under exp(-i L t)."""
    phase = np.exp(-1j * w * t)
    tot = 0.0
    for s in start_points:
        psi = v @ (phase * v[s, :])
        p = np.abs(psi) ** 2
        tot += float(p[object_points].sum())
    return tot / len(start_points)


def random_sheet_set(adj, d, size, rng):
    """A random connected set of `size` sheet points, grown from a random sheet point."""
    sheet = np.flatnonzero(d == SHEET_D)
    for _ in range(100):
        s = int(rng.choice(sheet))
        chosen = [s]
        while len(chosen) < size:
            frontier = [int(w) for v in chosen for w in adj[v] if w >= 0 and d[w] == SHEET_D and int(w) not in chosen]
            if not frontier:
                break
            chosen.append(int(rng.choice(frontier)))
        if len(chosen) == size:
            return sorted(chosen)
    return None


def measure(adj, w, v, spaces, points, label, kind, source, piece_size):
    obj = collar(adj, points)
    return dict(source=source, label=label, kind=kind, piece_size=piece_size, object_size=len(obj),
                points=" ".join(str(p) for p in points),
                mode=round(best_mode(spaces, obj), 6), mode_piece=round(best_mode(spaces, list(points)), 6),
                stay_half=round(stay(w, v, points, obj, T_HALF), 6), stay_pi=round(stay(w, v, points, obj, T_PI), 6),
                stay_piece_half=round(stay(w, v, points, list(points), T_HALF), 6))


def read_graph(adj, source, rng, n_controls=3):
    pieces, d = pieces_of(adj)
    if not pieces:
        return []
    w, v = np.linalg.eigh(laplacian(dense(adj)))
    spaces = eigenspaces(w, v)
    rows = []
    for points in pieces:
        kind = KINDS.get(len(points), "other")
        rows.append(measure(adj, w, v, spaces, points, "object", kind, source, len(points)))
    sizes = sorted({len(p) for p in pieces})
    for size in sizes:
        for _ in range(n_controls):
            pts = random_sheet_set(adj, d, size, rng)
            if pts is not None:
                rows.append(measure(adj, w, v, spaces, tuple(pts), "control", KINDS.get(size, "other"), source, size))
    return rows


def word(rows, kind):
    objs = [r for r in rows if r["label"] == "object" and r["kind"] == kind]
    if not objs:
        return "NONE", 0
    carries = sum(1 for r in objs if r["mode"] >= 0.5 and r["stay_half"] >= 0.5)
    leaks = sum(1 for r in objs if r["stay_half"] < 0.5)
    n = len(objs)
    if carries > n / 2:
        return "CARRIES", n
    if leaks > n / 2:
        return "LEAKS", n
    return "MIXED", n


def main(n_size, paths, out_dir="results"):
    name = "t86_wave_carrier_N%d" % n_size
    meta = dict(task="T86", N=n_size, files=[str(p) for p in paths], rule="i dpsi/dt = L psi, whole-arrangement Laplacian",
                controls_seed=20261030, n_controls=3, graphity=__version__, python=platform.python_version(),
                numpy=np.__version__, numba=numba.__version__, **provenance())
    rng = np.random.default_rng(meta["controls_seed"])
    all_rows = []
    with ResultWriter(name, meta, out_dir) as out:
        for p in paths:
            d = np.load(p)
            adj = d["adj"]
            assert adj.shape[0] == n_size, (p, adj.shape)
            rows = read_graph(adj, str(p), rng)
            for r in rows:
                out.write(r)
            all_rows.extend(rows)
    for kind in ["quarter_knot", "column", "double_column", "other"]:
        wd, n = word(all_rows, kind)
        objs = [r for r in all_rows if r["label"] == "object" and r["kind"] == kind]
        ctr = [r for r in all_rows if r["label"] == "control" and r["kind"] == kind]
        if objs:
            print("N = %d  %-14s %-8s over %d instances: mode median %.3f (controls %.3f); stay(pi/2) median %.3f (controls %.3f); stay(pi) median %.3f" % (
                n_size, kind, wd, n, np.median([r["mode"] for r in objs]), np.median([r["mode"] for r in ctr]) if ctr else float("nan"),
                np.median([r["stay_half"] for r in objs]), np.median([r["stay_half"] for r in ctr]) if ctr else float("nan"),
                np.median([r["stay_pi"] for r in objs])))
    return all_rows


if __name__ == "__main__":
    main(int(sys.argv[1]), [Path(p) for p in sys.argv[2:]])
