"""T93: a zero-dimensional cell attached inside a six-link rod, the rod fed evenly; where does the opening start?

Usage:
    python scripts/run_rod_cell.py <stage>        stage is "stage1" (L = 16) or "heldout" (L = 24)

THE QUESTION (VISION Update 154, the owner's picture: dark matter and photon energy organize around a zero-dimensional
space inside the rod, and new space opens there). In the six-link toy a point at local dimension 0 is a point every pair
of whose links (except opposite pairs) closes a square: every point of the 6-cube (the 4 x 4 x 4 torus). A whole 6-cube
uses all six links of each point, so it can sit inside a rod only joined by switches. Here a 6-cube is joined to the rod
4 x 4 x L at its middle layer z = L/2 by two edge switches, the rod is fed evenly (every side-0 rod point's store gets
the same energy), and the question is where along the rod the first open region forms: at the junction or anywhere.

THE CONTROLS. (1) The same rod joined in the same way to a 4 x 4 x 8 rod piece (local dimension 1, not 0): the joins
are defects, and a defect alone may draw the opening; this control has the same defects with no zero-dimensional cell.
(2) The plain rod, nothing attached: where the opening starts when nothing marks a place.

THE MODEL. Six links, named points, sealed, the owner's tie "all at the last" f = (0, a, 2a, 0), a = 4(λ − 1), λ = 1.25,
per-vertex stores (graphity.sealed_tie_d.run_sealed_bath_table_d with by_vertex=True): T88's setting, so that the two
compare. Energy E_STORE in each side-0 rod point's store, nothing in the attached piece.

THE JOIN (exact, deterministic). A switch exchanges the ends of a rod link (u_r, v_r) and a piece link (u_c, v_c), u on
side 0 and v on side 1 of each, giving (u_r, v_c) and (u_c, v_r); degrees and the bipartition are kept. Candidates are
taken in a fixed order (rod links at the middle layer, piece links at the piece's first layer) and the first two that
keep the hard-core rule (graphity.cqg_d.is_valid) are used.

READ every RECORD_EVERY sweeps: local_dimension_d at every point; among rod points, the open regions (connected sets of
d = 3 points of at least MIN_REGION points) and the layer profile of d = 3 points; the piece's count of d = 0 points.
The first record with an open region gives the start: the circular mean layer of that region's points and its circular
distance (in layers) to the junction layer L/2.
Pure functions, tested in tests/test_rod_cell.py.
"""
import csv
import json
import math
import platform
import sys
from pathlib import Path

import numba
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from graphity import __version__                                          # noqa: E402
from graphity.cqg_d import hamiltonian, is_valid, torus                    # noqa: E402
from graphity.dimension import local_dimension_d                           # noqa: E402
from graphity.sealed_tie_d import run_sealed_bath_table_d, table_total     # noqa: E402
from run_curled_bath_d import MIN_REGION                                   # noqa: E402
from run_curled_bath_tie_d import table_of                                 # noqa: E402

LAM, E_STORE, REPLICAS, N_SWEEPS, RECORD_EVERY, SEED = 1.25, 30.0, 16, 10000, 50, 20269301
STAGES = {"stage1": 16, "heldout": 24}
KINDS = ("cell", "rodpiece", "plain")


def coords_of(dims):
    return np.array(np.unravel_index(np.arange(int(np.prod(dims))), dims)).T


def _links(adj, part, points):
    """Links (u on side 0, v on side 1) at the given points, in a fixed order."""
    out = []
    for u in points:
        if part[u] != 0:
            continue
        for v in adj[u]:
            out.append((int(u), int(v)))
    return out


def joined(rod_dims, piece_dims, joins=2):
    """The rod with a piece joined by `joins` switches at its middle layer. Returns adj, part, n_rod, junction layer."""
    a_r, p_r = torus(rod_dims)
    a_c, p_c = torus(piece_dims)
    n_r = len(a_r)
    adj = np.vstack([a_r, a_c + n_r])
    part = np.concatenate([p_r, p_c])
    zr = coords_of(rod_dims)[:, 2]
    zc = coords_of(piece_dims)[:, 2]
    mid = rod_dims[2] // 2
    rod_links = _links(adj, part, np.flatnonzero(zr == mid))
    piece_links = _links(adj, part, n_r + np.flatnonzero(zc == 0))
    done = 0
    used = set()
    for (ur, vr) in rod_links:
        if done == joins:
            break
        for (uc, vc) in piece_links:
            if {ur, vr, uc, vc} & used:
                continue
            trial = adj.copy()
            ir = list(trial[ur]).index(vr); jr = list(trial[vr]).index(ur)
            ic = list(trial[uc]).index(vc); jc = list(trial[vc]).index(uc)
            trial[ur, ir] = vc; trial[vc, jc] = ur
            trial[uc, ic] = vr; trial[vr, jr] = uc
            if is_valid(trial):
                adj = trial
                used |= {ur, vr, uc, vc}
                done += 1
                break
    if done < joins:
        raise RuntimeError("could not make %d valid joins" % joins)
    return adj, part, n_r, mid


def build(kind, length):
    rod = (4, 4, length)
    if kind == "cell":
        return joined(rod, (4, 4, 4))
    if kind == "rodpiece":
        return joined(rod, (4, 4, 8))
    adj, part = torus(rod)
    return adj, part, len(adj), length // 2


def open_regions(adj, mask):
    """Connected components (as index arrays) of the points where mask is true, largest first."""
    n = len(adj)
    seen = np.zeros(n, dtype=bool)
    comps = []
    for s in np.flatnonzero(mask):
        if seen[s]:
            continue
        stack, comp = [s], []
        seen[s] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if v >= 0 and mask[v] and not seen[v]:
                    seen[v] = True
                    stack.append(v)
        comps.append(np.array(comp))
    comps.sort(key=len, reverse=True)
    return comps


def circular_mean_layer(layers, length):
    ang = 2 * math.pi * np.asarray(layers) / length
    m = math.atan2(np.sin(ang).mean(), np.cos(ang).mean())
    return (m * length / (2 * math.pi)) % length


def circular_distance(a, b, length):
    d = abs(a - b) % length
    return min(d, length - d)


def reading(adj, n_rod, rod_z, length, mid):
    d = local_dimension_d(adj)
    rod_mask = np.zeros(len(adj), dtype=bool)
    rod_mask[:n_rod] = True
    at3 = (d == 3) & rod_mask
    regions = [c for c in open_regions(adj, at3) if len(c) >= MIN_REGION]
    start, dist = "", ""
    if regions:
        start = circular_mean_layer(rod_z[regions[0]], length)
        dist = circular_distance(start, mid, length)
    return dict(n_open3=int(at3.sum()), regions=len(regions), largest=int(len(regions[0])) if regions else 0,
                start_layer=start, start_dist=dist, piece_d0=int(((d == 0) & ~rod_mask).sum()),
                damaged=int((d > 3).sum()), profile=" ".join(str(int(c)) for c in np.bincount(rod_z[at3], minlength=length)))


def main(stage, out_dir="results"):
    length = STAGES[stage]
    name = "t93_rod_cell_" + stage
    path = Path(out_dir) / (name + ".csv")
    if path.exists():
        raise SystemExit("refusing to overwrite " + str(path))
    rows = []
    for kind in KINDS:
        adj0, part0, n_rod, mid = build(kind, length)
        rod_z = coords_of((4, 4, length))[:, 2]
        rod_z = np.concatenate([rod_z, np.full(len(adj0) - n_rod, -1)])
        ftab = table_of("all_at_the_last", LAM, 3)
        for rep in range(REPLICAS):
            adj, part = adj0.copy(), part0.copy()
            side_u = np.flatnonzero(part == 0)
            stores = np.zeros(len(adj))
            fed = np.flatnonzero((part == 0) & (np.arange(len(adj)) < n_rod))
            stores[fed] = E_STORE
            e0 = float(hamiltonian(adj, LAM)) + float(table_total(adj, ftab)) + float(stores.sum())
            seed = int(np.random.SeedSequence([SEED, length, KINDS.index(kind), rep]).generate_state(1)[0])
            first = None
            for b in range(1, N_SWEEPS // RECORD_EVERY + 1):
                run_sealed_bath_table_d(adj, side_u, stores, RECORD_EVERY, seed if b == 1 else -1, LAM, ftab, True)
                r = reading(adj, n_rod, rod_z, length, mid)
                cons = float(hamiltonian(adj, LAM)) + float(table_total(adj, ftab)) + float(stores.sum()) - e0
                rows.append(dict(stage=stage, kind=kind, L=length, replica=rep, sweep=b * RECORD_EVERY,
                                 conserved=cons, **r))
                if first is None and r["regions"]:
                    first = (b * RECORD_EVERY, r["start_layer"], r["start_dist"])
            print(stage, kind, rep, "first open region:", first, flush=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    meta = dict(script="scripts/run_rod_cell.py", stage=stage, L=length, lam=LAM, e_store=E_STORE, replicas=REPLICAS,
                n_sweeps=N_SWEEPS, record_every=RECORD_EVERY, seed=SEED, kinds=KINDS, tie="all_at_the_last",
                package=__version__, python=platform.python_version(), numpy=np.__version__, numba=numba.__version__,
                preregistration="PREREGISTRATION.md section T93")
    (Path(out_dir) / (name + ".meta.json")).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
