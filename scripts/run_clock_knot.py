"""T101 Amendment 1: the toy's own clock near one planted knot in flat space at lambda = 1 (the published model). Usage:

    python scripts/run_clock_knot.py configs/t101a_clock_knot_lam100.json [out_dir]

The owner's question of 3 November (22:19 UTC): is lambda = 1.25 the right setting for a clock test when her picture puts
open space at lambda <= 1? T101's cell (O184) was paired with T99 at 1.25. This cell is the one her picture asks for: a
flat 32 x 32 torus, one knot planted by a single random valid switch near the origin (as scripts/t85_gamma.py plants its
knot), held at a fixed coupling g for `sweeps` sweeps at lambda = 1, with each point's count of accepted moves recorded
(`cqg.run_chain(..., moves=)`, which uses no random numbers and leaves the chain bit for bit unchanged).

Reading. Bands by graph distance to the knot's points AT PLANTING (the points not at d = 2 after the switch): NEAR is
distance 0 or 1, MID 2 or 3, FAR 4 or more; the mean moves per point in each band, the final energy, the final count of
points not flat and whether the planted knot's points are still not flat. A cell (g) is scored only if the sheet stays
at least 90 % flat at the end in every replica; a melted cell is reported, not scored. No size claim, so no held-out size.
"""
import json
import platform
import sys
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphity import __version__                                           # noqa: E402
from graphity.cqg import NO_CAP, hamiltonian, is_valid, run_chain, torus   # noqa: E402
from graphity.dimension import local_dimension                               # noqa: E402
from graphity.results import ResultWriter                                    # noqa: E402

BANDS = (("near", 0, 1), ("mid", 2, 3), ("far", 4, 10 ** 9))


def _switch(adj, u1, v1, u2, v2):
    """Replace links u1-v1 and u2-v2 by u1-v2 and u2-v1 (the kernel's move) in place."""
    adj[u1, np.flatnonzero(adj[u1] == v1)[0]] = v2
    adj[u2, np.flatnonzero(adj[u2] == v2)[0]] = v1
    adj[v1, np.flatnonzero(adj[v1] == u1)[0]] = u2
    adj[v2, np.flatnonzero(adj[v2] == u2)[0]] = u1


def plant_knot(adj, part, rng, tries=100000):
    """One random valid switch between two side-0 points within graph distance 2 of point 0. Returns the four points."""
    g = nx.Graph([(u, int(v)) for u in range(adj.shape[0]) for v in adj[u] if v >= 0])
    near = sorted(p for p, d in nx.single_source_shortest_path_length(g, 0, cutoff=2).items() if part[p] == 0)
    for _ in range(tries):
        u1, u2 = (int(x) for x in rng.choice(near, 2, replace=False))
        v1, v2 = int(adj[u1, rng.integers(4)]), int(adj[u2, rng.integers(4)])
        if len({u1, v1, u2, v2}) < 4 or (adj[u2] == v1).any() or (adj[u1] == v2).any():
            continue
        _switch(adj, u1, v1, u2, v2)
        if is_valid(adj, NO_CAP):
            return (u1, v1, u2, v2)
        _switch(adj, u1, v2, u2, v1)
    raise RuntimeError("no valid switch found")


def distance_to(adj, points):
    g = nx.Graph([(u, int(v)) for u in range(adj.shape[0]) for v in adj[u] if v >= 0])
    dist = nx.multi_source_dijkstra_path_length(g, set(int(p) for p in points))
    return np.array([dist[p] for p in range(adj.shape[0])])


def band_means(moves, dist):
    out = {}
    for name, lo, hi in BANDS:
        sel = (dist >= lo) & (dist <= hi)
        out[name + "_points"] = int(sel.sum())
        out[name] = float(moves[sel].mean()) if sel.any() else float("nan")
    return out


def main(path, out_dir="results"):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    L, lam, sweeps = int(cfg["side"]), float(cfg["lambda"]), int(cfg["sweeps"])
    adj_dir = Path(out_dir) / (cfg["name"] + "_adj")
    meta = dict(config=cfg, package=__version__, python=platform.python_version(), numpy=np.__version__,
                preregistration="PREREGISTRATION.md section T101, Amendment 1")
    with ResultWriter(cfg["name"], meta, out_dir) as out:
        for g in cfg["g"]:
            for rep in cfg["replica_ids"]:
                seed = int(cfg["seed"]) + 1000 * rep + int(round(100 * g))
                rng = np.random.default_rng(seed)
                adj, part = torus(L, L, NO_CAP)
                quad = plant_knot(adj, part, rng)
                knot = np.flatnonzero(local_dimension(adj) != 2)
                h0 = float(hamiltonian(adj, lam))
                dist = distance_to(adj, knot)
                side_u = np.flatnonzero(part == 0)
                moves = np.zeros(adj.shape[0], dtype=np.int64)
                run_chain(adj, side_u, 1.0 / g, 0, sweeps, seed, lam, NO_CAP, False, None, moves)
                d_end = local_dimension(adj)
                row = dict(N=L * L, L=L, lam=lam, g=g, replica=rep, seed=seed, sweeps=sweeps, knot_points=int(len(knot)),
                           h_knot=h0, h_final=float(hamiltonian(adj, lam)), not_flat_final=int((d_end != 2).sum()),
                           flat_share=float((d_end == 2).mean()), knot_still=int((d_end[knot] != 2).sum()),
                           total_moves=int(moves.sum()))
                row.update(band_means(moves, dist))
                out.write(row)
                adj_dir.mkdir(parents=True, exist_ok=True)
                target = adj_dir / ("g%s_rep%d.npz" % (str(g).replace(".", "p"), rep))
                if target.exists():
                    raise FileExistsError("results are append-only; %s exists" % target)
                np.savez(target, adj=adj, part=part, lam=lam, g=g, replica=rep, knot=knot, quad=np.array(quad), moves=moves)
                print("g", g, "rep", rep, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "results")
