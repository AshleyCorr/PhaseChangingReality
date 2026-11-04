"""T101: each point's own clock, counted. Does the toy's dynamics run slower or faster near a knot? Usage:

    python scripts/run_clock_count.py configs/t101_clock_l64_lam125_00.json [out_dir]

Option A of the clock addendum to VISION Update 156 (the owner's "a first proceed", 3 November 2026). T99's protocol
(scripts/run_scrap_freeze.py: a 4 x L tube opens by itself at g_hot, then cools to g_cold over t_cool fair sweeps and
holds), with the same seeds, so every tube opens and cools exactly as in T99 (a paired design; the final energy must equal
T99's row for row). The one addition is the per-point counter of accepted moves (`cqg.run_chain(..., moves=)`), which
uses no random numbers: `moves_open` over the opening and `moves_cool` over the cooling and hold.

Reading. At the end, the leftover points are the points not at d = 2 (analyse_t37.leftovers' definition). Every point
gets its graph distance to the nearest leftover point; NEAR is distance 0 or 1, MID is 2 or 3, FAR is 4 or more. Per
replica: the mean moves per point in each band, for the opening and for the cooling, and the final energy. Replicas with
no leftover (energy 0) have no NEAR band and are reported, not scored. Saved: the final graph and both counters.
"""
import json
import platform
import sys
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from graphity import __version__                                           # noqa: E402
from graphity.cqg import NO_CAP, hamiltonian, run_chain, torus              # noqa: E402
from graphity.dimension import local_dimension                               # noqa: E402
from graphity.results import ResultWriter                                    # noqa: E402
from run_scrap_freeze import LY, has_opened, schedule, stream_seed          # noqa: E402

BANDS = (("near", 0, 1), ("mid", 2, 3), ("far", 4, 10 ** 9))


def open_tube_counted(lx, lam, g_hot, seed, check_every, stop_flat, max_d1_piece, cap):
    adj, part = torus(lx, LY, NO_CAP)
    side_u = np.flatnonzero(part == 0)
    moves = np.zeros(adj.shape[0], dtype=np.int64)
    sweeps = 0
    while sweeps < cap:
        run_chain(adj, side_u, 1.0 / g_hot, 0, check_every, seed if sweeps == 0 else -1, lam, NO_CAP, False, None, moves)
        sweeps += check_every
        if has_opened(adj, stop_flat, max_d1_piece):
            return adj, part, moves, sweeps, True
    return adj, part, moves, sweeps, False


def cool_counted(adj, part, gs, block_sweeps, seed, lam):
    side_u = np.flatnonzero(part == 0)
    moves = np.zeros(adj.shape[0], dtype=np.int64)
    for b, g in enumerate(gs, start=1):
        run_chain(adj, side_u, 1.0 / g, 0, block_sweeps, seed if b == 1 else -1, lam, NO_CAP, False, None, moves)
    return moves


def distance_to_leftover(adj):
    d = local_dimension(adj)
    left = np.flatnonzero(d != 2)
    if len(left) == 0:
        return None
    g = nx.Graph([(u, int(v)) for u in range(adj.shape[0]) for v in adj[u] if v >= 0])
    dist = nx.multi_source_dijkstra_path_length(g, set(int(p) for p in left))
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
    lx, lam = int(cfg["side"][0]), float(cfg["lambda"])
    n = lx * LY
    g_hot, g_cold, hold, block, n_ref = (float(cfg["g_hot"]), float(cfg["g_cold"]), int(cfg["hold"]), int(cfg["block"]),
                                         int(cfg["n_ref"]))
    t_cool = int(cfg["t_cool"][0])
    block_sweeps = int(round(block * n / n_ref))
    gs = schedule(g_hot, g_cold, t_cool, hold, block)          # T99's schedule, unchanged
    adj_dir = Path(out_dir) / (cfg["name"] + "_adj")
    meta = dict(config=cfg, package=__version__, python=platform.python_version(), numpy=np.__version__,
                preregistration="PREREGISTRATION.md section T101")
    with ResultWriter(cfg["name"], meta, out_dir) as out:
        for rep in cfg["replica_ids"]:
            adj, part, m_open, open_sweeps, opened = open_tube_counted(
                lx, lam, g_hot, stream_seed(cfg["seed"], n, rep), int(cfg["open_check_every"]), float(cfg["open_stop_flat"]),
                int(cfg["open_max_d1_piece"]), int(cfg["open_cap"]))
            row = dict(N=n, L=lx, lam=lam, replica=rep, t_cool=t_cool, opened=opened, open_sweeps=open_sweeps,
                       cool_sweeps=block_sweeps * len(gs))
            if not opened:
                out.write(row)
                continue
            m_cool = cool_counted(adj, part, gs, block_sweeps, stream_seed(cfg["seed"], n, rep, t_cool), lam)
            dist = distance_to_leftover(adj)
            row.update(h_final=float(hamiltonian(adj, lam)), leftover_points=int((local_dimension(adj) != 2).sum()),
                       total_open=int(m_open.sum()), total_cool=int(m_cool.sum()))
            if dist is not None:
                row.update({"open_" + k: v for k, v in band_means(m_open, dist).items()})
                row.update({"cool_" + k: v for k, v in band_means(m_cool, dist).items()})
            out.write(row)
            adj_dir.mkdir(parents=True, exist_ok=True)
            target = adj_dir / ("N%d_rep%d_tc%d.npz" % (n, rep, t_cool))
            if target.exists():
                raise FileExistsError("results are append-only; %s exists" % target)
            np.savez(target, adj=adj, part=part, lam=lam, t_cool=t_cool, replica=rep, moves_open=m_open, moves_cool=m_cool)
            print("rep", rep, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "results")
