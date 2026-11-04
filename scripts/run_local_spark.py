"""T26 and T27: what does energy packed into one place do to cold space? Usage:

    python scripts/run_local_spark.py configs/t26_stores_n144.json

Implements PREREGISTRATION.md sections T26 (the local spark, sealed) and T27 (the melt with a leak). A flat
lx x ly torus at lambda > 1 is given E units of energy in one place, by one of two protocols
(graphity.spark):

  "stores"  the sheet starts perfect; the energy sits in the per-vertex stores of the side-0 vertices
            within `radius` of vertex 0 (all of them when radius is null: energy spread evenly, T27's
            setting), under the per-vertex bath (sealed.run_sealed_bath, by_vertex=True).
  "patch"   the energy is put into the wiring near vertex 0 by uphill switches (spark.hot_patch), and the
            run starts in a cold empty shared bath of 2N stores (T10's box), or, with "local_heat": true,
            in empty per-vertex stores, so that the heat the patch gives off stays where it is released.
            With "points": "interchangeable" the chain is graphity.interchangeable.run with the empty
            shared bath.
  "bath"    T21's bath: the sheet starts perfect and the whole energy sits in one store of a shared bath of
            2N (any move anywhere may draw on it). Energy spread through the box, for T27's leak.

The run proceeds in blocks of `block` sweeps for `n_blocks` blocks, carrying the random stream on between
blocks (seed < 0). After each block, if `leak` > 0, every store keeps the fraction 1 - leak and the rest is
counted as lost (T27). Recorded after every block: the local-dimension census, the folded vertices (d < 2),
the melted ones (d > 2), the pieces of each, the energy in the wiring, in the stores, and lost, and the
conservation check H + stores + lost = E0. The final graph is saved when `save_adjacency` is set.
"""
import json
import platform
import sys
from pathlib import Path

import numba
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphity import __version__, interchangeable                       # noqa: E402
from graphity.cqg import NO_CAP, hamiltonian, torus                     # noqa: E402
from graphity.dimension import local_dimension, pieces_of               # noqa: E402
from graphity.results import ResultWriter                               # noqa: E402
from graphity.sealed import run_sealed_bath                             # noqa: E402
from graphity.spark import energy_of, hot_patch, local_stores           # noqa: E402

D_BINS = 7
CENTER = 0


def census(adj, lam):
    d = local_dimension(adj)
    hist = np.bincount(d, minlength=D_BINS)[:D_BINS]
    folded = pieces_of(adj, d < 2)
    damage = pieces_of(adj, d != 2)
    return dict(h=float(energy_of(adj, lam)), folded=int((d < 2).sum()), flat=int((d == 2).sum()),
                melted=int((d > 2).sum()), **{"d%d" % i: int(hist[i]) for i in range(D_BINS)},
                pieces_folded=len(folded), largest_folded=(folded[0] if folded else 0),
                pieces_damage=len(damage), largest_damage=(damage[0] if damage else 0))


def one_run(cfg, energy, rep, out, lam, adj_dir):
    lx, ly = cfg["side"]
    n = lx * ly
    protocol, points = cfg["protocol"], cfg.get("points", "named")
    radius = cfg.get("radius")
    leak = float(cfg.get("leak", 0.0))
    block, n_blocks = int(cfg["block"]), int(cfg["n_blocks"])
    if points == "interchangeable" and (protocol != "patch" or leak or cfg.get("local_heat")):
        raise ValueError("interchangeable points are implemented for the sealed patch protocol with the shared bath only")
    adj, part = torus(lx, ly, NO_CAP)
    side_u = np.flatnonzero(part == 0)
    seed = int(np.random.SeedSequence([int(cfg["seed"]), n, int(round(energy)), rep]).generate_state(1)[0])
    achieved = float(energy)
    if protocol == "stores":
        stores = local_stores(adj, part, CENTER, radius, energy)
        by_vertex = True
    elif protocol == "patch":
        rng = np.random.default_rng(np.random.SeedSequence([int(cfg["seed"]), n, int(round(energy)), rep, 1]))
        achieved = hot_patch(adj, part, CENTER, radius, energy, lam, NO_CAP, rng)
        by_vertex = bool(cfg.get("local_heat", False))
        stores = np.zeros(n if by_vertex else 2 * n)
    elif protocol == "bath":
        stores = np.zeros(2 * n)
        stores[0] = float(energy)
        by_vertex = False
    else:
        raise ValueError("protocol must be 'stores', 'patch' or 'bath'")
    e0 = hamiltonian(adj, lam) + stores.sum()
    lost = 0.0
    base = dict(N=n, protocol=protocol, points=points, E=energy, achieved=achieved, radius=("" if radius is None else radius),
                leak=leak, local_heat=by_vertex, replica=rep)

    def row(b, sweep, stores_total, stores_max, symmetries=""):
        c = census(adj, lam)
        drift = abs(c["h"] + stores_total + lost - e0)
        out.write(dict(**base, block=b, sweep=sweep, **c, stores_total=float(stores_total),
                       stores_max=float(stores_max), lost=float(lost), drift=float(drift),
                       symmetries=symmetries, final=(b == n_blocks)))

    row(0, 0, stores.sum(), stores.max())
    if points == "interchangeable":
        from graphity import symmetry
        state = dict(b=0)

        def on_sweep(sweep, graph):
            if (sweep + 1) % block == 0:
                state["b"] += 1
                row(state["b"], (sweep + 1), stores.sum(), stores.max(), symmetry.count(graph, part))
        interchangeable.run(adj, part, block * n_blocks, seed, lam, NO_CAP, demons=stores, on_sweep=on_sweep)
    else:
        for b in range(1, n_blocks + 1):
            run_sealed_bath(adj, side_u, stores, block, seed if b == 1 else -1, lam, NO_CAP, by_vertex=by_vertex)
            if leak > 0.0:
                gone = stores.sum() * leak
                stores *= (1.0 - leak)
                lost += gone
            row(b, b * block, stores.sum(), stores.max())
    if cfg.get("save_adjacency"):
        adj_dir.mkdir(parents=True, exist_ok=True)
        np.savez(adj_dir / ("E%d_rep%d.npz" % (int(round(energy)), rep)), adj=adj, part=part, lam=lam,
                 E=energy, achieved=achieved, e0=e0)
    c = census(adj, lam)
    print("%s %s E=%g rep=%-2d achieved %.1f | end: folded %d melted %d flat %d | stores %.1f lost %.1f"
          % (protocol, points, energy, rep, achieved, c["folded"], c["melted"], c["flat"], stores.sum(), lost),
          flush=True)


def main(path, out_dir="results"):
    cfg = json.loads(Path(path).read_text())
    lam = float(cfg["lambda"])
    meta = dict(config=cfg, config_path=str(path), package=__version__, python=platform.python_version(),
                numpy=np.__version__, numba=numba.__version__,
                preregistration="PREREGISTRATION.md section %s, written 2026-09-24" % cfg.get("section", "T26"))
    adj_dir = Path(out_dir) / (cfg["name"] + "_adj")
    with ResultWriter(cfg["name"], meta, out_dir) as out:
        for energy in cfg["energies"]:
            for rep in range(int(cfg["replicas"])):
                one_run(cfg, float(energy), rep, out, lam, adj_dir)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "results")
