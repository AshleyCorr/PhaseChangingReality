"""T56: a curled six-link torus in a bath at a fixed coupling under the owner's direction tie, no push, no box. Usage:

    python scripts/run_curled_bath_tie_d.py configs/t56_slab_l8_lam120_g20.json [out_dir]

Implements PREREGISTRATION.md section T56 (written 2026-10-09, before any run; VISION Update 47; ASSUMPTIONS O94,
O106): the exact torus with one or two directions curled (the slab 4 x L x L, the owner's shape for X; the rod
4 x 4 x L), six links, followed in the thermal chain with a tie of any shape, `sealed_tie_d.run_chain_table_d`
(Metropolis in H + T_f), at one coupling g. Named points. The model with the tie is our family, never CQG; every
six-link result carries VISION Update 24's caveat: the reproduction gate (Gate C) is open.

The tie. `"tie": "all_at_the_last"` is the owner's choice of 9 October: f = (0, a, 2a, 0) per point with
a = 4 (lambda - 1), the shape of ASSUMPTIONS O89; `"tie": "none"` is the untied control (f = 0, which makes this
runner run_curled_bath_d's chain draw for draw, tested); `"tie": [f0, f1, f2, f3]` any table. A point with more open
directions than three (damaged) costs 0 under any table (sealed_tie_d, as T45 and T46 ran it).

Everything else is run_curled_bath_d (T53): each replica starts from the exact torus, its stream seeded once from
(config seed, N, replica) and carried on block to block; after every block the graph is read with T48's census, the
open regions, the pieces of the whole graph, H and now T_f; reading does not change the chain. One CSV row per
reading and one for sweep 0; the final graph saved as <out_dir>/<name>_adj/rep<k>.npz when `save_adjacency` is set.
Results are append-only (rule 5).

Config keys: name, section, dims (e.g. [4, 12, 12]), lambda, g, tie, replicas, n_sweeps, record_every, seed,
save_adjacency. `n_sweeps` must be a whole number of blocks.
"""
import json
import platform
import sys
from pathlib import Path

import numba
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from graphity import __version__                                          # noqa: E402
from graphity.dimension import local_dimension_d, pieces_of               # noqa: E402
from graphity.results import ResultWriter                                 # noqa: E402
from graphity.sealed_tie_d import run_chain_table_d, table_total          # noqa: E402
from run_curled_bath_d import MIN_REGION, reading, refuse_if_saved, save_graph, seed_of   # noqa: E402
from run_sealed_curled_d import start_of                                  # noqa: E402


def table_of(tie, lam, dim=3):
    """ftab (length D + 1) for the config's `tie`."""
    a = 4.0 * (float(lam) - 1.0)
    if tie == "all_at_the_last":
        return np.array([0.0] + [k * a for k in range(1, dim)] + [0.0])
    if tie == "none":
        return np.zeros(dim + 1)
    tab = np.asarray([float(x) for x in tie], dtype=float)
    if tab.shape[0] != dim + 1:
        raise ValueError("a tie table needs D + 1 entries")
    return tab


def reading_with_tie(adj, lam, ftab):
    """T53's reading, plus the tie total and the open regions counted at d = 3 (the slab starts at d = 2, so T53's
    `open_regions`, counted at d >= 2, is the whole slab from the first reading; `open_regions3` is what "one place or
    several" reads for it: the separate connected pieces of at least MIN_REGION points at d = 3, and `n_open3`)."""
    r = reading(adj, lam)
    r["t_tie"] = float(table_total(adj, ftab))
    at3 = local_dimension_d(adj) == 3
    sizes = pieces_of(adj, at3)
    r["open_regions3"] = sum(1 for s in sizes if s >= MIN_REGION)
    r["n_open3"] = int(at3.sum())
    return r


def follow(adj, side_u, lam, g, ftab, n_sweeps, every, seed):
    """Run the tied chain at coupling g in blocks of `every` sweeps; after each block yield
    (sweep, acceptance, reading). The stream is seeded at the first block and carried on afterwards."""
    if n_sweeps % every:
        raise ValueError("n_sweeps must be a whole number of blocks of record_every")
    for b in range(1, n_sweeps // every + 1):
        _, _, _, acc = run_chain_table_d(adj, side_u, 1.0 / g, every, seed if b == 1 else -1, lam, ftab)
        yield b * every, float(acc), reading_with_tie(adj, lam, ftab)


def main(path, out_dir="results"):
    cfg = json.loads(Path(path).read_text())
    lam, g = float(cfg["lambda"]), float(cfg["g"])
    adj0, part0, label = start_of(cfg)
    n, dim = adj0.shape[0], adj0.shape[1] // 2
    ftab = table_of(cfg["tie"], lam, dim)
    every, n_sweeps, reps = int(cfg.get("record_every", 1000)), int(cfg["n_sweeps"]), int(cfg["replicas"])
    if n_sweeps % every:
        raise ValueError("n_sweeps must be a whole number of blocks of record_every")
    save = bool(cfg.get("save_adjacency"))
    adj_dir = Path(out_dir) / (cfg["name"] + "_adj")
    if save:
        refuse_if_saved(adj_dir, range(reps))
    meta = dict(config=cfg, config_path=str(path), package=__version__, python=platform.python_version(),
                numpy=np.__version__, numba=numba.__version__, ftab=[float(x) for x in ftab],
                preregistration="PREREGISTRATION.md section %s, written 2026-10-09" % cfg.get("section", "T56"))
    with ResultWriter(cfg["name"], meta, out_dir) as out:
        for rep in range(reps):
            adj, part = adj0.copy(), part0.copy()
            side_u = np.flatnonzero(part == 0)
            seed = seed_of(cfg["seed"], n, rep)
            base = dict(N=n, dims=label, lam=lam, g=g, tie=str(cfg["tie"]), replica=rep)
            r = reading_with_tie(adj, lam, ftab)
            e0 = r["h"] + r["t_tie"]
            out.write(dict(**base, sweep=0, **r, acceptance="", final=False))
            for sweep, acc, r in follow(adj, side_u, lam, g, ftab, n_sweeps, every, seed):
                out.write(dict(**base, sweep=sweep, **r, acceptance=acc, final=(sweep == n_sweeps)))
            if save:
                save_graph(adj_dir, rep, adj, part, lam, g, seed)
            print("dims=%s lambda=%g g=%g tie=%s rep=%-2d | end: d3 %d d2 %d d1 %d damaged %d | largest open %d | "
                  "regions %d | pieces %d | (H+T)/N %.3f (start %.3f)"
                  % (label, lam, g, cfg["tie"], rep, r["d3"], r["d2"], r["d1"], r["damaged_d"], r["largest_open_d"],
                     r["open_regions"], r["pieces"], (r["h"] + r["t_tie"]) / n, e0 / n), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "results")
