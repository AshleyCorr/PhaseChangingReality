"""T53: a curled six-link torus in a bath at a fixed coupling, with no push and no box. Usage:

    python scripts/run_curled_bath_d.py configs/t53_l18_lam125_g15.json [out_dir]

Implements PREREGISTRATION.md section T53 (written 2026-10-05, before any run; VISION Update 42; ASSUMPTIONS O94,
O100): the exact torus C_4 x C_4 x C_L, six links, two directions curled and one open, 8(lambda - 1) per point above
the flat 3-torus, followed in the thermal chain `cqg_d.run_chain` (Metropolis) at one coupling g. Named points, no
tie, no new knob. Every six-link result carries VISION Update 24's caveat: the reproduction gate (Gate C) is open.

Each replica starts from the exact torus and runs `n_sweeps` sweeps in blocks of `record_every`. Its random stream is
seeded once, from (config seed, N, replica), and carried on from block to block (run_chain's seed < 0; ASSUMPTIONS
Q14), so the blocks make the same chain as one uninterrupted call (tested). After every block the graph is read, and
reading does not change the chain (tested). One CSV row per reading, and one for the start at sweep 0:

* the census T48 uses (`run_sealed_curled_d.census_any`), unchanged: `d0` .. `d7`, the points at each count of open
  directions d (d7 meaning 7 or more); `damaged_d`, the points with more open directions than three; `largest_open_d`,
  the largest connected piece of points at d = 3; and that census's own `h_per_vertex`, `melted`, `pieces_flat` and
  `largest_flat`;
* `open_regions`, the separate connected pieces of at least 16 points among the points at d >= 2, and `n_open2`, the
  number of points at d >= 2. Both read d >= 2 as written, with no upper limit, so a damaged point (d above 3) counts;
* `pieces`, the connected pieces of the whole graph (does an opening split the space; O100);
* `h`, the energy H = 16 (3N - S) + 4 lambda X from scratch (`cqg_d.hamiltonian`);
* `acceptance`, the share of attempted switches accepted during the block (empty at sweep 0);
* `final`, true on a replica's last row.

The final graph of each replica is saved as `<out_dir>/<name>_adj/rep<k>.npz` when `save_adjacency` is set. Results
are append-only (rule 5): the run stops before it starts if the CSV, its meta file or any of those graphs exists.

Config keys: name, section, dims (e.g. [4, 4, 18]), lambda, g, replicas, n_sweeps, record_every, seed,
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
from graphity.cqg_d import hamiltonian, run_chain                         # noqa: E402
from graphity.dimension import local_dimension_d, pieces_of               # noqa: E402
from graphity.results import ResultWriter                                 # noqa: E402
from run_sealed_curled_d import census_any, start_of                      # noqa: E402

MIN_REGION = 16      # points: the smallest connected piece counted as a separate open region (PREREGISTRATION T53)
OPEN_FROM = 2        # d >= 2: at least one of the two curled directions has opened at the point


def open_regions_of(adj, d=None):
    """(connected pieces of at least MIN_REGION points among the points at d >= OPEN_FROM, the number of such points).

    `d` is the count of open directions at each point; it is read from the graph when not given.
    """
    if d is None:
        d = local_dimension_d(adj)
    members = np.asarray(d) >= OPEN_FROM
    sizes = pieces_of(adj, members)
    return sum(1 for s in sizes if s >= MIN_REGION), int(members.sum())


def whole_pieces(adj):
    """Connected pieces of the whole graph, at any number of links.

    (connectivity.component_labels reads four link slots only, so it is not used at six links.)
    """
    return len(pieces_of(adj, np.ones(adj.shape[0], dtype=bool)))


def reading(adj, lam):
    """Everything one CSV row says about the graph. Reads only; the graph and the random stream are left alone."""
    regions, n_open2 = open_regions_of(adj)
    return dict(**census_any(adj, lam), open_regions=regions, n_open2=n_open2, pieces=whole_pieces(adj),
                h=float(hamiltonian(adj, lam)))


def seed_of(config_seed, n, replica):
    """The replica's seed, from (config seed, N, replica)."""
    return int(np.random.SeedSequence([int(config_seed), int(n), int(replica)]).generate_state(1)[0])


def follow(adj, side_u, lam, g, n_sweeps, every, seed, read=reading):
    """Run the chain at coupling g in blocks of `every` sweeps; after each block yield (sweep, acceptance, read(adj, lam)).

    The stream is seeded at the first block and carried on afterwards. Modifies adj in place.
    """
    if n_sweeps % every:
        raise ValueError("n_sweeps must be a whole number of blocks of record_every")
    for b in range(1, n_sweeps // every + 1):
        _, _, acc = run_chain(adj, side_u, 1.0 / g, 0, every, seed if b == 1 else -1, lam, False)
        yield b * every, float(acc), read(adj, lam)


def graph_path(adj_dir, replica):
    return Path(adj_dir) / ("rep%d.npz" % replica)


def refuse_if_saved(adj_dir, replicas):
    """Results are append-only (rule 5): never write over a saved graph."""
    for rep in replicas:
        p = graph_path(adj_dir, rep)
        if p.exists():
            raise FileExistsError("%s already exists. Results are append-only: to run this again, copy the config "
                                  "under a new name." % p)


def save_graph(adj_dir, replica, adj, part, lam, g, seed):
    refuse_if_saved(adj_dir, [replica])
    Path(adj_dir).mkdir(parents=True, exist_ok=True)
    np.savez(graph_path(adj_dir, replica), adj=adj, part=part, lam=lam, g=g, seed=seed)


def main(path, out_dir="results"):
    cfg = json.loads(Path(path).read_text())
    lam, g = float(cfg["lambda"]), float(cfg["g"])
    adj0, part0, label = start_of(cfg)
    n = adj0.shape[0]
    every, n_sweeps, reps = int(cfg.get("record_every", 1000)), int(cfg["n_sweeps"]), int(cfg["replicas"])
    if n_sweeps % every:
        raise ValueError("n_sweeps must be a whole number of blocks of record_every")
    save = bool(cfg.get("save_adjacency"))
    adj_dir = Path(out_dir) / (cfg["name"] + "_adj")
    if save:
        refuse_if_saved(adj_dir, range(reps))                             # before anything is run or written
    meta = dict(config=cfg, config_path=str(path), package=__version__, python=platform.python_version(),
                numpy=np.__version__, numba=numba.__version__,
                preregistration="PREREGISTRATION.md section %s, written 2026-10-05" % cfg.get("section", "T53"))
    with ResultWriter(cfg["name"], meta, out_dir) as out:
        for rep in range(reps):
            adj, part = adj0.copy(), part0.copy()
            side_u = np.flatnonzero(part == 0)
            seed = seed_of(cfg["seed"], n, rep)
            base = dict(N=n, dims=label, lam=lam, g=g, replica=rep)
            r = reading(adj, lam)
            h0 = r["h"]
            out.write(dict(**base, sweep=0, **r, acceptance="", final=False))
            for sweep, acc, r in follow(adj, side_u, lam, g, n_sweeps, every, seed):
                out.write(dict(**base, sweep=sweep, **r, acceptance=acc, final=(sweep == n_sweeps)))
            if save:
                save_graph(adj_dir, rep, adj, part, lam, g, seed)
            print("dims=%s lambda=%g g=%g rep=%-2d | end: d3 %d d2 %d d1 %d damaged %d | largest open %d | regions %d | "
                  "pieces %d | H/N %.3f (start %.3f)"
                  % (label, lam, g, rep, r["d3"], r["d2"], r["d1"], r["damaged_d"], r["largest_open_d"],
                     r["open_regions"], r["pieces"], r["h"] / n, h0 / n), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "results")
