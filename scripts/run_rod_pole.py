"""T88, the pole test: a six-link rod fed at one end only. Usage:

    python scripts/run_rod_pole.py configs/t88_rod_l8_lam125_e120.json [out_dir]

Implements PREREGISTRATION.md section T88 (written 2026-11-01, before any run; VISION Update 149: the owner's
picture that the big bang fires at one pole of the rod). The exact rod 4 x 4 x L (two directions curled, one open; six
links; named points) under her tie "all at the last", f = (0, a, 2a, 0), a = 4 (lambda - 1), sealed: the whole energy E
is placed at the start in the per-vertex stores of the side-0 points of the first `fed_layers` layers along the open
direction (z = 0, 1, ...), nothing anywhere else, and the per-vertex bath of `sealed_tie_d.run_sealed_bath_table_d`
(by_vertex=True: a move pays from, or is paid into, the store of the point that proposed it) runs with no leak. The
energy a point releases therefore stays where it is released, which is the setting in which the two-dimensional tube
opened as a front (T48). The rod is periodic, so the fed end is a pole only in the sense that the energy sits there;
the far side is the layer z = L / 2.

Read after every block (reading does not change the chain or the random stream): d = local_dimension_d at every point;
the count of points at d = 3 (both curled directions open) in each layer z; the connected open regions at d = 3 of at
least MIN_REGION points; the damaged points (d > 3); H, the tie total T_f, the energy in the stores, and the
conservation check H + T_f + stores = E0 + H0 + T0. The verdict words (ONE FRONT, SEVERAL, NOTHING OPENS) are read
by scripts/analyse_t88.py from these rows. The model with the tie is our family, never combinatorial quantum gravity;
every six-link result carries VISION Update 24's caveat (Gate C open). Results are append-only (rule 5).

Config keys: name, section, dims [4, 4, L], lambda, energy, fed_layers, replicas, n_sweeps, record_every, seed,
save_adjacency.
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
from graphity.cqg_d import hamiltonian, torus                             # noqa: E402
from graphity.dimension import local_dimension_d, pieces_of               # noqa: E402
from graphity.results import ResultWriter                                 # noqa: E402
from graphity.sealed_tie_d import run_sealed_bath_table_d, table_total    # noqa: E402
from run_curled_bath_d import MIN_REGION, refuse_if_saved, save_graph, seed_of   # noqa: E402
from run_curled_bath_tie_d import table_of                                # noqa: E402

OPEN_AXIS = 2        # the rod's open direction is the last of dims [4, 4, L]


def layers_of(dims):
    """z coordinate (along the open direction) of every point of torus(dims)."""
    return np.array(np.unravel_index(np.arange(int(np.prod(dims))), dims))[OPEN_AXIS]


def fed_stores(dims, part, energy, fed_layers):
    """One store per point: the energy shared equally among the side-0 points of the first `fed_layers` layers."""
    z = layers_of(dims)
    fed = np.flatnonzero((z < fed_layers) & (np.asarray(part) == 0))
    stores = np.zeros(int(np.prod(dims)))
    stores[fed] = float(energy) / len(fed)
    return stores


def reading(adj, dims, lam, ftab, stores):
    """Everything one CSV row says. Reads only."""
    d = local_dimension_d(adj)
    z = layers_of(dims)
    length = int(dims[OPEN_AXIS])
    at3 = d == 3
    per_layer = np.bincount(z[at3], minlength=length)
    sizes = pieces_of(adj, at3)
    far = length // 2
    row = dict(h=float(hamiltonian(adj, lam)), t_tie=float(table_total(adj, ftab)), stores=float(stores.sum()),
               n_open3=int(at3.sum()), open_regions3=sum(1 for s in sizes if s >= MIN_REGION),
               largest_open3=int(sizes[0]) if sizes else 0, damaged=int((d > 3).sum()),
               d0=int((d == 0).sum()), d1=int((d == 1).sum()), d2=int((d == 2).sum()), d3=int(at3.sum()),
               fed_open3=int(per_layer[0]), far_open3=int(per_layer[far]),
               profile=" ".join(str(int(c)) for c in per_layer))
    return row


def main(path, out_dir="results"):
    cfg = json.loads(Path(path).read_text())
    lam, energy = float(cfg["lambda"]), float(cfg["energy"])
    dims = [int(x) for x in cfg["dims"]]
    if len(dims) != 3 or dims[0] != 4 or dims[1] != 4:
        raise ValueError("T88 runs the rod 4 x 4 x L only")
    fed_layers = int(cfg.get("fed_layers", 1))
    adj0, part0 = torus(dims)
    n, dim = adj0.shape[0], adj0.shape[1] // 2
    ftab = table_of(cfg.get("tie", "all_at_the_last"), lam, dim)
    every, n_sweeps, reps = int(cfg.get("record_every", 100)), int(cfg["n_sweeps"]), int(cfg["replicas"])
    if n_sweeps % every:
        raise ValueError("n_sweeps must be a whole number of blocks of record_every")
    save = bool(cfg.get("save_adjacency"))
    adj_dir = Path(out_dir) / (cfg["name"] + "_adj")
    if save:
        refuse_if_saved(adj_dir, range(reps))
    meta = dict(config=cfg, config_path=str(path), package=__version__, python=platform.python_version(),
                numpy=np.__version__, numba=numba.__version__,
                preregistration="PREREGISTRATION.md section %s, written 2026-11-01" % cfg.get("section", "T88"))
    label = " ".join(map(str, dims))
    with ResultWriter(cfg["name"], meta, out_dir) as out:
        for rep in range(reps):
            adj, part = adj0.copy(), part0.copy()
            side_u = np.flatnonzero(part == 0)
            stores = fed_stores(dims, part, energy, fed_layers)
            seed = seed_of(cfg["seed"], n, rep)
            base = dict(N=n, dims=label, lam=lam, energy=energy, fed_layers=fed_layers, replica=rep)
            r = reading(adj, dims, lam, ftab, stores)
            e0 = r["h"] + r["t_tie"] + r["stores"]
            out.write(dict(**base, sweep=0, **r, conserved=0.0, acceptance="", final=False))
            for b in range(1, n_sweeps // every + 1):
                _, _, _, _, _, acc = run_sealed_bath_table_d(adj, side_u, stores, every, seed if b == 1 else -1,
                                                             lam, ftab, True)
                r = reading(adj, dims, lam, ftab, stores)
                sweep = b * every
                out.write(dict(**base, sweep=sweep, **r, conserved=r["h"] + r["t_tie"] + r["stores"] - e0,
                               acceptance=float(acc), final=(sweep == n_sweeps)))
            if save:
                save_graph(adj_dir, rep, adj, part, lam, energy, seed)
            print("dims=%s lambda=%g E=%g rep=%-2d | end: d3 %d d2 %d d1 %d damaged %d | regions %d | far %d | "
                  "stores %.1f | profile %s" % (label, lam, energy, rep, r["d3"], r["d2"], r["d1"], r["damaged"],
                                                r["open_regions3"], r["far_open3"], r["stores"], r["profile"]),
                  flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "results")
