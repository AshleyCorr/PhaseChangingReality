"""T30 and T32: a curled six-link torus in a sealed box, given a spark. Usage:

    python scripts/run_sealed_curled_d.py configs/t30_cascade_lam110_n288_c2n.json

Implements PREREGISTRATION.md sections T30 (does one push open both curled directions: the cascade and the room)
and T32 (is the activation fixed with size) on the six-link model (graphity.cqg_d, graphity.sealed_d; VISION
Update 22; ASSUMPTIONS O41). The torus C_4 x C_4 x C_L has two directions curled and one open, 8(lambda - 1) per
vertex above the flat 3-torus; the run puts a spark of E units in one store of a bath of C stores, all else empty,
and follows it for `n_sweeps` sweeps in blocks of `record_every`. Recorded every block: S, X, H per vertex, the
census of the local dimension read at six links (d = 3 flat, 2 one curled, 1 two curled, 0 all curled, above 3
melted), the mean store energy (legacy bath_T, not a calibrated temperature) and the conservation check.
These dimension labels are reference square-count signatures, not geometric certificates.
At the end it records the pieces of the selected region and whether
the arrangement ever left its start (T32's question). The final graph is saved when `save_adjacency` is set.

Config keys: name, section, dims (e.g. [4, 4, 18]) or gas ({"dims": [4, 4, 4], "copies": 8}: that many separate
tori, the literal form of the owner's fully curled X), lambda, capacities (expressions in N, as T18), sparks (a
list of E), replicas, n_sweeps, record_every, seed, save_adjacency, and "local_heat" (T34): true puts the spark in
the store of the first side-0 vertex under the per-vertex bath (sealed_d.run_sealed_bath_d with by_vertex), so only a
move made from that vertex can spend it and released energy stays where it is released; capacities are then ignored
and C is recorded as N.
"kappa" (T44, 2026-09-26): the follow form of the direction tie (VISION Update 30; ASSUMPTIONS O68), run through
graphity.sealed_tie_d; the rows then carry `kappa`, the tie total `tie_T` and the energy with the tie
`h_tie_per_vertex`, and the conservation check includes the tie. Absent, the runner is what it was.
"ftable_per_a" (T45, 2026-09-26): a tie of any shape, ftab[d] = entry d times a = 4(lambda - 1), run through
graphity.sealed_tie_d.run_sealed_bath_table_d; "tie_label" names it in the rows. Exclusive with "kappa". With
"local_heat" (T47 part B, 2026-09-26) the table tie runs under the per-vertex bath too.
Every row also carries (T48, 2026-09-27) `damaged_d`, the points with more open directions than the run has (d > D),
and `largest_open_d`, the largest connected piece at d = D; `melted` and `largest_flat` keep their six-link
meaning (d > 3, d = 3) so that earlier files read as they did.
"""
import json
import platform
import sys
from pathlib import Path

import numba
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphity import __version__                                          # noqa: E402
from graphity.cqg_d import surplus, torus, total_squares                  # noqa: E402
from graphity.dimension import local_dimension_d, pieces_of               # noqa: E402
from graphity.results import ResultWriter                                 # noqa: E402
from graphity import interchangeable_d                                  # noqa: E402
from graphity.sealed_d import energy_d, run_sealed_bath_d                 # noqa: E402
from graphity.sealed_tie_d import (energy_table_d, energy_tie_d, run_sealed_bath_table_d,       # noqa: E402
                                  run_sealed_bath_tie_d, table_total, tie_total)

D_BINS = 8       # d = 0 .. 6 and "7 or more"


def census(adj, lam, kappa=None):
    d = local_dimension_d(adj)
    dim = adj.shape[1] // 2
    hist = np.bincount(np.minimum(d, D_BINS - 1), minlength=D_BINS)[:D_BINS]
    flat = pieces_of(adj, d == 3)
    # T48 (2026-09-27): `melted` and `largest_flat` read d > 3 and d = 3, which is flat only at six links. The two columns
    # below read the run's own number of directions D = links / 2: points above D, and the largest piece at d = D.
    open_d = flat if dim == 3 else pieces_of(adj, d == dim)
    out = dict(h_per_vertex=float(energy_d(adj, lam)) / adj.shape[0],
               **{"d%d" % i: int(hist[i]) for i in range(D_BINS)},
               melted=int((d > 3).sum()), pieces_flat=len(flat), largest_flat=(flat[0] if flat else 0),
               damaged_d=int((d > dim).sum()), largest_open_d=(open_d[0] if open_d else 0))
    if kappa is not None:                                                 # T44
        t = int(tie_total(adj))
        out.update(tie_T=t, h_tie_per_vertex=float(energy_tie_d(adj, lam, kappa)) / adj.shape[0])
    return out


def census_any(adj, lam, kappa=None, ftab=None):
    """census, with the table tie's total and energy when ftab is given (T45)."""
    if ftab is None:
        return census(adj, lam, kappa)
    out = census(adj, lam)
    out.update(tie_T=float(table_total(adj, ftab)), h_tie_per_vertex=float(energy_table_d(adj, lam, ftab)) / adj.shape[0])
    return out


def gas(dims, copies):
    """`copies` separate tori C_dims, one after another in the vertex numbering."""
    a, p = torus(dims)
    n = a.shape[0]
    return np.concatenate([a + k * n for k in range(copies)]), np.concatenate([p] * copies)


def start_of(cfg):
    if "patch" in cfg:
        # T65 (2026-10-15): the slab 4 x L x L with a patch of k columns opened by hand (the helix patch of T54,
        # exact_t54_patch.patch), the sealed box then started with empty stores. Absent, the runner is what it was.
        from exact_t54_patch import patch as open_patch
        pt = cfg["patch"]
        length, k = int(pt["L"]), int(pt["k"])
        adj0, part = torus([4, length, length])
        adj, _ = open_patch(adj0, length, k, int(pt.get("x0", 0)), int(pt.get("y0", 0)))
        return adj, part, "4 %d %d patch %d" % (length, length, k)
    if "gas" in cfg:
        g = cfg["gas"]
        adj, part = gas([int(x) for x in g["dims"]], int(g["copies"]))
        return adj, part, "%dx(%s)" % (int(g["copies"]), " ".join(map(str, g["dims"])))
    dims = [int(x) for x in cfg["dims"]]
    adj, part = torus(dims)
    return adj, part, " ".join(map(str, dims))


def main(path, out_dir="results"):
    cfg = json.loads(Path(path).read_text())
    lam = float(cfg["lambda"])
    adj0, part0, label = start_of(cfg)
    n = adj0.shape[0]
    every, n_sweeps = int(cfg.get("record_every", 100)), int(cfg["n_sweeps"])
    local_heat = bool(cfg.get("local_heat", False))
    # T42 (2026-09-25): interchangeable points through graphity.interchangeable_d (the bath only); "weighted": false
    # runs the same chain with named points, as a control. Absent, the runner is what it was.
    interchangeable = bool(cfg.get("interchangeable", False))
    weighted = bool(cfg.get("weighted", True))
    tied = "kappa" in cfg
    kappa = float(cfg.get("kappa", 0.0))
    tabled = "ftable_per_a" in cfg                                        # T45
    ftab = np.array([float(v) for v in cfg.get("ftable_per_a", [])]) * 4.0 * (lam - 1.0)
    if tabled and tied:
        raise ValueError("kappa and ftable_per_a are exclusive")
    if tabled and interchangeable:
        raise ValueError("the table tie is run with named points only")
    if tied and interchangeable:
        raise ValueError("the direction tie is run with named points only")
    if interchangeable and local_heat:
        raise ValueError("interchangeable points are run with the shared bath only")
    baths = [n] if local_heat else [int(round(eval(str(c), {"N": n}))) for c in cfg["capacities"]]
    meta = dict(config=cfg, config_path=str(path), package=__version__, python=platform.python_version(),
                numpy=np.__version__, numba=numba.__version__,
                preregistration="PREREGISTRATION.md section %s, written 2026-09-25" % cfg.get("section", "T30"))
    adj_dir = Path(out_dir) / (cfg["name"] + "_adj")
    with ResultWriter(cfg["name"], meta, out_dir) as out:
        for c_bath in baths:
            for spark in cfg["sparks"]:
                for rep in range(int(cfg["replicas"])):
                    adj, part = adj0.copy(), part0.copy()
                    side_u = np.flatnonzero(part == 0)
                    stores = np.zeros(c_bath)
                    stores[int(side_u[0]) if local_heat else 0] = float(spark)
                    h0 = (energy_tie_d(adj, lam, kappa) if tied else energy_table_d(adj, lam, ftab) if tabled
                          else energy_d(adj, lam))
                    e0 = h0 + stores.sum()
                    s_start, x_start = int(total_squares(adj)), int(surplus(adj))
                    seed = int(np.random.SeedSequence([int(cfg["seed"]), n, c_bath, int(round(float(spark) * 100)), rep]).generate_state(1)[0])
                    left = False
                    left_at = ""
                    base = dict(N=n, dims=label, C=c_bath, spark=float(spark), replica=rep, lam=lam, local_heat=local_heat)
                    if interchangeable:                                   # T42; absent from earlier files
                        base.update(interchangeable=True, weighted=weighted)
                    if tied:                                              # T44; absent from earlier files
                        base.update(kappa=kappa)
                    if tabled:                                            # T45; absent from earlier files
                        base.update(tie_label=str(cfg.get("tie_label", "table")), ftable=" ".join("%.4f" % v for v in ftab))
                    c0 = census_any(adj, lam, kappa if tied else None, ftab if tabled else None)
                    out.write(dict(**base, sweep=0, **c0, bath_T=float(stores.mean()), total=float(stores.sum()),
                                   drift=0.0, left=False, final=False))
                    blocks = n_sweeps // every
                    rng = np.random.default_rng(seed) if interchangeable else None
                    for b in range(1, blocks + 1):
                        if tabled:
                            s, x, _, mean, tot, _ = run_sealed_bath_table_d(adj, side_u, stores, every,
                                                                            seed if b == 1 else -1, lam, ftab,
                                                                            by_vertex=local_heat)
                        elif tied:
                            s, x, _, mean, tot, _ = run_sealed_bath_tie_d(adj, side_u, stores, every, seed if b == 1 else -1,
                                                                          lam, kappa, by_vertex=local_heat)
                        elif interchangeable:
                            s, x, _, tot = interchangeable_d.run(adj, part, every, seed, lam, demons=stores,
                                                                 weighted=weighted, rng=rng)
                            mean = tot / len(stores)
                        else:
                            s, x, mean, tot, _ = run_sealed_bath_d(adj, side_u, stores, every, seed if b == 1 else -1, lam,
                                                                   by_vertex=local_heat)
                        if not left and ((s != s_start).any() or (x != x_start).any()):
                            left = True
                            left_at = b * every
                        c = census_any(adj, lam, kappa if tied else None, ftab if tabled else None)
                        drift = abs((c["h_tie_per_vertex"] if (tied or tabled) else c["h_per_vertex"]) * n + tot[-1] - e0)
                        out.write(dict(**base, sweep=b * every, **c, bath_T=float(mean[-1]), total=float(tot[-1]),
                                       drift=float(drift), left=left, final=(b == blocks)))
                    if cfg.get("save_adjacency"):
                        adj_dir.mkdir(parents=True, exist_ok=True)
                        np.savez(adj_dir / ("C%d_E%d_rep%d.npz" % (c_bath, int(round(float(spark))), rep)), adj=adj, part=part,
                                 lam=lam, h0=h0, e0=e0)
                    c = census(adj, lam)
                    print("dims=%s C=%d E=%g rep=%-2d left=%s at %s | end: d3 %d d2 %d d1 %d melted %d | H/N %.3f (start %.3f) | T %.2f"
                          % (base["dims"], c_bath, spark, rep, left, left_at, c["d3"], c["d2"], c["d1"], c["melted"],
                             c["h_per_vertex"], h0 / n, stores.mean()), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "results")
