"""T67: does the owner's untied slab open from a feed, energy arriving at a fixed rate? Usage:

    python scripts/run_slab_feed.py configs/t67_shared_cn_l8.json

Implements PREREGISTRATION.md section T67 (written 2026-10-16, after one sandbox look and before any registered run;
rule 11's fifth use on piece 5; VISION Updates 58 and 59; the owner's words "the rates may not be fixed but it's good
to run that way for starting"). The slab 4 x L x L with one direction curled, untied, lambda fixed, sealed with empty
stores; every `record_every` sweeps r * record_every units are added to the stores: mode "shared" into store 0 of a
shared bath of C stores (any move may draw it); mode "spot" into one point's own store with per-vertex stores (only
moves from that point spend it; the owner's "one spot per seed"). Each record writes the census (open = points at
local dimension 3, curled = at most 2, damaged = above 3), the number of separate open regions and the largest one,
the stores' mean, the energy fed so far and the drift. A replica stops at the first record with open >= 0.9 N (OPEN),
with damaged > 0.1 N (DAMAGED), or at `n_sweeps` (TIME). The final graph is saved when `save_adjacency` is set.

The chain is `graphity.sealed_tie_d.run_sealed_bath_table_d` with a table of zeros (untied), as T65 and T66. Energy
is conserved to the last bit and the drift, |H + stores - H0 - fed|, is written in every row. Seeds:
SeedSequence([seed, N, mode index, round(1000 r), replica]).

T68 (PREREGISTRATION T68, written 2026-10-17 on the owner's "proceed with these": the two variants T67 named as
untried) adds, with T67's cells unchanged: a feed `schedule` ("fixed" as above; "ramp", the rate rising linearly from
0 to 2r over the run so that the mean is r; "pulse", the same mean delivered as bursts of PULSE = 128 units, T65's
spark, whenever r t has grown by 128), written as a cumulative feed F(t) so that every block adds F(t) - fed; and two
more places: "spots", k points' own stores chosen at random (per-vertex stores, each spot getting 1/k of the feed),
and "all", every point's own store getting 1/N. A config may give "sizes" (a list of L) instead of one "dims". The
seed formula is T67's for T67's cells (schedule fixed, mode shared or spot) and appends [schedule index, k] otherwise,
so T67's configs reproduce bit for bit.
"""
import json
import platform
import sys
from pathlib import Path

import numba
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphity import __version__                                          # noqa: E402
from graphity.cqg_d import torus                                           # noqa: E402
from graphity.dimension import local_dimension_d, pieces_of                # noqa: E402
from graphity.results import ResultWriter, provenance                      # noqa: E402
from graphity.sealed_tie_d import energy_table_d, run_sealed_bath_table_d  # noqa: E402

OPEN_FRAC, DAMAGE_FRAC = 0.9, 0.1
MODES = ("shared", "spot", "spots", "all")
SCHEDULES = ("fixed", "ramp", "pulse")
PULSE = 128.0


def cumulative_feed(schedule, rate, t, total):
    """Energy fed by sweep t under a schedule with mean rate `rate` over a run of `total` sweeps."""
    if schedule == "fixed":
        return rate * t
    if schedule == "ramp":
        return rate * t * t / total
    if schedule == "pulse":
        return PULSE * np.floor(rate * t / PULSE)
    raise ValueError(schedule)


def seed_for(seed, n, mode, rate, rep, schedule="fixed", k=1):
    ents = [int(seed), n, MODES.index(mode), int(round(1000 * float(rate))), rep]
    if not (schedule == "fixed" and mode in ("shared", "spot")):
        ents += [SCHEDULES.index(schedule), int(k)]
    return int(np.random.SeedSequence(ents).generate_state(1)[0])


def census(adj):
    d = local_dimension_d(adj)
    op = d == 3
    sizes = pieces_of(adj, op) if op.any() else []
    return dict(open=int(op.sum()), curled=int((d <= 2).sum()), damaged=int((d > 3).sum()),
                regions=len(sizes), largest=int(max(sizes)) if len(sizes) else 0)


def run_cell(dims, lam, rate, mode, c_bath, n_sweeps, every, seed, schedule="fixed", k=1):
    """One replica. Yields (sweep, census, store_mean, fed, drift, stop) with stop in {None, OPEN, DAMAGED, TIME};
    the last record carries the stop word. The feed for a block is added after the block has run: F(t) - fed, with
    F the cumulative feed of the schedule, split equally over the feed points."""
    adj, part = torus(dims)
    n = adj.shape[0]
    side_u = np.flatnonzero(part == 0)
    by_vertex = mode != "shared"
    ftab = np.zeros(len(dims) + 1)
    stores = np.zeros(n if by_vertex else c_bath)
    if mode == "shared":
        feed_idx = np.array([0])
    elif mode == "spot":
        feed_idx = np.array([int(side_u[0])])
    elif mode == "spots":
        feed_idx = np.sort(np.random.default_rng(seed).choice(n, size=int(k), replace=False))
    else:
        feed_idx = np.arange(n)
    e0 = energy_table_d(adj, lam, ftab)
    fed = 0.0
    yield 0, census(adj), 0.0, 0.0, 0.0, None
    blocks = n_sweeps // every
    for b in range(blocks):
        run_sealed_bath_table_d(adj, side_u, stores, every, seed if b == 0 else -1, lam, ftab, by_vertex=by_vertex)
        target = float(cumulative_feed(schedule, rate, (b + 1) * every, n_sweeps))
        stores[feed_idx] += (target - fed) / len(feed_idx)
        fed = target
        c = census(adj)
        drift = abs(energy_table_d(adj, lam, ftab) + stores.sum() - e0 - fed)
        stop = None
        if c["open"] >= OPEN_FRAC * n:
            stop = "OPEN"
        elif c["damaged"] > DAMAGE_FRAC * n:
            stop = "DAMAGED"
        elif b == blocks - 1:
            stop = "TIME"
        yield (b + 1) * every, c, float(stores.mean()), float(fed), float(drift), stop
        if stop:
            break
    run_cell.final = (adj, part)


def main(path, out_dir="results"):
    cfg = json.loads(Path(path).read_text())
    dims_list = [[4, int(L), int(L)] for L in cfg["sizes"]] if "sizes" in cfg else [[int(x) for x in cfg["dims"]]]
    lam = float(cfg["lambda"])
    mode = cfg["mode"]
    assert mode in MODES
    schedule = cfg.get("schedule", "fixed")
    assert schedule in SCHEDULES
    k = int(cfg.get("k", 1))
    every, n_sweeps = int(cfg["record_every"]), int(cfg["n_sweeps"])
    meta = dict(config=cfg, config_path=str(path), package=__version__, python=platform.python_version(),
                numpy=np.__version__, numba=numba.__version__, **provenance(),
                preregistration="PREREGISTRATION.md section %s, written %s" % (cfg.get("section", "T67"), cfg.get("written", "2026-10-16")))
    adj_dir = Path(out_dir) / (cfg["name"] + "_adj")
    with ResultWriter(cfg["name"], meta, out_dir) as out:
        for dims in dims_list:
            n = int(np.prod(dims))
            label = "x".join(str(x) for x in dims)
            c_bath = int(round(eval(str(cfg["capacity"]), {"N": n}))) if mode == "shared" else n
            kk = n if mode == "all" else (k if mode == "spots" else 1)
            for rate in cfg["rates"]:
                for rep in range(int(cfg["replicas"])):
                    seed = seed_for(cfg["seed"], n, mode, rate, rep, schedule, kk)
                    for sweep, c, mean, fed, drift, stop in run_cell(dims, lam, float(rate), mode, c_bath, n_sweeps, every, seed, schedule, kk):
                        out.write(dict(N=n, dims=label, lam=lam, mode=mode, schedule=schedule, k=kk, C=c_bath, rate=float(rate), replica=rep, sweep=sweep,
                                       **c, store_mean=mean, fed=fed, drift=drift, stop=stop or ""))
                    if cfg.get("save_adjacency"):
                        adj, part = run_cell.final
                        adj_dir.mkdir(parents=True, exist_ok=True)
                        np.savez(adj_dir / ("%s_%s_k%d_C%d_r%g_N%d_rep%d.npz" % (schedule, mode, kk, c_bath, float(rate), n, rep)), adj=adj, part=part, lam=lam)
                    print("dims=%s %s %s k=%d C=%d r=%g rep=%-2d stop=%s at %d | open %d curled %d damaged %d | regions %d largest %d | fed %.0f"
                          % (label, schedule, mode, kk, c_bath, rate, rep, stop, sweep, c["open"], c["curled"], c["damaged"], c["regions"], c["largest"], fed), flush=True)
    print("wrote", Path(out_dir) / (cfg["name"] + ".csv"))


if __name__ == "__main__":
    main(sys.argv[1], *(sys.argv[2:3]))
