"""T51: how much scrap freezes in, and how does that depend on how slowly the new space cools? Usage:

    python scripts/run_scrap_freeze.py configs/t51_l256_fast_00.json [out_dir]

Implements PREREGISTRATION.md section T51 with its Amendments 1 and 2 (series paper 2; programme piece 5). Tubes
4 x L at lambda = 1.25, named points, the thermal chain of T7 to T37 (`cqg.run_chain`, Metropolis, no cap). Each
replica named in `replica_ids` goes through two stages.

Opening. The exact tube runs at the fixed coupling `g_hot` and is read every `open_check_every` sweeps of the chain.
It stops at the first reading at which two things hold together: at least `open_stop_flat` (90 %) of its points are
at d = 2 (the share of flat points, T37's line between CLEAN and DEFECTED), and no connected piece of points at d = 1
holds more than `open_max_d1_piece` (eight) points. That is: no stretch of tube three or more columns long is left; a
resting piece of eight is allowed, and is read as an other. The second condition comes from Amendments 1 and 2. The
first cost check, made before any run, reached 90 % flat with two stretches of tube, 13 and 7 columns long, still
unconverted, so the flat share alone ends the opening while tube is still converting (Amendment 1). The second, with
the limit at four points, saw those stretches close about 1,700 sweeps later and one piece of eight points at d = 1
then rest unchanged for 15,000 sweeps: a limit of four waits for a leftover to heal, not for tube to finish
converting (Amendment 2). A replica that has not met both conditions after `open_cap` sweeps gets one row marked not
opened and is not followed. The opening's random stream is seeded from the config's seed, N and the replica and from
nothing else, so every cooling time, and every config with the same seed and replica, starts from the same opened
sheet (a paired design).

Cooling. For each cooling time in `t_cool`, a copy of the opened sheet is cooled in blocks of `block` fair sweeps: over
`t_cool` fair sweeps the coupling falls from `g_hot` to `g_cold` by the same factor each block (T25's schedule: the
coupling of block b of nb is g_hot (g_cold / g_hot)^(b / nb)), and it is then held at `g_cold` for `hold` fair sweeps.
`t_cool` = 0 is the quench: no cooling block, straight to the hold. Each cooling has its own stream, seeded from the
config's seed, N, the replica and the cooling time, so a cooling does not depend on which other cooling times share
its config. The final graph is saved in `<out_dir>/<name>_adj/`.

The fair clock. One fair sweep is N / `n_ref` sweeps of the chain (n_ref = 96, T19's and T25's size), so that every
local pair of links is offered as often per fair sweep as it is per sweep at 96 points (ASSUMPTIONS O90). A block is
round(block * N / n_ref) sweeps of the chain, rounded once per block: at N = 96 a fair sweep is exactly one sweep, and
at N = 1,024 a block of 250 fair sweeps is 2,667 sweeps where the unrounded figure is 2,666.67. The `fair_sweep`
column is the nominal count (block index times `block`); `chain_sweep` is what was actually run.

Readings. One row at the end of the opening and one after every block, read exactly as T37 reads its end states
(`analyse_t37.leftovers`): the leftovers are the connected pieces of points not at d = 2, a column is a piece of exactly
four points all at d = 1, anything else is other; `n_d1` is the points at d = 1, `largest_d1` the size of the largest
connected piece of points at d = 1 (the number the stop rule looks at), `flat_share` the share at d = 2, and `h` the
exact energy 16(N - S) + 4 lambda X. The random stream is carried on between calls of the chain (seed < 0,
ASSUMPTIONS Q14), and reading the graph draws no random numbers, so the chain is the same as if nobody had looked.

Config keys: name, section, side ([L, 4]), lambda, g_hot, g_cold, t_cool (a list, in fair sweeps), hold, block, n_ref,
open_check_every, open_stop_flat, open_max_d1_piece, open_cap, seed, replicas (the size of the seeded set),
replica_ids (the replicas this config runs), save_adjacency.
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
from graphity.cqg import NO_CAP, hamiltonian, run_chain, torus            # noqa: E402
from graphity.dimension import local_dimension, pieces_of                 # noqa: E402
from graphity.results import ResultWriter                                 # noqa: E402
from analyse_t37 import leftovers                                         # noqa: E402

LY = 4                       # the tube is four points round
OPENING, COOLING = 1, 2      # which stage a random stream belongs to (see stream_seed)


def stream_seed(seed, n, rep, t_cool=None):
    """The seed of a random stream: the opening's from (seed, N, replica), a cooling's from those and its t_cool.

    The stage word is there because numpy's SeedSequence pads short lists with zeros: without it the quench
    (t_cool = 0) would be handed the opening's own stream over again.
    """
    words = [int(seed), int(n), int(rep)] + ([OPENING] if t_cool is None else [COOLING, int(t_cool)])
    return int(np.random.SeedSequence(words).generate_state(1)[0])


def chain_sweeps(fair_sweeps, n, n_ref):
    """Sweeps of the chain in a stretch of `fair_sweeps` fair sweeps: one fair sweep is N / n_ref of them."""
    return int(round(fair_sweeps * n / n_ref))


def cooling_blocks(t_cool, block):
    """How many blocks the cooling takes: T25's count, and none at all for the quench."""
    return max(1, int(round(t_cool / block))) if t_cool > 0 else 0


def schedule(g_hot, g_cold, t_cool, hold, block):
    """The coupling of each block, with every length in fair sweeps: the cooling blocks, then the holding blocks.

    For t_cool > 0 this is T25's schedule (scripts/run_scrap_race.schedule); t_cool = 0 goes straight to the hold.
    """
    nb = cooling_blocks(t_cool, block)
    cool = [g_hot * (g_cold / g_hot) ** ((b + 1) / nb) for b in range(nb)]
    return cool + [g_cold] * int(round(hold / block))


def flat_and_tube(adj):
    """(the share of points at d = 2, the size of the largest connected piece of points at d = 1; 0 if none)."""
    d = local_dimension(adj)
    pieces = pieces_of(adj, d == 1)
    return float((d == 2).mean()), (int(pieces[0]) if pieces else 0)


def stop_rule(flat, largest_d1, stop_flat, max_d1_piece):
    """The opening is over when enough points are flat and no piece of points at d = 1 is larger than allowed
    (Amendments 1 and 2)."""
    return flat >= stop_flat and largest_d1 <= max_d1_piece


def has_opened(adj, stop_flat, max_d1_piece):
    """The stop rule read off a graph: at least `stop_flat` of the points at d = 2, and no connected piece of points
    at d = 1 with more than `max_d1_piece` points."""
    return stop_rule(*flat_and_tube(adj), stop_flat, max_d1_piece)


def reading(adj, lam):
    """What is read from a graph: columns and other pieces as T37 reads them, points at d = 1 and their largest
    connected piece, flat share, energy."""
    d = local_dimension(adj)
    columns, others = leftovers(adj)
    flat, largest_d1 = flat_and_tube(adj)
    return dict(columns=int(columns), others=int(others), n_d1=int((d == 1).sum()), largest_d1=largest_d1,
                flat_share=flat, h=float(hamiltonian(adj, lam)))


def open_tube(lx, lam, g_hot, seed, check_every, stop_flat, max_d1_piece, cap, look=has_opened):
    """Run the exact 4 x lx tube at g_hot until `look` first says it has opened, looking every `check_every` sweeps,
    for at most `cap` sweeps. Returns (adj, part, sweeps run, whether it opened)."""
    adj, part = torus(lx, LY, NO_CAP)
    side_u = np.flatnonzero(part == 0)
    sweeps = 0
    while sweeps < cap:
        run_chain(adj, side_u, 1.0 / g_hot, 0, check_every, seed if sweeps == 0 else -1, lam, NO_CAP, False)
        sweeps += check_every
        if look(adj, stop_flat, max_d1_piece):
            return adj, part, sweeps, True
    return adj, part, sweeps, False


def cool(adj, part, gs, block_sweeps, seed, lam, read=reading):
    """Run the blocks of a schedule on adj, in place: block b at coupling gs[b - 1] for `block_sweeps` sweeps of the
    chain. Yields (b, coupling, what `read` returns) after each block."""
    side_u = np.flatnonzero(part == 0)
    for b, g in enumerate(gs, start=1):
        run_chain(adj, side_u, 1.0 / g, 0, block_sweeps, seed if b == 1 else -1, lam, NO_CAP, False)
        yield b, g, read(adj, lam)


def final_path(adj_dir, n, rep, t_cool):
    return Path(adj_dir) / ("N%d_rep%d_tc%d.npz" % (n, rep, t_cool))


def save_final(adj_dir, n, rep, t_cool, adj, part, lam):
    """Keep the final graph of one cooling. Results are append-only: an existing file is never replaced."""
    target = final_path(adj_dir, n, rep, t_cool)
    if target.exists():
        raise FileExistsError("results are append-only; %s exists" % target)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(target, adj=adj, part=part, lam=lam, t_cool=t_cool, replica=rep)


def main(path, out_dir="results"):
    cfg = json.loads(Path(path).read_text())
    lam = float(cfg["lambda"])
    lx, ly = cfg["side"]
    if ly != LY:
        raise ValueError("the tube is 4 points round")
    n = lx * ly
    g_hot, g_cold = float(cfg["g_hot"]), float(cfg["g_cold"])
    block, hold, n_ref = int(cfg["block"]), int(cfg["hold"]), int(cfg["n_ref"])
    block_sweeps = chain_sweeps(block, n, n_ref)
    t_cools = [int(t) for t in cfg["t_cool"]]
    reps = [int(r) for r in cfg["replica_ids"]] if "replica_ids" in cfg else list(range(int(cfg["replicas"])))
    save = bool(cfg.get("save_adjacency"))
    adj_dir = Path(out_dir) / (cfg["name"] + "_adj")
    if save:                     # refuse before any sweep is spent, not after the last one
        taken = [p for p in (final_path(adj_dir, n, rep, t) for rep in reps for t in t_cools) if p.exists()]
        if taken:
            raise FileExistsError("results are append-only; %s exists" % taken[0])
    meta = dict(config=cfg, config_path=str(path), package=__version__,
                python=platform.python_version(), numpy=np.__version__, numba=numba.__version__,
                preregistration="PREREGISTRATION.md section T51, written 2026-10-05, with its Amendments 1 and 2")
    with ResultWriter(cfg["name"], meta, out_dir) as out:
        for rep in reps:
            sheet, part, open_sweeps, opened = open_tube(
                lx, lam, g_hot, stream_seed(cfg["seed"], n, rep), int(cfg["open_check_every"]),
                float(cfg["open_stop_flat"]), int(cfg["open_max_d1_piece"]), int(cfg["open_cap"]))
            base = dict(N=n, L=lx, lam=lam, replica=rep)
            first = reading(sheet, lam)
            out.write(dict(**base, t_cool="", phase="opening", block=0, fair_sweep=0, chain_sweep=0,
                           open_sweeps=open_sweeps, g=g_hot, **first, opened=opened, final=not opened))
            print("L=%d rep %d: %s %d sweeps; columns %d, others %d, flat share %.3f, largest piece at d = 1 %d"
                  % (lx, rep, "opened after" if opened else "NOT OPENED, not followed, after", open_sweeps,
                     first["columns"], first["others"], first["flat_share"], first["largest_d1"]), flush=True)
            if not opened:
                continue
            for t_cool in t_cools:
                adj = sheet.copy()
                gs = schedule(g_hot, g_cold, t_cool, hold, block)
                if not gs:
                    raise ValueError("nothing to run: no cooling block and no holding block")
                n_cool = cooling_blocks(t_cool, block)
                for b, g, r in cool(adj, part, gs, block_sweeps, stream_seed(cfg["seed"], n, rep, t_cool), lam):
                    out.write(dict(**base, t_cool=t_cool, phase=("cooling" if b <= n_cool else "hold"), block=b,
                                   fair_sweep=b * block, chain_sweep=b * block_sweeps, open_sweeps=open_sweeps, g=g,
                                   **r, opened=True, final=(b == len(gs))))
                if save:
                    save_final(adj_dir, n, rep, t_cool, adj, part, lam)
                print("L=%d rep %d t_cool=%d: columns %d -> %d, others %d -> %d, flat share %.3f, energy left %.1f"
                      % (lx, rep, t_cool, first["columns"], r["columns"], first["others"], r["others"],
                         r["flat_share"], r["h"]), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "results")
