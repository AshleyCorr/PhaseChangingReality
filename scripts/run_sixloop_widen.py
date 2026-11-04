"""T62: with a price on loops of six, does the tube six around give way and come out wider? Usage:

    python scripts/run_sixloop_widen.py configs/t62_mu020_g200_n96.json [out_dir]

Implements PREREGISTRATION.md section T62. The rule is graphity.sixloop's (VISION Update 56; ASSUMPTIONS O124;
ours, not the published model): H6 = 16 (N - S) + 4 lam X + mu E6. Each replica starts as the product of paper 1's
change, the 4 x L tube re-glued with its seam shifted by two: exactly flat, six around, N = 4L points. Nothing is
pushed by hand. Three spells, all of the thermal chain for H6:

  warm    `hold` sweeps at coupling g;
  cooled  `cool` sweeps while g falls in 40 equal steps to `cool_to`;
  cold    `cold_sweeps` at g = `cold_g`, which takes no uphill move.

Recorded per replica: E6 at the start; the first block (of 500 sweeps) of the warm spell after which E6 was below
its start, and after which it was below half; the fewest squares per point seen at the end of any sweep; S, X and
E6 after each spell; and the end state read from its wiring: valid or not, whether the chain's running counts
equal a full recount, the points at each d, and for every point the number of points within 3, 4 and 5 steps
(`sees3` is the share of points with exactly 25 within three steps, as in T61). The end graph is saved; its
pieces and the kind of each flat piece are read from it afterwards (scripts/analyse_t62.py).

Seeds: numpy SeedSequence([seed, N, round(100 lambda), round(1000 mu), round(100 g), replica]).

T63 (PREREGISTRATION.md, written 2026-10-12 after T62 was queued) uses the same runner with `"start": "tube"` in
its config: the curled tube itself, four around. A config without `start` begins six around, as T62's do, and
T62's configs are unchanged. `curl_sweep` is the first block of the warm spell after which no surplus square was
left (0 if there was none to begin with).
"""
import json
import platform
import sys
from pathlib import Path

import numba
import numpy as np
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphity import __version__                                           # noqa: E402
from graphity.cqg import NO_CAP, is_valid, surplus, total_squares          # noqa: E402
from graphity.dimension import local_dimension                             # noqa: E402
from graphity.results import ResultWriter, provenance                      # noqa: E402
from graphity.sixloop import excess6, run_chain6                           # noqa: E402

BLOCK = 500
COOL_STEPS = 40


def seam_shifted(length, shift):
    """The 4 x `length` tube with its seam shifted by `shift`: point (x, y), x in 0..3 round it and y along it, and a
    step round past x = 3 arrives at (0, y - shift). Shift 0 is the curled tube, four around; shift 2 is exactly
    flat and six around. Returns (neighbour array, side U)."""
    def index(x, y):
        return x * length + (y % length)
    adj = np.empty((4 * length, 4), dtype=np.int64)
    for x in range(4):
        for y in range(length):
            ahead = index(0, y - shift) if x == 3 else index(x + 1, y)
            back = index(3, y + shift) if x == 0 else index(x - 1, y)
            adj[index(x, y)] = (ahead, back, index(x, y + 1), index(x, y - 1))
    side = np.array([x * length + y for x in range(4) for y in range(length) if (x + y) % 2 == 0], dtype=np.int64)
    return adj, side


def six_around(length):
    """T62's start: the tube re-glued, exactly flat and six around."""
    return seam_shifted(length, 2)


STARTS = {"six": 2, "tube": 0}


@njit(cache=True)
def balls(adj):
    """(N, 3) array: the number of points within 3, 4 and 5 steps of each point."""
    n = adj.shape[0]
    out = np.zeros((n, 3), dtype=np.int64)
    dist = np.empty(n, dtype=np.int64)
    queue = np.empty(n, dtype=np.int64)
    for start in range(n):
        dist[:] = -1
        dist[start] = 0
        queue[0] = start
        head, tail = 0, 1
        while head < tail:
            v = queue[head]
            head += 1
            if dist[v] == 5:
                continue
            for k in range(4):
                w = adj[v, k]
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    queue[tail] = w
                    tail += 1
        for i in range(tail):
            d = dist[queue[i]]
            if d <= 3:
                out[start, 0] += 1
            if d <= 4:
                out[start, 1] += 1
            out[start, 2] += 1
    return out


def replica(length, lam, mu, g, seed, hold, cool, cool_to, cold_sweeps, cold_g, start="six"):
    """One tube (six around, or the curled tube itself), warm, cooled, cold. Returns (row fields, end graph, side U)."""
    adj, side = seam_shifted(length, STARTS[start])
    n = adj.shape[0]
    e_start = int(excess6(adj, side))
    low = float(total_squares(adj)) / n
    left = half = -1
    curl = 0 if surplus(adj) == 0 else -1
    used = 0
    first = True
    s = x = e = None
    while used < hold:
        s, x, e, _acc = run_chain6(adj, side, 1.0 / g, 0, min(BLOCK, hold - used), seed if first else -1, lam, mu)
        first = False
        used += len(s)
        low = min(low, float(s.min()) / n)
        if left < 0 and e[-1] < e_start:
            left = used
        if half < 0 and 2 * e[-1] < e_start:
            half = used
        if curl < 0 and x[-1] == 0:
            curl = used
    warm = (int(s[-1]), int(x[-1]), int(e[-1]))
    per_step = max(1, cool // COOL_STEPS)
    for step in range(COOL_STEPS if cool > 0 else 0):
        g_now = g + (cool_to - g) * (step + 1) / COOL_STEPS
        s, x, e, _acc = run_chain6(adj, side, 1.0 / g_now, 0, per_step, -1, lam, mu)
        low = min(low, float(s.min()) / n)
    cooled = (int(s[-1]), int(x[-1]), int(e[-1]))
    if cold_sweeps > 0:
        s, x, e, _acc = run_chain6(adj, side, 1.0 / cold_g, 0, cold_sweeps, -1, lam, mu)
    end = (int(s[-1]), int(x[-1]), int(e[-1]))
    recount = (int(total_squares(adj)), int(surplus(adj)), int(excess6(adj, side)))
    near = balls(adj)
    d = np.bincount(local_dimension(adj), minlength=7)[:7]
    fields = dict(start=start, e6_start=e_start, left_sweep=left, half_sweep=half, curl_sweep=curl,
                  phi_min=round(low, 6),
                  s_warm=warm[0], x_warm=warm[1], e6_warm=warm[2], s_cooled=cooled[0], x_cooled=cooled[1],
                  e6_cooled=cooled[2], s_end=end[0], x_end=end[1], e6_end=end[2],
                  h_end=16.0 * (n - end[0]) + 4.0 * lam * end[1],
                  books=bool(end == recount), valid=bool(is_valid(adj, NO_CAP)),
                  flat=bool(end[0] == n and end[1] == 0),
                  near3=round(float(near[:, 0].mean()), 4), near4=round(float(near[:, 1].mean()), 4),
                  near5=round(float(near[:, 2].mean()), 4), sees3=round(float((near[:, 0] == 25).mean()), 4),
                  **{"d%d" % j: int(d[j]) for j in range(7)})
    return fields, adj, side


def seed_of(seed, n, lam, mu, g, rep):
    return int(np.random.SeedSequence([int(seed), int(n), int(round(100 * lam)), int(round(1000 * mu)),
                                       int(round(100 * g)), int(rep)]).generate_state(1)[0])


def main(path, out_dir="results"):
    cfg = json.loads(Path(path).read_text())
    lam, mu, g = float(cfg["lambda"]), float(cfg["mu"]), float(cfg["g"])
    meta = dict(config=cfg, config_path=str(path), package=__version__, python=platform.python_version(),
                numpy=np.__version__, numba=numba.__version__,
                preregistration="PREREGISTRATION.md section %s, written 2026-10-12" % cfg.get("section", "T62"),
                **provenance())
    adj_dir = Path(out_dir) / (cfg["name"] + "_adj")
    if cfg.get("save_adjacency") and adj_dir.exists():
        raise SystemExit("refusing to overwrite %s" % adj_dir)
    with ResultWriter(cfg["name"], meta, out_dir) as out:
        for length in cfg["lengths"]:
            n = 4 * int(length)
            scale = max(1.0, n / 192.0)
            hold, cool = int(cfg["hold_sweeps_at_192"] * scale), int(cfg["cool_sweeps_at_192"] * scale)
            for rep in range(int(cfg["replicas"])):
                seed = seed_of(cfg["seed"], n, lam, mu, g, rep)
                fields, adj, side = replica(int(length), lam, mu, g, seed, hold, cool, float(cfg["cool_to"]),
                                            int(cfg["cold_sweeps"]), float(cfg["cold_g"]), cfg.get("start", "six"))
                out.write(dict(N=n, length=int(length), lam=lam, mu=mu, g=g, replica=rep, seed=seed,
                               hold_sweeps=hold, cool_sweeps=cool, **fields))
                if cfg.get("save_adjacency"):
                    adj_dir.mkdir(parents=True, exist_ok=True)
                    np.savez(adj_dir / ("N%d_rep%d.npz" % (n, rep)), adj=adj, side=side, lam=lam, mu=mu)
            print("%s N=%d done" % (cfg["name"], n), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
