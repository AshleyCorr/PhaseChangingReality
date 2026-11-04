"""T52: what does a cut hide? The hidden count round the relics of saved end states, exact. Usage:

    python scripts/exact_hidden_relics.py configs/t52_hidden_relics.json [out_dir] [--resume]

`--resume` reuses the rows of a `.partial` left by a run that died (every window is an independent exact count),
keeps that file beside the new one as `.partial.resumed` until the run completes, and records the reuse in the
`.meta.json`; added 2026-10-09 after the laptop run of 5 October died at 66 rows.

Implements PREREGISTRATION.md section T52 with its Amendment 1 (both written 2026-10-05, before any count was taken
on a saved state with `graphity.hidden`). For every saved end state named by the config's `states` pattern:

  relic windows   every four-point relic (T37's column: a connected piece of four points all at d = 1) gets the
                  ball of graph distance r for each r in `radii_all`; the `first_per_state` relics of lowest vertex
                  number get each r in `radii_first` as well;
  flat controls   the first `flat_squares_per_state` squares, in order of their lowest vertex number, all of whose
                  points lie at graph distance at least `flat_distance` from every point not at d = 2, get each r in
                  `radii_all`; the first `flat_first_per_state` of them get each r in `radii_first` as well.

A window is the set of points within graph distance r of its four anchor points. Its hidden count is
`graphity.hidden.hidden_count`: the valid wirings of the window's inside that leave every link with an end outside
untouched and the whole graph's energy what it was (ASSUMPTIONS O87). What is scored is `shapes`: those wirings counted
up to renaming of the window's points that have no link out of it (Amendment 1). A radius in `radii_all` is counted in
full, with the wirings at every other energy; a radius in `radii_first` at the original's energy only, one wiring per
class of renamings (`same_energy_only`, `fold_renamings`), which is exact for the count and the shapes. Nothing is
sampled; no random number is drawn.

The time limit. The search is one compiled call and cannot be stopped from inside, so each window is counted in a
process of its own (`--one`, below) and abandoned after `time_limit_s` seconds: its row then says counted = False.

    python scripts/exact_hidden_relics.py --one STATE.npz RADIUS V1,V2,V3,V4 LAMBDA     (one window; prints JSON)
"""
import csv
import json
import platform
import subprocess
import sys
import time
from glob import glob
from pathlib import Path

import numba
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphity import __version__                                          # noqa: E402
from graphity.dimension import local_dimension, piece_labels              # noqa: E402
from graphity.hidden import hidden_count                                  # noqa: E402
from graphity.results import ResultWriter                                 # noqa: E402


def ball(adj, anchors, radius):
    """The points within graph distance `radius` of any of `anchors`, sorted."""
    seen = {int(v) for v in anchors}
    edge = set(seen)
    for _ in range(radius):
        edge = {int(w) for v in edge for w in adj[v]} - seen
        seen |= edge
    return sorted(seen)


def relics(adj):
    """The four-point relics, each a sorted tuple of its points, in order of their lowest vertex number."""
    d = local_dimension(adj)
    labels = piece_labels(adj, d != 2)
    found = []
    for k in range(int(labels.max()) + 1):
        points = np.flatnonzero(labels == k)
        if len(points) == 4 and (d[points] == 1).all():
            found.append(tuple(int(v) for v in points))
    return sorted(found)


def distance_from_damage(adj):
    """For every point, the graph distance to the nearest point not at d = 2 (a number above N if there is none)."""
    n = adj.shape[0]
    dist = np.full(n, n + 1, dtype=np.int64)
    edge = [int(v) for v in np.flatnonzero(local_dimension(adj) != 2)]
    dist[edge] = 0
    step = 0
    while edge:
        step += 1
        nxt = []
        for v in edge:
            for w in adj[v]:
                if dist[w] > step:
                    dist[w] = step
                    nxt.append(int(w))
        edge = nxt
    return dist


def flat_squares(adj, how_many, min_distance):
    """The first `how_many` squares, in order of their lowest vertex number (then of their sorted points), all of
    whose points are at least `min_distance` from any point not at d = 2. Each is a sorted tuple of four points."""
    far = distance_from_damage(adj) >= min_distance
    found = []
    for v in range(adj.shape[0]):
        if not far[v]:
            continue
        here = set()
        nb = sorted(int(w) for w in adj[v])
        for i, a in enumerate(nb):
            for b in nb[i + 1:]:
                for c in adj[a]:
                    c = int(c)
                    if c != v and c in adj[b] and min(a, b, c) > v and far[a] and far[b] and far[c]:
                        here.add(tuple(sorted((v, a, b, c))))
        for square in sorted(here):
            found.append(square)
            if len(found) == how_many:
                return found
    return found


def count_one(state, radius, anchors, lam, full):
    """The hidden count of one window, as a dict that JSON can carry. `full` also lists the other energies."""
    z = np.load(state)
    r = hidden_count(z["adj"], z["part"], ball(z["adj"], anchors, radius), lam, keep=0,
                     same_energy_only=not full, fold_renamings=not full)
    return dict(points=r["points"], cut=r["cut"], links=r["links"], count=r["count"], shapes=r["shapes"],
                sealed="%d %d" % tuple(r["sealed"]), valid=r["valid"],
                levels={"%.9g" % k: v for k, v in r["levels"].items()} if full else None)


def count_with_limit(state, radius, anchors, lam, limit, full=True):
    """`count_one` in a process of its own; None if it has not finished after `limit` seconds."""
    cmd = [sys.executable, str(Path(__file__).resolve()), "--one", str(state), str(radius),
           ",".join(str(v) for v in anchors), repr(float(lam)), "full" if full else "ties"]
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=limit, check=True)
    except subprocess.TimeoutExpired:
        return None
    return json.loads(done.stdout.strip().splitlines()[-1])


def windows(adj, cfg):
    """Every (kind, anchors, radius) the pre-registration asks of one end state, in a fixed order."""
    out = []
    for i, relic in enumerate(relics(adj)):
        radii = list(cfg["radii_all"]) + (list(cfg["radii_first"]) if i < int(cfg["first_per_state"]) else [])
        out += [("relic", relic, int(r)) for r in radii]
    for i, square in enumerate(flat_squares(adj, int(cfg["flat_squares_per_state"]), int(cfg["flat_distance"]))):
        radii = list(cfg["radii_all"]) + (list(cfg["radii_first"]) if i < int(cfg["flat_first_per_state"]) else [])
        out += [("flat", square, int(r)) for r in radii]
    return out


def finished_rows(partial_path):
    """{(state, kind, anchors, radius): row} from the .partial a run that died left behind, so that a restart can
    reuse every window it had finished. Each window is an independent exact count, so a reused row is the row the
    restart would have written, apart from `seconds`. Empty if there is no such file."""
    partial_path = Path(partial_path)
    if not partial_path.is_file():
        return {}
    with open(partial_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return {(r["state"], r["kind"], r["anchors"], int(r["radius"])): r for r in rows}


def main(path, out_dir="results", resume=False):
    cfg = json.loads(Path(path).read_text())
    lam, limit = float(cfg["lambda"]), float(cfg["time_limit_s"])
    states = sorted(glob(cfg["states"]))
    if not states:
        raise FileNotFoundError("no saved end state matches %s" % cfg["states"])
    meta = dict(config=cfg, config_path=str(path), package=__version__, python=platform.python_version(),
                numpy=np.__version__, numba=numba.__version__, states=[Path(s).as_posix() for s in states],
                preregistration="PREREGISTRATION.md section T52, written 2026-10-05")
    # --resume (added 2026-10-09, after the laptop run of 5 October died at 66 rows): the windows already in the dead
    # run's .partial are written again as they were and not recounted; the .partial it came from is kept beside the
    # new one until this run completes. The order of windows, the counts and the time limit are unchanged.
    partial = Path(out_dir) / (cfg["name"] + ".csv.partial")
    done = finished_rows(partial) if resume else {}
    if done:
        partial.replace(partial.with_name(partial.name + ".resumed"))
        meta["resumed"] = dict(rows_reused=len(done), from_partial=partial.as_posix(), date="2026-10-09")
    with ResultWriter(cfg["name"], meta, out_dir) as out:
        for state in states:
            adj = np.load(state)["adj"]
            for kind, anchors, radius in windows(adj, cfg):
                key = (Path(state).as_posix(), kind, " ".join(str(v) for v in anchors), radius)
                if key in done:
                    out.write(done[key])
                    print("%s %s r=%d %s: reused from the earlier run" % (key[0], kind, radius, key[2]), flush=True)
                    continue
                started = time.time()
                full = radius in [int(x) for x in cfg["radii_all"]]
                r = count_with_limit(state, radius, anchors, lam, limit, full)
                row = dict(state=Path(state).as_posix(), kind=kind, anchors=" ".join(str(v) for v in anchors),
                           radius=radius, full=full, counted=r is not None,
                           seconds=round(time.time() - started, 1))
                keys = ("points", "cut", "links", "sealed", "count", "shapes", "valid")
                row.update({k: ("" if r is None or r[k] is None else r[k]) for k in keys})
                row["levels"] = json.dumps(r["levels"], sort_keys=True) if r and r["levels"] is not None else ""
                out.write(row)
                print("%s %s r=%d %s: %s" % (Path(state).parent.name + "/" + Path(state).name, kind, radius,
                                             row["anchors"], "points %d cut %d sealed %s count %d shapes %s (%.0f s)"
                                             % (r["points"], r["cut"], r["sealed"], r["count"], r["shapes"],
                                                row["seconds"])
                                             if r else "NOT COUNTED after %.0f s" % limit), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--one":
        print(json.dumps(count_one(sys.argv[2], int(sys.argv[3]), [int(v) for v in sys.argv[4].split(",")],
                                   float(sys.argv[5]), len(sys.argv) < 7 or sys.argv[6] == "full")))
    else:
        args = [a for a in sys.argv[1:] if a != "--resume"]
        main(args[0], args[1] if len(args) > 1 else "results", resume="--resume" in sys.argv)
