"""EXACT: every kind of single move out of every rung of the curling ladder, at two to five directions. Usage:

    python scripts/exact_ladder_d.py [--max-d=5] [--out=docs/figures/ladder_kinds.json]

WHY. The owner asked (27 Sep) for a chart of the energy each opening costs and releases across lambda, and whether the
pattern learned at two, three and four directions continues at five and more. A move's cost is -16 dS + 4 lambda dX
(ASSUMPTIONS Q21), and its kind (dS, dX) does not depend on lambda, so listing the kinds once gives the wall at every
lambda: the wall is the cheapest kind's cost, and a negative cost is a way downhill (the kind dS = dX = 0 changes
nothing at any lambda and is skipped). Each rung is a torus with c sides of 4 (curled) and the rest long enough that the walls do not
depend on their length (O49 found the dependence only for open sides of 6 or less at two curled); the fully curled rung is the hypercube, the only
connected one (O85). Uses scripts/exact_walls_d.walls with near = 4 (checked against the full search on the 3D tori).
Writes one JSON file, adding rungs not already in it, so a long search can be resumed.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from graphity.cqg_d import torus                  # noqa: E402
from exact_walls_d import walls                   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# (directions, curled, sides): the rungs. Long sides chosen so the rung exists and its walls are not the short-side ones.
RUNGS = [
    (2, 0, [8, 8]), (2, 1, [4, 48]), (2, 2, [4, 4]),
    (3, 0, [6, 6, 6]), (3, 1, [4, 8, 12]), (3, 2, [4, 4, 18]), (3, 3, [4, 4, 4]),
    (4, 0, [6, 6, 8, 8]), (4, 1, [4, 6, 8, 12]), (4, 2, [4, 4, 12, 12]), (4, 3, [4, 4, 4, 12]), (4, 4, [4, 4, 4, 4]),
    (5, 1, [4, 6, 6, 6, 6]), (5, 2, [4, 4, 6, 6, 6]), (5, 3, [4, 4, 4, 6, 6]), (5, 4, [4, 4, 4, 4, 8]), (5, 5, [4, 4, 4, 4, 4]),
    (5, 0, [6, 6, 6, 6, 6]),
]


def main(argv):
    max_d = 5
    out = ROOT / "docs" / "figures" / "ladder_kinds.json"
    for a in argv:
        if a.startswith("--max-d="):
            max_d = int(a.split("=")[1])
        if a.startswith("--out="):
            out = Path(a.split("=")[1])
    data = json.loads(out.read_text()) if out.exists() else []
    done = {(r["D"], r["curled"]) for r in data}
    for dim, curled, sides in RUNGS:
        if dim > max_d or (dim, curled) in done:
            continue
        t = time.time()
        adj, part = torus(sides)
        kinds = walls(adj, part, 1.0, top=10 ** 6, near=4)
        row = dict(D=dim, curled=curled, sides=sides, N=int(adj.shape[0]),
                   kinds=[dict(dS=int(ds), dX=int(dx), ways=int(k)) for _, ds, dx, k in kinds if (ds, dx) != (0, 0)])
        data.append(row)
        out.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
        cheapest = min(-16 * k["dS"] + 4 * 1.25 * k["dX"] for k in row["kinds"])
        print("D=%d curled=%d %-18s N=%-6d kinds %-4d cheapest at 1.25: %.1f (%.0f s)"
              % (dim, curled, sides, row["N"], len(row["kinds"]), cheapest, time.time() - t), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
