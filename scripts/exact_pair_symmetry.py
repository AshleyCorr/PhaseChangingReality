"""Does counting with interchangeable points pull two identical defects together? (series paper 5). Usage:

    python scripts/exact_pair_symmetry.py configs/exact_pair_symmetry.json

EXACT, exploratory. With interchangeable points every arrangement is weighted by its number of
symmetries A (the renamings that keep every relationship; ASSUMPTIONS Q15), relative to named points.
So for two identical defects in a flat torus, the counting contributes an effective free energy
-g ln A(r) at separation r: if A grows as the defects approach, counting pulls them together; if it
is flat, there is no pull; if it peaks at special placements, those placements are preferred for
reasons of symmetry, not distance.

The defect is the flat sheet's cheapest break: one valid switch with (Delta S, Delta X) = (-2, 0),
chosen as the first such switch in a fixed order near the origin. The second copy is the same switch
translated by the lattice vector (dx, dy), with dx + dy even so that the two sides of the bipartition
are preserved. Every placement whose two copies do not share a vertex or a square is counted.
"""
import json
import platform
import sys
from math import exp
from pathlib import Path

import numba
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphity import __version__                                          # noqa: E402
from graphity.cqg import NO_CAP, _switch, is_valid, surplus, torus, total_squares   # noqa: E402
from graphity.results import ResultWriter                                 # noqa: E402
from graphity.small_graphs import log_automorphisms, sides_first          # noqa: E402


def site(x, y, lx, ly):
    return (x % lx) * ly + (y % ly)


def first_break(adj, part, lx, ly):
    """The first (u1, v1, u2, v2) near the origin whose switch has (dS, dX) = (-2, 0)."""
    s0, x0 = total_squares(adj), surplus(adj)
    near = [site(x, y, lx, ly) for x in range(3) for y in range(3)]
    for u1 in near:
        if part[u1] != 0:
            continue
        for u2 in near:
            if u2 <= u1 or part[u2] != 0:
                continue
            for v1 in adj[u1]:
                for v2 in adj[u2]:
                    if v1 == v2:
                        continue
                    t = adj.copy()
                    _switch(t, int(u1), int(v1), int(u2), int(v2))
                    if is_valid(t, NO_CAP) and total_squares(t) - s0 == -2 and surplus(t) - x0 == 0:
                        return int(u1), int(v1), int(u2), int(v2)
    raise RuntimeError("no break found")


def shifted(v, dx, dy, lx, ly):
    x, y = divmod(v, ly)
    return site(x + dx, y + dy, lx, ly)


def main(path, out_dir="results"):
    cfg = json.loads(Path(path).read_text())
    lx, ly = cfg["side"]
    meta = dict(config=cfg, config_path=str(path), package=__version__,
                python=platform.python_version(), numpy=np.__version__, numba=numba.__version__,
                purpose=cfg.get("_purpose", ""))
    base, part = torus(lx, ly, NO_CAP)
    move = first_break(base, part, lx, ly)
    s_sheet = total_squares(base)
    one = base.copy()
    _switch(one, *move)
    a_one = exp(log_automorphisms(sides_first(one, part)))
    a_sheet = exp(log_automorphisms(sides_first(base, part)))
    with ResultWriter(cfg["name"], meta, out_dir) as out:
        out.write(dict(dx="", dy="", placement="flat sheet", valid=True, squares=s_sheet, symmetries=round(a_sheet)))
        out.write(dict(dx="", dy="", placement="one defect", valid=True, squares=total_squares(one), symmetries=round(a_one)))
        print("flat sheet %dx%d: %d symmetries; one defect: %d" % (lx, ly, round(a_sheet), round(a_one)), flush=True)
        for dx, dy in cfg["shifts"]:
            if (dx + dy) % 2:
                continue
            g = one.copy()
            m2 = tuple(shifted(v, dx, dy, lx, ly) for v in move)
            ok = len(set(move) | set(m2)) == 8
            if ok:
                try:
                    _switch(g, *m2)
                    ok = is_valid(g, NO_CAP) and total_squares(g) == s_sheet - 4
                except Exception:
                    ok = False
            a = round(exp(log_automorphisms(sides_first(g, part)))) if ok else ""
            out.write(dict(dx=dx, dy=dy, placement="two defects", valid=ok,
                           squares=(total_squares(g) if ok else ""), symmetries=a))
            print("  shift (%2d,%2d): %s" % (dx, dy, ("symmetries %s" % a) if ok else "overlapping or invalid"), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "results")
