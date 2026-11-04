"""Do two defects in a sheet of space pull towards each other, push apart, or ignore each other?

    python scripts/run_defect_interaction.py configs/defect_interaction.json

EXPLORATORY (TASKS T13 rung 3; the parked extension's item 3(i); the author's question of
2026-09-22, whether gravity could be a pull back towards symmetry). This is the cheapest rung of
that ladder and it is EXACT: no sampling, no seeds. Make one defect in a perfect flat sheet, make
a second one a known distance away, and compare the energy of the pair with twice the energy of a
single. Less than twice is an attraction, more is a repulsion, exactly twice is no force at all.

    interaction(d) = H(two defects d apart) - 2 * H(one defect)

A defect here is the cheapest single switch out of the flat sheet, which costs 16 (the wall of
ASSUMPTION Q13 in its sheet version). The second defect is the same switch translated by (dx, dy)
around the torus, so the two are identical by construction and the only thing that changes is how
far apart they sit. Separation is measured on the lattice, which is the only distance this model
has; for a pair on a torus it is the shorter way round.

PREDICTION, written before running (ours, unverified). The interaction is a contact one: zero
beyond the point where the two defects' damage overlaps, and negative (they cost less together)
where it does overlap, because two defects that share a broken square pay for it once. If that is
what comes out, then leftovers in this model attract only by touching and there is no long-range
force between them -- which would be a real limit on any reading of gravity as a pull between
leftovers, and should be reported as such rather than buried.
"""
import json
import platform
import sys
from pathlib import Path

import numba
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphity import __version__                                          # noqa: E402
from graphity.cqg import NO_CAP, _switch, hamiltonian, is_valid, torus     # noqa: E402
from graphity.results import ResultWriter                                  # noqa: E402


def xy(v, ly):
    return divmod(v, ly)


def shifted(v, dx, dy, lx, ly):
    x, y = xy(v, ly)
    return ((x + dx) % lx) * ly + (y + dy) % ly


def cheapest_defects(adj, lx, ly, lam):
    """Every valid single switch of the lowest cost, as (u1, v1, u2, v2) with its cost."""
    n = lx * ly
    base = hamiltonian(adj, lam)
    found, best = [], None
    for u1 in range(n):
        for v1 in adj[u1]:
            for u2 in range(u1 + 1, n):
                for v2 in adj[u2]:
                    if len({u1, v1, u2, v2}) < 4:
                        continue
                    trial = adj.copy()
                    _switch(trial, u1, v1, u2, v2)
                    if not is_valid(trial, NO_CAP):
                        continue
                    cost = hamiltonian(trial, lam) - base
                    if best is None or cost < best - 1e-9:
                        best, found = cost, [(u1, v1, u2, v2)]
                    elif abs(cost - best) < 1e-9:
                        found.append((u1, v1, u2, v2))
    return found, best


def main(path, out_dir="results"):
    cfg = json.loads(Path(path).read_text())
    meta = dict(config=cfg, config_path=str(path), package=__version__,
                python=platform.python_version(), numpy=np.__version__,
                numba=numba.__version__, purpose=cfg.get("_purpose", ""))
    with ResultWriter(cfg["name"], meta, out_dir) as out:
        for lam in cfg["lambdas"]:
            lam = float(lam)
            for lx, ly in cfg["sides"]:
                adj0, _ = torus(lx, ly, NO_CAP)
                defects, cost = cheapest_defects(adj0, lx, ly, lam)
                u1, v1, u2, v2 = defects[0]                 # one representative; the rest are its images
                one = adj0.copy()
                _switch(one, u1, v1, u2, v2)
                h_one = hamiltonian(one, lam) - hamiltonian(adj0, lam)
                print("lambda %.2f  %dx%d: %d cheapest defects, each costing %.0f"
                      % (lam, lx, ly, len(defects), h_one), flush=True)
                for dx in range(lx // 2 + 1):
                    for dy in range(ly // 2 + 1):
                        if dx == 0 and dy == 0:
                            continue
                        w1, x1 = shifted(u1, dx, dy, lx, ly), shifted(v1, dx, dy, lx, ly)
                        w2, x2 = shifted(u2, dx, dy, lx, ly), shifted(v2, dx, dy, lx, ly)
                        if len({u1, v1, u2, v2, w1, x1, w2, x2}) < 8:
                            continue                        # the two copies share a point: not a pair
                        pair = one.copy()
                        _switch(pair, w1, x1, w2, x2)
                        if not is_valid(pair, NO_CAP):
                            out.write(dict(lam=lam, lx=lx, ly=ly, dx=dx, dy=dy,
                                           distance=min(dx, lx - dx) + min(dy, ly - dy),
                                           one_defect=h_one, pair=None, interaction=None,
                                           note="the second switch is not valid there"))
                            continue
                        h_pair = hamiltonian(pair, lam) - hamiltonian(adj0, lam)
                        out.write(dict(lam=lam, lx=lx, ly=ly, dx=dx, dy=dy,
                                       distance=min(dx, lx - dx) + min(dy, ly - dy),
                                       one_defect=h_one, pair=h_pair,
                                       interaction=h_pair - 2 * h_one, note=""))


if __name__ == "__main__":
    main(*sys.argv[1:])
