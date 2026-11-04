"""How interchangeable is each arrangement, and what does measuring it cost? (task T10). Usage:

    python scripts/measure_symmetry_cost.py

Prints to the screen and writes nothing, because it measures the cost of a calculation rather
than producing a result about the model.

WHY. Treating the points as named weights an arrangement by how many distinct named versions it
has; treating them as interchangeable weights every shape once. The two differ by exactly the
number of renamings that leave every relationship intact (ASSUMPTION Q15), so that number says
how much of an arrangement is genuinely interchangeable and how much the choice matters.

WHAT IS MEASURED. The three arrangements the model actually visits, at the published size:
fully shattered, perfectly ordered, and melted. The timing decides whether the correction can
be applied inside a chain (it cannot) or only to measured configurations (it can).
"""
import time
from math import log

import numpy as np

from graphity.cqg import NO_CAP, is_valid, run_chain, torus
from graphity.small_graphs import log_automorphisms, sides_first


def shattered(n_cubes):
    """n_cubes separate 4-cubes: what the cold phase with the penalty off looks like."""
    block, block_part = torus(4, 4, cap=NO_CAP)
    adj = np.concatenate([block + 16 * k for k in range(n_cubes)])
    part = np.concatenate([block_part] * n_cubes)
    return sides_first(adj, part)


def report(name, adj):
    assert is_valid(adj, NO_CAP), name
    t = time.time()
    value = log_automorphisms(adj)
    took = time.time() - t
    print("%-30s %7d %12.2f %14.1f %10.2f"
          % (name, len(adj), value, value / log(10), took))


print("%-30s %7s %12s %14s %10s"
      % ("arrangement", "points", "ln renamings", "as a power of 10", "seconds"))

for n_cubes in (4, 10, 12):
    report("%d separate 4-cubes" % n_cubes, shattered(n_cubes))

adj, part = torus(16, 10, cap=NO_CAP)
flat = sides_first(adj, part)
report("perfect flat sheet 16x10", flat)

side_u = np.flatnonzero(part == 0)
run_chain(adj, side_u, 0.0, 300, 1, 2024, 1.0, NO_CAP)          # melt it at infinite temperature
report("melted random graph", sides_first(adj, part))

print("\nThe ordering is the opposite of the obvious guess. A shattered state is the fastest,")
print("because the group factorises over separate pieces. A melted graph is the slowest, and")
print("its answer is always zero: the work goes into proving there is no symmetry at all.")
print("A run makes of order 1e8 moves, so no correction at every move is affordable at any of")
print("these speeds; apply it to measured configurations, or per energy level, instead.")
