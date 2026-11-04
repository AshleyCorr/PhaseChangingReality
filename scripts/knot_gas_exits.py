"""Every single move out of a gas of 4-cubes (baby universes), priced exactly. Usage:

    python scripts/knot_gas_exits.py

EXACT (ASSUMPTIONS O40). Asked on 2026-09-24 before pre-registering the author's "all directions curl together"
test in 2D: is a gas of fully curled knots stuck for now at lambda = 1.25, as the curled torus is, and if so,
how high is its wall? Every move the chain can propose is listed (`chain_switches`, the corrected census) and
priced as paper 1 prices moves A and B. A move offered c times among the N * N / 4 * 16 proposals is offered
c / N times per sweep of 2N attempts.
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from graphity.cqg import NO_CAP, is_valid, surplus, torus, total_squares  # noqa: E402
from run_move_census import chain_switches  # noqa: E402


def knot_gas(k):
    adj1, part1 = torus(4, 4, NO_CAP)
    n1 = adj1.shape[0]
    adj = np.concatenate([adj1 + i * n1 for i in range(k)]).astype(np.int64)
    part = np.concatenate([part1] * k)
    return adj, part


def piece_of(v):
    return v // 16


lam = 1.25
for k in (2, 4):
    adj, part = knot_gas(k)
    n = adj.shape[0]
    assert is_valid(adj, NO_CAP)
    s0, x0 = total_squares(adj), surplus(adj)
    h0 = 16 * (n - s0) + 4 * lam * x0
    side_u = np.flatnonzero(part == 0)
    classes = Counter()
    for trial in chain_switches(adj, side_u):
        ds, dx = total_squares(trial) - s0, surplus(trial) - x0
        diff = np.flatnonzero((trial != adj).any(axis=1))
        touched = {piece_of(int(v)) for v in diff}
        kind = "joins two knots" if len(touched) > 1 else "inside one knot"
        classes[(ds, dx, kind)] += 1
    print(f"k = {k} knots, N = {n}: S = {s0}, X = {x0}, H/N = {h0 / n:.3f} (expect 8(lam-1) = {8 * (lam - 1):.3f})")
    rows = []
    for (ds, dx, kind), cnt in classes.items():
        cost_expr = f"{-16 * ds} {'+' if dx >= 0 else '-'} {abs(4 * dx)}*lam"
        cost = -16 * ds + 4 * lam * dx
        rows.append((cost, ds, dx, kind, cnt, cost_expr))
    for cost, ds, dx, kind, cnt, expr in sorted(rows)[:8]:
        print(f"   dS={ds:+d} dX={dx:+d}  cost {expr} = {cost:6.2f}  {kind:16s} moves {cnt:5d}  offered per sweep {cnt / n:.3f}")
    downhill = [r for r in rows if r[0] < 0]
    print(f"   {'NOT stuck: ' + str(len(downhill)) + ' kinds of move lower the energy' if downhill else 'stuck: every move costs'}")
