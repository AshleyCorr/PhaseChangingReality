"""EXPLORATORY, exact: Bianconi's discrete entropy action on this project's wirings. Usage:

    python scripts/explore_bianconi_action.py

The owner asked (5 October 2026) for a first evaluation of the network action of [Bia24] (G. Bianconi, "Quantum entropy
couples matter with geometry", J. Phys. A 57, 365002; arXiv:2404.08556) on flat, curled and leftover arrangements.
That paper keeps the wiring fixed and leaves "the possible implied dynamics of the network topology to future works"
(its Sec. 3.1). This script does the smallest piece of that: it holds everything else at its simplest value and asks
how the action differs BETWEEN wirings. ASSUMPTIONS Q23 records how the paper was read; O102 the numbers.

What is computed, for a wiring with N points, E = 2N links and F squares (every 4-cycle is taken as a face):

  cells      N + E + F, the size of her matrices. In her vacuum (no matter, c0 = 0, her Eq. 65) the metric is the same
             number on every cell, so the action is a constant times the number of cells: between wirings with the same
             points and links it depends only on the number of squares.
  logdet     Tr ln(I + c0 L), with L her Gauss-Bonnet Laplacian (blocks L0, L1, L2, the Hodge Laplacians of points,
             links and squares) at the flat metric, zero gauge field and no matter, where her induced metric is
             G = I + c0 L (her Eq. 52). At the identity metric her two actions are then
                 S+ = -logdet - cells        and        S- = +logdet - cells        (her Eqs. 41, 42).
             Whether the metric should instead be solved from her Eq. 57 for each wiring is the question to put to her.

Exactness. With every link pointed from its side-0 end to its side-1 end, a square's four links alternate in sign round
it, the boundary of a boundary is zero (checked for every wiring), and so
    Tr ln(I + c0 L) = 2 [ logdet(I + c0 L0) + logdet(I + c0 L2) ],
L0 being the ordinary graph Laplacian and L2 = B2^T B2 (tests/test_bianconi_action.py checks this identity against
the three blocks computed directly).

Not pre-registered; a first look. Nothing here says the action is to be minimized or maximized between wirings: the
paper does not say, and that is part of the question for its author.
"""
import sys
from glob import glob
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphity.cqg import NO_CAP, hamiltonian, torus      # noqa: E402

C0S = (0.1, 1.0, 10.0)
LAM = 1.25


def side_of(adj):
    """A 0/1 side for every point of a two-sided graph, by walking it (side 0 for each piece's lowest point)."""
    n = adj.shape[0]
    side = np.full(n, -1, dtype=np.int64)
    for s in range(n):
        if side[s] >= 0:
            continue
        side[s] = 0
        stack = [s]
        while stack:
            v = stack.pop()
            for w in adj[v]:
                w = int(w)
                if side[w] < 0:
                    side[w] = 1 - side[v]
                    stack.append(w)
                elif side[w] == side[v]:
                    raise ValueError("the graph is not two-sided")
    return side


def boundaries(adj, part=None):
    """(B1, B2): points x links and links x squares, links pointed from side 0 to side 1, squares being all 4-cycles."""
    n = adj.shape[0]
    part = side_of(adj) if part is None else np.asarray(part)
    links = {}
    for a in np.flatnonzero(part == 0):
        for b in adj[a]:
            links[(int(a), int(b))] = len(links)
    b1 = np.zeros((n, len(links)))
    for (a, b), e in links.items():
        b1[a, e], b1[b, e] = -1.0, 1.0
    zeros = [int(a) for a in np.flatnonzero(part == 0)]
    nbrs = {a: set(int(b) for b in adj[a]) for a in zeros}
    reach = {}                                   # side-1 point -> its side-0 neighbours
    for a in zeros:
        for b in nbrs[a]:
            reach.setdefault(b, []).append(a)
    pairs = set()
    for b, around in reach.items():
        for i, a0 in enumerate(around):
            for a1 in around[i + 1:]:
                pairs.add((min(a0, a1), max(a0, a1)))
    squares = []
    for a0, a1 in sorted(pairs):
        common = sorted(nbrs[a0] & nbrs[a1])
        for i, x in enumerate(common):
            for y in common[i + 1:]:
                squares.append((a0, x, a1, y))   # the cycle a0 - x - a1 - y - a0
    b2 = np.zeros((len(links), len(squares)))
    for f, (a0, x, a1, y) in enumerate(squares):
        b2[links[(a0, x)], f] = 1.0              # along the cycle
        b2[links[(a1, x)], f] = -1.0             # against it
        b2[links[(a1, y)], f] = 1.0
        b2[links[(a0, y)], f] = -1.0
    return b1, b2


def logdet(m, c0):
    sign, value = np.linalg.slogdet(np.eye(m.shape[0]) + c0 * m)
    assert sign > 0
    return float(value)


def action_parts(adj, part=None, c0s=C0S):
    """(points, links, squares, {c0: Tr ln(I + c0 L)}) for one wiring."""
    b1, b2 = boundaries(adj, part)
    assert np.abs(b1 @ b2).max() < 1e-12, "the boundary of a boundary must vanish"
    l0, l2 = b1 @ b1.T, b2.T @ b2
    return b1.shape[0], b1.shape[1], b2.shape[1], {c0: 2.0 * (logdet(l0, c0) + logdet(l2, c0)) for c0 in c0s}


def row(label, adj, part=None):
    n, e, f, ld = action_parts(adj, part)
    h = float(hamiltonian(np.ascontiguousarray(adj, dtype=np.int64), LAM)) / n
    print("%-34s N=%-5d squares/N %.4f  cells/N %.4f  H/N %7.4f  | Tr ln(I + c0 L) / N at c0 = 0.1, 1, 10: %s"
          % (label, n, f / n, (n + e + f) / n, h, "  ".join("%.4f" % (ld[c] / n) for c in C0S)), flush=True)
    return n, f, ld


def main():
    print("EXPLORATORY. H is this project's energy at lambda = %.2f, for comparison. Per point throughout." % LAM)
    for label, sides in (("flat 8 x 8", (8, 8)), ("flat 16 x 16", (16, 16)), ("flat 16 x 10", (16, 10)),
                         ("curled tube 16 x 4", (16, 4)), ("curled tube 64 x 4", (64, 4)), ("4-cube (4 x 4)", (4, 4))):
        adj, part = torus(sides[0], sides[1], NO_CAP)
        row(label, adj, part)
    shown = 0
    for f in sorted(glob("results/t37_lam125_g150_L64_adj/*.npz")):       # small opened sheets, with and without leftovers
        z = np.load(f)
        h = float(hamiltonian(z["adj"], LAM))
        if h in (0.0, 14.0) and shown < 4:
            row("opened 64 x 4 tube, %s" % ("clean sheet" if h == 0.0 else "one 14-unit relic"), z["adj"], z["part"])
            shown += 1


if __name__ == "__main__":
    main()
