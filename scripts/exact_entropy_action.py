"""Bianconi's discrete entropy action, evaluated on this project's wirings. Exact linear algebra; no chain is run.

    python scripts/exact_entropy_action.py

Source: G. Bianconi, "Quantum entropy couples matter with geometry", J. Phys. A 57, 365002 (2024), arXiv:2404.08556
([Bia24]; the passages used were read from the arXiv HTML on 10 October 2026). It is the discrete form of her
"Gravity from entropy" (arXiv:2408.14391). The paper works on a cell complex of nodes, links and polygons with the
wiring held fixed, and names letting the wiring change as open ("leaving the discussion about the possible implied
dynamics of the network topology to future works", Sec. 3.1). This project's networks change their wiring and count
squares, so its arrangements can be handed to her action as they are: nodes, links, and every square as a 2-cell.

What is taken from the paper (Sourced):
  d = G^(-1/2) B^T G^(1/2) and D = d + d^T, the Dirac operator                              (Eqs. 23, 31)
  D^2 = L, the Gauss-Bonnet Laplacian, block diagonal in the Hodge Laplacians L0, L1, L2    (Eqs. 35 to 40)
  {D, gamma0} = 0, gamma0 = +1 on nodes and squares, -1 on links                            (Eq. 34)
  S+ = sigma Tr ln G + Tr G (ln G - ln G~) - Tr G, G the metric and G~ the induced one      (Eq. 41)
  with no matter and no gauge field, G~ = I + c0 L                                          (Eq. 52)
  her vacuum is G~ = I (c0 = 0), where the metric obeys -G ln G = sigma I                   (Eq. 65)

What is ours (unverified; the question put to her is whether it is the right reading):
  - every 4-cycle of the network is taken as a 2-cell, as the energy of this model counts them;
  - with c0 = 0, Eq. (65) makes every cell's metric the same number g, so the action is one number per cell,
    s = sigma ln g - sigma - g, times the number of cells, 3N + S on a four-regular network of N nodes, 2N links
    and S squares. It then depends on the wiring only through the count of squares;
  - with c0 > 0 the wiring also enters through the spectra. Evaluated here at the uniform metric G = I, where
    d = B^T and S+ = -sum ln(1 + c0 mu) - (number of cells), mu the eigenvalues of L0, L1 and L2. The metric that
    solves her equation of motion for c0 > 0 is not computed.
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exact_exits_d import disjoint, torus                    # noqa: E402
from exact_torus_level import census                         # noqa: E402

ZERO = 1e-9


def cells(adj):
    """(links as (u, v) with u < v, squares as cycles (v0, v1, v2, v3) starting at their smallest node)."""
    n = adj.shape[0]
    links = sorted({(min(u, int(v)), max(u, int(v))) for u in range(n) for v in adj[u]})
    nbrs = [set(int(v) for v in adj[u]) for u in range(n)]
    squares = set()
    for u in range(n):
        around = sorted(nbrs[u])
        for i, a in enumerate(around):
            for b in around[i + 1:]:
                for w in (nbrs[a] & nbrs[b]) - {u}:
                    cycle = (u, a, w, b)
                    k = cycle.index(min(cycle))
                    turned = cycle[k:] + cycle[:k]
                    squares.add(min(turned, (turned[0], turned[3], turned[2], turned[1])))
    return links, sorted(squares)


def boundaries(adj):
    """B1 (nodes by links) and B2 (links by squares), each link pointing from its smaller node to its larger."""
    links, squares = cells(adj)
    where = {link: k for k, link in enumerate(links)}
    b1 = np.zeros((adj.shape[0], len(links)))
    for k, (u, v) in enumerate(links):
        b1[u, k], b1[v, k] = -1.0, 1.0
    b2 = np.zeros((len(links), len(squares)))
    for k, cycle in enumerate(squares):
        for i in range(4):
            u, v = cycle[i], cycle[(i + 1) % 4]
            b2[where[(min(u, v), max(u, v))], k] = 1.0 if u < v else -1.0
    return b1, b2


def hodge(b1, b2):
    """The Hodge Laplacians L0, L1, L2 at the uniform metric, where d = B^T."""
    return b1 @ b1.T, b1.T @ b1 + b2 @ b2.T, b2.T @ b2


def dirac(b1, b2):
    """(D, gamma0) at the uniform metric: D = d + d^T on nodes, links and squares."""
    n, e = b1.shape
    s = b2.shape[1]
    d = np.zeros((n + e + s, n + e + s))
    d[:n, n:n + e], d[n:n + e, :n] = b1, b1.T
    d[n:n + e, n + e:], d[n + e:, n:n + e] = b2, b2.T
    return d, np.diag([1.0] * n + [-1.0] * e + [1.0] * s)


def spectra(adj):
    return [np.linalg.eigvalsh(lap) for lap in hodge(*boundaries(adj))]


def betti(spec):
    """How many eigenvalues of each Hodge Laplacian vanish: pieces, independent loops, enclosed cavities."""
    return tuple(int((np.abs(mu) < ZERO).sum()) for mu in spec)


def vacuum_metric(sigma):
    """The solution of -g ln g = sigma nearest to g = 1 (Eq. 65). It exists for 0 < sigma < 1/e."""
    assert 0.0 < sigma < 1.0 / math.e
    lo, hi = 1.0 / math.e, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if -mid * math.log(mid) > sigma else (lo, mid)
    return 0.5 * (lo + hi)


def vacuum_action_per_cell(sigma):
    """S+ per cell at her vacuum: sigma ln g + g ln g - g, with g ln g = -sigma."""
    g = vacuum_metric(sigma)
    return sigma * math.log(g) - sigma - g


def action_at_identity(spec, c0):
    """S+ at G = I with the induced metric I + c0 L: -sum ln(1 + c0 mu) - (number of cells)."""
    mu = np.clip(np.concatenate(spec), 0.0, None)
    return -float(np.log1p(c0 * mu).sum()) - len(mu)


def arrangements(n=64):
    """Same N, different wiring: the flat torus, the curled torus, the same re-glued once, and a gas of 4-cubes."""
    assert n % 16 == 0
    flat, _ = torus((8, n // 8))
    tube, labels = torus((n // 4, 4))
    _counts, neutral = census(tube, np.flatnonzero(labels == 0))
    reglued = neutral[0][0]                                   # one free switch: two rings joined the other way
    knots, _ = disjoint([torus((4, 4))] * (n // 16))
    return [("flat torus", flat), ("curled torus (tube)", tube), ("tube, re-glued once", reglued),
            ("%d knots (4-cubes)" % (n // 16), knots)]


# The runs whose saved end states O111 was computed on (10 October 2026). Runs saved later are left out, so that
# the numbers on the record do not move when new results arrive (T60's end states did move them, on 10 October).
RECORDED_RUNS = ("t7d_", "t8_", "t23_", "t24_", "t38_")


def saved_wirings(per_kind=25, n=64):
    """End states saved by the registered runs in RECORDED_RUNS at N points (results/<run>_adj), at most `per_kind`
    of each kind (S - N, X), in file order, with the three rungs of the ladder added. Reading only."""
    from graphity.cqg import surplus, total_squares
    results = Path(__file__).resolve().parents[1] / "results"
    kinds = {}
    for path in sorted(results.glob("*_adj/N%d_rep*.npz" % n)):
        if not path.parent.name.startswith(RECORDED_RUNS):
            continue
        adj = np.load(path)["adj"]
        if adj.shape != (n, 4):
            continue
        key = (int(total_squares(adj)), int(surplus(adj)))
        if len(kinds.setdefault(key, [])) < per_kind:
            kinds[key].append(adj)
    for name, adj in arrangements(n):
        if "re-glued" not in name:
            kinds.setdefault((int(total_squares(adj)), int(surplus(adj))), []).append(adj)
    return [(s, x, adj) for (s, x), adjs in sorted(kinds.items()) for adj in adjs]


def effective_rule(wirings, c0):
    """Least-squares fit of S+ at G = I to a + b S + c X over the wirings given. This model's energy is
    16 (N - S) + 4 lambda X, so the action orders these wirings like that energy with lambda_eff = -4 c / b.
    Returns (b, c, lambda_eff, rms residual, largest residual)."""
    design = np.array([[1.0, s, x] for s, x, _adj in wirings])
    action = np.array([action_at_identity(spectra(adj), c0) for _s, _x, adj in wirings])
    coef = np.linalg.lstsq(design, action, rcond=None)[0]
    left = action - design @ coef
    return (float(coef[1]), float(coef[2]), float(-4.0 * coef[2] / coef[1]),
            float(np.sqrt((left ** 2).mean())), float(np.abs(left).max()))


def main(n=64, sigma=0.1, c0s=(0.1, 1.0)):
    per_cell = vacuum_action_per_cell(sigma)
    print("Her vacuum (c0 = 0), sigma = %.2f: g = %.6f, action per cell %.6f (ours: the wiring enters only "
          "through the number of cells)\n" % (sigma, vacuum_metric(sigma), per_cell))
    print("N = %d. Cells are nodes + links + squares; holes are (pieces, loops, cavities)." % n)
    print("%-22s %7s %6s %-14s %12s" % ("arrangement", "squares", "cells", "holes", "vacuum S+")
          + "".join("   S+ at G=I, c0=%-4g" % c for c in c0s))
    for name, adj in arrangements(n):
        spec = spectra(adj)
        n_cells = sum(len(mu) for mu in spec)
        print("%-22s %7d %6d %-14s %12.3f" % (name, len(spec[2]), n_cells, betti(spec), per_cell * n_cells)
              + "".join("   %18.4f" % action_at_identity(spec, c) for c in c0s))
    wirings = saved_wirings(n=n)
    kinds = len({(s, x) for s, x, _adj in wirings})
    print("\nAcross %d wirings of %d kinds (saved end states of the registered runs, and the ladder), S+ at G = I"
          " fitted to a + b S + c X (ours, exploratory):" % (len(wirings), kinds))
    print("   c0      per square b   per surplus square c   lambda_eff   rms left over   largest")
    for c0 in (0.01, 0.1, 1.0, 10.0, 100.0):
        b, c, lam, rms, worst = effective_rule(wirings, c0)
        print("  %-6g %12.4f %18.4f %14.3f %13.3f %10.3f" % (c0, b, c, lam, rms, worst))


if __name__ == "__main__":
    main()
