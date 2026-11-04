"""EXACT: the first test of the exchange-phase rule (loop_exchange_brief.md, candidate A) on saved arrangements. Usage:

    python scripts/exact_exchange_sign.py [results/t15_rung0_adj ...]

Written 2026-09-25 (VISION Update 29). Under the fermionic form of the rule, an arrangement's weight is the signed sum
over its renamings, sum of sgn(sigma) over the side-preserving automorphisms sigma, instead of their count. Because the
sign is a homomorphism to {+1, -1}, that sum is |Aut| when every automorphism is an even permutation and exactly zero
when any is odd (the odd ones are then half the group). So the rule is a selection rule: arrangements with an odd
renaming have zero weight, the way two fermions cannot share a state. This script lists every side-preserving
automorphism of each saved graph (igraph, VF2, with the sides as colors), counts the odd ones, and reports the signed
sum beside the plain count. It reads the .npz files of the given directories (adj, part) and, for reference, the flat
torus, the curled torus and a 4-cube.
"""
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphity.cqg import NO_CAP, torus                       # noqa: E402

try:
    import igraph as ig
except ImportError:                                          # pragma: no cover
    raise SystemExit("igraph is needed (pip install igraph)")


def parity(perm):
    """+1 for an even permutation, -1 for an odd one (cycle decomposition)."""
    seen = [False] * len(perm)
    sign = 1
    for i in range(len(perm)):
        if seen[i]:
            continue
        length = 0
        j = i
        while not seen[j]:
            seen[j] = True
            j = perm[j]
            length += 1
        if length % 2 == 0:
            sign = -sign
    return sign


def signed_sum(adj, part, limit=200000):
    """(count, odd, signed sum) over the side-preserving automorphisms; None if more than `limit`."""
    n = adj.shape[0]
    edges = [(u, int(v)) for u in range(n) for v in adj[u] if u < v]
    g = ig.Graph(n=n, edges=edges)
    colors = [int(c) for c in np.asarray(part)]
    count = int(g.count_automorphisms(color=colors))
    if count > limit:
        return count, None, None
    autos = g.get_automorphisms_vf2(color=colors)
    odd = sum(1 for p in autos if parity(p) < 0)
    return count, odd, count - 2 * odd


def report(name, adj, part):
    count, odd, s = signed_sum(adj, part)
    if odd is None:
        print("  %-40s A = %-8d (too many to list; signed sum not computed)" % (name, count))
    else:
        print("  %-40s A = %-8d odd = %-6d signed sum = %-8d -> %s" % (name, count, odd, s, "ALLOWED" if s else "FORBIDDEN"))


def main(dirs):
    print("reference arrangements (four links):")
    for label, dims in (("flat 8 x 8 torus", (8, 8)), ("flat 16 x 10 torus", (16, 10)), ("curled 16 x 4 torus", (16, 4)), ("4-cube (4 x 4)", (4, 4))):
        adj, part = torus(*dims, NO_CAP)
        report(label, adj, part)
    for d in dirs:
        files = sorted(glob.glob(str(Path(d) / "*.npz")))
        if not files:
            continue
        print("saved arrangements in %s:" % d)
        for f in files:
            z = np.load(f)
            report(Path(f).name, z["adj"], z["part"])


if __name__ == "__main__":
    main(sys.argv[1:] or ["results/t15_rung0_adj"])
