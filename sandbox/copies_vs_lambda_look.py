"""Sandbox look (4 November 2026, her question "allowing for the range of lambda, which lambda value has the most
copies/entropy in the 2d network?"): on a small flat torus, count exactly the made states the toy can reach with k
pairs: the k-matchings of the square-link graph, times 2^k for which square of each link gets the matter (the seeded
coin, ours). The ledger 4(lambda - 1) per point buys two units per pair, so k pairs need lambda = 1 + k / (2 n_points).
The k with the most made states is the lambda with the most copies, at this size of sheet. Exact, by memoised
recursion on the remaining squares. A look, not a test."""
import sys
from functools import lru_cache

from graphity.cqg import NO_CAP, torus
from graphity.curl_pair_moves import square_links
from graphity.curl_pairs import CurlState

sys.setrecursionlimit(10000)


def matching_counts(nb):
    n = len(nb)

    @lru_cache(maxsize=None)
    def f(remaining):
        if not remaining:
            return {0: 1}
        v = min(remaining)
        rest = remaining - {v}
        out = dict(f(rest))
        for w in nb[v]:
            if w in rest:
                for k, c in f(rest - {w}).items():
                    out[k + 1] = out.get(k + 1, 0) + c
        return out

    return f(frozenset(range(n)))


if __name__ == "__main__":
    sides = [int(a) for a in sys.argv[1:]] or [4, 6]
    for side in sides:
        adj, part = torus(side, side, NO_CAP)
        st = CurlState(adj, part)
        links, nb = square_links(st.sq)
        n_sq, n_pt = len(st.sq), len(adj)
        counts = matching_counts(tuple(tuple(x) for x in nb))
        print(f"side {side}: {n_pt} points, {n_sq} squares, {len(links)} square links")
        best_m = max(counts, key=counts.get)
        weighted = {k: c * 2 ** k for k, c in counts.items()}
        best_w = max(weighted, key=weighted.get)
        for k in sorted(counts):
            lam = 1 + k / (2 * n_pt)
            print(f"  pairs {k:3d}  lambda {lam:.4f}  layouts {counts[k]:>14,d}  made states {weighted[k]:>18,d}")
        print(f"  most layouts at {best_m} pairs (lambda {1 + best_m / (2 * n_pt):.4f}); most made states at "
              f"{best_w} pairs (lambda {1 + best_w / (2 * n_pt):.4f}); full sheet at {max(counts)} pairs", flush=True)
