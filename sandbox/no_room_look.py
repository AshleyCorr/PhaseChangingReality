"""Sandbox look (5 November 2026, her words "no photons, there is no space for them, movement through space requires
cubes, there is only squares"): the moves on a sheet with room switched off, the sheet as her curled slab. No pair
ever cancels; same-hand neighbours join into stores. Measured: stores per pair, halves left, the largest cluster, the
like-sign fraction among neighbouring stores. A look, not a test."""
import sys

from graphity.cqg import NO_CAP, torus
from graphity.curl_pair_moves import PairMoves
from graphity.curl_pairs import CurlState


class NoRoom(PairMoves):
    def room(self, i):
        return False


if __name__ == "__main__":
    sides = [int(a) for a in sys.argv[1:]] or [8, 12, 16]
    for side in sides:
        for lam in (1.1, 1.25, 1.5, 2.0):
            for seed in (1, 2, 3):
                adj, part = torus(side, side, NO_CAP)
                m = NoRoom(CurlState(adj, part), lam=lam, barrier=0.0, seed=seed, price_from="gravity", clock="simultaneous")
                r = m.run(max_sweeps=500)
                s = m.store
                pairs = [(i, j) for i, j in m.links if s[i] and s[j]]
                like = sum(s[i] == s[j] for i, j in pairs) / len(pairs) if pairs else float("nan")
                print(f"side {side:2d} lam {lam:.2f} seed {seed}: pairs {r['pairs_made']:4d} stores {r['stores']:4d} "
                      f"(+{r['stores_plus']} -{r['stores_minus']}) stores/pairs {r['stores'] / r['pairs_made']:.3f} "
                      f"halves left {r['frozen_curls']:3d} photons {r['photons']} cluster {r['store_cluster']} "
                      f"like-sign {like:.2f} sweeps {r['sweeps']}", flush=True)
