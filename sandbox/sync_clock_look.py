"""Sandbox look (4 November 2026): T104's moves with room judged all at once on the made state (a synchronous clock)
instead of one link after another (T103/T104's sequential sweep). A convention of the toy's clock, ours. A look only:
no registration, no verdict; it can kill the sequential clock as the cause, not crown the synchronous one."""
import sys
import numpy as np
from graphity.cqg import NO_CAP, torus
from graphity.curl_pairs import CurlState
from graphity.curl_pair_moves import PairMoves


class SyncMoves(PairMoves):
    def sweep(self):
        made = self.make_pairs()
        c, s = self.state.c, self.store
        room = np.array([self.room(i) for i in range(len(c))])            # judged once, on the made state
        cancels, joins = [], []
        for i, j in self.links:
            if s[i] or s[j] or c[i] == 0.0:
                continue
            if c[i] == -c[j] and (room[i] or room[j]):
                cancels.append((i, j))
            elif c[i] == c[j] and not room[i] and not room[j]:
                joins.append((i, j))
        used, done = set(), 0
        for i, j in (cancels[k] for k in self.rng.permutation(len(cancels))):
            if i in used or j in used or c[i] == 0.0 or c[j] == 0.0:
                continue
            c[i] = c[j] = 0.0; used.update((i, j)); self.light += 2.0 * self.unit; self.photons += 2; done += 1
        price = self.barrier + self.lam_cost
        for i, j in (joins[k] for k in self.rng.permutation(len(joins))):
            if i in used or j in used:
                continue
            s[i] = int(np.sign(c[i])); c[j] = 0.0; used.update((i, j))
            self.gravity += price; self.held += self.unit; self.stores_made += 1; done += 1
        self.sweeps += 1
        return made + done


rows = []
for side in (8, 16, 24):
    for lam in (1.05, 1.1, 1.25, 1.5, 2.0):
        for rep in range(3):
            adj, part = torus(side, side, NO_CAP)
            m = SyncMoves(CurlState(adj, part), lam=lam, barrier=0.0, seed=100 + rep, price_from="gravity")
            r = m.run()
            assert abs(r["books"] - r["release"]) < 1e-9
            rows.append((side, lam, rep, r["stores"], r["stores_plus"], r["stores_minus"], r["frozen_curls"], r["store_cluster"], r["sweeps"]))
import collections
agg = collections.defaultdict(list)
for side, lam, rep, st, sp, sm, fr, cl, sw in rows:
    agg[(side, lam)].append((st, sp, sm, fr, cl, sw))
print("side lam  stores(mean of 3) per_square  +/-  frozen  cluster  sweeps")
for (side, lam), v in sorted(agg.items()):
    a = np.array(v, dtype=float).mean(axis=0)
    print(f"{side:4d} {lam:4.2f}  {a[0]:7.1f}  {a[0]/side**2:8.4f}  {a[1]:.0f}/{a[2]:.0f}  {a[3]:6.1f}  {a[4]:5.1f}  {a[5]:4.1f}")
