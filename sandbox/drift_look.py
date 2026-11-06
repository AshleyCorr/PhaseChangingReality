"""Sandbox look (5 November 2026, after her decision "halves drift under gravity before they join"): the defaults of
docs/design/drift_2026-11-05.md as a subclass of PairMoves, run beside the plain moves (T105's form) on small sheets.
Gravity on a square = number of full neighbouring squares (a store counts two); a half hops to an empty neighbour with a
strictly higher count; drift until nothing can rise, then her clock. Measured: stores per pair, the largest cluster, the
share of leftover halves on the store cluster's edge, the like-sign fraction among neighbouring stores. A look, not a test."""
import sys

import numpy as np

from graphity.cqg import NO_CAP, torus
from graphity.curl_pair_moves import PairMoves
from graphity.curl_pairs import CurlState


class DriftMoves(PairMoves):
    max_rounds = 20   # hops are not counted as a change for the freeze; a half that keeps hopping does not keep the sheet alive

    def gravity_count(self):
        c, s = self.state.c, self.store
        full = (c != 0) | (s != 0)
        w = full.astype(float) + (s != 0)
        return np.array([w[self.nb[i]].sum() for i in range(len(c))])

    def drift(self):
        c, s = self.state.c, self.store
        hops = 0
        for _ in range(self.max_rounds):
            g = self.gravity_count()
            moved = 0
            for i in self.rng.permutation(len(c)):
                if c[i] == 0.0 or s[i]:
                    continue
                best, cands = g[i], []
                for j in self.nb[i]:
                    if c[j] == 0.0 and s[j] == 0 and g[j] > best:
                        cands.append(j)
                if cands:
                    j = cands[self.rng.integers(len(cands))]
                    c[j], c[i] = c[i], 0.0
                    moved += 1
            hops += moved
            if moved == 0:
                break
        return hops

    def sweep(self):
        made = self.make_pairs()
        self.drift()
        changes = made + self.resolve_at_once()
        self.sweeps += 1
        return changes

    def edge_share(self):
        """Share of the leftover halves that share a link with a store."""
        c, s = self.state.c, self.store
        halves = [i for i in range(len(c)) if c[i] != 0 and not s[i]]
        if not halves:
            return float("nan")
        return sum(any(s[j] != 0 for j in self.nb[i]) for i in halves) / len(halves)

    def like_sign(self):
        s = self.store
        pairs = [(i, j) for i, j in self.links if s[i] and s[j]]
        if not pairs:
            return float("nan")
        return sum(s[i] == s[j] for i, j in pairs) / len(pairs)


if __name__ == "__main__":
    sides = [int(a) for a in sys.argv[1:]] or [8, 12, 16]
    for side in sides:
        for lam in (1.25, 1.5, 2.0):
            for seed in (1, 2, 3):
                out = {}
                for name, cls in (("plain", PairMoves), ("drift", DriftMoves)):
                    adj, part = torus(side, side, NO_CAP)
                    m = cls(CurlState(adj, part), lam=lam, barrier=0.0, seed=seed, price_from="gravity", clock="simultaneous")
                    r = m.run(max_sweeps=200)
                    out[name] = (r["stores"] / r["pairs_made"], r["store_cluster"], r["stores"], r["frozen_curls"],
                                 m.edge_share() if name == "drift" else float("nan"),
                                 m.like_sign() if name == "drift" else float("nan"), r["sweeps"])
                p, d = out["plain"], out["drift"]
                print(f"side {side:2d} lam {lam:.2f} seed {seed}: stores/pairs plain {p[0]:.3f} drift {d[0]:.3f} | "
                      f"cluster plain {p[1]} drift {d[1]} (of {d[2]} stores) | halves left plain {p[3]} drift {d[3]} | "
                      f"edge share {d[4]:.2f} like-sign {d[5]:.2f} | sweeps {p[6]}/{d[6]}", flush=True)
