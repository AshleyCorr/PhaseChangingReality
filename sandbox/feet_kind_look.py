"""Sandbox look (6 November 2026, her rule accepted the same morning: "the first pair made across the boundary sets
the outside foot's kind, and everything after follows it"). The loose-band tube of `boundary_feet_look.py` (4 x L,
T37's cell, a band of L/8 columns loosened by k switches) run to a fixed time after the escape; then her curl state
put on the squares of that wiring and her clock run (room judged once on the made state, price from gravity, T105's
form), with the coin that places each pair's matter replaced by her rule: the first pair whose two squares straddle
the band's edge decides which kind lies outward (a coin, once per replica, so that twins of opposite sign appear
across replicas); every pair made after it puts that kind on its square farther from the band's centre (a tie keeps
the coin). Read at the freeze: net twist on the squares inside the band and outside it, stores by sign on each side.
The control is the same with the coin kept (T105's rule), which must read zero on both sides. A look, not a test."""
import sys

import numpy as np

from graphity.cqg import NO_CAP, run_chain, torus
from graphity.curl_pair_moves import PairMoves
from graphity.curl_pairs import CurlState
from graphity.dimension import local_dimension
from loose_band_look import loosen


class BoundaryKind(PairMoves):
    """PairMoves with her rule for the kind on each side of the boundary. `dist` is each square's distance from the
    band's centre (circular, in columns); `edge` the band's half width. outward_kind is 0 until the first straddling
    pair is made."""

    def __init__(self, state, dist, edge, rule, **kw):
        super().__init__(state, **kw)
        self.dist, self.edge, self.rule = dist, edge, rule
        self.outward_kind = 0

    def make_pairs(self):
        c, made = self.state.c, 0
        for k in self.rng.permutation(len(self.links)):
            if self.ledger < 2.0 * self.unit:
                break
            i, j = self.links[k]
            if c[i] == 0.0 and c[j] == 0.0:
                if self.rule == "coin" or self.dist[i] == self.dist[j]:
                    if self.rng.integers(2):
                        i, j = j, i
                else:
                    outer, inner = (i, j) if self.dist[i] > self.dist[j] else (j, i)
                    if self.outward_kind == 0 and (self.dist[inner] < self.edge <= self.dist[outer]):
                        self.outward_kind = 1 if self.rng.integers(2) else -1
                    if self.outward_kind == 0:
                        if self.rng.integers(2):
                            i, j = j, i
                    else:
                        i, j = (outer, inner) if self.outward_kind > 0 else (inner, outer)
                c[i], c[j] = 1.0, -1.0
                self.ledger -= 2.0 * self.unit
                made += 1
        self.pairs_made += made
        return made


def escaped_wiring(L, k, rep, lam=1.25, g=1.5, block=10, max_sweeps=20000, min_size=8, after=200):
    adj, part = torus(L, 4, NO_CAP)
    n = adj.shape[0]
    x_of = np.arange(n) // 4
    centre = L // 2
    width = L // 8
    in_band = np.abs(((x_of - centre + L // 2) % L) - L // 2) < width / 2
    rng = np.random.default_rng([20261105, L, k, rep])
    loosen(adj, part, rng, in_band, k)
    side_u = np.flatnonzero(part == 0)
    seed = int(np.random.SeedSequence([20261105, L, k, rep]).generate_state(1)[0])
    init = local_dimension(adj) == 2
    first, escape = True, None
    for sweeps in range(0, max_sweeps, block):
        run_chain(adj, side_u, 1.0 / g, 0, block, seed if first else -1, lam=lam, cap=NO_CAP)
        first = False
        t = sweeps + block
        if escape is None:
            if ((local_dimension(adj) == 2) & ~init).sum() < min_size:
                continue
            escape = t
        if t - escape >= after:
            break
    return adj, part, x_of, centre, width / 2, escape


def circ_dist(x, centre, L):
    return np.abs(((x - centre + L / 2) % L) - L / 2)


def one(L, k, rep, rule, lam=1.25, barrier=12):
    adj, part, x_of, centre, edge, escape = escaped_wiring(L, k, rep, lam=lam)
    st = CurlState(adj, part)
    # a square's column: the circular mean of its corners' columns, then its distance from the band's centre
    ang = 2 * np.pi * x_of / L
    dist = np.empty(len(st.sq))
    for q, (black, white) in enumerate(st.sq):
        corners = list(black) + list(white)
        m = np.arctan2(np.sin(ang[corners]).mean(), np.cos(ang[corners]).mean()) * L / (2 * np.pi)
        dist[q] = circ_dist(m, centre, L)
    inside = dist < edge
    pm = BoundaryKind(st, dist, edge, rule, lam=lam, barrier=barrier,
                      seed=int(np.random.SeedSequence([20261106, L, k, rep]).generate_state(1)[0]),
                      price_from="gravity", clock="simultaneous")
    pm.run()
    c, s = st.c, pm.store
    free = s == 0
    twist = 0.5 * np.sign(c) * free + s          # a free full curl carries a half twist; a store a full one
    return dict(escape=escape, squares=len(st.sq), inside_sq=int(inside.sum()), outward_kind=pm.outward_kind,
                twist_in=float(twist[inside].sum()), twist_out=float(twist[~inside].sum()),
                stores_in_plus=int((s[inside] > 0).sum()), stores_in_minus=int((s[inside] < 0).sum()),
                stores_out_plus=int((s[~inside] > 0).sum()), stores_out_minus=int((s[~inside] < 0).sum()),
                photons=pm.photons, pairs=pm.pairs_made, books_ok=abs(pm.books() - pm.release) < 1e-9)


if __name__ == "__main__":
    L = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    for rule in ("coin", "boundary"):
        rows = []
        for rep in range(reps):
            r = one(L, k, rep, rule)
            rows.append(r)
            print(f"L {L} k {k} {rule} rep {rep:2d}: escape {r['escape']} squares {r['squares']} (inside {r['inside_sq']}) "
                  f"outward {r['outward_kind']:+d} twist in {r['twist_in']:+.1f} out {r['twist_out']:+.1f} "
                  f"stores in +{r['stores_in_plus']} -{r['stores_in_minus']} out +{r['stores_out_plus']} -{r['stores_out_minus']} "
                  f"pairs {r['pairs']} photons {r['photons']} books {r['books_ok']}", flush=True)
        # score each replica's twist by the outward kind, so that twins of opposite sign add instead of cancelling
        sgn = np.array([r["outward_kind"] or 1 for r in rows])
        tin = np.array([r["twist_in"] for r in rows]) * sgn
        tout = np.array([r["twist_out"] for r in rows]) * sgn
        print(f"== L {L} k {k} {rule} ({reps} replicas): twist outside, in the outward kind's sign: median {np.median(tout):+.1f} "
              f"(positive in {np.mean(tout > 0):.2f}); twist inside: median {np.median(tin):+.1f} (negative in {np.mean(tin < 0):.2f}); "
              f"opposite signs on the two feet in {np.mean((tout > 0) & (tin < 0)):.2f}", flush=True)
