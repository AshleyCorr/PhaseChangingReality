"""The two kinds as moves on the curl state (docs/design/curl_pair_moves_2026-11-04.md; her answers of 4 November 2026,
sandbox/NOTEBOOK.md the same day). Her rules put in and read back; nothing derived.

HER RULES (hers, 3 and 4 November): a pair is made full, +1 curl with +1/2 twist and -1 curl with -1/2 twist, on the two
squares sharing a link, costing two units of the release. A pair cancels into two photons when there is room; if not,
the pair stays apart, and each half finds a matching (same-sign) twist and joins it into a two-sided Klein store, so a
failed pair makes two stores of opposite sign. A store cannot move and cannot cancel. Twist is not a second number: a
full curl carries a half twist of its sign, and a store is a flag meaning two halves joined into one full twist. Seeded
sweeps: make pairs while the ledger can pay, cancel where there is room, join where there is none, freeze when a sweep
changes nothing. No cooling, no bath.

OURS (stand-ins, marked in the design document): "room" on a sheet is an empty square beside either square (T97's form).
Which square of a link gets the matter is a seeded coin. Any +1 and -1 on squares sharing a link may cancel, not only
the two born together. A cancelled pair's energy leaves as light and never returns (her step 3: once there is room, the
light leaves). A store is paid from the ledger: the barrier B (no value from her; scanned) plus the curling cost
4(lambda - 1) for its square; the emptied partner's curl joins the store. The ledger starts at the release 4(lambda - 1)
per point in units where one full curl is one unit.

HER RULE OF 4 NOVEMBER, AFTER T103 (price_from="gravity"): the store's price comes from gravity, "which is one directional and
doesn't get lost as it's spent but increase as it's spent to make more dense particles". Stand-in (ours): a join costs the
ledger nothing; a one-way counter `gravity` grows by B + 4(lambda - 1) at every join and is never spent. A count, not an
energy, in this build. price_from="ledger" is T103's form, kept so that T103 reproduces.

HER CLOCK OF 4 NOVEMBER, AFTER T104 (clock="simultaneous", "Yes to 1"): room is judged once on the made state, for every
pair at the same instant; the pairs with room cancel into light and the same-sign neighbours without room join into
stores, each square used once, cancellations first. clock="sequential" is T103's and T104's sweep (cancel link after link,
then join), kept so that they reproduce; its order was ours and is what left no square without room (the T104 reading).

HER KIND RULE OF 6 NOVEMBER (kind_rule="boundary"; T107): "the first pair made across the boundary sets the outside
foot's kind, and everything after follows it". With `dist` (each square's distance from the band's centre) and `edge`
(the band's half width) given: the first pair whose two squares straddle the edge draws the outward kind once (a
coin, so that twins of opposite sign appear across replicas); every pair made after it puts that kind on its square
farther from the centre (a tie keeps the coin). kind_rule="coin" is T103 to T105's rule, kept so that they reproduce.
Stand-in (ours): the straddling test and the farther-square placing; her words name neither a coin nor a tie.

EXACT: ledger + light + (curl held on squares) + (held in stores) = the release, to the last digit, after every move,
under both forms (under "gravity" the store holds only its partner's curl) and both clocks.
"""
from dataclasses import dataclass, field

import numpy as np

from graphity.curl_pairs import CurlState


def square_links(sq):
    """Pairs of squares that share a link, each once, and the neighbour lists. Two squares share a link exactly when
    they share two corners that are adjacent in the wiring; on a bipartite wiring two shared corners of different
    colour are always adjacent along the square, so sharing one black and one white corner is the test."""
    index = {}
    for k, (black, white) in enumerate(sq):
        for b in black:
            for w in white:
                index.setdefault((b, w), []).append(k)
    links, nb = set(), [set() for _ in sq]
    for members in index.values():
        for a in members:
            for b in members:
                if a < b:
                    links.add((a, b)); nb[a].add(b); nb[b].add(a)
    return sorted(links), [sorted(s) for s in nb]


@dataclass
class PairMoves:
    state: CurlState
    lam: float
    barrier: float
    seed: int
    unit: float = 1.0
    ledger: float = None
    price_from: str = "ledger"
    clock: str = "sequential"
    kind_rule: str = "coin"
    dist: np.ndarray = None
    edge: float = None
    links: list = field(init=False)
    nb: list = field(init=False)
    store: np.ndarray = field(init=False)

    def __post_init__(self):
        self.links, self.nb = square_links(self.state.sq)
        self.store = np.zeros(len(self.state.sq), dtype=np.int64)
        self.n_points = len(self.state.adj)
        self.release = 4.0 * (self.lam - 1.0) * self.n_points if self.ledger is None else float(self.ledger)
        self.ledger = self.release
        self.lam_cost = 4.0 * (self.lam - 1.0)
        if self.price_from not in ("ledger", "gravity"):
            raise ValueError(self.price_from)
        if self.clock not in ("sequential", "simultaneous"):
            raise ValueError(self.clock)
        if self.kind_rule not in ("coin", "boundary"):
            raise ValueError(self.kind_rule)
        if self.kind_rule == "boundary" and (self.dist is None or self.edge is None):
            raise ValueError("the boundary rule needs dist and edge")
        self.outward_kind = 0
        self.light = 0.0
        self.held = 0.0
        self.gravity = 0.0
        self.photons = self.pairs_made = self.stores_made = self.sweeps = 0
        self.rng = np.random.default_rng(self.seed)

    # readers (exact)
    def room(self, i):
        c = self.state.c
        return any(c[j] == 0.0 for j in self.nb[i])

    def energy_on_squares(self):
        return float(np.abs(self.state.c).sum() * self.unit)

    def books(self):
        """ledger + light + curl on squares + held in stores; equals the release when the books balance."""
        return self.ledger + self.light + self.energy_on_squares() + self.held

    def census(self):
        c, s = self.state.c, self.store
        free = s == 0
        plus, minus = int(np.sum(s > 0)), int(np.sum(s < 0))
        return dict(sweeps=self.sweeps, pairs_made=self.pairs_made, photons=self.photons, light=self.light,
                    ledger_left=self.ledger, stores_made=self.stores_made, stores_plus=plus, stores_minus=minus,
                    stores=plus + minus, frozen_curls=int(np.sum((c != 0) & free)),
                    net_curl=float(c[free].sum() + 2.0 * s.sum()),
                    net_twist=float(0.5 * np.sign(c[free]).sum() + s.sum()),
                    stored_on_squares=self.energy_on_squares(), held_in_stores=self.held, gravity=self.gravity,
                    store_cluster=self.largest_store_cluster(), books=self.books(), release=self.release,
                    outward_kind=self.outward_kind)

    def largest_store_cluster(self):
        """The largest connected set of store squares (connected through shared links). A look for her question of
        4 November, whether each Klein is a store and the set together a knot; not a registered claim."""
        seen, best = set(), 0
        idx = np.flatnonzero(self.store != 0)
        members = set(int(i) for i in idx)
        for start in members:
            if start in seen:
                continue
            stack, size = [start], 0
            seen.add(start)
            while stack:
                i = stack.pop(); size += 1
                for j in self.nb[i]:
                    if j in members and j not in seen:
                        seen.add(j); stack.append(j)
            best = max(best, size)
        return best

    # the moves
    def make_pairs(self):
        c, made = self.state.c, 0
        for k in self.rng.permutation(len(self.links)):
            if self.ledger < 2.0 * self.unit:
                break
            i, j = self.links[k]
            if c[i] == 0.0 and c[j] == 0.0:
                i, j = self._place(i, j)
                c[i], c[j] = 1.0, -1.0
                self.ledger -= 2.0 * self.unit
                made += 1
        self.pairs_made += made
        return made

    def _place(self, i, j):
        """Which square of the pair takes the matter (+1): the coin, or her boundary rule."""
        if self.kind_rule == "coin" or self.dist[i] == self.dist[j]:
            return (j, i) if self.rng.integers(2) else (i, j)
        outer, inner = (i, j) if self.dist[i] > self.dist[j] else (j, i)
        if self.outward_kind == 0 and self.dist[inner] < self.edge <= self.dist[outer]:
            self.outward_kind = 1 if self.rng.integers(2) else -1
        if self.outward_kind == 0:
            return (j, i) if self.rng.integers(2) else (i, j)
        return (outer, inner) if self.outward_kind > 0 else (inner, outer)

    def cancel_pairs(self):
        c, s, done = self.state.c, self.store, 0
        for k in self.rng.permutation(len(self.links)):
            i, j = self.links[k]
            if s[i] or s[j] or c[i] == 0.0 or c[i] != -c[j]:
                continue
            if self.room(i) or self.room(j):
                c[i] = c[j] = 0.0
                self.light += 2.0 * self.unit
                self.photons += 2
                done += 1
        return done

    def join_stores(self):
        c, s, done = self.state.c, self.store, 0
        price = self.barrier + self.lam_cost
        from_ledger = self.price_from == "ledger"
        for k in self.rng.permutation(len(self.links)):
            if from_ledger and self.ledger < price:
                break
            i, j = self.links[k]
            if s[i] or s[j] or c[i] == 0.0 or c[i] != c[j]:
                continue
            if self.room(i) or self.room(j):
                continue
            s[i] = int(np.sign(c[i]))
            c[j] = 0.0
            if from_ledger:
                self.ledger -= price
                self.held += self.unit + price
            else:
                self.gravity += price
                self.held += self.unit
            self.stores_made += 1
            done += 1
        return done

    def _join_one(self, i, j, price, from_ledger):
        c, s = self.state.c, self.store
        s[i] = int(np.sign(c[i]))
        c[j] = 0.0
        if from_ledger:
            self.ledger -= price
            self.held += self.unit + price
        else:
            self.gravity += price
            self.held += self.unit

    def resolve_at_once(self):
        """Her clock: room judged once on the made state; cancellations first, then joins; each square used once."""
        c, s = self.state.c, self.store
        room = np.array([self.room(i) for i in range(len(c))])
        cancels, joins = [], []
        for i, j in self.links:
            if s[i] or s[j] or c[i] == 0.0:
                continue
            if c[i] == -c[j] and (room[i] or room[j]):
                cancels.append((i, j))
            elif c[i] == c[j] and not room[i] and not room[j]:
                joins.append((i, j))
        used, done = set(), 0
        for k in self.rng.permutation(len(cancels)):
            i, j = cancels[k]
            if i in used or j in used:
                continue
            c[i] = c[j] = 0.0
            used.update((i, j))
            self.light += 2.0 * self.unit
            self.photons += 2
            done += 1
        price = self.barrier + self.lam_cost
        from_ledger = self.price_from == "ledger"
        for k in self.rng.permutation(len(joins)):
            i, j = joins[k]
            if i in used or j in used or (from_ledger and self.ledger < price):
                continue
            self._join_one(i, j, price, from_ledger)
            used.update((i, j))
            self.stores_made += 1
            done += 1
        return done

    def sweep(self):
        if self.clock == "simultaneous":
            changes = self.make_pairs() + self.resolve_at_once()
        else:
            changes = self.make_pairs() + self.cancel_pairs() + self.join_stores()
        self.sweeps += 1
        return changes

    def run(self, max_sweeps=10_000):
        while self.sweeps < max_sweeps:
            if self.sweep() == 0:
                break
        out = self.census()
        out["frozen"] = self.sweeps < max_sweeps
        return out
