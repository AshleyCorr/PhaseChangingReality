"""Curl and anti-curl on the squares of the four-link toy: the owner's two kinds, built as a state
(docs/design/curl_pairs_build_2026-11-03.md; VISION Update 156, the S1 decision of 3 November 2026).

HER RULE (3 November, her correction of the record's "the toy holds nothing that comes in opposite kinds"): the two
kinds are the two same-colour diagonals of one square. The black-to-black diagonal carries curl, matter; the white-to-white
diagonal carries anti-curl, antimatter. Matter, energy and twist are stored between points of the same kind; the links
(black to white) carry the waves.

EXACT, FROM THE WIRING: every square of a bipartite wiring has two black corners and two white, so exactly one diagonal of
each kind. The wiring alone therefore holds the two kinds always paired, one of each per square: her "equal pairs that
cancel". The energy H = 16(N - S) + 4 lambda X counts squares and cannot tell the two diagonals apart. A surplus of one
kind, her net matter, can live only in a STATE on the squares, which is what this module holds.

THE STATE (the first form, ours from her rules; fixed in the design document before any run): one real number c in
[-1, 1] per square. c > 0 is curl of size |c| stored on the square's black diagonal (matter); c < 0 is anti-curl of size
|c| on its white diagonal (antimatter); c = 0 is an empty square. Her rules carried: curl is continuous; a pair is made
with equal absolute values (+c, -c); only a full curl, |c| = 1, stores energy stably, so the readers count |c| < 1 as
unstable. No dynamics is built here: where pairs are created, when a pair cancels (and whether into a photon or into a
frozen pair), and how lambda sets the number of pairs are her choices, listed in the design document, and nothing runs
before she makes them.

READERS (exact): net_curl = sum c (her net matter); stored = sum |c| (energy held, in full curls); stable = the number of
squares with |c| = 1; census by kind; stored_pairs(kind) lists the same-colour pairs carrying a kind, in the form the
Carroll reader of T94 takes, so that a window with matter only (black diagonals) can be read apart from one with pairs.
"""
from dataclasses import dataclass, field

import numpy as np

MATTER, ANTIMATTER = "matter", "antimatter"


def squares(adj, part):
    """Every 4-cycle of the wiring as (black_pair, white_pair), each pair a sorted tuple of two corners; sorted.

    A 4-cycle u - v - a - b has u and a on one side and v and b on the other, so the two diagonals are (u, a) and
    (v, b). Under the hard-core rule two points share at most two neighbours, so four corners hold at most one square.
    """
    seen = set()
    out = []
    n = len(adj)
    for u in range(n):
        for v in adj[u]:
            if v < 0:
                continue
            for a in adj[v]:
                if a < 0 or a == u:
                    continue
                for b in adj[a]:
                    if b < 0 or b == v or not (adj[u] == b).any():
                        continue
                    key = frozenset((int(u), int(v), int(a), int(b)))
                    if key in seen:
                        continue
                    seen.add(key)
                    d1, d2 = tuple(sorted((int(u), int(a)))), tuple(sorted((int(v), int(b))))
                    black, white = (d1, d2) if part[u] == 0 else (d2, d1)
                    out.append((black, white))
    return sorted(out)


def wiring_is_paired(adj, part):
    """Exact check of the fact the state rests on: every square has one black and one white diagonal."""
    for black, white in squares(adj, part):
        if not (part[black[0]] == part[black[1]] == 0 and part[white[0]] == part[white[1]] == 1):
            return False
    return True


@dataclass
class CurlState:
    """The wiring with one real curl value per square."""
    adj: np.ndarray
    part: np.ndarray
    sq: list = field(init=False)
    c: np.ndarray = field(init=False)

    def __post_init__(self):
        self.sq = squares(self.adj, self.part)
        self.c = np.zeros(len(self.sq))

    def set_pair(self, i, j, size):
        """A pair made together: curl +size on square i (black diagonal) and anti-curl -size on square j (white)."""
        if not 0 < size <= 1:
            raise ValueError("a curl's size is in (0, 1]")
        if i == j:
            raise ValueError("a pair sits on two squares; one square holds one value")
        self.c[i] = float(size)
        self.c[j] = -float(size)

    def set_curl(self, i, value):
        if not -1 <= value <= 1:
            raise ValueError("curl is in [-1, 1]")
        self.c[i] = float(value)

    def net_curl(self):
        return float(self.c.sum())

    def stored(self):
        return float(np.abs(self.c).sum())

    def stable(self):
        return int(np.sum(np.abs(self.c) == 1.0))

    def census(self):
        return dict(squares=len(self.sq), matter=int(np.sum(self.c > 0)), antimatter=int(np.sum(self.c < 0)),
                    empty=int(np.sum(self.c == 0)), stable=self.stable(), net_curl=self.net_curl(), stored=self.stored())

    def stored_pairs(self, kind):
        """The same-colour pairs carrying `kind`: black diagonals of squares with c > 0, or white ones with c < 0."""
        if kind == MATTER:
            return [black for (black, _), v in zip(self.sq, self.c) if v > 0]
        if kind == ANTIMATTER:
            return [white for (_, white), v in zip(self.sq, self.c) if v < 0]
        raise ValueError(kind)


def daughter_front(L, R, lam, unit=1.0):
    """T97's first form of the owner's pair rules (PREREGISTRATION T97; VISION Update 156 and its addenda). Exact.

    A flat sheet of squares L wide (y) and R rows deep (x = 0..R-1) is the daughter; row R is outside it. A square holds
    one curl value. Its budget is b = 4 (lam - 1) in units where one full curl is `unit`; it makes one pair of size
    s = min(b / (2 unit), 1), and any budget beyond a full curl leaves as light at once. The front passes one row at a
    time: pairs standing across the link between rows x - 1 (matter, +s) and x (antimatter, -s) cancel into two photons
    if either square has an empty link-neighbour, else freeze; then every empty square of row x puts +s on itself (its
    black diagonal) and -s on the square ahead across their shared link, (x + 1, y). Row R - 1's antimatter lands on
    row R, outside, where nothing cancels it. Returns the ledger.
    """
    budget = 4.0 * (lam - 1.0)
    s = min(budget / (2.0 * unit), 1.0)
    light_per_square = max(0.0, budget - 2.0 * s * unit)
    c = {}                                                    # (x, y) -> curl value; absent is empty
    pairs = []                                                # (matter square, antimatter square, size)
    frozen = photons = 0
    photon_energy = light = 0.0

    def empty_neighbour(sq):
        x, y = sq
        for nb in ((x + 1, y), (x - 1, y), (x, (y + 1) % L), (x, (y - 1) % L)):
            if nb not in c and 0 <= nb[0] <= R:
                return True
        return False

    for x in range(R):
        for (m, a, size) in [p for p in pairs if p[1][0] == x]:
            if c.get(m) == size and c.get(a) == -size:
                if empty_neighbour(m) or empty_neighbour(a):
                    del c[m]; del c[a]
                    photons += 2; photon_energy += 2.0 * size * unit
                else:
                    frozen += 1
        for y in range(L):
            light += light_per_square
            if (x, y) in c or (x + 1, y) in c:
                continue
            c[(x, y)] = s; c[(x + 1, y)] = -s
            pairs.append(((x, y), (x + 1, y), s))
    inside = [v for (x, y), v in c.items() if x < R]
    surplus_curl = float(sum(inside))
    surplus_energy = float(sum(abs(v) for v in inside if v > 0) * unit)
    gross = float(sum(p[2] for p in pairs) * unit)
    return dict(L=L, R=R, lam=lam, pair_size=s, pairs_made=len(pairs), surplus_curl=surplus_curl,
                surplus_energy=surplus_energy, gross=gross, surplus_fraction=surplus_energy / gross if gross else 0.0,
                frozen=frozen, photons=photons, photon_energy=photon_energy, light=light)
