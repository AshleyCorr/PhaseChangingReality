"""EXPLORATORY, read-only (11 October 2026): what kind of surface is each exactly flat end state? Usage:

    python scripts/read_flat_kind.py [results_dir]

Why. `read_endpoint_width.py` reads a width from three points and calls an end state flat when H = 0. Two things it
cannot see were found on 11 October in T61's saved end states: an end state with H = 0 can be **two or more
separate pieces**, each flat; and a flat piece can be **one-sided**, a Klein bottle, which is flat at every point
and still does not look the same from every point.

What is exact. In a graph with H = 0 every link lies on exactly two squares. Glue the squares along the links they
share. If the four squares at every point form one ring, each connected piece is a closed surface with as many
squares as points and twice as many links, so its Euler number is zero and it is a torus (two-sided) or a Klein
bottle (one-sided). It is two-sided exactly when every square can be given a direction of turn such that two
squares sharing a link run along it in opposite directions.

What is read. Every saved final graph of the tube runs T8, T23, T24, T60 and T61 with H = 0. For each: its
connected pieces, the kind of each, and each piece's width from one of its points (`read_endpoint_width.width`;
for a torus that is its width, for a Klein bottle it is only what that point sees).

Nothing here is a registered rule and no verdict changes. It reads results/ and writes nothing.
"""
import glob
import re
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from read_endpoint_width import is_flat, width                    # noqa: E402

RUNS = ("t8_lam*_adj", "t23_lam*_adj", "t24_lam*_adj", "t60_lam*_adj", "t61_lam*_adj")
TORUS, KLEIN = "torus", "Klein bottle"


def klein_bottle(around, along):
    """A flat Klein bottle for tests: point (x, y), x in 0..around-1, y in 0..along-1; stepping past x = around-1
    arrives at (0, -y). `around` and `along` must be even, for two sides, and at least 6, for the hard-core rule."""
    def index(x, y):
        return x * along + (y % along)
    adj = np.empty((around * along, 4), dtype=np.int64)
    for x in range(around):
        for y in range(along):
            ahead = index(0, -y) if x == around - 1 else index(x + 1, y)
            back = index(around - 1, -y) if x == 0 else index(x - 1, y)
            adj[index(x, y)] = (index(x, y + 1), index(x, y - 1), ahead, back)
    return adj


def pieces(adj):
    """The connected pieces, each as its own neighbour array with points renumbered from 0."""
    n = adj.shape[0]
    label = np.full(n, -1, dtype=np.int64)
    out = []
    for start in range(n):
        if label[start] >= 0:
            continue
        members = [start]
        label[start] = len(out)
        for v in members:                                             # grows while it is walked
            for w in adj[v]:
                if label[w] < 0:
                    label[w] = len(out)
                    members.append(int(w))
        members.sort()
        new = {v: i for i, v in enumerate(members)}
        out.append(np.array([[new[int(w)] for w in adj[v]] for v in members], dtype=np.int64))
    return out


def squares_of(adj):
    """Every square once, as four points in the order round it."""
    found = {}
    nbr = [set(int(x) for x in row) for row in adj]
    for u in range(adj.shape[0]):
        for a in nbr[u]:
            for c in nbr[u]:
                if a < c:
                    for w in nbr[a] & nbr[c]:
                        if w != u:
                            found[frozenset((u, a, w, c))] = (u, a, w, c)
    return list(found.values())


def kind(adj):
    """TORUS or KLEIN for one connected flat piece; otherwise a sentence saying why it is not a closed surface."""
    faces = squares_of(adj)
    on_link = defaultdict(list)
    at_point = defaultdict(list)
    for f, cyc in enumerate(faces):
        for i in range(4):
            on_link[frozenset((cyc[i], cyc[(i + 1) % 4]))].append((f, cyc[i]))
            at_point[cyc[i]].append(f)
    if any(len(v) != 2 for v in on_link.values()):
        return "not flat: a link without exactly two squares"
    for v in range(adj.shape[0]):                                     # the squares at a point must form ONE ring
        mine = at_point[v]
        if len(mine) != 4:
            return "not a surface: a point without exactly four squares"
        ring, grew = {mine[0]}, True
        while grew:
            grew = False
            for w in adj[v]:
                (f1, _), (f2, _) = on_link[frozenset((v, int(w)))]
                if (f1 in ring) != (f2 in ring):
                    ring.update((f1, f2))
                    grew = True
        if len(ring) != 4:
            return "not a surface: the squares at a point form two rings"
    turn = {0: 1}
    queue = deque([0])
    one_sided = False
    while queue:
        f = queue.popleft()
        cyc = faces[f]
        for i in range(4):
            p = cyc[i]
            (f1, p1), (f2, p2) = on_link[frozenset((p, cyc[(i + 1) % 4]))]
            other, first = (f2, p2) if f1 == f else (f1, p1)
            want = -turn[f] if first == p else turn[f]                # both list the link from p: they must differ
            if other not in turn:
                turn[other] = want
                queue.append(other)
            elif turn[other] != want:
                one_sided = True
    if len(turn) != len(faces):
        return "not one piece"
    return KLEIN if one_sided else TORUS


def describe(adj):
    """For a graph with H = 0: a sorted tuple of (points, kind, width seen from the piece's first point)."""
    return tuple(sorted((p.shape[0], kind(p), width(p, 0)) for p in pieces(adj)))


def main(results="results"):
    table = defaultdict(Counter)
    kleins = []
    for pattern in RUNS:
        for folder in sorted(glob.glob(str(Path(results) / pattern))):
            run = re.match(r"(t\d+)_", Path(folder).name).group(1)
            for path in sorted(Path(folder).glob("N*.npz")):
                adj = np.load(path)["adj"]
                n = adj.shape[0]
                if not is_flat(adj):
                    continue
                parts = describe(adj)
                cell = table[(run, n)]
                cell["flat"] += 1
                if len(parts) == 1 and parts[0][1] == TORUS:
                    cell["one torus, %s around" % parts[0][2]] += 1
                elif len(parts) == 1:
                    cell["one piece: %s" % parts[0][1]] += 1
                else:
                    cell["%d pieces" % len(parts)] += 1
                    cell["  of those, every piece a torus six around"] += all(k == TORUS and w == 6 for _, k, w in parts)
                if any(k == KLEIN for _, k, _ in parts):
                    kleins.append((str(path), parts))
    print("Exactly flat end states of the tube runs, by kind (exploratory; a torus is at least 6 around).\n")
    for (run, n), cell in sorted(table.items()):
        print("%-4s N = %-4d flat %5d | %s" % (run, n, cell["flat"],
              "; ".join("%s: %d" % (k.strip(), v) for k, v in sorted(cell.items()) if k != "flat")))
    print("\nEnd states with a one-sided piece: %d" % len(kleins))
    for path, parts in kleins:
        print("  %s: %s" % (path, "; ".join("%d points, %s, sees %s around" % p for p in parts)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "results"))
