"""EXACT: the lattice tori of N points weighed two ways at lambda = 1, every single switch out of each, and the tube's
zero-cost switches read from their positions. Usage:

    python3 scripts/exact_torus_counting.py [N ...]                 the tables (default N = 144 160 192)
    python3 scripts/exact_torus_counting.py --zero L [L ...]        the 4 x L tube's zero-cost switches, by position,
                                                                    and the renamings that fix the first of them
    python3 scripts/exact_torus_counting.py --reach L [L ...] [--cap=K]
                                                                    the H = 0 graphs reachable from the tube by chains
                                                                    of zero-cost switches, up to renaming within sides,
                                                                    with the (dS, dX) of every zero-cost switch out
    python3 scripts/exact_torus_counting.py --exits L [L ...] [--lam=K]
                                                                    the lowest switch costs at lambda K (default 1.25)
                                                                    out of each class of that family

WHY. At lambda = 1 (the published model) every rectangular lattice torus of N points has H = 0, so which torus the
network sits in is decided by counting and by the excitations each torus admits. Two sandbox looks came before this
script, both on 23 October 2026 (sandbox/torus_counting_two_ways.py, sandbox/torus_free_energy_one_switch.py): every
non-square rectangle had A = 2N side-preserving renamings and the square 4N, and the 4 x L tube had 2L valid single
switches of zero cost and the lowest one-switch free energy. This script is their promotion under rule 16(c): the same
counts, exact, with tests (tests/test_exact_torus_counting.py), and the zero-cost switches read from their positions
before anything is named.

WHAT THE READING FOUND (exact, only at the lengths run: the tests name theirs, and --reach was run at every even L
from 8 to 48). Read from positions (tested at L = 8 and 12), each zero-cost switch crosses the two links joining
opposite points of ring y to ring y + 1, so ring y + 1 is joined on through a reflection of the ring that fixes the
other two points, and the 2L switches are one orbit under the tube's renamings. At every even L from 8 to 48, chains
of zero-cost switches reach exactly three classes: the tube, the tube closed through that reflection (A = N; its
non-ring squares form a connected closed surface that cannot be oriented), and the tube closed through a half turn of
the ring (A = 2N; the lattice torus with wraps (4, 0) and (2, L)). At L = 6 a fourth class appears, and at L = 4 (the
4-cube) there are five.

WHAT IS COMPUTED, per torus a x b (a, b even, both >= 4, a b = N; built by cqg.torus, no cap):
  * H at lambda = 1, squares per point S/N, the number A of renamings that keep each side (symmetry.count, as Q15),
    and log10 of the labelled copies (n!)^2 / A with n = N/2 points per side (Q9);
  * every valid single switch: the kernel's move (u1, v1), (u2, v2) -> (u1, v2), (u2, v1) with u1, u2 on side 0,
    one per unordered pair of edges, kept when the result passes is_valid with no cap (4-regular, simple, no two
    points sharing more than two neighbours). Each is recorded by its change of squares dS and of surplus squares dX,
    recomputed from scratch, so its cost at any lambda is dH = -16 dS + 4 lambda dX (Q1);
  * the one-switch free energy of the look, F(g) = -g ln(1 + sum over valid switches of exp(-dH / g)).

WHAT THE ONE-SWITCH F IS NOT. It is the first term of an expansion round one labelled graph. It counts no deeper
excitation, says nothing about whether the chain moves between tori, and is computed at lambda = 1 only. And where a
switch costs nothing its result is itself a ground state, which the full partition function counts once as a labelled
copy of its own class; the zero-cost term of F is therefore not extra entropy of the full ensemble (see --reach).

COORDINATES. cqg.torus(a, b) numbers the point (x, y), 0 <= x < a, 0 <= y < b, as i = x b + y; its side is
(x + y) mod 2. For the tube cqg.torus(4, L): x runs round the curled direction (circumference 4) and y along the tube.
Ring y is the closed loop of four (0, y) (1, y) (2, y) (3, y), itself a square; its four edges are the ring edges.
The links (x, y) - (x, y + 1) join ring y to ring y + 1 (junction y).
"""
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphity import cqg, symmetry                                              # noqa: E402
from graphity.cqg import NO_CAP, _switch, has_edge, is_valid, surplus, total_squares   # noqa: E402

LAM = 1.0
COUPLINGS = (0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0)


# ---------------------------------------------------------------- the tori and their counts

def tori_of(n):
    """Every rectangular lattice torus a x b of n points with a <= b, both even and >= 4."""
    return [(a, n // a) for a in range(4, math.isqrt(n) + 1, 2) if n % a == 0 and (n // a) % 2 == 0 and n // a >= 4]


def build(a, b):
    return cqg.torus(a, b, cap=NO_CAP)


def counts(a, b):
    """H at lambda = 1, squares per point, A, log10 of labelled copies."""
    adj, part = build(a, b)
    n = a * b
    aut = symmetry.count(adj, part)
    log_copies = (2 * math.lgamma(n // 2 + 1) - math.log(aut)) / math.log(10)
    return dict(a=a, b=b, N=n, H=float(cqg.hamiltonian(adj, LAM)), S_per_point=total_squares(adj) / n,
                A=aut, log10_copies=log_copies)


# ---------------------------------------------------------------- every single switch

@njit(cache=True)
def _every_switch(adj, part):
    """Every valid single switch: rows (u1, v1, u2, v2, dS, dX), and the number of pairs of edges tried."""
    n = adj.shape[0]
    eu = np.empty(4 * n, np.int64)
    ev = np.empty(4 * n, np.int64)
    m = 0
    for u in range(n):
        if part[u] == 0:
            for k in range(4):
                eu[m] = u
                ev[m] = adj[u, k]
                m += 1
    s0 = total_squares(adj)
    x0 = surplus(adj)
    out = np.empty((m * (m - 1) // 2, 6), np.int64)
    c = 0
    tried = 0
    for i in range(m):
        for j in range(i + 1, m):
            u1, v1, u2, v2 = eu[i], ev[i], eu[j], ev[j]
            if u1 == u2 or v1 == v2 or has_edge(adj, u1, v2) or has_edge(adj, u2, v1):
                continue
            tried += 1
            _switch(adj, u1, v1, u2, v2)
            if is_valid(adj, NO_CAP):
                out[c, 0] = u1
                out[c, 1] = v1
                out[c, 2] = u2
                out[c, 3] = v2
                out[c, 4] = total_squares(adj) - s0
                out[c, 5] = surplus(adj) - x0
                c += 1
            _switch(adj, u1, v2, u2, v1)                     # the exact reverse, slots included
    return out[:c], tried


def every_switch(adj, part):
    """(rows, tried): every valid single switch out of a copy of `adj` as (u1, v1, u2, v2, dS, dX)."""
    work = adj.copy()
    rows, tried = _every_switch(work, np.asarray(part, dtype=np.int64))
    assert np.array_equal(work, adj)
    return rows, int(tried)


def cost(ds, dx, lam=LAM):
    return -16 * ds + 4 * lam * dx


def cost_histogram(adj, part, lam=LAM):
    """{dH: number of valid switches}, exact integers at lambda = 1."""
    rows, _ = every_switch(adj, part)
    h = Counter()
    for ds, dx in rows[:, 4:6]:
        c = cost(int(ds), int(dx), lam)
        h[int(c) if float(c).is_integer() else c] += 1
    return dict(sorted(h.items()))


def free_energy(hist, g):
    """The look's one-switch free energy, -g ln(1 + sum exp(-dH / g))."""
    return -g * math.log(1.0 + sum(k * math.exp(-dh / g) for dh, k in hist.items()))


# ---------------------------------------------------------------- the tube read from positions

def xy(i, length):
    return int(i) // length, int(i) % length


def index(x, y, length):
    return (x % 4) * length + (y % length)


def tube_symmetries(length):
    """The renamings (x, y) -> (e1 x + i, e2 y + j) mod (4, length) with e1, e2 = +-1 and i + j even, as index arrays.
    They keep every edge and each side of the tube; tests check that there are 2N of them, which is A."""
    maps = []
    for e1 in (1, -1):
        for e2 in (1, -1):
            for i in range(4):
                for j in range(length):
                    if (i + j) % 2 == 0:
                        maps.append(np.array([index(e1 * x + i, e2 * y + j, length)
                                              for x in range(4) for y in range(length)], dtype=np.int64))
    return maps


def squares_of(adj):
    """Every square as a frozenset of its four points (in a two-sided simple graph the set fixes the square)."""
    out = set()
    n = adj.shape[0]
    for u in range(n):
        for v in adj[u]:
            for a in adj[u]:
                if a == v:
                    continue
                for b in adj[v]:
                    if b != u and has_edge(adj, a, b):
                        out.add(frozenset((u, int(v), int(a), int(b))))
    return out


def edge_set(adj):
    return {frozenset((u, int(v))) for u in range(adj.shape[0]) for v in adj[u]}


def ring_maps(adj, length):
    """If every ring edge of the tube is present and every other edge joins ring y to ring y + 1, the junction maps:
    J[y][x] is the x' with (x, y) - (x', y + 1) an edge. None otherwise."""
    maps = []
    for y in range(length):
        jy = []
        for x in range(4):
            nb = {xy(v, length) for v in adj[index(x, y, length)]}
            if {((x + 1) % 4, y), ((x - 1) % 4, y)} - nb:
                return None
            up = [p for p in nb if p[1] == (y + 1) % length]
            down = [p for p in nb if p[1] == (y - 1) % length]
            if len(up) != 1 or len(down) != 1:
                return None
            jy.append(up[0][0])
        maps.append(tuple(jy))
    return maps


def monodromy(maps):
    """Where the point x of ring 0 arrives after following the links once round the tube."""
    out = []
    for x in range(4):
        for jy in maps:
            x = jy[x]
        out.append(x)
    return tuple(out)


def cycle_lengths(perm):
    seen, out = set(), []
    for s in range(len(perm)):
        if s in seen:
            continue
        k, x = 0, s
        while x not in seen:
            seen.add(x)
            x = perm[x]
            k += 1
        out.append(k)
    return sorted(out)


def surface(adj, length):
    """The squares other than the L ring squares, as faces: (closed, V - E + F, orientable). Closed means a connected
    closed surface: every edge on exactly two faces, the faces at each point join, edge to edge, into one ring round
    it, and every face is reached from every other through shared edges. Connectedness is checked because the
    classification of closed surfaces, through which V - E + F and orientability name the surface, is for a
    connected one."""
    faces = [sq for sq in squares_of(adj) if len({xy(v, length)[1] for v in sq}) > 1]
    on_edge = Counter()
    cycles = []
    for sq in faces:
        u = min(sq)
        a, b = [v for v in sq if v != u and has_edge(adj, u, v)]
        w = next(v for v in sq if v not in (u, a, b))
        cyc = (u, a, w, b)                                    # a cyclic order round the square
        cycles.append(cyc)
        for k in range(4):
            on_edge[frozenset((cyc[k], cyc[(k + 1) % 4]))] += 1
    closed = all(c == 2 for c in on_edge.values()) and len(on_edge) == 2 * adj.shape[0]
    for v in range(adj.shape[0]):                             # the faces at v, joined through the edges at v
        pairs = [tuple(p for p in (cyc[(cyc.index(v) - 1) % 4], cyc[(cyc.index(v) + 1) % 4]))
                 for cyc in cycles if v in cyc]
        reached, todo = {pairs[0][0]}, [pairs[0][0]]
        while todo:
            p = todo.pop()
            for a, b in pairs:
                for s, t in ((a, b), (b, a)):
                    if s == p and t not in reached:
                        reached.add(t)
                        todo.append(t)
        closed = closed and len(pairs) == 4 and len(reached) == 4
    chi = adj.shape[0] - len(on_edge) + len(faces)
    # orient: two faces sharing an edge must run along it in opposite directions
    by_edge = {}
    for f, cyc in enumerate(cycles):
        for k in range(4):
            by_edge.setdefault(frozenset((cyc[k], cyc[(k + 1) % 4])), []).append(f)
    sign = {0: 1}
    stack, orientable = [0], True

    def runs(f, p, q):                                        # +1 if face f, as oriented, runs p -> q
        cyc = cycles[f]
        k = cyc.index(p)
        return sign[f] if cyc[(k + 1) % 4] == q else -sign[f]
    while stack:
        f = stack.pop()
        cyc = cycles[f]
        for k in range(4):
            p, q = cyc[k], cyc[(k + 1) % 4]
            for h in by_edge[frozenset((p, q))]:
                if h == f:
                    continue
                cyc_h = cycles[h]
                kh = cyc_h.index(p)
                natural = 1 if cyc_h[(kh + 1) % 4] == q else -1
                want = -runs(f, p, q) * natural               # h must run q -> p
                if h not in sign:
                    sign[h] = want
                    stack.append(h)
                elif sign[h] != want:
                    orientable = False
    closed = closed and len(sign) == len(faces)              # the pass from face 0 reached every face: connected
    return closed, chi, orientable


def switch_orbits(switches, maps):
    """Group switches (each a frozenset of its two removed edges) into orbits under the renamings `maps`."""
    left = set(switches)
    orbits = []
    while left:
        s = left.pop()
        orbit = {s}
        for p in maps:
            orbit.add(frozenset(frozenset(int(p[v]) for v in e) for e in s))
        left -= orbit
        orbits.append(orbit)
    return orbits


def switch_stabilizer(length, switch):
    """The tube's side-keeping renamings that carry the switch (u1, v1, u2, v2, ...) onto itself (the same two removed
    links), each read from positions as a dict (x, y) -> (x', y')."""
    target = frozenset((frozenset((int(switch[0]), int(switch[1]))), frozenset((int(switch[2]), int(switch[3])))))
    out = []
    for p in tube_symmetries(length):
        if frozenset(frozenset(int(p[v]) for v in ed) for ed in target) == target:
            out.append({(x, y): xy(p[index(x, y, length)], length) for x in range(4) for y in range(length)})
    return out


def zero_cost_switches(length, lam=LAM):
    """The tube's switches with dH = 0 at lambda, as rows (u1, v1, u2, v2, dS, dX)."""
    adj, part = build(4, length)
    rows, _ = every_switch(adj, part)
    return [tuple(int(t) for t in r) for r in rows if cost(int(r[4]), int(r[5]), lam) == 0]


def read_zero_cost(length):
    """Each zero-cost switch of the 4 x L tube read from its positions, and what the switched graph is."""
    adj, part = build(4, length)
    tube_key = symmetry.canonical_key(adj, part)
    sq0 = squares_of(adj)
    out = []
    for u1, v1, u2, v2, ds, dx in zero_cost_switches(length):
        new = adj.copy()
        _switch(new, u1, v1, u2, v2)
        sq1 = squares_of(new)
        maps = ring_maps(new, length)
        mono = monodromy(maps) if maps else None
        out.append(dict(
            points=[xy(v, length) for v in (u1, u2, v1, v2)],
            removed=[(xy(u1, length), xy(v1, length)), (xy(u2, length), xy(v2, length))],
            added=[(xy(u1, length), xy(v2, length)), (xy(u2, length), xy(v1, length))],
            lost=sorted(sorted(xy(v, length) for v in sq) for sq in sq0 - sq1),
            gained=sorted(sorted(xy(v, length) for v in sq) for sq in sq1 - sq0),
            dS=ds, dX=dx, A=symmetry.count(new, part), tube=symmetry.canonical_key(new, part) == tube_key,
            junctions=maps, monodromy=mono, lines=[length * c for c in cycle_lengths(mono)] if mono else None,
            surface=surface(new, length)))
    removed = [frozenset(frozenset((u, v)) for u, v in ((r[0], r[1]), (r[2], r[3]))) for r in zero_cost_switches(length)]
    orbits = switch_orbits(removed, tube_symmetries(length))
    return out, orbits, surface(adj, length)


# ---------------------------------------------------------------- the degenerate set reachable from the tube

def describe(adj, part, length):
    maps = ring_maps(adj, length)
    mono = monodromy(maps) if maps else None
    return dict(A=symmetry.count(adj, part), S=int(total_squares(adj)), X=int(surplus(adj)),
                ring_bundle=maps is not None, monodromy=mono,
                lines=[length * c for c in cycle_lengths(mono)] if mono else None,
                surface=surface(adj, length) if maps else None)


def reach(length, lam=LAM, cap=200):
    """Breadth-first over classes up to side-preserving renaming: start at the 4 x L tube, step by every switch with
    dH = 0 at lambda out of one representative of each class. Returns (classes, cap_hit, moves): classes maps a
    canonical key to (distance, representative, every valid switch out of it); moves[key] counts the zero-cost
    switches out of that class by the class they lead to (None where the cap stopped the search)."""
    adj, part = build(4, length)
    part = np.asarray(part, dtype=np.int64)
    key0 = symmetry.canonical_key(adj, part)
    classes = {key0: (0, adj, None)}
    moves = {}
    frontier = [key0]
    hit = False
    while frontier:
        nxt = []
        for key in frontier:
            dist, rep, _ = classes[key]
            rows, _ = every_switch(rep, part)
            classes[key] = (dist, rep, rows)
            moves[key] = Counter()
            for r in rows:
                if cost(int(r[4]), int(r[5]), lam) != 0:
                    continue
                new = rep.copy()
                _switch(new, int(r[0]), int(r[1]), int(r[2]), int(r[3]))
                k = symmetry.canonical_key(new, part)
                if k not in classes:
                    if len(classes) >= cap:
                        hit = True
                        moves[key][None] += 1
                        continue
                    classes[k] = (dist + 1, new, None)
                    nxt.append(k)
                moves[key][k] += 1
        frontier = nxt
    return classes, hit, moves


def family_free_energy(length, couplings=COUPLINGS, lam=LAM):
    """The one-switch free energy round the whole zero-cost family of the 4 x L tube, per labelled tube:
    F(g) = -g ln sum over the family's classes c of (A_tube / A_c) (1 + sum over switches out of c with dH > 0 of
    exp(-dH / g)). A_tube / A_c is the class's labelled copies over the tube's (Q9); zero-cost switches are not
    counted as excitations, because they lead to members of the family, each counted once. Returns {g: F}."""
    classes, hit, _ = reach(length, lam=lam)
    assert not hit
    part = build(4, length)[1]
    a_tube = symmetry.count(*build(4, length))
    terms = []
    for _dist, rep, rows in classes.values():
        dh = [cost(int(r[4]), int(r[5]), lam) for r in rows]
        terms.append((a_tube / symmetry.count(rep, part), [d for d in dh if d > 0], [d for d in dh if d < 0]))
    assert all(not neg for _, _, neg in terms)                  # nothing in the family has a way downhill
    return {g: -g * math.log(sum(w * (1.0 + sum(math.exp(-d / g) for d in pos)) for w, pos, _ in terms))
            for g in couplings}


def family_exits(length, lam, lowest=3):
    """For each class of the tube's zero-cost family at lambda = 1 (in reach's order), the costs at `lam` of every
    valid switch out of its representative: {dH: count} for the `lowest` distinct costs. The family is the one found
    at lambda = 1; only the costs are read at `lam`."""
    classes, hit, _ = reach(length)
    assert not hit
    out = []
    for key in sorted(classes, key=lambda k: classes[k][0]):
        h = Counter(cost(int(r[4]), int(r[5]), lam) for r in classes[key][2])
        out.append(dict(sorted(h.items())[:lowest]))
    return out


# ---------------------------------------------------------------- printing

def print_tables(sizes):
    print("The lattice tori at one N, lambda = %.0f: counts and every single switch (exact)" % LAM)
    for n in sizes:
        rows = [counts(a, b) for a, b in tori_of(n)]
        print("\nN = %d: %d lattice tori; n = %d per side" % (n, len(rows), n // 2))
        print("  %-9s %5s %6s %6s %6s %13s" % ("torus", "H", "S/N", "A", "A/N", "log10 copies"))
        for r in rows:
            print("  %-9s %5.0f %6.3f %6d %6.0f %13.1f" % ("%d x %d" % (r["a"], r["b"]), r["H"], r["S_per_point"],
                                                       r["A"], r["A"] / n, r["log10_copies"]))
        hists = []
        print("  %-9s %7s %7s   valid switches by cost dH (count at each cost)" % ("torus", "pairs", "valid"))
        for a, b in tori_of(n):
            adj, part = build(a, b)
            rows_s, tried = every_switch(adj, part)
            hist = cost_histogram(adj, part)
            hists.append(((a, b), hist))
            print("  %-9s %7d %7d   %s" % ("%d x %d" % (a, b), tried, len(rows_s),
                                          ", ".join("%+d: %d" % (dh, k) for dh, k in hist.items())))
        print("  %-9s " % "F(g)" + " ".join("%9s" % ("g=%.1f" % g) for g in COUPLINGS))
        for (a, b), hist in hists:
            print("  %-9s " % ("%d x %d" % (a, b)) + " ".join("%9.3f" % free_energy(hist, g) for g in COUPLINGS))
        tube = [h for (a, b), h in hists if a == 4]
        if tube:
            print("  F(tube) - F(other), as in the look (negative where the tube's is lower):")
            for (a, b), hist in hists:
                if a != 4:
                    print("  %-9s " % ("vs %d x %d" % (a, b)) +
                          " ".join("%9.3f" % (free_energy(tube[0], g) - free_energy(hist, g)) for g in COUPLINGS))
            # the same comparison with the tube's zero-cost neighbours counted as ground states, once each
            fam = family_free_energy(n // 4)
            a_tube = rows[0]["A"]
            print("  F(tube's zero-cost family) - F(other), each per labelled tube, labelled copies weighed by 1/A:")
            for r, ((a, b), hist) in zip(rows, hists):
                if a != 4:
                    assert 0 not in hist
                    w = a_tube / r["A"]
                    print("  %-9s " % ("vs %d x %d" % (a, b)) + " ".join(
                        "%9.3f" % (fam[g] + g * math.log(w * (1.0 + sum(k * math.exp(-dh / g) for dh, k in hist.items()))))
                        for g in COUPLINGS))


def print_zero(lengths):
    for length in lengths:
        out, orbits, tube_surface = read_zero_cost(length)
        print("\n4 x %d tube (N = %d): %d zero-cost switches at lambda = 1; %d orbit(s) under the tube's %d renamings;"
              " the tube's own faces: closed %s, V - E + F = %d, orientable %s"
              % (length, 4 * length, len(out), len(orbits), 8 * length, *tube_surface))
        for r in out:
            print("  remove %s-%s, %s-%s; add %s-%s, %s-%s | dS %d dX %d | A %d | tube? %s | monodromy %s, lines %s |"
                  " faces: closed %s, chi %d, orientable %s"
                  % (*r["removed"][0], *r["removed"][1], *r["added"][0], *r["added"][1], r["dS"], r["dX"], r["A"],
                     r["tube"], r["monodromy"], r["lines"], *r["surface"]))
        r = out[0]
        print("  first switch: squares lost %s" % r["lost"])
        print("                squares gained %s" % r["gained"])
        stab = switch_stabilizer(length, zero_cost_switches(length)[0])
        print("                %d renamings fix it; y fixed in each: %s; their action on the ring (x = 0..3 go to): %s"
              % (len(stab), all(q[1] == p[1] for s in stab for p, q in s.items()),
                 sorted(tuple(s[x, 0][0] for x in range(4)) for s in stab)))


def print_reach(lengths, cap):
    for length in lengths:
        classes, hit, moves = reach(length, cap=cap)
        print("\n4 x %d tube (N = %d): %d class(es) with H = 0 reachable by zero-cost switches, up to side-preserving"
              " renaming; cap %d, hit: %s" % (length, 4 * length, len(classes), cap, hit))
        part = build(4, length)[1]
        order = sorted(classes, key=lambda k: classes[k][0])
        for i, key in enumerate(order):
            dist, rep, rows = classes[key]
            d = describe(rep, part, length)
            out = moves.get(key)
            out = ", ".join("%d to class %s" % (k, order.index(t) if t is not None else "beyond the cap")
                            for t, k in out.items()) if out is not None else "not expanded"
            kinds = dict(Counter((int(r[4]), int(r[5])) for r in rows if cost(int(r[4]), int(r[5])) == 0)) \
                if rows is not None else "not expanded"
            print("  class %d, distance %d: A %d, S %d, X %d, ring bundle %s, monodromy %s, lines along the tube %s,"
                  " faces (closed, V - E + F, orientable) %s; zero-cost switches out: %s; their (dS, dX): %s"
                  % (i, dist, d["A"], d["S"], d["X"], d["ring_bundle"], d["monodromy"], d["lines"], d["surface"], out,
                     kinds))


def print_exits(lengths, lam):
    for length in lengths:
        print("\n4 x %d tube (N = %d): the lowest switch costs at lambda = %g out of each class of the zero-cost family"
              " found at lambda = 1 ({dH: count}; 12L = %d)" % (length, 4 * length, lam, 12 * length))
        for i, h in enumerate(family_exits(length, lam)):
            print("  class %d: %s" % (i, ", ".join("%g: %d" % (dh, k) for dh, k in h.items())))


def main(argv):
    cap, lam = 200, 1.25
    for a in argv:
        if a.startswith("--cap="):
            cap = int(a.split("=")[1])
        if a.startswith("--lam="):
            lam = float(a.split("=")[1])
    args = [a for a in argv if not a.startswith(("--cap=", "--lam="))]
    if args and args[0] == "--zero":
        print_zero([int(x) for x in args[1:]] or [8])
    elif args and args[0] == "--reach":
        print_reach([int(x) for x in args[1:]] or [6, 8, 10, 12], cap)
    elif args and args[0] == "--exits":
        print_exits([int(x) for x in args[1:]] or [8, 12, 24], lam)
    else:
        print_tables([int(x) for x in args] or [144, 160, 192])


if __name__ == "__main__":
    main(sys.argv[1:])
