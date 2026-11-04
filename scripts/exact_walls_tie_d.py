"""EXACT, exploratory: the ladders and walls of the six- and eight-link models with the direction tie of VISION
Update 30 (the owner's decision of 2026-09-25; its first form fixed there before any run). Usage:

    python scripts/exact_walls_tie_d.py 4,4,18 4,12,12 6,6,8 4,4,4x8
    python scripts/exact_walls_tie_d.py --lam=1.10,1.25 --kappa=0,0.5,1,2,4,8 --json=out.json 4,4,8,8 4,4,4,4x4

The knob. H_tie = kappa * T, T = sum over points v of d(v) (D - d(v)), where d = graphity.dimension.local_dimension_d
(the number of pairs of links at v that close no square: D on flat space, 0 in a cube) and D = links / 2. It is added
to the model energy H = 16 (D(D-1)/2 N - S) + 4 lam X (ASSUMPTIONS Q21). At kappa != 0 (or lam != 1) this is our
family around the model, never combinatorial quantum gravity (VISION Updates 22 and 30).

The enumeration is scripts/exact_walls_d.py's, which is imported and not changed: one u1 (every side-0 vertex of a
torus is equivalent under even translations, and every cube of a gas under permutation), u2 over every side-0 vertex,
v1 and v2 over the neighbor slots, the same exclusions. Two differences, both towards exactness. (1) No nearby-partner
shortcut: exact_walls_d.py's `near` argument rests on squares (a switch whose edges share no square loses every square
on both), and the tie term prices points, not squares, so every partner is searched here. (2) Every candidate is priced
from whole-graph recomputations before and after the switch: the hard-core rule (cqg_d.is_valid; the kernel's local
check _new_edge_ok is used first, only to reject), S, X, and d at every point. A switch's signature is then exact:
(dS, dX, dT, and the change in the number of points at each d), and its cost at any (lam, kappa) is

    -16 dS + 4 lam dX + kappa dT.

A property of the fixed form that the results turn on (ours; arithmetic; the numbers are in
docs/design/direction_tie_first_look.md): d counts pairs of links, and at a
damaged point (a pair that closed a square no longer does, or a new link closes nothing) it can exceed D, up to
2D(2D - 1)/2. Then d (D - d) is negative, so the literal form REWARDS damage. The script also reports, as a diagnostic
only, T+ = sum_v max(0, d (D - d)), the same form with a damaged point counted as zero (as if clipped to all-open). T+
is NOT the form fixed in Update 30; which reading of a damaged point is meant is the owner's call under S1.

Written 2026-09-25 for Update 30's first exact test; tests in tests/test_exact_walls_tie_d.py.
"""
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from numba import njit, prange

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from graphity.cqg_d import (_new_edge_ok, _switch, has_edge, hamiltonian, is_valid, surplus,   # noqa: E402
                            total_squares)
from graphity.dimension import local_dimension_d                                            # noqa: E402

_SPEC = importlib.util.spec_from_file_location("exact_walls_d", ROOT / "scripts" / "exact_walls_d.py")
walls_d = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(walls_d)
parse = walls_d.parse          # "4,4,18" a torus, "4,4,4x8" a gas of eight separate 4 x 4 x 4 tori


@njit(cache=True)
def _tie_sums(d, dim):
    """(T, T+): sum of d (D - d), and of its positive part."""
    t = 0
    tp = 0
    for v in range(d.shape[0]):
        x = d[v] * (dim - d[v])
        t += x
        if x > 0:
            tp += x
    return t, tp


@njit(cache=True)
def _histogram(d, size):
    h = np.zeros(size, dtype=np.int64)
    for v in range(d.shape[0]):
        h[d[v]] += 1
    return h


@njit(parallel=True, cache=True)
def _enumerate(adj, u1, partners):
    """Every valid switch (u1, v1), (u2, v2) -> (u1, v2), (u2, v1) with u2 in partners, priced from scratch.

    Returns ok[i, a, b] (valid), and dS, dX, dT, dT+ and the change of the d-histogram for each (u2 = partners[i],
    v1 = adj[u1, a], v2 = adj[u2, b])."""
    n, deg = adj.shape
    dim = deg // 2
    size = deg * (deg - 1) // 2 + 1
    m = partners.shape[0]
    ok = np.zeros((m, deg, deg), dtype=np.bool_)
    ds = np.zeros((m, deg, deg), dtype=np.int64)
    dx = np.zeros((m, deg, deg), dtype=np.int64)
    dt = np.zeros((m, deg, deg), dtype=np.int64)
    dtp = np.zeros((m, deg, deg), dtype=np.int64)
    dh = np.zeros((m, deg, deg, size), dtype=np.int64)
    s0 = total_squares(adj)
    x0 = surplus(adj)
    d0 = local_dimension_d(adj)
    t0, tp0 = _tie_sums(d0, dim)
    h0 = _histogram(d0, size)
    for i in prange(m):
        u2 = partners[i]
        if u2 == u1:
            continue
        t = adj.copy()
        for a in range(deg):
            v1 = adj[u1, a]
            if has_edge(adj, u2, v1):
                continue
            for b in range(deg):
                v2 = adj[u2, b]
                if v2 == v1 or has_edge(adj, u1, v2):
                    continue
                _switch(t, u1, v1, u2, v2)
                if _new_edge_ok(t, u1, v2) and _new_edge_ok(t, u2, v1) and is_valid(t):
                    d1 = local_dimension_d(t)
                    t1, tp1 = _tie_sums(d1, dim)
                    h1 = _histogram(d1, size)
                    ok[i, a, b] = True
                    ds[i, a, b] = total_squares(t) - s0
                    dx[i, a, b] = surplus(t) - x0
                    dt[i, a, b] = t1 - t0
                    dtp[i, a, b] = tp1 - tp0
                    for k in range(size):
                        dh[i, a, b, k] = h1[k] - h0[k]
                _switch(t, u1, v2, u2, v1)          # the exact reverse, slots included
    return ok, ds, dx, dt, dtp, dh


def kinds(adj, part, u1=None):
    """Counter {(dS, dX, dT, dT+, d-change): ways from one u1}; d-change is a tuple of (d, change in count)."""
    side0 = np.flatnonzero(part == 0).astype(np.int64)
    u1 = int(side0[0]) if u1 is None else int(u1)
    ok, ds, dx, dt, dtp, dh = _enumerate(adj, u1, side0)
    out = Counter()
    for i, a, b in zip(*np.nonzero(ok)):
        change = tuple((int(k), int(c)) for k, c in enumerate(dh[i, a, b]) if c != 0)
        out[(int(ds[i, a, b]), int(dx[i, a, b]), int(dt[i, a, b]), int(dtp[i, a, b]), change)] += 1
    return out


def rung(adj, lam):
    """(N, D, H at lam, T, T+, the d-histogram) of the state itself."""
    n, deg = adj.shape
    dim = deg // 2
    d = local_dimension_d(adj)
    t, tp = _tie_sums(d, dim)
    hist = {int(k): int(c) for k, c in enumerate(np.bincount(d)) if c}
    return n, dim, float(hamiltonian(adj, lam)), int(t), int(tp), hist


def cost(kind, lam, kappa, plus=False):
    ds, dx, dt, dtp, _ = kind
    return -16.0 * ds + 4.0 * lam * dx + kappa * (dtp if plus else dt)


def is_null(kind):
    """A switch that changes nothing this script measures (S, X, and the number of points at each d). It costs 0 at
    every (lam, kappa) and does not leave the rung; the 16 x 4 torus has one (2 ways from one u1), and
    exact_walls_d.py lists it too, while paper 1's wall of 12 is the cheapest move that is not of this kind. Such
    switches are counted and reported, and left out of the walls."""
    return kind[0] == 0 and kind[1] == 0 and kind[2] == 0 and not kind[4]


def wall(kinds_, lam, kappa, plus=False):
    """(cost, kind, ways) of the cheapest switch that is not null; ties broken by the kind's signature."""
    best = min((k for k in kinds_ if not is_null(k)), key=lambda k: (cost(k, lam, kappa, plus), k))
    return cost(best, lam, kappa, plus), best, kinds_[best]


def positive_set(kinds_, lam, plus=False):
    """The exact set of kappa >= 0 at which every switch costs more than zero, as (lo, hi, lo_open, hi_open) or None.

    It is the intersection over kinds of the half-lines {c + kappa t > 0}, c the cost at kappa = 0 and t = dT (or dT+)."""
    lo, hi = 0.0, float("inf")
    lo_open = False
    for k in kinds_:
        if is_null(k):
            continue
        c = cost(k, lam, 0.0)
        t = k[3] if plus else k[2]
        if t == 0:
            if c <= 0:
                return None
        elif t > 0:                    # c + kappa t > 0  <=>  kappa > -c / t
            b = -c / t + 0.0           # + 0.0: no "-0.0" when c = 0
            if b >= lo:
                lo, lo_open = b, True
        else:                          # kappa < -c / t
            b = -c / t
            if b < hi:
                hi = b
    if hi < lo or (hi == lo):
        return None
    return lo, hi, lo_open


def describe(kind):
    ds, dx, dt, dtp, change = kind
    ch = ", ".join("d=%d %+d" % (k, c) for k, c in change)
    return "dS %d, dX %d, dT %d, dT+ %d; points: %s" % (ds, dx, dt, dtp, ch)


def main(argv):
    lams, kappas, out_json = [1.10, 1.25], [0.0, 0.5, 1.0, 2.0, 4.0, 8.0], None
    specs = []
    for a in argv:
        if a.startswith("--lam="):
            lams = [float(x) for x in a.split("=")[1].split(",")]
        elif a.startswith("--kappa="):
            kappas = [float(x) for x in a.split("=")[1].split(",")]
        elif a.startswith("--json="):
            out_json = a.split("=", 1)[1]
        else:
            specs.append(a)
    record = {}
    for spec in specs:
        (adj, part), label = parse(spec)
        k = kinds(adj, part)
        n, dim, _, t, tp, hist = rung(adj, 1.0)
        entry = {"N": n, "D": dim, "T": t, "T_plus": tp, "d_histogram": hist, "kinds": [], "rungs": {}, "walls": {}}
        null_ways = sum(c for q, c in k.items() if is_null(q))
        entry["null_ways"] = null_ways
        print("%-12s N=%-5d D=%d  T=%d (%.3f per point), T+=%d, d-histogram %s, %d kinds of valid switch, "
              "%d null switches from one u1" % (label, n, dim, t, t / n, tp, hist, len(k), null_ways))
        for lam in lams:
            h = rung(adj, lam)[2]
            entry["rungs"][str(lam)] = {"H": h, "H_per_point": h / n}
            print("  lambda=%.2f  H above flat %.3f per point; with the tie %.3f + %.3f kappa per point"
                  % (lam, h / n, h / n, t / n))
            for plus in (False, True):
                ps = positive_set(k, lam, plus)
                name = "T+ (diagnostic)" if plus else "T (fixed form)"
                if ps is None:
                    txt = "never (some switch is downhill or free at every kappa >= 0)"
                else:
                    lo, hi, lo_open = ps
                    txt = "kappa in %s%.4f, %s)" % ("(" if lo_open else "[", lo, "inf" if hi == float("inf") else "%.4f" % hi)
                print("    %-16s every switch costs more than zero for %s" % (name, txt))
                entry["walls"]["%s|%s|positive_set" % (lam, "plus" if plus else "T")] = ps
                for kap in kappas:
                    c, kind, ways = wall(k, lam, kap, plus)
                    entry["walls"]["%s|%s|%s" % (lam, "plus" if plus else "T", kap)] = {
                        "cost": c, "kind": [kind[0], kind[1], kind[2], kind[3], [list(p) for p in kind[4]]],
                        "ways": ways}
                    print("      kappa=%-4g wall %8.2f  (%s; %d ways)" % (kap, c, describe(kind), ways))
        every = sorted(k, key=lambda q: (cost(q, lams[-1], 0.0), q))
        entry["kinds"] = [{"kind": [q[0], q[1], q[2], q[3], [list(p) for p in q[4]]], "ways": k[q]} for q in every]
        record[label] = entry
    if out_json:
        with open(out_json, "w") as f:
            json.dump(record, f, indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
