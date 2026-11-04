"""T69: the lowest path out of the owner's slab. Does the energy released by the first steps of an opening pay for the
next steps, or does the path climb? Usage:

    python scripts/exact_lowest_path.py configs/t69_lowest_path_l8.json [out_dir]

Implements PREREGISTRATION.md section T69 (written 2026-10-17 on the owner's "I think yes, can we check?", before any
computation). Exact within its stated move set and budget; nothing random.

THE SEARCH. Start: the untied slab cqg_d.torus([4, L, L]) (axis 0 curled), energy E0. A state is a wiring; a step is one
switch of the kernel's kind, (u1, v1), (u2, v2) -> (u1, v2), (u2, v1), with u1 and u2 on side 0, valid by the hard-core
rule, and LOCAL: both u1 and u2 lie within graph distance `near` of a point the path has already changed (at the start,
within `near` of one seed point and its six neighbours). The energy of a path is the list of E - E0 after each step; its SADDLE is the highest
value on the list. A best-first search on the saddle (Dijkstra on the bottleneck) pops states in order of the lowest
saddle at which they can be reached, so the first time a state with a given property is popped, its saddle is the
lowest over every path the move set allows, exactly, as far as the search has gone. The search stops when `budget`
states have been popped or no state is left; a child whose energy exceeds `cap` is not kept, since it could never be popped before the cap.

WHAT IS REPORTED. For every state popped: its energy above the slab, its saddle, the number of steps, the points at
each local dimension (open = 3, curled = 2, broken = above 3). The MILESTONES, per run: for m in `milestones`, the
lowest saddle at which a state with at least m points open and no point broken was reached, S(m), with that state's
energy and step count; and the same with broken points allowed, reported beside it. The exhausted saddle: the largest
saddle below which every reachable state has been popped (the saddle of the last popped state when the budget ends).

THE LOCAL ENERGY. A switch changes the square count only on edges that share a square with one of its four edges; every
such edge has both ends within one step of {u1, v1, u2, v2} as read before the switch. So dH is computed on that set
of points alone, before and after, and checked against the full energy in the tests (rule 6). H = 16 (3N - S) + 4 lam X
with X the surplus over 4 squares per edge (Q21); untied, as the slab stands since VISION Update 58.
"""
import hashlib
import heapq
import json
import platform
import sys
from pathlib import Path

import numba
import numpy as np
from numba import njit

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphity import __version__                                              # noqa: E402
from graphity.cqg_d import _switch, hamiltonian, has_edge, is_valid, squares_on_edge, torus   # noqa: E402
from graphity.dimension import local_dimension_d                              # noqa: E402
from graphity.results import ResultWriter, provenance                         # noqa: E402

SAT = 4            # squares per edge in flat six-link space; surplus counts above it (Q21)
MAX_CODEGREE = 2


@njit(cache=True)
def _codeg(adj, x, y):
    c = 0
    for i in range(adj.shape[1]):
        a = adj[x, i]
        if a >= 0 and has_edge(adj, y, a):
            c += 1
    return c


@njit(cache=True)
def _ok_around(adj, pts):
    """Simple at pts, and the hard-core rule for every pair at distance two through a point of pts."""
    deg = adj.shape[1]
    for t in range(pts.shape[0]):
        p = pts[t]
        for i in range(deg):
            if adj[p, i] == p:
                return False
            for j in range(i + 1, deg):
                if adj[p, i] == adj[p, j]:
                    return False
        for i in range(deg):
            m = adj[p, i]
            for j in range(deg):
                w = adj[m, j]
                if w != p and _codeg(adj, p, w) > MAX_CODEGREE:
                    return False
    return True


@njit(cache=True)
def _local_energy(adj, pts, npts, lam):
    """16 (-S) + 4 lam X summed over the edges with both ends among pts[:npts] (each edge once)."""
    deg = adj.shape[1]
    mark = np.zeros(adj.shape[0], dtype=np.uint8)
    for t in range(npts):
        mark[pts[t]] = 1
    s4 = 0.0
    x = 0.0
    for t in range(npts):
        p = pts[t]
        for i in range(deg):
            q = adj[p, i]
            if q > p and mark[q]:
                c = squares_on_edge(adj, p, q)
                s4 += c                       # each square counted once per edge; S = sum_e S_e / 4
                if c > SAT:
                    x += c - SAT
    return -16.0 * s4 / 4.0 + 4.0 * lam * x


@njit(cache=True)
def _patch_points(adj, u1, v1, u2, v2, out):
    """The four points and their neighbours (before the switch), without repeats; returns the count."""
    deg = adj.shape[1]
    n = 0
    seeds = (u1, v1, u2, v2)
    for s in seeds:
        found = False
        for t in range(n):
            if out[t] == s:
                found = True
        if not found:
            out[n] = s
            n += 1
    for s in seeds:
        for i in range(deg):
            q = adj[s, i]
            found = False
            for t in range(n):
                if out[t] == q:
                    found = True
            if not found:
                out[n] = q
                n += 1
    return n


@njit(cache=True)
def expand(adj, window, lam):
    """Every valid switch with u1 < u2 both in `window` (side-0 points): arrays u1, v1, u2, v2, dH."""
    deg = adj.shape[1]
    cap = window.shape[0] * window.shape[0] * deg * deg
    ou1 = np.empty(cap, dtype=np.int64)
    ov1 = np.empty(cap, dtype=np.int64)
    ou2 = np.empty(cap, dtype=np.int64)
    ov2 = np.empty(cap, dtype=np.int64)
    odh = np.empty(cap, dtype=np.float64)
    pts = np.empty(4 + 4 * deg, dtype=np.int64)
    four = np.empty(4, dtype=np.int64)
    k = 0
    for i in range(window.shape[0]):
        u1 = window[i]
        for j in range(i + 1, window.shape[0]):
            u2 = window[j]
            for a in range(deg):
                v1 = adj[u1, a]
                for b in range(deg):
                    v2 = adj[u2, b]
                    if v1 == v2 or has_edge(adj, u1, v2) or has_edge(adj, u2, v1):
                        continue
                    npts = _patch_points(adj, u1, v1, u2, v2, pts)
                    before = _local_energy(adj, pts, npts, lam)
                    _switch(adj, u1, v1, u2, v2)
                    four[0] = u1
                    four[1] = u2
                    four[2] = v1
                    four[3] = v2
                    if _ok_around(adj, four):
                        after = _local_energy(adj, pts, npts, lam)
                        ou1[k] = u1
                        ov1[k] = v1
                        ou2[k] = u2
                        ov2[k] = v2
                        odh[k] = after - before
                        k += 1
                    _switch(adj, u1, v2, u2, v1)
    return ou1[:k], ov1[:k], ou2[:k], ov2[:k], odh[:k]


def ball(adj, anchors, radius):
    seen = {int(v) for v in anchors}
    edge = set(seen)
    for _ in range(radius):
        edge = {int(w) for v in edge for w in adj[v]} - seen
        seen |= edge
    return seen


def key_of(adj):
    """A 16-byte digest of the wiring with each point's partners sorted, so that slot order does not matter."""
    return hashlib.blake2b(np.sort(adj, axis=1).tobytes(), digest_size=16).digest()


def census(adj):
    d = local_dimension_d(adj)
    return int((d == 3).sum()), int((d == 2).sum()), int((d < 2).sum()), int((d > 3).sum())


def search(length, lam, near, budget, milestones, seed_point=None, record=None, cap=None):
    """Best-first search on the saddle. Returns (clean milestones, milestones with breaks, exhausted saddle, states
    popped, states seen, the slab's energy). Only popped states keep a copy of their wiring; a child is kept as its
    parent's key and the four points of the switch, and rebuilt from the parent when it is popped."""
    adj0, part = torus([4, length, length])
    e0 = float(hamiltonian(adj0, lam))
    side0 = set(int(v) for v in np.flatnonzero(part == 0))
    seed = int(sorted(side0)[0]) if seed_point is None else int(seed_point)
    start_key = key_of(adj0)
    first = frozenset([seed] + [int(w) for w in adj0[seed]])        # the seed and its neighbours
    popped_states = {start_key: (adj0, first)}
    pending = {}                                                      # key -> (parent key, u1, v1, u2, v2)
    best_saddle = {start_key: 0.0}
    counter = 0
    heap = [(0.0, 0.0, 0, counter, start_key)]
    done = set()
    table = {m: None for m in milestones}
    table_broken = {m: None for m in milestones}
    popped = 0
    exhausted = 0.0
    scratch = adj0.copy()
    while heap and popped < budget:
        saddle, energy, steps, _, key = heapq.heappop(heap)
        if key in done or saddle > best_saddle[key] + 1e-9:
            continue
        if key not in popped_states:
            pkey, u1, v1, u2, v2 = pending.pop(key)
            padj, ptouched = popped_states[pkey]
            adj = padj.copy()
            _switch(adj, u1, v1, u2, v2)
            touched = ptouched | {u1, v1, u2, v2}
            popped_states[key] = (adj, touched)
        else:
            adj, touched = popped_states[key]
        done.add(key)
        popped += 1
        exhausted = saddle
        op, cu, deeper, broken = census(adj)
        if record is not None:
            record(dict(saddle=saddle, energy=energy, steps=steps, open=op, curled=cu, deeper=deeper, broken=broken))
        for m in milestones:
            if op >= m:
                if broken == 0 and table[m] is None:
                    table[m] = (saddle, energy, steps)
                if table_broken[m] is None:
                    table_broken[m] = (saddle, energy, steps, broken)
        window = np.array(sorted(v for v in ball(adj, touched, near) if v in side0), dtype=np.int64)
        u1s, v1s, u2s, v2s, dhs = expand(adj, window, lam)
        if cap is not None:
            keep = energy + dhs <= cap + 1e-9                         # a child above the cap is never popped
            u1s, v1s, u2s, v2s, dhs = u1s[keep], v1s[keep], u2s[keep], v2s[keep], dhs[keep]
        scratch[:] = adj
        for u1, v1, u2, v2, dh in zip(u1s.tolist(), v1s.tolist(), u2s.tolist(), v2s.tolist(), dhs.tolist()):
            new_e = energy + dh
            new_s = max(saddle, new_e)
            _switch(scratch, u1, v1, u2, v2)
            ck = key_of(scratch)
            _switch(scratch, u1, v2, u2, v1)
            if ck in done:
                continue
            if ck not in best_saddle or new_s < best_saddle[ck] - 1e-9:
                best_saddle[ck] = new_s
                pending[ck] = (key, u1, v1, u2, v2)
                counter += 1
                heapq.heappush(heap, (new_s, new_e, steps + 1, counter, ck))
    return table, table_broken, exhausted, popped, len(best_saddle), e0


def main(path, out_dir="results"):
    cfg = json.loads(Path(path).read_text())
    meta = dict(config=cfg, config_path=str(path), package=__version__, python=platform.python_version(),
                numpy=np.__version__, numba=numba.__version__, **provenance(),
                preregistration="PREREGISTRATION.md section T69, written 2026-10-17")
    with ResultWriter(cfg["name"], meta, out_dir) as out:
        for length in cfg["lengths"]:
            for lam in cfg["lambdas"]:
                for near in cfg["near"]:
                    n = 4 * length * length
                    rows = []
                    table, table_broken, exhausted, popped, seen, e0 = search(
                        length, float(lam), int(near), int(cfg["budget"]), cfg["milestones"],
                        record=rows.append if cfg.get("record_states") else None, cap=cfg.get("cap"))
                    for m in cfg["milestones"]:
                        clean = table[m]
                        anyb = table_broken[m]
                        out.write(dict(N=n, L=length, lam=lam, near=near, budget=cfg["budget"], popped=popped, seen=seen,
                                       exhausted_saddle=exhausted, milestone=m,
                                       saddle=clean[0] if clean else "", energy=clean[1] if clean else "", steps=clean[2] if clean else "",
                                       saddle_any=anyb[0] if anyb else "", energy_any=anyb[1] if anyb else "",
                                       steps_any=anyb[2] if anyb else "", broken_any=anyb[3] if anyb else ""))
                        print("L=%d lam=%.2f near=%d m=%d | clean %s | with breaks %s | exhausted %.1f popped %d seen %d"
                              % (length, lam, near, m, clean, anyb, exhausted, popped, seen), flush=True)
                    if rows:
                        Path(out_dir, "%s_states_L%d_lam%g_near%d.json" % (cfg["name"], length, lam, near)).write_text(json.dumps(rows))
    print("wrote", Path(out_dir) / (cfg["name"] + ".csv"))


if __name__ == "__main__":
    main(sys.argv[1], *(sys.argv[2:3]))
