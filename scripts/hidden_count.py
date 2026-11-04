"""
hidden_count.py  --  exact "hidden count" of a region of a built torus.

Question (piece 8, from Jacobson/Verlinde): for a region R of a graph, how many
interior wirings look the same from outside R?  Does that count scale with the
links crossing R's boundary (area) or with the points inside (volume), and does
a relic inside R change it?

Definition used here (register before running):
  Given graph G (2D-regular, hard-core: no self-loops, no multi-edges) and a
  region R (a set of vertices), the HIDDEN COUNT of R is the number of edge
  sets W on R x R such that the graph G' = (G minus all interior edges of R)
  plus W
    (i)   keeps every vertex at degree 2D,
    (ii)  leaves every edge with one or both ends outside R unchanged,
    (iii) has the same total energy as G (energy function pluggable),
  and the HIDDEN ENTROPY is log of that count.  Isomorphic interior wirings are
  counted separately (they are different microstates behind the same cut).

Plug in the repo's energy via --energy module:function taking an adjacency
dict {v: set(neighbours)} and the dimension D, returning a float.  The default
energy counts the Trugenberger-style local terms only as a placeholder and is
NOT the repo's energy; results with it are a check of the script, not a result.

Usage:
  python hidden_count.py --edges torus.edgelist --D 2 --region 0,1,2,3,8,9,10,11
  python hidden_count.py --square 8 --D 2 --block 2x2     (builds a flat torus itself)
Exact; brute force over interior edge sets, so keep the interior small
(<= ~10 interior vertices, or use --max-edges to cap).
"""
import argparse, itertools, importlib, math, sys
from collections import defaultdict

def square_torus(L):
    adj = defaultdict(set)
    for x in range(L):
        for y in range(L):
            v = x*L + y
            for dx, dy in ((1,0),(0,1)):
                w = ((x+dx)%L)*L + (y+dy)%L
                adj[v].add(w); adj[w].add(v)
    return dict(adj)

def read_edges(path):
    adj = defaultdict(set)
    for line in open(path):
        p = line.split()
        if len(p) < 2 or p[0].startswith('#'): continue
        a, b = int(p[0]), int(p[1]); adj[a].add(b); adj[b].add(a)
    return dict(adj)

def default_energy(adj, D):
    # placeholder: -(number of squares) ; replace with the repo's energy
    sq = 0
    for u in adj:
        for v in adj[u]:
            if v <= u: continue
            # squares through edge uv: common neighbours of N(u)\{v} and N(v)\{u} pairs
            for a in adj[u]:
                if a == v: continue
                for b in adj[v]:
                    if b == u or b == a: continue
                    if b in adj[a]: sq += 1
    return -sq/8.0  # each square is met 4 times per edge-orientation pair; fine as a placeholder

def hidden_count(adj, region, D, energy, max_edges=None):
    region = set(region)
    E0 = energy(adj, D)
    # strip interior edges
    base = {v: set(adj[v]) for v in adj}
    for v in region:
        base[v] = {w for w in base[v] if w not in region}
    need = {v: 2*D - len(base[v]) for v in region}   # degree each interior vertex must regain
    if any(n < 0 for n in need.values()): raise ValueError("region has a vertex above degree 2D")
    total_needed = sum(need.values())
    if total_needed % 2: return 0, E0, []
    verts = sorted(region)
    need_left = dict(need)
    count, found = 0, []
    def energy_ok(W):
        g = {v: set(x) for v, x in base.items()}
        for a, b in W: g[a].add(b); g[b].add(a)
        return abs(energy(g, D) - E0) < 1e-9
    def rec(i, W):
        nonlocal count
        if i == len(verts):
            if energy_ok(W): count += 1; found.append(tuple(W))
            return
        v = verts[i]
        k = need_left[v]
        if k == 0: rec(i+1, W); return
        later = [w for w in verts[i+1:] if need_left[w] > 0]
        for pick in itertools.combinations(later, k):
            for w in pick: need_left[w] -= 1
            need_left[v] = 0
            rec(i+1, W + [(v, w) for w in pick])
            need_left[v] = k
            for w in pick: need_left[w] += 1
    rec(0, [])
    return count, E0, found

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--edges'); ap.add_argument('--square', type=int)
    ap.add_argument('--D', type=int, default=2)
    ap.add_argument('--region'); ap.add_argument('--block')
    ap.add_argument('--energy', help='module:function')
    ap.add_argument('--max-edges', type=int, default=40)
    a = ap.parse_args()
    adj = read_edges(a.edges) if a.edges else square_torus(a.square)
    if a.region: region = [int(x) for x in a.region.split(',')]
    elif a.block and a.square:
        bx, by = map(int, a.block.lower().split('x'))
        region = [x*a.square + y for x in range(bx) for y in range(by)]
    else: raise SystemExit("give --region or --block with --square")
    energy = default_energy
    if a.energy:
        mod, fn = a.energy.split(':'); energy = getattr(importlib.import_module(mod), fn)
    cut = sum(1 for v in region for w in adj[v] if w not in region)
    n, E0, found = hidden_count(adj, region, a.D, energy, a.max_edges)
    print(f"points in region: {len(region)}  links crossing the cut: {cut}  energy: {E0}")
    print(f"hidden count: {n}   hidden entropy (ln): {math.log(n) if n else float('-inf'):.4f}")
    for W in found[:10]: print("  wiring:", W)

if __name__ == '__main__':
    main()
