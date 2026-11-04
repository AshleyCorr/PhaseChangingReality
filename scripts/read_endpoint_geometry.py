"""Retrospective endpoint checks (O122); no Monte Carlo dynamics.

Certify adjacency as a square-grid translation quotient, then measure its
shortest noncontractible loop and shortest linearly independent loop in edges.
These are finite-graph certificates, not physical lengths or an asymptotic law.
"""
from pathlib import Path
import argparse
import collections
import csv
import hashlib
import itertools
import json
import math
import numpy as np

def brute(adj):
    a = [set(map(int,r)) for r in adj]; n=len(a)
    squares=set()
    for v in range(n):
        for u,w in itertools.combinations(a[v],2):
            for z in (a[u]&a[w])-{v}:
                squares.add(tuple(sorted((tuple(sorted((v,u))),tuple(sorted((u,z))),tuple(sorted((z,w))),tuple(sorted((w,v)))))))
    edge=collections.Counter(e for q in squares for e in q)
    d=np.array([sum(not ((a[u]&a[w])-{v}) for u,w in itertools.combinations(a[v],2)) for v in range(n)])
    return len(squares),sum(max(v-2,0) for v in edge.values()),d

def validate(adj):
    n=len(adj); a=[set(map(int,r)) for r in adj]
    if adj.shape!=(n,4): return False
    if n == 0 or any(w < 0 or w >= n for row in a for w in row): return False
    for v,row in enumerate(a):
        if len(row)!=4 or v in row or any(w<0 or w>=n or v not in a[w] for w in row): return False
        c=collections.Counter(z for w in row for z in a[w] if z!=v)
        if max(c.values())>2: return False
    colors={}
    for v in range(n):
        if v in colors: continue
        colors[v]=0; q=[v]
        for u in q:
            for w in a[u]:
                if w not in colors: colors[w]=1-colors[u]; q.append(w)
                elif colors[w]==colors[u]: return False
    return True

def geometry(adj):
    """Certify square-grid quotient by consistent parallel frames, period lattice and adjacency reconstruction."""
    a=[set(map(int,r)) for r in adj]; n=len(a)
    frame=[None]*n; pos=[None]*n
    e=min(a[0]); w=next(v for v in a[0]-{e} if not ((a[v]&a[e])-{0}))
    north=min(a[0]-{e,w}); south=next(iter(a[0]-{e,w,north}))
    frame[0]=[e,north,w,south]; pos[0]=(0,0); queue=[0]; periods=[]
    dirs=[(1,0),(0,1),(-1,0),(0,-1)]
    for u in queue:
        for k,v in enumerate(frame[u]):
            f=[None]*4; f[(k+2)%4]=u
            for j in ((k+1)%4,(k+3)%4):
                cross=(a[v]&a[frame[u][j]])-{u}
                if len(cross)!=1: raise ValueError('nonunique square')
                f[j]=next(iter(cross))
            rem=a[v]-set(f)
            if len(rem)!=1: raise ValueError('not a grid vertex')
            f[k]=next(iter(rem))
            p=(pos[u][0]+dirs[k][0],pos[u][1]+dirs[k][1])
            if frame[v] is None: frame[v]=f; pos[v]=p; queue.append(v)
            elif frame[v]!=f: raise ValueError('nontranslation holonomy')
            else:
                d=(p[0]-pos[v][0],p[1]-pos[v][1])
                if d!=(0,0): periods.append(d)
    if len(queue)!=n: raise ValueError('disconnected')
    periods=list(set(periods))
    area=0
    for (x,y),(z,w) in itertools.combinations(periods,2): area=math.gcd(area,abs(x*w-y*z))
    if area!=n: raise ValueError(('area',area,n))
    # Construct a lattice vector (b,c) whose y coordinate is gcd of all period y values.
    b=c=0
    def egcd(a,b):
        if b==0: return abs(a),(1 if a>=0 else -1),0
        g,x,y=egcd(b,a%b); return g,y,x-(a//b)*y
    for x,y in periods:
        g,s,t=egcd(c,y); b=s*b+t*x; c=g
    horizontal=area//c; b%=horizontal
    def coset(p):
        x,y=p; q,r=divmod(y,c); return ((x-q*b)%horizontal,r)
    labels={coset(p):i for i,p in enumerate(pos)}
    if len(labels)!=n: raise ValueError('coordinate collision')
    for i,p in enumerate(pos):
        if {labels[coset((p[0]+dx,p[1]+dy))] for dx,dy in dirs}!=a[i]: raise ValueError('reconstruction')
    vectors=[]
    for ky in range(-n//c,n//c+1):
        for kx in range(-n//horizontal-abs(ky*b)//horizontal-2,n//horizontal+abs(ky*b)//horizontal+3):
            x=kx*horizontal+ky*b; y=ky*c
            if (x or y) and abs(x)+abs(y)<=n: vectors.append((abs(x)+abs(y),x,y))
    vectors.sort(); shortest=vectors[0]; second=next(v for v in vectors if v[1]*shortest[2]!=v[2]*shortest[1])
    return dict(lattice=[horizontal,b,c],systole=shortest[0],independent=second[0],short_vector=shortest[1:])


def topology(adj):
    """Fill every four-cycle; certify a closed surface before naming its type."""
    if not validate(adj):
        raise ValueError('invalid ensemble graph')
    a = [set(map(int, row)) for row in adj]
    faces = {}
    for v, row in enumerate(a):
        for u, w in itertools.combinations(row, 2):
            for z in (a[u] & a[w]) - {v}:
                faces[tuple(sorted((v, u, z, w)))] = [v, u, z, w]
    edge_faces = collections.defaultdict(list)
    links = [collections.defaultdict(list) for _ in a]
    for i, cycle in enumerate(faces.values()):
        for k, v in enumerate(cycle):
            u, w = cycle[k-1], cycle[(k+1) % 4]
            links[v][u].append(w)
            links[v][w].append(u)
            edge_faces[tuple(sorted((v, w)))].append((i, 1 if v < w else -1))
    edges = {tuple(sorted((v, w))) for v, row in enumerate(a) for w in row}
    if set(edge_faces) != edges or any(len(fs) != 2 for fs in edge_faces.values()):
        raise ValueError('not two faces per edge')
    for v, link in enumerate(links):
        if set(link) != a[v] or any(len(row) != 2 for row in link.values()):
            raise ValueError('vertex link is not a cycle')
        seen = {next(iter(link))}
        queue = list(seen)
        for u in queue:
            for w in link[u]:
                if w not in seen:
                    seen.add(w)
                    queue.append(w)
        if len(seen) != len(link):
            raise ValueError('disconnected vertex link')
    dual = collections.defaultdict(list)
    for (i, si), (j, sj) in edge_faces.values():
        dual[i].append((j, -si*sj))
        dual[j].append((i, -si*sj))
    signs = {}
    orientable = True
    for i in range(len(faces)):
        if i in signs:
            continue
        signs[i] = 1
        queue = [i]
        for j in queue:
            for k, sign in dual[j]:
                if k not in signs:
                    signs[k] = sign*signs[j]
                    queue.append(k)
                elif signs[k] != sign*signs[j]:
                    orientable = False
    seen = set()
    components = []
    for v in range(len(a)):
        if v in seen:
            continue
        queue = [v]
        seen.add(v)
        for u in queue:
            for w in a[u]:
                if w not in seen:
                    seen.add(w)
                    queue.append(w)
        components.append(len(queue))
    chi = len(a)-len(edges)+len(faces)
    kind = 'disconnected surface'
    if len(components) == 1:
        kind = ('torus' if orientable else 'Klein bottle') if chi == 0 else 'other surface'
    return dict(kind=kind, components=sorted(components), orientable=orientable, chi=chi)


def inspect(adj):
    result = topology(adj)
    if result['kind'] == 'torus':
        result.update(geometry(adj))
    return result


def paper1(root):
    """Fixed retrospective selection: original T8 and reached, H=0 T60 endpoints."""
    records = []
    t8 = sorted((root/'results/t8_lam105_adj').glob('*.npz'))
    if len(t8) != 120:
        raise ValueError('expected 120 original T8 endpoints')
    selected = [(p, 'T8') for p in t8]
    cells = sorted((root/'results').glob('t60_lam*_n*.csv'))
    if len(cells) != 35:
        raise ValueError('expected 35 T60 cells at base 5382053a')
    for path in cells:
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 120:
            raise ValueError(f'expected 120 replicas: {path}')
        for row in rows:
            if float(row['reached'] or 0) < .75:
                continue
            p = path.parent/(path.stem+'_adj')/f'N{row["N"]}_rep{row["replica"]}.npz'
            adj = np.load(p)['adj']
            if not validate(adj):
                raise ValueError(f'invalid graph: {p}')
            squares, surplus, _ = brute(adj)
            # lambda > 1: zero energy iff each edge has exactly two squares.
            if squares == len(adj) and surplus == 0:
                selected.append((p, 'T60'))
    for path, dataset in selected:
        adj = np.load(path)['adj']
        records.append(dict(file=path.relative_to(root).as_posix(), dataset=dataset,
            N=len(adj), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), **inspect(adj)))
    return dict(selection='T8 lambda=1.05 all 120; T60 reached>=0.75, S=N and X=0; 35 cells',
                retrospective=True, records=records,
                counts={ds: dict(collections.Counter(r['kind'] for r in records if r['dataset']==ds))
                        for ds in ('T8', 'T60')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = paper1(args.root)
    # Refuse to replace an earlier audit artifact.
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result['counts'], indent=2))
